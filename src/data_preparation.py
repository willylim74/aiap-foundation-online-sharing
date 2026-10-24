"""Data preparation: row-level cleaning (run once, before the split) and the unfitted preprocessing pipeline.

Everything is driven by ``config.yaml``; nothing about the data is hard-coded here.

* ``load_config``                 - read the YAML configuration
* ``split_fractions``             - float-safe test sizes for the train/validation/test split
* ``load_raw`` / ``clean_rows`` / ``make_xy`` - row-level cleaning (structural fixes, invalid targets, de-duplication) and target engineering
* ``DataCleaner``                 - stateless column-level cleaning + missingness indicators (first pipeline step)
* ``FeatureEngineer``             - engineered features; the log-transform columns are learned in ``fit`` (training rows only)
* ``Winsorizer``                  - outlier clipping to training quantiles
* ``build_preprocessing_pipeline`` - returns an UNFITTED sklearn Pipeline, to be fitted afresh per model / fold

Run the embedded self-tests with:  ``python src/data_preparation.py``  (from any directory).
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_CONFIG = SRC_DIR / "config.yaml"


# ============================================================================== configuration
def load_config(path=None):
    """Read config.yaml (default: the one next to this file) and attach the resolved project root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(cfg, relative):
    """Resolve a config path relative to the project root (independent of the current working directory)."""
    p = Path(relative)
    return p if p.is_absolute() else Path(cfg["_project_root"]) / p


# ============================================================================== row-level cleaning
def load_raw(cfg):
    path = resolve_path(cfg, cfg["paths"]["data_file"])
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    return pd.read_csv(path)


def clean_rows(raw, cfg):
    """Structural fixes, invalid-target removal and de-duplication. Uses no statistics from the data, so it is safe before the split.

    Returns (cleaned_frame, report_dict).
    """
    ccfg, tcfg = cfg["cleaning"], cfg["target"]
    raw = raw.copy()
    report = {"rows_loaded": len(raw)}
    if ccfg.get("normalise_text", True):
        for c in cfg["columns"]["categorical"]:
            raw[c] = raw[c].str.strip().str.lower()
    if ccfg.get("drop_invalid_target", True):
        bad = raw[tcfg["name"]].isna() | (raw[tcfg["name"]] <= tcfg["min_valid_value"])
        report["removed_invalid_target"] = int(bad.sum())
        raw = raw.loc[~bad]
    dd = ccfg["deduplicate"]
    subset = [c for c in raw.columns if c not in dd.get("ignore_columns", [])]
    report["removed_exact_duplicates"] = int(raw.duplicated(subset=subset).sum())
    raw = raw.drop_duplicates(subset=subset)
    if dd.get("unique_subset"):
        report["removed_duplicate_keys"] = int(raw.duplicated(subset=dd["unique_subset"]).sum())
        raw = raw.drop_duplicates(subset=dd["unique_subset"])
    raw = raw.reset_index(drop=True)
    report["rows_kept"] = len(raw)
    return raw, report


def split_fractions(cfg):
    """(test_size of the first split, test_size of the second split) for a train/validation/test split.

    Rounded: 1 - 0.70 is 0.30000000000000004 in floating point, which makes scikit-learn round the test size UP by one row."""
    sp = cfg["split"]
    assert abs(sp["train"] + sp["validation"] + sp["test"] - 1) < 1e-9, "split ratios must sum to 1"
    return round(1 - sp["train"], 10), round(sp["test"] / (sp["validation"] + sp["test"]), 10)


def make_xy(raw, cfg):
    """Feature matrix and engineered target: y_raw = shares, y_log = log1p(shares). ID and the target are not features."""
    target = cfg["target"]["name"]
    y_raw = raw[target].astype(float)
    y_log = np.log1p(y_raw) if cfg["target"]["transform"] == "log1p" else y_raw.copy()
    X = raw.drop(columns=list(dict.fromkeys(cfg["columns"]["drop"] + [target])))
    return X, y_raw, y_log


# ============================================================================== transformers
class DataCleaner(BaseEstimator, TransformerMixin):
    """Stateless structural cleaning and missingness indicators (first step of the pipeline).

    * normalises categorical text; a missing category becomes the explicit level ``unknown_label``
    * adds indicator columns (``indicators``) BEFORE any value is blanked
    * turns placeholders into NaN so the imputer treats them as missing: the columns listed in ``empty_set_nan``
      for rows where ``empty_column == 0``, and negative values of ``sentinel_columns``
    * optionally drops the leakage-suspect columns
    It learns nothing from the data, so it cannot leak.
    """

    def __init__(self, categorical=("weekday", "data_channel"), unknown_label="unknown", indicators=(), empty_column=None,
                 empty_set_nan=(), sentinel_columns=(), leaky_columns=(), drop_leaky=True, add_indicators=True):
        self.categorical = categorical
        self.unknown_label = unknown_label
        self.indicators = indicators
        self.empty_column = empty_column
        self.empty_set_nan = empty_set_nan
        self.sentinel_columns = sentinel_columns
        self.leaky_columns = leaky_columns
        self.drop_leaky = drop_leaky
        self.add_indicators = add_indicators

    def fit(self, X, y=None):
        self.feature_names_out_ = np.array(self.transform(X.head(5)).columns)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.categorical:
            X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
            X[c] = X[c].str.strip().str.lower()
        empty = (X[self.empty_column] == 0) if self.empty_column else pd.Series(False, index=X.index)
        if self.add_indicators:
            for ind in self.indicators:
                col, kind = ind["column"], ind["kind"]
                if kind == "isna":
                    X[ind["name"]] = X[col].isna().astype(int)
                elif kind == "eq":
                    X[ind["name"]] = (X[col] == ind["value"]).astype(int)
                elif kind == "lt":
                    X[ind["name"]] = (X[col] < ind["value"]).astype(int)
                else:
                    raise ValueError(f"unknown indicator kind: {kind}")
        for c in self.categorical:
            if c in X.columns:
                X[c] = X[c].fillna(self.unknown_label)
        if len(self.empty_set_nan):
            X.loc[empty, list(self.empty_set_nan)] = np.nan
        for c in self.sentinel_columns:
            X.loc[X[c] < 0, c] = np.nan
        if self.drop_leaky:
            X = X.drop(columns=[c for c in self.leaky_columns if c in X.columns])
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineered features. Which columns get a log transform is learned in ``fit`` (training rows only).

    Adds is_weekend, pub_month (from the URL date), imgs_per_100w, links_per_100w, external_hrefs, self_link_ratio, media_count,
    kw_range_avg and log_<col> for every non-negative, non-binary numeric column whose training skewness exceeds ``skew_threshold``.
    The URL column is dropped at the end.
    """

    def __init__(self, skew_threshold=2.0, add_ratios=True, add_log=True, weekend_days=("saturday", "sunday"),
                 url_column="URL", url_date_regex=r"/(\d{4})/(\d{2})/", weekday_column="weekday"):
        self.skew_threshold = skew_threshold
        self.add_ratios = add_ratios
        self.add_log = add_log
        self.weekend_days = weekend_days
        self.url_column = url_column
        self.url_date_regex = url_date_regex
        self.weekday_column = weekday_column

    def fit(self, X, y=None):
        num = X.select_dtypes("number")
        self.log_cols_ = [c for c in num.columns if num[c].nunique() > 2 and num[c].min() >= 0 and num[c].skew() > self.skew_threshold]
        self.feature_names_out_ = np.array(self.transform(X.head(5)).columns)
        return self

    def transform(self, X):
        X = X.copy()
        X["is_weekend"] = X[self.weekday_column].isin(list(self.weekend_days)).astype(int)
        X["pub_month"] = X[self.url_column].str.extract(self.url_date_regex)[1].astype(float)
        if self.add_ratios:
            words = X["n_tokens_content"].replace(0, np.nan)
            X["imgs_per_100w"] = X["num_imgs"] / words * 100
            X["links_per_100w"] = X["num_hrefs"] / words * 100
            X["external_hrefs"] = X["num_hrefs"] - X["num_self_hrefs"]
            X["self_link_ratio"] = X["num_self_hrefs"] / X["num_hrefs"].replace(0, np.nan)
            X["media_count"] = X["num_imgs"].fillna(0) + X["num_videos"].fillna(0)
            X["kw_range_avg"] = X["kw_max_avg"] - X["kw_min_avg"]
        if self.add_log:
            for c in getattr(self, "log_cols_", []):
                if c in X.columns:
                    X["log_" + c] = np.log1p(X[c].clip(lower=0))
        return X.drop(columns=[self.url_column])

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each non-binary column to its [lower, upper] TRAINING quantiles (rows are never deleted)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = np.array(X.columns)
        self.lo_ = X.quantile(self.lower)
        self.hi_ = X.quantile(self.upper)
        binary = X.nunique() <= 2
        self.lo_[binary] = -np.inf
        self.hi_[binary] = np.inf
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(lower=self.lo_, upper=self.hi_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.columns_


# ============================================================================== pipeline
def build_preprocessing_pipeline(cfg, drop_leaky=True):
    """Return an UNFITTED Pipeline: DataCleaner -> FeatureEngineer -> ColumnTransformer(numeric: winsorise -> impute -> scale; categorical: one-hot).

    Every data-dependent step lives in this pipeline, so fitting it on the training rows only (per model, per fold) cannot leak.
    """
    cols, cc, fe, pp = cfg["columns"], cfg["column_cleaning"], cfg["feature_engineering"], cfg["preprocessing"]
    if pp["scaling"] != "standard" or pp["encoding"] != "onehot" or pp["outlier_handling"]["method"] != "winsorize":
        raise ValueError("only standard scaling, one-hot encoding and winsorising are implemented")
    cleaner = DataCleaner(categorical=tuple(cols["categorical"]), unknown_label=cc["unknown_category_label"], indicators=list(cc["indicators"]),
                          empty_column=cc["empty_article"]["column"], empty_set_nan=tuple(cc["empty_article"]["set_nan"]),
                          sentinel_columns=tuple(cc["negative_sentinel_columns"]), leaky_columns=tuple(cols["leaky"]), drop_leaky=drop_leaky)
    engineer = FeatureEngineer(skew_threshold=fe["log_skew_threshold"], add_ratios=fe["ratio_features"], add_log=fe["log_features"],
                               weekend_days=tuple(fe["weekend_days"]), url_column=cols["url"], url_date_regex=fe["url_date_regex"])
    numeric = Pipeline([
        ("winsorize", Winsorizer(lower=pp["outlier_handling"]["lower_quantile"], upper=pp["outlier_handling"]["upper_quantile"])),
        ("impute", SimpleImputer(strategy=pp["imputation"]["strategy"])),
        ("scale", StandardScaler()),
    ])
    encode = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cols["categorical"])),
    ], verbose_feature_names_out=False)
    return Pipeline([("clean", cleaner), ("fe", engineer), ("encode", encode)])


# ============================================================================== embedded self-tests
def _toy_frame(cfg, n=60, seed=0):
    """Small synthetic frame with the same columns as the real data (used by the self-tests)."""
    rng = np.random.default_rng(seed)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    df = pd.DataFrame({
        "ID": np.arange(n), "URL": [f"http://mashable.com/2013/{1 + i % 12:02d}/05/slug-{i}/" for i in range(n)],
        "timedelta": rng.integers(8, 731, n).astype(float), "weekday": rng.choice(days, n),
        "shares": rng.integers(4, 5000, n), "n_tokens_title": rng.integers(5, 15, n).astype(float),
        "n_tokens_content": rng.integers(50, 2000, n).astype(float), "n_unique_tokens": rng.uniform(.3, .9, n),
        "n_non_stop_words": np.ones(n), "n_non_stop_unique_tokens": rng.uniform(.4, .95, n),
        "num_hrefs": rng.integers(0, 50, n).astype(float), "num_self_hrefs": rng.integers(0, 10, n).astype(float),
        "num_imgs": rng.integers(0, 30, n).astype(float), "num_videos": rng.integers(0, 5, n).astype(float),
        "n_comments": rng.integers(0, 300, n).astype(float), "average_token_length": rng.uniform(4, 5.5, n),
        "data_channel": rng.choice(["world", "business", "lifestyle", None], n),
        "self_reference_min_shares": rng.uniform(0, 5000, n), "self_reference_max_shares": rng.uniform(5000, 9000, n),
        "self_reference_avg_shares": rng.uniform(2000, 6000, n), "num_keywords": rng.integers(1, 11, n).astype(float),
        "kw_min_min": np.where(rng.random(n) < .5, -1.0, 4.0), "kw_max_min": rng.uniform(100, 3000, n), "kw_avg_min": rng.uniform(50, 900, n),
        "kw_min_max": rng.uniform(0, 9000, n), "kw_max_max": np.full(n, 843300.0), "kw_avg_max": rng.uniform(1e5, 5e5, n),
        "kw_min_avg": rng.uniform(0, 2000, n), "kw_max_avg": rng.uniform(3000, 9000, n), "kw_avg_avg": rng.uniform(2000, 4000, n),
    })
    return df


def run_self_tests(cfg=None, verbose=True):
    """Embedded self-tests for the cleaning functions and the preprocessing pipeline. Raises AssertionError on failure."""
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    cfg = cfg or load_config()
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        if verbose:
            print(("  PASS  " if cond else "  FAIL  ") + name)
        assert cond, f"self-test failed: {name}"

    toy = _toy_frame(cfg)
    # --- row-level cleaning
    dirty = pd.concat([toy, toy.iloc[[0]].assign(ID=999), toy.iloc[[1]].assign(ID=1000, URL=toy.URL.iloc[2])], ignore_index=True)
    dirty.loc[3, "weekday"] = "  MONDAY "
    dirty.loc[4, "shares"] = 0
    cleaned, rep = clean_rows(dirty, cfg)
    check("clean_rows removes the invalid (zero) target row", rep["removed_invalid_target"] == 1)
    check("clean_rows removes the exact duplicate (ignoring ID)", rep["removed_exact_duplicates"] == 1)
    check("clean_rows removes the duplicate URL", rep["removed_duplicate_keys"] == 1)
    check("clean_rows normalises casing / whitespace of categoricals", set(cleaned.weekday.dropna()) <= {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
    X, y_raw, y_log = make_xy(cleaned, cfg)
    check("make_xy drops ID and the target from the features", "ID" not in X.columns and "shares" not in X.columns)
    check("make_xy engineers log1p(shares)", np.allclose(y_log, np.log1p(y_raw)))

    # --- DataCleaner
    xt = toy.drop(columns=["ID", "shares"]).copy()
    xt.loc[0:4, "n_tokens_content"] = 0
    xt.loc[5:7, "num_videos"] = np.nan
    pipe = build_preprocessing_pipeline(cfg)
    cl = pipe.named_steps["clean"].fit(xt)
    out = cl.transform(xt)
    check("DataCleaner drops the leakage-suspect column", "n_comments" not in out.columns and "n_comments" in xt.columns)
    check("DataCleaner flags empty articles", out["empty_content"].sum() == 5)
    check("DataCleaner blanks the placeholder zeros of empty articles", out.loc[0:4, "n_unique_tokens"].isna().all())
    check("DataCleaner adds num_videos_missing", out["num_videos_missing"].sum() == 3)
    check("DataCleaner turns kw_min_min = -1 into NaN and flags it", out.kw_min_min.isna().sum() == (xt.kw_min_min < 0).sum() == out.kw_min_min_sentinel.sum())
    check("DataCleaner gives a missing channel the 'unknown' level", "unknown" in set(out.data_channel) and out.data_channel.notna().all())
    check("DataCleaner is stateless / does not modify its input", xt.n_unique_tokens.notna().all() and "empty_content" not in xt.columns)

    # --- FeatureEngineer
    fe = pipe.named_steps["fe"].fit(out)
    eng = fe.transform(out)
    check("FeatureEngineer adds is_weekend and pub_month", {"is_weekend", "pub_month"} <= set(eng.columns) and eng.pub_month.between(1, 12).all())
    check("FeatureEngineer drops the URL column", "URL" not in eng.columns)
    check("FeatureEngineer ratios are NaN (not inf) for empty articles", np.isinf(eng.imgs_per_100w.dropna()).sum() == 0 and eng.loc[0:4, "imgs_per_100w"].isna().all())
    check("FeatureEngineer learned log columns only from non-negative skewed columns", all(out[c].min() >= 0 for c in fe.log_cols_))

    # --- Winsorizer
    w = Winsorizer(0.1, 0.9).fit(pd.DataFrame({"a": np.arange(100.0), "flag": np.tile([0, 1], 50)}))
    clipped = w.transform(pd.DataFrame({"a": [-50.0, 50.0, 500.0], "flag": [0, 1, 1]}))
    check("Winsorizer clips continuous columns to the training quantiles", clipped.a.min() == w.lo_["a"] and clipped.a.max() == w.hi_["a"])
    check("Winsorizer leaves binary columns untouched", clipped.flag.tolist() == [0, 1, 1])

    # --- pipeline
    fresh = build_preprocessing_pipeline(cfg)
    check("build_preprocessing_pipeline returns an UNFITTED pipeline", not hasattr(fresh.named_steps["fe"], "log_cols_") and not hasattr(fresh.named_steps["encode"], "transformers_"))
    pf = clone(build_preprocessing_pipeline(cfg)).fit(xt.iloc[:40])
    mat = pf.transform(xt.iloc[40:])
    check("pipeline output has no NaN", not np.isnan(mat).any())
    check("pipeline output is numeric with a stable width", mat.shape[1] == pf.transform(xt.iloc[:5]).shape[1])
    probe = xt.iloc[:3].copy()
    probe["data_channel"] = ["  WORLD ", "never_seen", None]
    check("pipeline survives messy / unseen categories", not np.isnan(pf.transform(probe)).any())
    # leakage: fitted statistics come from the training rows only
    tr, va = xt.iloc[:40], xt.iloc[40:]
    p_tr = build_preprocessing_pipeline(cfg).fit(tr)
    p_all = build_preprocessing_pipeline(cfg).fit(xt)
    wz_tr = p_tr.named_steps["encode"].named_transformers_["num"].named_steps["winsorize"]
    wz_all = p_all.named_steps["encode"].named_transformers_["num"].named_steps["winsorize"]
    common = [c for c in wz_tr.columns_ if c in set(wz_all.columns_) and wz_tr.hi_[c] != np.inf]
    check("fit on train differs from fit on all rows (no validation information leaks into the fit)", any(wz_tr.hi_[c] != wz_all.hi_[c] for c in common))
    p_tr.transform(va)
    check("transform() does not change the fitted statistics", np.allclose(wz_tr.hi_.replace(np.inf, 0), p_tr.named_steps["encode"].named_transformers_["num"].named_steps["winsorize"].hi_.replace(np.inf, 0)))

    # --- real data (if present): reproduces the dimensions validated in the notebook
    data_path = resolve_path(cfg, cfg["paths"]["data_file"])
    if data_path.exists():
        raw = load_raw(cfg)
        cleaned, rep = clean_rows(raw, cfg)
        check("real data: no rows removed (35,680 kept), as in the notebook", rep["rows_kept"] == len(raw) == 35680 and rep["removed_exact_duplicates"] == 0)
        X, y_raw, y_log = make_xy(cleaned, cfg)
        seed = cfg["project"]["random_seed"]
        X_tr, X_tmp, _, _ = train_test_split(X, y_log, test_size=split_fractions(cfg)[0], random_state=seed, shuffle=True)
        check("real data: 70% train split has 24,976 rows", len(X_tr) == 24976)
        p = build_preprocessing_pipeline(cfg)
        m = p.fit_transform(X_tr)
        check("real data: preprocessing yields 65 numeric model inputs and no NaN", m.shape == (24976, 65) and not np.isnan(m).any())
    else:
        print("  (real data file not found: skipping the real-data self-tests)")
    if verbose:
        print(f"\nAll {len(results)} data-preparation self-tests passed.")
    return results


if __name__ == "__main__":
    run_self_tests()
