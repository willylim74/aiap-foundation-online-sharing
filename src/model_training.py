"""Model training: split, baseline fitting, hyper-parameter tuning (grid + successive halving, checkpointed) and one-time test evaluation.

The model registry, every hyper-parameter grid, the split ratios, the tuning strategy and the metrics are read from ``config.yaml``.
Design validated in the notebook (EDA.ipynb, Sections 11-25):

* baselines use the registry ``params`` only (library defaults + seeds); tuned settings are applied ONLY in the tuning code path
* cheap linear models: ``GridSearchCV`` on the full training split; slow models: ``HalvingRandomSearchCV`` on a training subsample,
  then the winner is refit on the FULL training split before it is scored
* every finished model is checkpointed to a resumable ledger (pickle + JSON), keys namespaced ``<stage>::<model>``
* the test split is used exactly once: the selected configuration is refit on train + validation and predicts the test split a single time
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import platform
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (enables HalvingRandomSearchCV)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from data_preparation import build_preprocessing_pipeline, resolve_path, split_fractions

NL = chr(10)
SUMMARY_KEYS = ("R2", "RMSE", "MAE", "MSE", "R2_raw", "RMSE_raw")


# ============================================================================== registry helpers
def get_model_specs(cfg):
    """Registry as an ordered dict: name -> spec (class path, baseline params, tuning block)."""
    return {m["name"]: m for m in cfg["models"]}


def make_estimator(spec):
    """Instantiate a registry entry with its BASELINE params (library defaults + seeds)."""
    module, cls = spec["class"].rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)(**spec.get("params", {}))


def make_model_pipeline(name, cfg, drop_leaky=True):
    """The ONE 'fit a model' helper shared by baselines and tuning: preprocessing + the registry estimator at its baseline params.

    It never applies tuned settings; the tuning path calls ``set_params`` on a clone afterwards, so baselines stay at library defaults."""
    spec = get_model_specs(cfg)[name]
    return Pipeline([("prep", build_preprocessing_pipeline(cfg, drop_leaky=drop_leaky)), ("model", make_estimator(spec))])


def set_threads(cfg):
    """Single-CPU sandbox: limit BLAS / OpenMP threads (the returned object must stay referenced)."""
    return threadpool_limits(limits=int(cfg["tuning"]["threads"]))


# ============================================================================== split
def split_data(X, y_raw, y_log, cfg):
    """Random train / validation / test split (ratios from config). Regression -> no stratification."""
    seed, sp = cfg["project"]["random_seed"], cfg["split"]
    t1, t2 = split_fractions(cfg)
    X_tr, X_tmp, yr_tr, yr_tmp, yl_tr, yl_tmp = train_test_split(X, y_raw, y_log, test_size=t1, random_state=seed, shuffle=sp["shuffle"])
    X_va, X_te, yr_va, yr_te, yl_va, yl_te = train_test_split(X_tmp, yr_tmp, yl_tmp, test_size=t2, random_state=seed, shuffle=sp["shuffle"])
    data = dict(X_train=X_tr, X_val=X_va, X_test=X_te, y_train_raw=yr_tr, y_val_raw=yr_va, y_test_raw=yr_te, y_train=yl_tr, y_val=yl_va, y_test=yl_te)
    assert not (set(X_tr.index) & set(X_va.index)) and not (set(X_tr.index) & set(X_te.index)) and not (set(X_va.index) & set(X_te.index))
    assert len(X_tr) + len(X_va) + len(X_te) == len(X)
    return data


# ============================================================================== metrics
def reg_metrics(y_true_log, pred_log, prefix, cfg):
    """MAE, MSE, RMSE, R2 on the modelling scale (log1p) and on the raw share scale (back-transformed with expm1)."""
    lo, hi = cfg["evaluation"]["raw_prediction_clip"]
    y_true_log, pred_log = np.asarray(y_true_log), np.asarray(pred_log)
    out = {}
    for scale, yt, yp in [("log", y_true_log, pred_log), ("raw", np.expm1(y_true_log), np.expm1(np.clip(pred_log, lo, hi)))]:
        mse = mean_squared_error(yt, yp)
        out.update({f"{prefix}_MAE_{scale}": mean_absolute_error(yt, yp), f"{prefix}_MSE_{scale}": mse,
                    f"{prefix}_RMSE_{scale}": float(np.sqrt(mse)), f"{prefix}_R2_{scale}": r2_score(yt, yp)})
    return out


# ============================================================================== baselines
def fit_baselines(cfg, data, verbose=True):
    """Fit every registry model at its baseline (default) settings on the training split; score train and validation."""
    rows, fitted = [], {}
    for name in get_model_specs(cfg):
        pipe = make_model_pipeline(name, cfg)
        t0 = time.time()
        pipe.fit(data["X_train"], data["y_train"])
        fit_s = time.time() - t0
        p_tr, p_va = pipe.predict(data["X_train"]), pipe.predict(data["X_val"])
        fitted[name] = pipe
        rows.append(dict(model=name, fit_seconds=fit_s, **reg_metrics(data["y_train"], p_tr, "train", cfg), **reg_metrics(data["y_val"], p_va, "val", cfg)))
        if verbose:
            print(f"  {name:30s} val R2(log) = {rows[-1]['val_R2_log']:.4f} | train R2(log) = {rows[-1]['train_R2_log']:.4f} | fit {fit_s:.1f}s", flush=True)
    results = pd.DataFrame(rows).set_index("model")
    results["overfit_gap_R2"] = results.train_R2_log - results.val_R2_log
    results["rank_by_val_R2"] = results.val_R2_log.rank(ascending=False).astype(int)
    return results.sort_values("rank_by_val_R2"), fitted


def check_baselines_against_notebook(results, cfg):
    """Compare the production baseline table with the notebook's own baseline table (outputs/baseline_results.csv).

    Returns (status, max scaled difference |a-b|/(1+|b|), per-metric details). status is 'match', 'MISMATCH' or 'no reference file'."""
    ref_path = resolve_path(cfg, cfg["paths"].get("reference_baseline_file", ""))
    if not ref_path.exists():
        return "no reference file", None, None
    ref = pd.read_csv(ref_path, index_col=0)
    cols = [c for c in ref.columns if c.startswith(("train_", "val_")) and c in results.columns]
    diff = (results.loc[ref.index, cols] - ref[cols]).abs()
    scaled = diff / (1.0 + ref[cols].abs())              # relative for large numbers (raw-scale MSE ~1e8), absolute for small ones
    worst = float(scaled.max().max())
    return ("match" if worst <= cfg["evaluation"]["baseline_reproduction_tolerance"] else "MISMATCH"), worst, diff


# ============================================================================== checkpoint ledger
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class CheckpointLedger:
    """Resumable ledger: one pickle per entry (fitted state) + one JSON file (metadata and text reports).

    Keys are '<stage>::<name>' (never a bare model name) and an entry is reused only if its settings fingerprint matches.
    It exists to survive an INTERRUPTED run of the same pipeline, not to hand results between processes."""

    def __init__(self, directory, filename="ledger.json"):
        self.dir, self.path = Path(directory), Path(directory) / filename
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data = json.load(open(self.path)) if self.path.exists() else {}

    @staticmethod
    def key(stage, name):
        return f"{stage}::{name}"

    def _pickle_path(self, key):
        return str(self.dir / (hashlib.md5(key.encode()).hexdigest()[:8] + "_" + re.sub(r"\W+", "_", key) + ".pkl"))

    def has(self, key, fingerprint):
        e = self.data.get(key)
        return bool(e) and e.get("fingerprint") == fingerprint and os.path.exists(e.get("pickle", ""))

    def save(self, key, record, obj, fingerprint):
        p = self._pickle_path(key)
        with open(p + ".tmp", "wb") as f:
            pickle.dump(obj, f)
        os.replace(p + ".tmp", p)
        self.data[key] = dict(jsonable(record), pickle=p, fingerprint=fingerprint, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(str(self.path) + ".tmp", "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(str(self.path) + ".tmp", self.path)

    def load(self, key):
        e = self.data[key]
        with open(e["pickle"], "rb") as f:
            return e, pickle.load(f)

    def clear(self):
        for e in self.data.values():
            if os.path.exists(e.get("pickle", "")):
                os.remove(e["pickle"])
        self.data = {}
        if self.path.exists():
            os.remove(self.path)


# ============================================================================== tuning
def build_values(v):
    """Turn a config search-space entry into a list (grid) or a scipy distribution (random search)."""
    if isinstance(v, list):
        return v
    t = v["type"]
    if t == "logspace":
        return [float(x) for x in np.logspace(v["low"], v["high"], v["num"])]
    if t == "randint":
        return randint(v["low"], v["high"])
    if t == "uniform":
        return uniform(v["low"], v["high"] - v["low"])
    if t == "loguniform":
        return loguniform(v["low"], v["high"])
    raise ValueError(f"unknown distribution type: {t}")


def _pyval(v):
    return v.item() if hasattr(v, "item") else v


def tuning_fingerprint(name, cfg, n_train):
    spec, t = get_model_specs(cfg)[name], cfg["tuning"]
    blob = dict(spec=spec, n_train=n_train, seed=cfg["project"]["random_seed"], halving=t["halving"], grid_cv=t["grid_cv_folds"],
                scoring=t["scoring"], version=t["pipeline_version"], preprocessing=cfg["preprocessing"], cleaning=cfg["column_cleaning"],
                features=cfg["feature_engineering"], leaky=cfg["columns"]["leaky"])
    return hashlib.md5(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def run_tuning(name, cfg, data, baseline, ledger, verbose=True):
    """Tune one model (or resume it from the ledger). Returns (record, object).

    The pipeline is BUILT with the shared baseline helper; every tuned setting is applied here via set_params, so baselines are never affected."""
    spec, tcfg = get_model_specs(cfg)[name], cfg["tuning"]
    stage, seed = tcfg["stage"], cfg["project"]["random_seed"]
    key, fp = CheckpointLedger.key(stage, name), tuning_fingerprint(name, cfg, len(data["X_train"]))
    if tcfg.get("resume_from_checkpoints", True) and ledger.has(key, fp):
        rec, obj = ledger.load(key)
        if verbose:
            print(f"[resumed from checkpoint {key}, saved {rec['saved_at']}]{NL}{rec['report']}")
        return rec, obj
    X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    pipe = make_model_pipeline(name, cfg)
    params = {"model__" + k: build_values(v) for k, v in spec["tuning"]["space"].items()}
    strategy, n_jobs, scoring = spec["tuning"]["strategy"], tcfg["n_jobs"], tcfg["scoring"]
    t0 = time.time()
    if strategy == "grid":
        folds = tcfg["grid_cv_folds"]
        search = GridSearchCV(pipe, params, cv=KFold(folds, shuffle=True, random_state=seed), scoring=scoring, n_jobs=n_jobs, refit=True)
        search.fit(X_train, y_train)                         # refit=True: the winner is refit on the FULL training split
        refit_s = float(search.refit_time_)
        search_s, best_est, n_search_rows = time.time() - t0 - refit_s, search.best_estimator_, len(X_train)
    elif strategy == "halving":
        h = tcfg["halving"]
        folds = h["cv_folds"]
        sub = X_train.sample(h["subsample_n"], random_state=seed).index
        search = HalvingRandomSearchCV(pipe, params, n_candidates=spec["tuning"]["n_candidates"], factor=h["factor"], resource="n_samples",
                                       min_resources=h["min_resources"], max_resources=h["subsample_n"], cv=KFold(folds, shuffle=True, random_state=seed),
                                       scoring=scoring, random_state=seed, refit=False, n_jobs=n_jobs)
        search.fit(X_train.loc[sub], y_train.loc[sub])        # search on the subsample only
        search_s = time.time() - t0
        t1 = time.time()
        best_est = clone(pipe).set_params(**{k: _pyval(v) for k, v in search.best_params_.items()}).fit(X_train, y_train)   # refit on the FULL training split
        refit_s, n_search_rows = time.time() - t1, h["subsample_n"]
    else:
        raise ValueError(f"model {name}: unknown tuning strategy {strategy}")
    best_params = {k: _pyval(v) for k, v in search.best_params_.items()}
    cvr = pd.DataFrame(search.cv_results_)
    cvr = cvr[[c for c in cvr.columns if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score", "iter", "n_resources", "mean_fit_time")]]
    p_tr, p_va = best_est.predict(X_train), best_est.predict(X_val)
    m = dict(**reg_metrics(y_train, p_tr, "train", cfg), **reg_metrics(y_val, p_va, "val", cfg))
    b = baseline.loc[name]
    n_fits = int(len(cvr) * folds)
    lines = [f"{name}: {strategy} search, {folds}-fold CV on {n_search_rows:,} rows | {len(cvr)} candidate evaluations ({n_fits} fits)",
             f"best params     : {best_params}",
             f"CV best R2      : {search.best_score_:.4f}   (search rows: {n_search_rows:,})",
             f"validation R2   : tuned {m['val_R2_log']:.4f} vs baseline {b.val_R2_log:.4f}  (delta {m['val_R2_log'] - b.val_R2_log:+.4f})",
             f"validation RMSE : tuned {m['val_RMSE_log']:.4f} vs baseline {b.val_RMSE_log:.4f} (log scale)",
             f"train R2        : tuned {m['train_R2_log']:.4f} (baseline {b.train_R2_log:.4f}) -> generalisation gap {m['train_R2_log'] - m['val_R2_log']:.4f}",
             f"time            : search {search_s:.0f}s + refit on full training split {refit_s:.0f}s"]
    if "iter" in cvr.columns:
        it = cvr.groupby("iter").agg(candidates=("mean_test_score", "size"), rows=("n_resources", "first"), best_cv_R2=("mean_test_score", "max"))
        lines.append("halving rungs   : " + "; ".join(f"rung {i}: {int(r.candidates)} candidates on {int(r.rows)} rows (best CV R2 {r.best_cv_R2:.3f})" for i, r in it.iterrows()))
    report = NL.join(lines)
    rec = dict(model=name, stage=stage, strategy=strategy, best_params=best_params, cv_best_score=float(search.best_score_), cv_folds=folds,
               search_rows=n_search_rows, n_candidates=len(cvr), n_fits=n_fits, search_seconds=float(search_s), refit_seconds=float(refit_s),
               report=report, **{k: float(v) for k, v in m.items()})
    obj = dict(best_estimator=best_est, cv_results=cvr, best_params=best_params, val_pred=p_va)
    ledger.save(key, rec, obj, fp)
    if verbose:
        print(report + NL + f"[checkpoint written: {key}]", flush=True)
    return ledger.data[key], obj


def tune_all(cfg, data, baseline, ledger, verbose=True):
    """Tune every registry model that has a tuning strategy (the naive benchmark has none)."""
    tuned = {}
    for name, spec in get_model_specs(cfg).items():
        if spec["tuning"]["strategy"] == "none":
            continue
        tuned[name] = run_tuning(name, cfg, data, baseline, ledger, verbose)
    return tuned


def build_tuned_table(tuned, baseline):
    rows = []
    for name, (rec, _) in tuned.items():
        b = baseline.loc[name]
        rows.append(dict(model=name, strategy=rec["strategy"], baseline_val_R2=b.val_R2_log, tuned_val_R2=rec["val_R2_log"], delta_R2=rec["val_R2_log"] - b.val_R2_log,
                         baseline_val_RMSE=b.val_RMSE_log, tuned_val_RMSE=rec["val_RMSE_log"], tuned_val_MAE=rec["val_MAE_log"], tuned_val_MSE=rec["val_MSE_log"],
                         baseline_train_R2=b.train_R2_log, tuned_train_R2=rec["train_R2_log"], tuned_gap=rec["train_R2_log"] - rec["val_R2_log"],
                         cv_best_R2=rec["cv_best_score"], tuned_val_R2_raw=rec["val_R2_raw"], tuned_val_RMSE_raw=rec["val_RMSE_raw"],
                         search_s=rec["search_seconds"], refit_s=rec["refit_seconds"], n_fits=rec["n_fits"], best_params=json.dumps(rec["best_params"])))
    t = pd.DataFrame(rows).set_index("model")
    t["rank_tuned_val_R2"] = t.tuned_val_R2.rank(ascending=False, method="min").astype(int)
    return t.sort_values("rank_tuned_val_R2")


# ============================================================================== final test evaluation (exactly once)
def evaluate_test_once(cfg, data, tuned_table, tuned, ledger, X_ids, verbose=True):
    """Select the model (highest validation R2 among tuned models), refit it on train + validation, predict the test split ONCE.

    The result (and the fitted pipeline) is stored in the ledger, so a re-run reuses it instead of touching the test split again.
    Two reference models (naive benchmark and the default-parameter twin, both refit on train + validation) are scored in the same
    step for context only; they take no part in the selection."""
    stage, seed = cfg["tuning"]["stage"], cfg["project"]["random_seed"]
    final_name = tuned_table.index[0]
    final_params = tuned[final_name][0]["best_params"]
    key = CheckpointLedger.key(stage, "FINAL_TEST")
    n_trval = len(data["X_train"]) + len(data["X_val"])
    fp = hashlib.md5(json.dumps(dict(model=final_name, params=final_params, n_trval=n_trval, seed=seed, version=cfg["tuning"]["pipeline_version"]), sort_keys=True).encode()).hexdigest()
    if ledger.has(key, fp):
        rec, obj = ledger.load(key)
        if verbose:
            print("[test result resumed from the ledger: the test split is NOT evaluated a second time]")
        return rec, obj
    X_trval, y_trval = pd.concat([data["X_train"], data["X_val"]]), pd.concat([data["y_train"], data["y_val"]])
    final_pipe = clone(make_model_pipeline(final_name, cfg)).set_params(**final_params).fit(X_trval, y_trval)
    pred_test = final_pipe.predict(data["X_test"])                                    # <- the one and only test-set evaluation of the final model
    ref_naive = clone(make_model_pipeline("Naive benchmark (train mean)", cfg)).fit(X_trval, y_trval).predict(data["X_test"])
    ref_base = clone(make_model_pipeline(final_name, cfg)).fit(X_trval, y_trval).predict(data["X_test"])
    metrics = {k: reg_metrics(data["y_test"], p, "test", cfg) for k, p in [("final", pred_test), ("baseline_same_model", ref_base), ("naive", ref_naive)]}
    ci = bootstrap_intervals(data["y_test"].values, dict(final=pred_test, baseline_same_model=ref_base, naive=ref_naive), cfg)
    rec = dict(model=final_name, params=final_params, selection_rule=cfg["evaluation"]["selection_rule"], n_train_final=len(X_trval),
               n_test=len(data["X_test"]), metrics=metrics, bootstrap=ci)
    obj = dict(pipeline=final_pipe, pred_test=pred_test, ref_naive=ref_naive, ref_base=ref_base, test_ids=np.asarray(X_ids))
    ledger.save(key, rec, obj, fp)
    if verbose:
        print("[test set evaluated once; stored in the ledger]")
    return ledger.data[key], obj


def bootstrap_intervals(y_true, preds, cfg):
    """Bootstrap 95% intervals (resampling test rows) for R2 / RMSE / MAE on the log scale and paired improvements."""
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    B = int(cfg["evaluation"]["bootstrap"]["n_resamples"])
    idx = rng.integers(0, len(y_true), (B, len(y_true)))

    def stats_(pred, i):
        e = y_true[i] - pred[i]
        return 1 - (e ** 2).sum() / ((y_true[i] - y_true[i].mean()) ** 2).sum(), np.sqrt((e ** 2).mean()), np.abs(e).mean()

    bs = {k: np.array([stats_(p, i) for i in idx]) for k, p in preds.items()}
    out = {}
    for k, p in preds.items():
        point = stats_(p, np.arange(len(y_true)))
        for j, mname in enumerate(["R2", "RMSE", "MAE"]):
            lo, hi = np.percentile(bs[k][:, j], [2.5, 97.5])
            out[f"{k}_{mname}_log"] = dict(point=float(point[j]), ci95_low=float(lo), ci95_high=float(hi))
    for ref in ("naive", "baseline_same_model"):
        d = bs[ref][:, 1] - bs["final"][:, 1]
        out[f"rmse_improvement_vs_{ref}"] = dict(mean=float(d.mean()), ci95_low=float(np.percentile(d, 2.5)), ci95_high=float(np.percentile(d, 97.5)))
    d = bs["final"][:, 0] - bs["baseline_same_model"][:, 0]
    out["r2_gain_vs_baseline_same_model"] = dict(mean=float(d.mean()), ci95_low=float(np.percentile(d, 2.5)), ci95_high=float(np.percentile(d, 97.5)))
    return out


def residuals_by_group(cfg, X_test, y_test, pred_test):
    """Mean residual (positive = under-prediction) and RMSE by channel and weekday."""
    res = np.asarray(y_test) - pred_test
    lab = cfg["column_cleaning"]["unknown_category_label"]
    frames = []
    for col in cfg["columns"]["categorical"]:
        g = pd.DataFrame({"res": res, "group": X_test[col].fillna(lab).values}).groupby("group").res.agg(
            n="size", mean_residual="mean", rmse=lambda s: float(np.sqrt((s ** 2).mean())))
        frames.append(g.reset_index().assign(feature=col))
    return pd.concat(frames, ignore_index=True)[["feature", "group", "n", "mean_residual", "rmse"]]


# ============================================================================== sensitivity (validation data only)
def sensitivity_checks(cfg, X_all, y_log, data, baseline, tuned_table, tuned, final_name, final_params):
    """Time-ordered split and the leakage-suspect-feature check, on validation data only (the test split is never used)."""
    sc = cfg["evaluation"]["sensitivity"]

    def val_r2(tuned_flag, tr_idx, va_idx, drop_leaky=True):
        p = clone(make_model_pipeline(final_name, cfg, drop_leaky=drop_leaky))
        if tuned_flag:
            p.set_params(**final_params)
        p.fit(X_all.loc[tr_idx], y_log.loc[tr_idx])
        return r2_score(y_log.loc[va_idx], p.predict(X_all.loc[va_idx]))

    order = X_all.sort_values(sc["time_order_column"], ascending=False).index      # oldest articles first
    n = len(order)
    a, b = int(sc["time_train_fraction"] * n), int((sc["time_train_fraction"] + sc["time_validation_fraction"]) * n)
    t_tr, t_va = order[:a], order[a:b]
    rows = [dict(scenario="random split, validation (main workflow)", defaults=baseline.loc[final_name, "val_R2_log"], tuned=tuned_table.loc[final_name, "tuned_val_R2"]),
            dict(scenario="time-ordered split: train oldest 70% -> validate next 15%", defaults=val_r2(False, t_tr, t_va), tuned=val_r2(True, t_tr, t_va)),
            dict(scenario="random split + leakage-suspect columns added back", defaults=val_r2(False, data["X_train"].index, data["X_val"].index, False),
                 tuned=val_r2(True, data["X_train"].index, data["X_val"].index, False))]
    out = pd.DataFrame(rows).set_index("scenario")
    out["tuned_minus_defaults"] = out.tuned - out.defaults
    return out


# ============================================================================== persistence
def _short(metrics, prefix):
    """{'test_MAE_log': v, 'test_MAE_raw': w} -> {'MAE': v, 'MAE_raw': w} (same labels as the summary table rows)."""
    out = {}
    for k, v in metrics.items():
        name = k[len(prefix) + 1:]
        out[name[:-4] if name.endswith("_log") else name] = v
    return out


def write_outputs(cfg, out_dir, baseline, tuned_table, tuned, final_rec, final_obj, data, sens, residuals, X_ids):
    """Write every artefact a reviewer needs to replicate / audit the run."""
    o = cfg["outputs"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(out_dir / o["baseline_results"])
    tuned_table.to_csv(out_dir / o["tuned_results"])
    json.dump({m: r[0]["best_params"] for m, r in tuned.items()}, open(out_dir / o["best_params"], "w"), indent=1)
    final_name, M = final_rec["model"], final_rec["metrics"]
    json.dump(jsonable(dict(model=final_name, params=final_rec["params"], selection_rule=final_rec["selection_rule"], n_train_final=final_rec["n_train_final"],
                            n_test=final_rec["n_test"], test_metrics=M["final"], reference_test_metrics=dict(default_params_same_model=M["baseline_same_model"], naive_benchmark=M["naive"]),
                            bootstrap_ci95=final_rec["bootstrap"])), open(out_dir / o["final_test_results"], "w"), indent=1)
    tp = pd.DataFrame({"ID": final_obj["test_ids"], "actual_shares": np.expm1(data["y_test"].values), "actual_log1p": data["y_test"].values, "pred_log1p": final_obj["pred_test"],
                       "pred_shares": np.expm1(np.clip(final_obj["pred_test"], *cfg["evaluation"]["raw_prediction_clip"]))})
    tp["residual_log1p"] = tp.actual_log1p - tp.pred_log1p
    tp.to_csv(out_dir / o["test_predictions"], index=False)
    residuals.to_csv(out_dir / o["residuals_by_group"], index=False)
    joblib.dump(final_obj["pipeline"], out_dir / o["final_model"])
    if sens is not None:
        sens.to_csv(out_dir / o["sensitivity"])
    base_b, tuned_t, naive_b = baseline.loc[final_name], tuned_table.loc[final_name], baseline.loc["Naive benchmark (train mean)"]
    summary = pd.DataFrame({
        "baseline (defaults), validation": dict(R2=base_b.val_R2_log, RMSE=base_b.val_RMSE_log, MAE=base_b.val_MAE_log, MSE=base_b.val_MSE_log, R2_raw=base_b.val_R2_raw, RMSE_raw=base_b.val_RMSE_raw),
        "tuned, validation": dict(R2=tuned_t.tuned_val_R2, RMSE=tuned_t.tuned_val_RMSE, MAE=tuned_t.tuned_val_MAE, MSE=tuned_t.tuned_val_MSE, R2_raw=tuned_t.tuned_val_R2_raw, RMSE_raw=tuned_t.tuned_val_RMSE_raw),
        "FINAL tuned, TEST (train+val refit)": {k: v for k, v in _short(M["final"], "test").items() if k in SUMMARY_KEYS},
        "naive benchmark, TEST": {k: v for k, v in _short(M["naive"], "test").items() if k in SUMMARY_KEYS},
    })
    summary.to_csv(out_dir / o["comparison_summary"])
    ids = data["split_ids"]
    json.dump(ids, open(out_dir / o["split_ids"], "w"))
    return summary


def write_manifest(cfg, out_dir, mode, timings, extra=None):
    data_path = resolve_path(cfg, cfg["paths"]["data_file"])
    m = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), mode=mode, config=cfg["_config_path"], data_file=str(data_path),
             data_sha256=hashlib.sha256(open(data_path, "rb").read()).hexdigest(), seed=cfg["project"]["random_seed"], stage=cfg["tuning"]["stage"],
             pipeline_version=cfg["tuning"]["pipeline_version"], python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
             sklearn=sklearn.__version__, compute=dict(n_jobs=cfg["tuning"]["n_jobs"], threads=cfg["tuning"]["threads"], halving=cfg["tuning"]["halving"],
                                                       grid_cv_folds=cfg["tuning"]["grid_cv_folds"], scoring=cfg["tuning"]["scoring"]), timings_seconds=timings)
    if extra:
        m.update(extra)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(jsonable(m), open(Path(out_dir) / cfg["outputs"]["manifest"], "w"), indent=1)
    return m


# ============================================================================== self-test
def _self_test():
    """Light checks that need no data file: registry coverage, split sizes, ledger namespacing / fingerprints / resume, metrics."""
    import tempfile
    from data_preparation import load_config

    cfg = load_config()
    specs = get_model_specs(cfg)
    required = ["Naive benchmark (train mean)", "Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest", "HistGradientBoosting", "K-Nearest Neighbors"]
    assert list(specs) == required, "registry must contain exactly the required regression models"
    assert specs["Lasso"]["params"].get("alpha") is None and specs["Decision Tree"]["params"].get("max_depth") is None, "baselines must stay at library defaults"
    for n, s in specs.items():
        est = make_estimator(s)
        assert hasattr(est, "fit")
        if s["tuning"]["strategy"] != "none":
            names = set(est.get_params())
            assert set(s["tuning"]["space"]) <= names, f"{n}: tuning space has unknown parameters"
            build = {k: build_values(v) for k, v in s["tuning"]["space"].items()}
            assert build
    X = pd.DataFrame({"a": np.arange(1000.0)}); y = pd.Series(np.arange(1000.0))
    d = split_data(X, y, y, cfg)
    assert (len(d["X_train"]), len(d["X_val"]), len(d["X_test"])) == (700, 150, 150), "70/15/15 split sizes"
    with tempfile.TemporaryDirectory() as tmp:
        led = CheckpointLedger(tmp)
        k1, k2 = CheckpointLedger.key("stage_a", "Random Forest"), CheckpointLedger.key("stage_b", "Random Forest")
        assert k1 != k2
        led.save(k1, dict(report="A"), {"x": 1}, "fp1")
        assert led.has(k1, "fp1") and not led.has(k1, "other fingerprint") and not led.has(k2, "fp1"), "namespacing / fingerprint check"
        led2 = CheckpointLedger(tmp)                                           # a 'restarted kernel' re-reads the ledger from disk
        assert led2.has(k1, "fp1") and led2.load(k1)[1] == {"x": 1}
    m = reg_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "t", cfg)
    assert m["t_RMSE_log"] == 0 and m["t_R2_log"] == 1
    print("model_training self-test passed (registry, split sizes, ledger namespacing/fingerprint/resume, metrics)")


if __name__ == "__main__":
    _self_test()
