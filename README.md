## News-popularity (shares) Prediction

This project predicts the number of social-media shares of a Mashable-style news article from its content, metadata and keyword features
(35,680 articles x 30 columns in `data/mini_project_1_data.csv`). It is a supervised **regression** task on a heavy-tailed target
(median 1,400 shares, mean 3,402, skewness 34), so the models are fitted to `log1p(shares)` and evaluated on both the log and the raw scale.

`main.py` runs the whole workflow end to end: data cleaning, feature engineering, a 70/15/15 split, baseline models, hyper-parameter tuning,
model selection and one final evaluation on the held-out test split. `EDA.ipynb` holds the detailed exploratory analysis and the validated
experiments that every decision in the pipeline comes from (if the code and the notebook ever disagree, the notebook's findings win).

```
.
├── EDA.ipynb                  exploratory analysis + validated modelling workflow (sections 1-26)
├── README.md
├── requirements.txt           runtime dependencies of main.py / src/
├── main.py                    entry point: runs every stage; works from any working directory
├── data/
│   └── mini_project_1_data.csv        input data (35,680 rows)
├── src/
│   ├── config.yaml                    all paths, cleaning rules, feature lists, models, search spaces, split, metrics, output names
│   ├── data_preparation.py            row-level cleaning, custom transformers, build_preprocessing_pipeline(), self-tests
│   └── model_training.py              split, baselines, tuning (checkpointed), one-time test evaluation, outputs
├── _archive/                  an earlier, simpler version of the project (not used by main.py)
└── artifacts/                 created by main.py: result tables, best parameters, final model, checkpoints
```

# Prerequisites and Installation

* Python 3.12 (developed and tested with pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, scipy 1.18.0, pyyaml 6.0.3, joblib 1.5.3, threadpoolctl 3.6.0).
  The lower bounds in `requirements.txt` are more permissive (pandas >= 2.1, scikit-learn >= 1.4, ...), but only the versions above were tested.
* Optional: `matplotlib`, `seaborn`, `statsmodels` and `jupyter`, only needed to re-run `EDA.ipynb` (they are not in `requirements.txt`).
* The pipeline runs on a single CPU core and needs no GPU.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

# Instructions for Executing the Pipeline

Run from the project root (or from any other directory: `main.py` resolves the config, data, `src/` import and every output path relative to its own location).

```bash
python main.py --skip-tuning     # quick check, ~2 min: data preparation, split, baselines (test split untouched)
python main.py                   # full run, ~7-10 min on one CPU: baselines + tuning + one-time test evaluation
python main.py --self-test       # embedded self-tests of the data-preparation and training modules, then exit
```

| option | effect |
|---|---|
| `--skip-tuning` | stop after the baselines; no tuning, no selection, the test split is never used |
| `--fresh` | delete existing checkpoints and recompute everything (otherwise an interrupted run resumes from `artifacts/checkpoints/`) |
| `--config PATH` | use another configuration file (default `src/config.yaml`) |
| `--output-dir DIR` | write outputs to `DIR` instead of `artifacts/` (relative paths are resolved against the project root) |
| `--self-test` | run the self-tests and exit |

**Changing parameters.** Nothing about the data or the models is hard-coded in Python; edit `src/config.yaml`:

| section | what you can change |
|---|---|
| `paths` | data file, output and checkpoint folders, optional reference baseline file |
| `target` | target column, transform (`log1p`), rule for invalid target values |
| `columns`, `cleaning`, `column_cleaning` | ID/URL/categorical columns, the leakage-suspect column list, de-duplication keys, sentinel and placeholder rules, missingness indicators |
| `feature_engineering`, `preprocessing` | weekend days, ratio and log features, log-skew threshold, winsorising quantiles, imputation, scaling, encoding |
| `split` | train / validation / test fractions (must sum to 1) |
| `models` | the model registry: class path, baseline parameters and the tuning search space of each model |
| `tuning` | grid CV folds, successive-halving settings (folds, factor, subsample size), thread count, resume behaviour |
| `evaluation` | metrics, selection rule, bootstrap resamples, raw-scale clip, time-ordered sensitivity check |

Example: to try another seed, change `project.random_seed`; to tune only on a smaller subsample, change `tuning.halving.subsample_n`.
After changing preprocessing or model code, bump `tuning.pipeline_version` (or use `--fresh`) so that old checkpoints are not reused.

**Outputs** are written to `artifacts/` (never to the notebook's folders): `baseline_results.csv`, `tuned_results.csv`, `best_params.json`,
`final_test_results.json`, `test_predictions.csv`, `test_residuals_by_group.csv`, `final_model.joblib`, `comparison_summary.csv`,
`sensitivity_checks.csv`, `split_ids.json`, `run_manifest.json` and `checkpoints/`.

# Description of Logical Steps/Flow of the Pipeline

```
data/mini_project_1_data.csv
        |
        v
[1] Row-level cleaning (clean_rows)   text normalisation, invalid-target removal, de-duplication; ID and target left out of X; y = log1p(shares)
        |
        v
[2] Split (split_data)                70 / 15 / 15 random split, seed 42 (24,976 / 5,352 / 5,352 rows); test split locked
        |
        v
[3] Preprocessing pipeline (unfitted; fitted again for every model, CV fold and refit on the training rows only)
        DataCleaner -> FeatureEngineer -> ColumnTransformer{ numeric: Winsorizer -> median imputer -> StandardScaler ; categorical: OneHotEncoder }
        |
        v
[4] Baselines (fit_baselines)         8 models at library defaults, scored on the validation split
        |
        v
[5] Tuning (tune_all)                 grid search (linear models) / successive halving on a subsample + full refit (slow models); checkpointed
        |
        v
[6] Selection + final evaluation      best tuned model by validation R2 -> refit on train + validation -> predict the test split ONCE
        |                             bootstrap 95% CIs, residuals by channel and weekday, validation-only sensitivity checks
        v
[7] Outputs                           tables, best parameters, final model, run manifest -> artifacts/
```

1. **Data preparation (`data_preparation.py`).** `clean_rows` applies only rules that need no statistics from the data (strip/lower-case the
   categoricals, drop rows with a missing or non-positive target, drop exact and URL duplicates), so it can safely run before the split. On this
   file it removes nothing; the checks stay as guards for refreshed data.
2. **Split.** A plain random split (no stratification, since this is regression). The test split stays locked until step 6.
3. **Preprocessing.** Everything that learns from data (which columns get a log transform, winsorising bounds, imputation medians, scaling,
   one-hot levels) lives in the pipeline and is fitted on training rows only, so no validation or test information can leak.
4. **Baselines.** All 8 registered models are fitted with library defaults (plus fixed seeds) and ranked by validation R2 on `log1p(shares)`.
   Tuned settings are applied only in the tuning code path, so the baseline-versus-tuned comparison is genuine.
5. **Tuning.** `GridSearchCV` (5-fold, full training split) for Linear Regression, Ridge and Lasso; `HalvingRandomSearchCV` (3-fold) on a
   7,290-row training subsample for Decision Tree, Random Forest, HistGradientBoosting and KNN (27 -> 9 -> 3 candidates on 810 -> 2,430 -> 7,290 rows),
   then the winner is refit on the full training split. After every model the fitted state and report are checkpointed, keyed `<stage>::<model>`
   with a settings fingerprint, so an interrupted run resumes and stale checkpoints are recomputed.
6. **Selection and final evaluation.** The model with the highest tuned validation R2 is refit on train + validation and predicts the test split exactly
   once; the result is stored in the ledger so a re-run does not touch the test split again. The naive benchmark and the default-parameter twin of the
   selected model are scored alongside for context. Validation-only sensitivity checks (time-ordered split) are run as well.
7. **Reproduction check.** If `paths.reference_baseline_file` (`outputs/baseline_results.csv`, the notebook's baseline table) exists, `main.py` compares
   its baseline table against it and aborts on a mismatch. The file is not shipped in this repository, so the check currently reports `no reference file`.

# Overview of Key Findings from EDA

Full detail is in `EDA.ipynb` (sections 1-10); the findings that shaped the pipeline are:

* **Heavy-tailed target.** Median 1,400, mean 3,402, skewness 34, maximum 843,300. The top 1% of articles hold 22% of all shares. After `log1p` the skewness is 1.04,
  so the models are fitted on the log scale and reported on both scales.
* **`n_comments` is a leakage suspect.** `shares / n_comments` takes a handful of exact values (six ratios cover 43.8% of rows, versus 4.1% after shuffling) and
  `n_comments <= shares / 9` always. It is excluded; adding it back would lift validation R2 from about 0.16 to about 0.43.
* **`ID` is a reverse clock** (Spearman with `timedelta` = -0.99999), so it is not a feature. `timedelta` and the URL date carry the same information.
* **Placeholders and sentinels.** 1,061 empty articles (2.97%) carry placeholder zeros in four ratio/length columns; `kw_min_min = -1` in 57.9% of rows (a "not available" sentinel,
  also in 767 `kw_avg_min` and 6 `kw_min_avg` rows); `kw_max_max = 843,300` in 76.2% of rows is a ceiling.
* **Missing data.** `num_videos` is 47% missing but looks random; `data_channel` is 15% missing and informative (median 1,900 vs 1,400, 43% of the top-1% articles); `num_imgs` is 4% missing;
  a block of 721 rows loses five link/self-reference columns together.
* **No duplicates and no casing issues**, no impossible values.
* **Extreme feature values are informative, not errors** (they coincide with 1.3-1.5x higher median shares), so values are clipped, never deleted.
* **Weak signal spread over many features.** The best legitimate feature (`kw_avg_avg`) has Spearman 0.25 (about 6% of rank variance); then `kw_max_avg` (0.22) and the
  `self_reference_*` columns (0.17-0.19). Channel explains 5.5% and weekday 1.5% of the variance of log shares; weekday matters mainly as weekend vs weekday (weekend median 2,000 vs 1,400).
  Some weak features show thresholds or U-shapes (e.g. `num_hrefs`, `num_imgs`, `n_tokens_content`, `timedelta`), each worth at most about one point of R2, which favours tree ensembles and log transforms.
* **Expectation:** a modest ceiling on predictability. A low R2 on the raw scale is not a modelling bug.
* **Data caveat.** The `weekday` column disagrees with the weekday implied by the URL date for 24% of rows (91% in 2014-Q4); the reported `weekday` is the better-supported label and is used as given.

# Feature Handling Description

The 28 input columns (after dropping `ID` and the target) become **65 numeric model inputs**: 24 original numeric columns + 6 indicators + 21 engineered columns (8 hand-built, 13 log copies)
= 51 numeric columns, plus 14 one-hot columns. (`n_comments` is dropped and `URL` is consumed by feature engineering.)

| feature(s) | cleaning / processing | engineered or selected as |
|---|---|---|
| `ID` | dropped (time stamp in disguise) | not used |
| `shares` (target) | rows with missing or <= 0 target removed (none) | `log1p(shares)` as the training target |
| `n_comments` | dropped by `DataCleaner` (leakage-suspect) | not used |
| `URL` | not used directly | `pub_month` extracted from the date in the URL, then the column is dropped |
| `weekday` | stripped, lower-cased | one-hot (7 columns); `is_weekend` (Saturday/Sunday) |
| `data_channel` | stripped, lower-cased; missing becomes the level `unknown` | one-hot (7 columns); `channel_unknown` indicator |
| `n_unique_tokens`, `n_non_stop_words`, `n_non_stop_unique_tokens`, `average_token_length` | placeholder 0 of empty articles (`n_tokens_content == 0`) set to NaN | `empty_content` indicator; median imputation |
| `kw_min_min`, `kw_avg_min`, `kw_min_avg` | negative sentinel values set to NaN | `kw_min_min_sentinel` indicator; median imputation |
| `num_videos`, `num_imgs`, `num_hrefs` (and `num_self_hrefs`, `self_reference_*` in the 721-row block) | NaN kept, then median imputation | `num_videos_missing`, `num_imgs_missing`, `hrefs_block_missing` indicators |
| `num_hrefs`, `num_self_hrefs`, `num_imgs`, `n_tokens_content`, `self_reference_*`, `kw_*` (training skewness > 2) | all numeric columns that are non-negative, non-binary and have training skewness above 2 (13 columns on the training split) | extra `log_<column>` columns (`log1p`); the list is learned in `fit` |
| `num_imgs`, `num_hrefs`, `n_tokens_content`, `num_self_hrefs`, `num_videos`, `kw_max_avg`, `kw_min_avg` | ratios and counts derived from existing columns (NaN where undefined) | `imgs_per_100w`, `links_per_100w`, `external_hrefs`, `self_link_ratio`, `media_count`, `kw_range_avg` |
| all remaining numeric columns (`timedelta`, `n_tokens_title`, `num_keywords`, `kw_*`, ...) | used as given | numeric inputs |
| all numeric inputs | winsorised to the training 1st / 99th percentile (binary columns untouched), median-imputed, standardised | final model matrix: 65 columns, no NaN |

The 13 log-transformed columns on the training split are `n_tokens_content`, `num_hrefs`, `num_self_hrefs`, `num_imgs`, `num_videos`, `self_reference_min/max/avg_shares`,
`kw_max_min`, `kw_avg_min`, `kw_min_max`, `kw_max_avg` and `kw_avg_avg`. All data-dependent choices are made on training rows only.

# Explanation of Model Choices

The task is regression on `log1p(shares)`, so the registry holds eight regression models, including a naive benchmark. All are in `src/config.yaml`.

| model | why it is included | tuned how |
|---|---|---|
| Naive benchmark (train mean) | the floor every real model must beat (R2 = 0 by construction) | not tuned |
| Linear Regression | simple, interpretable baseline; the log and indicator features were built partly for linear models | grid: `fit_intercept`, `positive` |
| Ridge | handles the strong collinearity among `kw_*` and `self_reference_*` columns | grid: `alpha` (11 values, 1e-2 to 1e3) |
| Lasso | sparse linear model; its default `alpha = 1` is far too strong for this target and collapses to the benchmark | grid: `alpha` (10 values, 1e-4 to 10^-0.5) |
| Decision Tree | interpretable non-linear model; picks up the thresholds and U-shapes seen in the EDA, but overfits at default depth | successive halving: `max_depth`, `min_samples_leaf`, `max_features` |
| Random Forest | averages many trees to reduce variance; robust to skew, outliers and scaling | successive halving: trees, depth, leaf size, `max_features`, `max_samples` |
| HistGradientBoosting | strongest tabular learner here; handles non-linearity and interactions, fast on 25k rows | successive halving: learning rate, iterations, leaves, leaf size, L2 |
| K-Nearest Neighbors | distance-based non-parametric baseline | successive halving: `n_neighbors`, `weights`, `p` |

Selection rule (fixed in advance): the highest **validation R2 on `log1p(shares)`** among the tuned models. Tree ensembles are expected to win because the signal is weak,
non-linear in places, and spread across many features with outliers; linear models serve as a check that the added complexity pays off.
Because compute is limited to one CPU, the slow models search on a subsample with 3-fold CV and are then refit on all training rows.

# Evaluation of Models

**Metrics.** MAE, MSE, RMSE and R2, computed on the modelling scale `log1p(shares)` (primary, describes the typical article) and on raw shares (back-transformed with `expm1`,
log predictions clipped to [0, 20]). R2 is the ranking metric because it is directly comparable with the naive benchmark (R2 = 0). Raw-scale metrics are dominated by about 1% of viral articles,
so they are reported for completeness only.

**Validation split** (24,976 train rows; R2 on `log1p(shares)`; tuned models ranked by tuned R2). Baselines use library defaults.

| rank | model | baseline val R2 | tuned val R2 | change | tuned val RMSE (log) | tuned train-val R2 gap |
|---|---|---|---|---|---|---|
| 1 | HistGradientBoosting | 0.1581 | 0.1570 | -0.0011 | 0.8413 | 0.090 |
| 2 | Random Forest | 0.1498 | 0.1519 | +0.0020 | 0.8438 | 0.228 |
| 3 | K-Nearest Neighbors | -0.0048 | 0.1354 | +0.1402 | 0.8520 | 0.865 (see note) |
| 4 | Linear Regression | 0.1275 | 0.1275 | +0.0000 | 0.8559 | 0.025 |
| 5 | Ridge | 0.1274 | 0.1269 | -0.0005 | 0.8562 | 0.025 |
| 6 | Lasso | -0.0000 | 0.1268 | +0.1268 | 0.8562 | 0.025 |
| 7 | Decision Tree | -0.6887 | 0.0945 | +0.7832 | 0.8719 | 0.075 |
| - | Naive benchmark (train mean) | -0.0000 | not tuned | | 0.9163 | 0.000 |

KNN note: with `weights='distance'` every training point is its own zero-distance neighbour, so its train R2 is exactly 1.0 and the gap is an artefact; only the validation score is meaningful.

**Final model: HistGradientBoosting (tuned)** (learning rate 0.022, 305 iterations, 24 leaves, `min_samples_leaf` 183, L2 0.096), refit on train + validation (30,328 rows)
and evaluated once on the test split (5,352 rows):

| test split | R2 (log) | RMSE (log) | MAE (log) | RMSE (raw shares) | R2 (raw) |
|---|---|---|---|---|---|
| naive benchmark (train mean) | -0.0000 | 0.9416 | 0.7161 | 9,152 | -0.0334 |
| HistGradientBoosting, library defaults (reference) | 0.1706 | 0.8576 | 0.6347 | 9,023 | -0.0046 |
| **HistGradientBoosting, tuned (final)** | **0.1747** (95% CI 0.156-0.192) | 0.8554 | 0.6331 | 9,028 | -0.0056 |

Bootstrap (1,000 resamples of the test rows): RMSE is lower than the naive benchmark by 0.086 (95% CI 0.076-0.096), i.e. about 9% lower. The gain of the tuned model over its default-parameter twin is
+0.0044 R2 (95% CI -0.0013 to 0.0102), which is not distinguishable from zero.

**Reading the results**

* **The model is better than guessing the mean, but not by much.** About 17.5% of the variance of log shares is explained on unseen data; on raw share counts R2 is about 0 for every model, because the viral tail is not predictable from these features.
* **Tuning mainly repaired unsuitable defaults** (Lasso's `alpha`, the unpruned Decision Tree, KNN's `k = 5`) and reduced overfitting (Random Forest gap 0.73 -> 0.23, boosting 0.18 -> 0.09). For the best models the gain is within noise; the fold-to-fold standard deviation of CV R2 is about 0.008.
* **Ensembles lead, linear models are close.** Boosting and the forest are 0.005 apart (either would be defensible); the tuned linear models sit around 0.127.
* **Sensitivity checks (validation data only).** A time-ordered split (train on the oldest 70%, validate on the next 15%) gives R2 0.173 (defaults) / 0.175 (tuned), so the random split does not hide a time-drift problem.
  Adding the leakage-suspect `n_comments` back would give 0.433, which is why it stays excluded.
* **Residuals (test).** Mean residual about 0, 18.7% of articles are off by more than one log unit (a factor of 2.7); the mean residual by channel ranges from -0.115 (`lifestyle`) to +0.116 (`social_media`),
  and the `unknown` channel has the largest error (RMSE 1.11 vs 0.74-0.75 for `world`, `business`, `technology`).
* **Caveats.** One random 70/15/15 split and one seed; validation differences below about 0.01 R2 are not reliable; the slow-model searches use 3-fold CV on a 7,290-row subsample.

These numbers were reproduced by a fresh `python main.py` run (about 7 minutes on one CPU); the test metrics, selected parameters and baseline table match `EDA.ipynb`.

# Considerations for Deployment

* **Predictive value is modest.** About 17% of the variance of log shares is explained, and raw-scale R2 is about 0. The model suits ranking or triaging typical articles, not forecasting viral hits or exact share counts.
  Predictions are confined to a narrow band (roughly 6-8.6 on the log scale), and the model is least reliable for the `unknown`, `social_media` and `lifestyle` channels.
* **Leakage and feature availability.** Only features known at publication time should be used. `n_comments` is excluded; the `kw_*` and `self_reference_*` columns are statistics of other articles' shares,
  so they must be computed in production exactly as in this dataset and without using the article's own shares.
* **Reproducibility.** The whole preprocessing is one unfitted-then-fitted scikit-learn pipeline saved with the model in `artifacts/final_model.joblib` (together with `run_manifest.json`, which records library versions).
  Load it with the same scikit-learn and pandas versions. Inputs need the same raw columns as the training file (including `URL`, `weekday`, `data_channel`); the pipeline tolerates messy
  or unseen categories and missing values, and clips features to the training percentiles.
* **Monitoring and drift.** `timedelta` shows era effects and the channel mix shifts over time; a time-ordered split gave similar R2 (about 0.17), but the model should be re-validated and refitted on new data
  periodically, and the target mean, channel mix and feature ranges monitored. Treat the URL-derived `pub_month` and the `weekday` label with care (the two disagree in late 2014).
* **Scalability and latency.** Inference is a single pipeline transform plus a histogram gradient-boosting prediction: milliseconds per batch, small memory, no GPU. Training takes under 10 minutes on one CPU
  (tuning dominates), so periodic retraining is cheap. The pipeline is checkpointed and resumable.
* **Integration.** Wrap `joblib.load("artifacts/final_model.joblib")` in a batch job or a small API; return `expm1` of the prediction for share counts (and keep in mind that this under-predicts the mean on the raw scale).
  `main.py` is config-driven, so schedules, paths and parameters can be changed without touching code.
* **Remaining limitations.** One random split and one seed (differences below about 0.01 R2 are not reliable); 3-fold CV on a 7,290-row subsample for the slow models; the data file is a subset
  of a larger set (about 10% of IDs are absent).
