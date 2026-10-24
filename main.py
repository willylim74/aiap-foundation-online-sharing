"""Entry point: config -> data preparation -> split -> baselines -> tuning -> final one-time test evaluation.

Usage (works from ANY working directory; every path is resolved relative to this file):

    python main.py                  # full run: baselines, tuning (checkpointed / resumable), final test evaluation
    python main.py --skip-tuning    # fast mode: data prep, split and baselines only (no tuning, no test-set use)
    python main.py --self-test      # run the embedded data-preparation and training self-tests and exit
    python main.py --fresh          # ignore / delete existing checkpoints and start the tuning from scratch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # project root = the folder that contains this file
sys.path.insert(0, str(ROOT / "src"))              # the src/ import, independent of the caller's working directory

import pandas as pd  # noqa: E402

import data_preparation as dp  # noqa: E402
import model_training as mt  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="News-popularity (shares) regression pipeline")
    ap.add_argument("--config", default=str(ROOT / "src" / "config.yaml"), help="path to config.yaml (default: src/config.yaml next to main.py)")
    ap.add_argument("--skip-tuning", action="store_true", help="fast mode: stop after the baselines (no tuning, test split untouched)")
    ap.add_argument("--fresh", action="store_true", help="delete existing checkpoints and recompute everything")
    ap.add_argument("--output-dir", default=None, help="override paths.output_dir (relative paths are resolved against the project root)")
    ap.add_argument("--self-test", action="store_true", help="run the embedded self-tests and exit")
    return ap.parse_args(argv)


def banner(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100, flush=True)


def main(argv=None):
    args = parse_args(argv)
    t_start, timings = time.time(), {}
    if args.self_test:
        dp.run_self_tests(dp.load_config(args.config))
        mt._self_test()
        return 0

    cfg = dp.load_config(args.config)
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
        cfg["paths"]["checkpoint_dir"] = str(Path(args.output_dir) / "checkpoints")
    out_dir = dp.resolve_path(cfg, cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _threads = mt.set_threads(cfg)                  # single-CPU sandbox; keep the reference alive so the limit stays active
    mode = "skip-tuning" if args.skip_tuning else "full"
    print(f"project root : {ROOT}\nconfig       : {cfg['_config_path']}\nmode         : {mode}\noutput dir   : {out_dir}")

    # ---- stage 1: data preparation (row-level cleaning, target engineering; runs once, before the split)
    banner("STAGE 1 - data preparation")
    t0 = time.time()
    raw = dp.load_raw(cfg)
    cleaned, report = dp.clean_rows(raw, cfg)
    X_all, y_raw, y_log = dp.make_xy(cleaned, cfg)
    print("cleaning report:", report)
    print(f"features: {X_all.shape[1]} raw columns | target: {cfg['target']['name']} -> {cfg['target']['transform']} (mean {y_log.mean():.3f})")
    print("leakage-suspect columns excluded by the pipeline:", cfg["columns"]["leaky"])
    timings["data_preparation"] = time.time() - t0

    # ---- stage 2: split
    banner("STAGE 2 - train / validation / test split")
    data = mt.split_data(X_all, y_raw, y_log, cfg)
    data["split_ids"] = dict(train_ids=[int(i) for i in cleaned.loc[data["X_train"].index, cfg["columns"]["id"]]],
                             val_ids=[int(i) for i in cleaned.loc[data["X_val"].index, cfg["columns"]["id"]]],
                             test_ids=[int(i) for i in cleaned.loc[data["X_test"].index, cfg["columns"]["id"]]], seed=cfg["project"]["random_seed"])
    print(f"train {len(data['X_train']):,} | validation {len(data['X_val']):,} | test {len(data['X_test']):,} (test split locked until the final evaluation)")

    # ---- stage 3: baselines (library defaults)
    banner("STAGE 3 - baseline fitting (library defaults; validation split)")
    t0 = time.time()
    baseline, _ = mt.fit_baselines(cfg, data)
    timings["baselines"] = time.time() - t0
    baseline.to_csv(out_dir / cfg["outputs"]["baseline_results"])
    shown = baseline[["rank_by_val_R2", "val_R2_log", "val_RMSE_log", "val_MAE_log", "val_MSE_log", "train_R2_log", "overfit_gap_R2", "val_R2_raw", "val_RMSE_raw"]]
    print(shown.round(4).to_string())
    status, worst, _ = mt.check_baselines_against_notebook(baseline, cfg)
    if worst is None:
        print(f"\nnotebook reproduction check: {status}")
    else:
        print(f"\nnotebook reproduction check: baselines {status} the notebook's baseline table (max |difference|/(1+|reference|) = {worst:.2e}, tolerance {cfg['evaluation']['baseline_reproduction_tolerance']:.0e})")
        if status != "match":
            print("ERROR: the production baselines do not reproduce the notebook's baselines", file=sys.stderr)
            return 2

    if args.skip_tuning:
        timings["total"] = time.time() - t_start
        mt.write_manifest(cfg, out_dir, mode, timings, dict(baseline_reproduction=dict(status=status, max_abs_diff=worst)))
        banner("DONE (skip-tuning mode): baselines written; tuning, selection and test evaluation were skipped")
        print(f"artifacts in {out_dir}")
        return 0

    # ---- stage 4: tuning (grid for the linear models, successive halving on a subsample + refit for the slow ones), checkpointed
    banner("STAGE 4 - hyper-parameter tuning (single CPU; resumable)")
    ledger = mt.CheckpointLedger(dp.resolve_path(cfg, cfg["paths"]["checkpoint_dir"]), cfg["outputs"]["ledger"])
    if args.fresh:
        ledger.clear()
        print("checkpoints cleared (--fresh)")
    print("ledger entries present:", list(ledger.data) or "none (clean start)")
    t0 = time.time()
    tuned = mt.tune_all(cfg, data, baseline, ledger)
    timings["tuning"] = time.time() - t0
    tuned_table = mt.build_tuned_table(tuned, baseline)
    print()
    print(tuned_table.drop(columns=["best_params"]).to_string(float_format=lambda x: f"{x:,.4f}"))

    # ---- stage 5: selection + the one-time test-set evaluation (+ validation-only sensitivity checks)
    banner("STAGE 5 - final model selection and one-time test evaluation")
    t0 = time.time()
    final_name = tuned_table.index[0]
    final_params = tuned[final_name][0]["best_params"]
    print(f"selected model (highest validation R2): {final_name} | params: {final_params}")
    test_ids = cleaned.loc[data["X_test"].index, cfg["columns"]["id"]].values
    final_rec, final_obj = mt.evaluate_test_once(cfg, data, tuned_table, tuned, ledger, test_ids)
    timings["final_evaluation"] = time.time() - t0
    M = final_rec["metrics"]
    tab = pd.DataFrame({"naive benchmark (reference)": M["naive"], f"{final_name} - library defaults (reference)": M["baseline_same_model"], f"{final_name} - TUNED (final)": M["final"]})
    tab.index = [i.replace("test_", "") for i in tab.index]
    print(tab.to_string(float_format=lambda x: f"{x:,.4f}"))
    b = final_rec["bootstrap"]
    print(f"\ntest R2 (log) = {b['final_R2_log']['point']:.4f}  95% CI [{b['final_R2_log']['ci95_low']:.4f}, {b['final_R2_log']['ci95_high']:.4f}]")
    print(f"RMSE improvement over the naive benchmark: {b['rmse_improvement_vs_naive']['mean']:.4f} [{b['rmse_improvement_vs_naive']['ci95_low']:.4f}, {b['rmse_improvement_vs_naive']['ci95_high']:.4f}]")
    print(f"R2 gain over the default-parameter twin   : {b['r2_gain_vs_baseline_same_model']['mean']:.4f} [{b['r2_gain_vs_baseline_same_model']['ci95_low']:.4f}, {b['r2_gain_vs_baseline_same_model']['ci95_high']:.4f}]")
    residuals = mt.residuals_by_group(cfg, data["X_test"], data["y_test"], final_obj["pred_test"])
    sens = None
    if cfg["evaluation"]["sensitivity"]["enabled"]:
        sens = mt.sensitivity_checks(cfg, X_all, y_log, data, baseline, tuned_table, tuned, final_name, final_params)
        print("\nsensitivity checks (validation data only):")
        print(sens.round(4).to_string())
    summary = mt.write_outputs(cfg, out_dir, baseline, tuned_table, tuned, final_rec, final_obj, data, sens, residuals, test_ids)
    timings["total"] = time.time() - t_start
    mt.write_manifest(cfg, out_dir, mode, timings, dict(selected_model=final_name, baseline_reproduction=dict(status=status, max_abs_diff=worst)))
    banner("DONE")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))
    print(f"\nartifacts written to {out_dir}\ntotal time {timings['total'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
