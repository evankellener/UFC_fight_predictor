"""MMA-AI architectural replication — AutoGluon WeightedEnsemble_L2, clean methodology.

Path B from the fork: match his exact architecture, but refuse his leakage.

Architecture (from his mma_ai_config.py + PDF p.37):
  - AutoGluon TabularPredictor, binary classification
  - hyperparameters = {"CAT": {}, "GBM": [{"extra_trees": True}, {}], "NN_TORCH": {}}
    (CatBoost + 2 LightGBM configs + NeuralNet)
  - num_stack_levels = 2, num_bag_folds = 4, num_bag_sets = 2
  - use_bag_holdout = True
  - eval_metric = "log_loss" (matches his calibration focus)
  - time_limit = 900s (per config)

Differences from his published v7 (to REMOVE leakage per LEAKAGE_REFERENCE.md §9):
  - NO `best_quality` preset (that's what mixed future data per Dec 5 2025 admission)
  - NO shuffle over train/val/test — use temporal split
  - tuning_data (val) is last 15% of training period BY DATE, not random

Leakage guardrails (per LEAKAGE_REFERENCE.md):
  §1  Temporal split: train/tuning strictly < 2024-05-04, test ∈ [2024-05-04, 2025-11-08].
      No shuffle over the train/test boundary. Val split inside train is temporal.
  §3  Features already built with d < fight_date in pipeline (verified earlier).
  §4  AutoGluon fits its own preprocessing (imputers/encoders) internally; with
      use_bag_holdout=True and no best_quality preset, it does not leak test data
      into fold fitting. Test DataFrame is never passed to .fit().
  §6  AutoGluon's internal CV (num_bag_folds=4) tunes hyperparameters on the
      training data only. The test window is never observed during training.
  §9  best_quality preset EXPLICITLY AVOIDED (was the source of MMA-AI's Dec 5
      leakage admission).
  §10 Single run. Single report.

Usage:  python3 scripts/run_mma_ai_autogluon.py
"""
import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, log_loss,
                             brier_score_loss, roc_auc_score)

warnings.filterwarnings("ignore")

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"
TEST_START = pd.Timestamp("2024-05-04")
TEST_END   = pd.Timestamp("2025-11-08")
TRAIN_START = pd.Timestamp("2016-01-01")
FILTER_THRESHOLD = 3

# Target: his post-leakage-fix clean number (Dec 5 2025 admission: 70% → 64%).
# Published leaky number was 70.32%; clean number per memory is ~64-67%.
TARGET_ACC, TARGET_LL = 0.7032, 0.5985       # his published v7 (leaky)
TARGET_CLEAN_ACC = 0.64                       # his post-fix "honest" number
TARGET_AUC, TARGET_BR = 0.7297, 0.2057

# Our blend result for reference
BLEND_ACC, BLEND_LL = 0.6919, 0.5974
BLEND_AUC, BLEND_BR = 0.7348, 0.2051


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(
        acc=float(accuracy_score(y, pred)),
        ll=float(log_loss(y, p)),
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
    )


def apply_filter(df):
    conn = sqlite3.connect(DB)
    hist = pd.read_sql("""
        SELECT w.jfighter, e.DATE FROM ufc_winlossko w
        JOIN ufc_event_details e ON e.jevent = w.jevent
    """, conn)
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fighter_dates = {f: grp["DATE"].values
                     for f, grp in hist.sort_values(["jfighter", "DATE"]).groupby("jfighter")}

    def prior_count(j, d):
        dates = fighter_dates.get(j, np.array([], dtype="datetime64[ns]"))
        return int((dates < np.datetime64(d)).sum()) if len(dates) else 0

    df = df.copy()
    df["f1_priors"] = df.apply(lambda r: prior_count(r["jfighter"], r["DATE"]), axis=1)
    df["f2_priors"] = df.apply(lambda r: prior_count(r["opp_jfighter"], r["DATE"]), axis=1)
    df = df[(df["f1_priors"] >= FILTER_THRESHOLD) & (df["f2_priors"] >= FILTER_THRESHOLD)]

    results = pd.read_sql("SELECT jevent, jbout, METHOD FROM ufc_fight_results", conn)
    results["METHOD_norm"] = results["METHOD"].str.lower().fillna("")
    conn.close()
    df = df.merge(results[["jevent", "jbout", "METHOD_norm"]],
                  on=["jevent", "jbout"], how="left")
    unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
    mask = df["METHOD_norm"].apply(
        lambda m: any(u in str(m) for u in unwanted) if pd.notna(m) else False
    )
    df = df[~mask]
    df = df[df["DATE"] >= TRAIN_START]
    return df.reset_index(drop=True)


def main():
    print("="*70)
    print("STEP 1: Load + filter features")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    print(f"  Raw: {len(df):,}")
    df = apply_filter(df)
    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  After filter — Train: {len(train):,}   Test: {len(test):,}")

    # ── Temporal train/val split (last 15% of train by DATE) ────────────
    train_sorted = train.sort_values("DATE").reset_index(drop=True)
    cal_start = int(len(train_sorted) * 0.85)
    fit_df = train_sorted.iloc[:cal_start].copy()
    val_df = train_sorted.iloc[cal_start:].copy()
    print(f"  Fit: {len(fit_df):,}   Val (last 15% by date): {len(val_df):,}")
    print(f"  Fit date range:  {fit_df['DATE'].min().date()} → {fit_df['DATE'].max().date()}")
    print(f"  Val date range:  {val_df['DATE'].min().date()} → {val_df['DATE'].max().date()}")
    print(f"  Test date range: {test['DATE'].min().date()} → {test['DATE'].max().date()}")

    # Features
    feat_cols = [c for c in df.columns if c.endswith("_diff") or c in
                 ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1")]
    usable = [c for c in feat_cols if c in fit_df.columns and fit_df[c].std() > 1e-8]
    print(f"  Usable features: {len(usable)}")

    # Add sample weight for recency (MMA-AI uses use_recency_weights=True)
    decay_lambda = 0.13
    fit_df = fit_df.copy()
    fit_df["sample_weight"] = np.exp(
        -decay_lambda * (TEST_START - fit_df["DATE"]).dt.days.values / 365.25
    )

    # Prepare DFs for AutoGluon (drop DATE, keep only features + label)
    label = "win"
    ag_cols = usable + [label, "sample_weight"]
    ag_train = fit_df[ag_cols].copy()
    ag_val   = val_df[usable + [label]].copy()
    ag_test  = test[usable + [label]].copy()

    # ── STEP 2: AutoGluon fit ───────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 2: AutoGluon fit (CatBoost + 2×LightGBM + NN_Torch, "
          "num_stack_levels=2, num_bag_folds=4, num_bag_sets=2)")
    print("="*70)
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label=label,
        eval_metric="log_loss",
        problem_type="binary",
        sample_weight="sample_weight",
        verbosity=2,
    )
    predictor.fit(
        train_data=ag_train,
        tuning_data=ag_val,           # explicit val, NOT random holdout
        use_bag_holdout=True,
        hyperparameters={
            "CAT": {},
            "GBM": [{"extra_trees": True}, {}],
            "NN_TORCH": {},
        },
        num_stack_levels=2,
        num_bag_folds=4,
        num_bag_sets=2,
        time_limit=900,
        calibrate=True,
        # CRITICAL: no `presets="best_quality"` — that was his Dec 5 leakage
        ag_args_fit={"num_cpus": 4},
    )

    # ── STEP 3: predict + metrics ───────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3: Predict on held-out test window (2024-05-04 → 2025-11-08)")
    print("="*70)
    p_test = predictor.predict_proba(ag_test.drop(columns=[label]))
    if hasattr(p_test, "iloc"):
        # DataFrame with class columns — take P(win=1)
        p_test = p_test[1].values if 1 in p_test.columns else p_test.iloc[:, 1].values
    y_test = ag_test[label].astype(int).values
    m = metrics(y_test, p_test)

    # Model leaderboard
    print("\nAutoGluon leaderboard (internal val metrics):")
    lb = predictor.leaderboard(silent=True)
    print(lb[["model", "score_val", "score_test" if "score_test" in lb.columns
              else "score_val"]].head(15).to_string())

    # ── STEP 4: summary table ───────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4: Summary")
    print("="*70)
    print(f"{'':34s}  {'n':>4s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    print(f"{'TARGET: MMA-AI v7 (leaky)':34s}  {411:>4d}  {TARGET_ACC*100:>6.2f}%  "
          f"{TARGET_LL:>8.4f}  {TARGET_AUC:>7.4f}  {TARGET_BR:>7.4f}")
    print(f"{'TARGET: MMA-AI post-fix (clean)':34s}  {411:>4d}  {TARGET_CLEAN_ACC*100:>6.2f}%  "
          f"{'-':>8s}  {'-':>7s}  {'-':>7s}")
    print(f"{'LR+CatBoost blend (ours, clean)':34s}  {422:>4d}  {BLEND_ACC*100:>6.2f}%  "
          f"{BLEND_LL:>8.4f}  {BLEND_AUC:>7.4f}  {BLEND_BR:>7.4f}")
    print(f"{'AutoGluon (this run, his arch)':34s}  {len(test):>4d}  {m['acc']*100:>6.2f}%  "
          f"{m['ll']:>8.4f}  {m['auc']:>7.4f}  {m['brier']:>7.4f}")

    # Save
    try:
        lb_data = lb.to_dict(orient="records")
    except Exception:
        lb_data = []
    out = {
        "n_test": int(len(test)),
        "n_train": int(len(fit_df)),
        "n_val": int(len(val_df)),
        "filter": {"threshold": FILTER_THRESHOLD, "method_strict": True,
                   "train_start": str(TRAIN_START.date())},
        "results": m,
        "target_published_leaky": {"acc": TARGET_ACC, "ll": TARGET_LL,
                                    "auc": TARGET_AUC, "brier": TARGET_BR},
        "target_postfix_clean_acc": TARGET_CLEAN_ACC,
        "baseline_blend": {"acc": BLEND_ACC, "ll": BLEND_LL,
                           "auc": BLEND_AUC, "brier": BLEND_BR},
        "ag_leaderboard": lb_data,
        "arch": "AutoGluon WeightedEnsemble_L2 (CAT + GBM×2 + NN_TORCH), "
                "num_stack_levels=2, num_bag_folds=4, num_bag_sets=2, "
                "use_bag_holdout=True, calibrate=True, NO best_quality preset",
    }
    (DT / "mmaai_autogluon_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {DT/'mmaai_autogluon_results.json'}")


if __name__ == "__main__":
    main()
