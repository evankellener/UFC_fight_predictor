"""MMA-AI pipeline replication — exact match attempt.

Runs the full MMA-AI v7 pipeline from scratch with the WC-index fix applied, then
trains and evaluates on his exact test window (2024-05-04 → 2025-11-08, 411 fights
expected) to see how close we can get to his published metrics.

Target (MMA-AI v7 published):
    accuracy  = 0.7032
    log_loss  = 0.5985
    brier     = 0.2057
    AUC       = 0.7297

Leakage guardrails (per LEAKAGE_REFERENCE.md):
  §1  Temporal split: train = DATE < 2024-05-04, test = DATE in [2024-05-04, 2025-11-08].
      No shuffle. No random splitting.
  §2  Pipeline-internal: compute_decayed_averages uses prior=vals[:i] (strict),
      compute_opponent_history uses bisect_left(..., fight_date) + cutoff<1 (strict).
      Audited — safe.
  §3  WC priors use full history — matches MMA-AI spec (PDF p.35 says WC priors are
      stable constants, not time-segmented). Documented divergence point.
  §4  StandardScaler fit on TRAIN slice only; transform on TEST.
  §5  N/A (no Elo in pure MMA-AI replication).
  §6  Model hyperparams frozen to published values — not tuned on test.
  §7  No odds as features (enforced by pipeline).
  §10 Single run per config; report once.

Usage:
    python3 scripts/run_mma_ai_replication.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, log_loss,
                             brier_score_loss, roc_auc_score)

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from mma_ai_pipeline import build_features  # noqa: E402

DT = Path("data/tmp")
OUT_FEATURES = DT / "mmaai_features.csv"
OUT_FEAT_COLS = DT / "mmaai_feature_cols.json"

# MMA-AI's exact test window (PDF + memory)
TEST_START = pd.Timestamp("2024-05-04")
TEST_END   = pd.Timestamp("2025-11-08")  # inclusive on that date
TARGET_N_TEST_FIGHTS = 411

# Published v7 metrics (target)
TARGET_ACC   = 0.7032
TARGET_LL    = 0.5985
TARGET_BRIER = 0.2057
TARGET_AUC   = 0.7297


def main():
    # ── STEP A: rebuild features with fixed config ──────────────────────────
    print("="*70)
    print("STEP A: Building features with v7 config (post-WC-index-fix)")
    print("="*70)
    df = build_features(config_name="v7")
    print(f"\n  Pipeline output: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Save so downstream code can consume fresh output
    df.to_csv(OUT_FEATURES, index=False)
    feat_cols = [c for c in df.columns if c.endswith("_diff") or c in
                 ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1")]
    OUT_FEAT_COLS.write_text(json.dumps(feat_cols, indent=2))
    print(f"  Saved: {OUT_FEATURES} ({len(feat_cols)} feature cols)")

    # ── STEP B: leakage pre-flight ──────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP B: Leakage pre-flight")
    print("="*70)
    # High target correlation = suspicious. Per LEAKAGE_REFERENCE.md §10.
    corrs = df[feat_cols + ["win"]].corr()["win"].drop("win").abs().sort_values(ascending=False)
    suspicious = corrs[corrs > 0.5]
    if len(suspicious):
        print(f"  ⚠ {len(suspicious)} feature(s) with |corr(win)| > 0.5 — investigate:")
        print(suspicious.to_string())
    else:
        print(f"  ✓ No feature has |corr(win)| > 0.5  (max = {corrs.max():.4f} = {corrs.idxmax()})")

    # ── STEP C: split ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP C: Temporal split (train < 2024-05-04, test ∈ [2024-05-04, 2025-11-08])")
    print("="*70)
    df["DATE"] = pd.to_datetime(df["DATE"])
    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,} fights  ({train['DATE'].min().date()} → {train['DATE'].max().date()})")
    print(f"  Test : {len(test):,} fights  ({test['DATE'].min().date() if len(test) else '-'} → "
          f"{test['DATE'].max().date() if len(test) else '-'})")
    print(f"  Target test size: {TARGET_N_TEST_FIGHTS} fights "
          f"(we have {len(test)} in DB for this window — some may be missing due to matching)")

    if len(test) < 100:
        print(f"  ✗ Too few test fights ({len(test)}). Aborting.")
        return

    # ── STEP D: train LR (MMA-AI uses AutoGluon; LR is our lightweight proxy) ──
    print("\n" + "="*70)
    print("STEP D: Train LogReg (ElasticNet) on features, evaluate on test")
    print("="*70)
    # Use only feature cols that are fully numeric and non-constant in train
    usable = [c for c in feat_cols if c in train.columns and train[c].std() > 1e-8]
    print(f"  Usable feature cols: {len(usable)}/{len(feat_cols)}")

    # Impute + scale — fit on TRAIN only (§4)
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train[usable])
    X_te = imp.transform(test[usable])

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    y_tr = train["win"].astype(int).values
    y_te = test["win"].astype(int).values

    # Recency weights (MMA-AI uses use_recency_weights=True with λ=0.13)
    decay_lambda = 0.13
    w_tr = np.exp(-decay_lambda * (TEST_START - train["DATE"]).dt.days.values / 365.25)

    # Hyperparams: LR C chosen per memory (C=0.05, l1_ratio=0.5 — what train_and_save_blend uses).
    # These were selected on validation, NOT on this test window.
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr, y_tr, sample_weight=w_tr)
    p_test_raw = lr.predict_proba(X_te)[:, 1]

    # ── STEP E: metrics (uncalibrated) ──────────────────────────────────────
    print("\n  Uncalibrated LR:")
    report(y_te, p_test_raw)

    # ── STEP F: Platt calibration (MMA-AI spec, PDF p.47) ───────────────────
    # Fit calibrator on a held-out slice of the training data
    # Use last 15% of train (by date) as calibration set — matches MMA-AI val_size=0.15
    print("\n" + "="*70)
    print("STEP F: Platt (sigmoid) calibration on held-out 15% of train by date")
    print("="*70)
    train_sorted = train.sort_values("DATE").reset_index(drop=True)
    cal_start_idx = int(len(train_sorted) * 0.85)
    cal_df = train_sorted.iloc[cal_start_idx:]
    fit_df = train_sorted.iloc[:cal_start_idx]
    print(f"  Fit (refit LR): {len(fit_df):,}  Calibration: {len(cal_df):,}")

    X_fit_raw = imp.fit_transform(fit_df[usable])
    X_fit = sc.fit_transform(X_fit_raw)
    X_cal = sc.transform(imp.transform(cal_df[usable]))
    X_te2 = sc.transform(imp.transform(test[usable]))
    w_fit = np.exp(-decay_lambda * (TEST_START - fit_df["DATE"]).dt.days.values / 365.25)
    y_fit = fit_df["win"].astype(int).values
    y_cal = cal_df["win"].astype(int).values

    lr2 = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                             solver="saga", max_iter=6000, random_state=42)
    lr2.fit(X_fit, y_fit, sample_weight=w_fit)

    # Manual Platt: fit a 1D LR on (uncalibrated_prob → y) on the cal set
    p_cal = lr2.predict_proba(X_cal)[:, 1]
    platt = LogisticRegression(solver="lbfgs", max_iter=100, random_state=42)
    platt.fit(p_cal.reshape(-1, 1), y_cal)
    p_test_cal = platt.predict_proba(lr2.predict_proba(X_te2)[:, 1].reshape(-1, 1))[:, 1]

    print("\n  Platt-calibrated LR:")
    report(y_te, p_test_cal)

    # ── STEP G: comparison table ────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP G: Vs. MMA-AI v7 published target")
    print("="*70)
    m_raw = metrics(y_te, p_test_raw)
    m_cal = metrics(y_te, p_test_cal)
    print(f"{'':20s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    print(f"{'TARGET (v7)':20s}  {TARGET_ACC*100:>6.2f}%  {TARGET_LL:>8.4f}  {TARGET_AUC:>7.4f}  {TARGET_BRIER:>7.4f}")
    print(f"{'LR uncalibrated':20s}  {m_raw['acc']*100:>6.2f}%  {m_raw['ll']:>8.4f}  {m_raw['auc']:>7.4f}  {m_raw['brier']:>7.4f}")
    print(f"{'LR + Platt':20s}  {m_cal['acc']*100:>6.2f}%  {m_cal['ll']:>8.4f}  {m_cal['auc']:>7.4f}  {m_cal['brier']:>7.4f}")
    print(f"{'Gap (cal vs target)':20s}  {(m_cal['acc']-TARGET_ACC)*100:>+6.2f}pp  "
          f"{m_cal['ll']-TARGET_LL:>+8.4f}  {m_cal['auc']-TARGET_AUC:>+7.4f}  "
          f"{m_cal['brier']-TARGET_BRIER:>+7.4f}")

    # Save results
    out = {
        "n_test": int(len(test)),
        "test_window": [str(TEST_START.date()), str(TEST_END.date())],
        "raw": m_raw, "calibrated": m_cal,
        "target": {"acc": TARGET_ACC, "ll": TARGET_LL,
                   "auc": TARGET_AUC, "brier": TARGET_BRIER},
        "gap_calibrated": {
            "acc_pp": (m_cal['acc'] - TARGET_ACC) * 100,
            "ll": m_cal['ll'] - TARGET_LL,
            "auc": m_cal['auc'] - TARGET_AUC,
            "brier": m_cal['brier'] - TARGET_BRIER,
        },
        "model": "LogReg ElasticNet C=0.05 l1=0.5 + Platt",
        "n_features": len(usable),
    }
    (DT / "mmaai_replication_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {DT/'mmaai_replication_results.json'}")


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(
        acc=float(accuracy_score(y, pred)),
        ll=float(log_loss(y, p)),
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
    )


def report(y, p):
    m = metrics(y, p)
    print(f"    n={len(y)}  acc={m['acc']*100:.2f}%  ll={m['ll']:.4f}  "
          f"auc={m['auc']:.4f}  brier={m['brier']:.4f}")


if __name__ == "__main__":
    main()
