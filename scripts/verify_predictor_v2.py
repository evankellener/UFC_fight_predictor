"""Verify predictor_v2 produces predictions that match the training-time pipeline.

For every fight in the filtered test window (2024-05-04 → 2025-11-08), we:
  1. Run the TRAINING-TIME feature construction (via the threshold-sweep script)
     and single-shot LR on train < 2024-05-04
  2. Run the PREDICTOR_V2 inference path (same LR model artifact, but row built
     from per-fighter abs history + recomputed recency + Elo-as-of-date)
  3. Compare:
       - Mean absolute difference in p(f1 wins)
       - Accuracy match rate (both predict same winner)
       - Log loss parity
  4. Report any large discrepancies so we can track down feature-construction bugs.

If the two paths build the SAME row, predictions match to float-precision.
Any drift reveals a real inference-time bug.

Leakage: both paths use the same train cutoff. No leakage concern; this is
purely a construction-parity check.
"""
import json, sqlite3, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from predictor_v2 import PredictorV2
from run_threshold_sweep_both_elos import (
    load_base_both_elos, apply_threshold, attach_vegas,
    TEST_FIRST, TEST_LAST, TRAIN_ERA_FLOOR, TRAIN_YEARS, LAM
)

DT = Path("data/tmp")
OUT = Path("app/models/blend_v2")


def main():
    print("="*70)
    print("Predictor V2 — consistency verification vs training-time path")
    print("="*70)

    print("\n[1/4] Building training-time feature matrix (threshold sweep, both Elos)...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()
    print(f"  Test fights (threshold=3): {len(test)}")

    print("\n[2/4] Loading predictor_v2...")
    p2 = PredictorV2(artifact_dir=OUT, verbose=True)

    print("\n[3/4] Running training-time predictions (single-shot LR on < 2024-05-04)...")
    # Apply the SAME LR we saved to disk — so any divergence is PURELY from
    # feature-row construction, not from different model weights.
    feats = p2.feat_cols
    usable = [c for c in feats if c in df.columns]
    # Training slice
    train = df[df["DATE"] < TEST_FIRST]
    # Use predictor's imputer + scaler (they were fitted on train in build script)
    X_tr_row = [train[usable].values]  # sanity
    X_test = test[usable].values
    X_test = p2.imputer.transform(X_test)
    X_test_s = p2.scaler.transform(X_test)
    p_train_path = p2.lr.predict_proba(X_test_s)[:, 1]

    print(f"  Training-path predictions: {len(p_train_path)}")

    print("\n[4/4] Running predictor_v2 predictions one-at-a-time...")
    p_v2_path = np.zeros(len(test))
    failed = []
    for i, row in test.reset_index(drop=True).iterrows():
        jf1 = row["jfighter"]     # training convention: jfighter = f1
        jf2 = row["opp_jfighter"]
        # predictor_v2 uses (fighter_a, fighter_b) and alphabetizes internally
        # so to match the training f1-win convention, we call with (jf1, jf2)
        # and take p_fighter_a which = p(jf1) = p(f1) = what training predicts.
        r = p2.predict(jf1, jf2, event_date=row["DATE"])
        if r["success"]:
            p_v2_path[i] = r["p_fighter_a"]
        else:
            p_v2_path[i] = 0.5
            failed.append((jf1, jf2, r.get("error")))

    if failed:
        print(f"\n  ⚠ {len(failed)} failed predictions (using 0.5 fallback):")
        for f in failed[:5]:
            print(f"    {f[0]} vs {f[1]}: {f[2]}")

    # ── Compare ──────────────────────────────────────────────────────
    y = test["win"].astype(int).values

    def _m(p, name):
        p_c = np.clip(p, 0.02, 0.98)
        pred = (p_c >= 0.5).astype(int)
        return dict(name=name, acc=accuracy_score(y, pred), ll=log_loss(y, p_c))

    m_tr = _m(p_train_path, "training-path")
    m_v2 = _m(p_v2_path, "predictor_v2")

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"  {'Metric':<20s}  {'Training path':>15s}  {'Predictor V2':>15s}")
    print("  " + "-"*52)
    print(f"  {'Accuracy':<20s}  {m_tr['acc']*100:>14.2f}%  {m_v2['acc']*100:>14.2f}%")
    print(f"  {'Log loss':<20s}  {m_tr['ll']:>15.4f}  {m_v2['ll']:>15.4f}")

    diffs = p_train_path - p_v2_path
    print("\n  Per-bout probability difference (training - v2):")
    print(f"    mean abs diff:  {np.mean(np.abs(diffs)):.4f}")
    print(f"    median abs diff: {np.median(np.abs(diffs)):.4f}")
    print(f"    max abs diff:   {np.max(np.abs(diffs)):.4f}")
    print(f"    corr:           {np.corrcoef(p_train_path, p_v2_path)[0,1]:.4f}")
    # Agreement rate (both predict same winner)
    agree = ((p_train_path >= 0.5) == (p_v2_path >= 0.5)).mean()
    print(f"    sign agreement: {agree*100:.1f}%")

    # ── Show worst mismatches ────────────────────────────────────────
    idx_worst = np.argsort(-np.abs(diffs))[:10]
    print("\n  Top 10 worst mismatches:")
    print(f"  {'Date':<12s}  {'F1':<20s}  {'F2':<20s}  {'Train':>6s}  {'V2':>6s}  {'Δ':>6s}")
    for i in idx_worst:
        r = test.iloc[i]
        print(f"  {str(pd.Timestamp(r['DATE']).date()):<12s}  "
              f"{r['jfighter']:<20s}  {r['opp_jfighter']:<20s}  "
              f"{p_train_path[i]:>6.3f}  {p_v2_path[i]:>6.3f}  "
              f"{diffs[i]:>+6.3f}")

    (DT / "predictor_v2_verification.json").write_text(json.dumps({
        "n_test": int(len(test)),
        "training_path": m_tr, "predictor_v2": m_v2,
        "mean_abs_prob_diff": float(np.mean(np.abs(diffs))),
        "median_abs_prob_diff": float(np.median(np.abs(diffs))),
        "max_abs_prob_diff": float(np.max(np.abs(diffs))),
        "correlation": float(np.corrcoef(p_train_path, p_v2_path)[0, 1]),
        "sign_agreement": float(agree),
        "n_failed": len(failed),
    }, indent=2, default=str))
    print(f"\nSaved to {DT/'predictor_v2_verification.json'}")

    # Pass/fail heuristics
    print("\n" + "="*70)
    if np.mean(np.abs(diffs)) < 0.02 and agree > 0.97:
        print("✓ VERIFICATION PASSED — predictions match training path within tolerance")
    elif np.mean(np.abs(diffs)) < 0.05 and agree > 0.90:
        print("~ VERIFICATION PARTIAL — close but non-trivial drift; investigate top mismatches")
    else:
        print("✗ VERIFICATION FAILED — major drift; predictor_v2 row construction buggy")


if __name__ == "__main__":
    main()
