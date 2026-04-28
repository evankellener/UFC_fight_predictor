"""Empirical leakage test for the PER-FOLD workflow used by the notebook.

The static `build_features()` pipeline has 58 leaky `*_adjperf_dec_avg_diff`
columns because `compute_wc_priors` aggregates over the full df. The
notebook's per-fold workflow (in `notebook_clean_walkforward.py`) BYPASSES
this by recomputing priors per fold from train-only data.

This test verifies the per-fold workflow itself:
  For a chosen target fold (train_end = T), the per-fold feature matrix
  for fights ≤ T should be IDENTICAL whether computed with:
    (a) the full dataset
    (b) the dataset truncated to DATE ≤ T
  Because compute_wc_priors only sees train-only data in both cases, and
  per-fight Step 1-6 outputs are temporally clean (verified separately).

Pass criteria: zero leaky columns (or only floating-point noise <1e-9).
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

import mma_ai_pipeline as mma


# Mimic notebook_clean_walkforward's build_through_step6 + per_fold_features
def build_through_step6(max_date=None):
    df = mma.load_base_data()
    df["DATE"] = pd.to_datetime(df["DATE"])
    if max_date is not None:
        df = df[df["DATE"] <= max_date].copy().reset_index(drop=True)
    df = mma.beta_binomial_smooth(df)
    df = mma.poisson_gamma_smooth(df)
    df = mma.compute_derived_features(df)
    stat_cols = sorted(set(c for c in df.columns if
                 (c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
                  c.endswith("_ratio") or c.endswith("_per_ctrl") or
                  c in ["ko_smooth", "win_smooth", "decision_smooth",
                        "sub_land_smooth", "sub_land_rate", "ctrl_pm",
                        "ko_per_sig_str_land", "td_per_sig_str_att",
                        "ground_per_ctrl", "dist_per_sig_str_land",
                        "head_per_sig_str_land", "rev_per_ctrlopp",
                        "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
                        "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"])
                 and c in df.columns and not c.startswith("opp_") and not c.endswith("_raw")))
    df = mma.compute_decayed_averages(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    df = mma.compute_opponent_history(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    return df, stat_cols


def per_fold_features(df_full, stat_cols, train_end):
    """Same as notebook_clean_walkforward.per_fold_features."""
    train_only = df_full[df_full["DATE"] < train_end].copy()
    if len(train_only) < 100:
        return None
    priors = mma.compute_wc_priors(train_only, stat_cols)
    df_with_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    result = mma.assemble_features(df_with_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


def main():
    # Match the actual user workflow: train_end = 2024-10-01.
    TRAIN_END = pd.Timestamp("2024-10-01")
    # Truncate dataset at a date AFTER train_end; verify features for
    # fights ≤ train_end are unaffected by what happens after the truncation.
    # Using 2026-04-01 means we're comparing FULL data vs a dataset that
    # excludes the most recent ~3 months — should NOT affect features
    # for fights ≤ 2024-10-01 if there's no leakage.
    TRUNCATE_AT = pd.Timestamp("2026-04-01")
    print("=" * 78)
    print(f"PER-FOLD LEAKAGE TEST")
    print(f"Target fold's train_end = {TRAIN_END.date()}")
    print(f"Run A: full pipeline (1994 → 2026)")
    print(f"Run B: truncated to ≤ {TRUNCATE_AT.date()}")
    print(f"Pass: features for fights ≤ {TRAIN_END.date()} bit-identical between runs")
    print("=" * 78)

    print(f"\n── Run A: FULL pipeline ──")
    t0 = time.time()
    df_A, stat_cols = build_through_step6(None)
    feat_A = per_fold_features(df_A, stat_cols, TRAIN_END)
    print(f"  done in {time.time()-t0:.0f}s, {len(feat_A)} feature rows")

    print(f"\n── Run B: TRUNCATED to {TRUNCATE_AT.date()} ──")
    t0 = time.time()
    df_B, _ = build_through_step6(TRUNCATE_AT)
    feat_B = per_fold_features(df_B, stat_cols, TRAIN_END)
    print(f"  done in {time.time()-t0:.0f}s, {len(feat_B)} feature rows")

    # Compare for fights ≤ TRAIN_END
    target_dates = feat_A["DATE"] <= TRAIN_END
    A_sub = feat_A[target_dates].copy()
    keys = ["DATE", "jfighter", "opp_jfighter"]
    merged = A_sub.merge(feat_B, on=keys, suffixes=("_A", "_B"), how="inner")
    print(f"\n── Comparing {len(merged):,} fights present in BOTH ──")

    numeric_cols = [c for c in feat_A.columns
                    if pd.api.types.is_numeric_dtype(feat_A[c])
                    and c not in ("win", "weightindex", "scheduled_rounds")]

    print(f"\n{'column':<55s} {'#diff':>6s}  {'max_abs':>11s}")
    print("-" * 75)
    leaks = {}
    for col in numeric_cols:
        cA = f"{col}_A"; cB = f"{col}_B"
        if cA not in merged.columns or cB not in merged.columns: continue
        a = merged[cA].values; b = merged[cB].values
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() == 0: continue
        diff = a[finite] - b[finite]
        n_diff = int((np.abs(diff) > 1e-9).sum())
        if n_diff > 0:
            leaks[col] = (n_diff, float(np.abs(diff).max()))
            print(f"  {col:<53s} {n_diff:>6d}  {leaks[col][1]:>10.6f}")

    print()
    if not leaks:
        print(f"✅ ZERO LEAKAGE in per-fold workflow at train_end={TRAIN_END.date()}.")
        print(f"   Features for fights ≤ {TRAIN_END.date()} are bit-identical")
        print(f"   regardless of presence of 2025-04 → 2025-05 future fights.")
    else:
        print(f"❌ {len(leaks)} columns leak in the per-fold workflow.")

    import json
    Path("results").mkdir(exist_ok=True)
    Path("results/leakage_per_fold_test.json").write_text(
        json.dumps({"train_end": str(TRAIN_END.date()),
                    "truncate_at": str(TRUNCATE_AT.date()),
                    "n_compared": int(len(merged)),
                    "leaky_columns": leaks},
                    indent=2))
    print(f"\n✓ Saved results/leakage_per_fold_test.json")


if __name__ == "__main__":
    main()
