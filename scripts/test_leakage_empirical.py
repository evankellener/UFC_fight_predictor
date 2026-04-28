"""Empirical leakage test for the MMA-AI pipeline.

The gold-standard test: for a given fight at date T, the feature row should be
IDENTICAL whether you run the pipeline on:
  (a) the full dataset (1994 → 2026)
  (b) the dataset truncated to fights with DATE <= T

If features differ between (a) and (b), there's leakage from future data.

Tests a specific fight and reports which feature columns leak.
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


def run_full_pipeline(max_date=None):
    """Run pipeline through Step 9. Returns the final diffed feature DataFrame."""
    df = mma.load_base_data()
    df["DATE"] = pd.to_datetime(df["DATE"])
    if max_date is not None:
        before = len(df)
        df = df[df["DATE"] <= max_date].copy().reset_index(drop=True)
        print(f"  Truncated to DATE <= {max_date.date()}: {len(df)} / {before} rows")

    df = mma.beta_binomial_smooth(df)
    df = mma.poisson_gamma_smooth(df)
    df = mma.compute_derived_features(df)

    stat_cols = [c for c in df.columns if
                 c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
                 c.endswith("_ratio") or c.endswith("_per_ctrl") or
                 c in ["ko_smooth", "win_smooth", "decision_smooth",
                       "sub_land_smooth", "sub_land_rate", "ctrl_pm",
                       "ko_per_sig_str_land", "td_per_sig_str_att",
                       "ground_per_ctrl", "dist_per_sig_str_land",
                       "head_per_sig_str_land", "rev_per_ctrlopp",
                       "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
                       "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"]]
    stat_cols = sorted(set(c for c in stat_cols if c in df.columns and
                           not c.startswith("opp_") and not c.endswith("_raw")))

    df = mma.compute_decayed_averages(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    df = mma.compute_opponent_history(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    priors = mma.compute_wc_priors(df, stat_cols)
    df = mma.compute_adjperf(df, stat_cols, priors)
    result = mma.assemble_features(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


def main():
    # Pick a target fight: an early 2024 fight, so there's plenty of "future" data
    # in the full pipeline that the truncated pipeline won't see.
    TARGET_DATE = pd.Timestamp("2024-04-15")
    print("=" * 78)
    print(f"EMPIRICAL LEAKAGE TEST — target date {TARGET_DATE.date()}")
    print("Compares features for fights ON OR BEFORE target date computed with:")
    print("  (a) FULL pipeline (1994 → 2026)")
    print("  (b) TRUNCATED pipeline (1994 → 2024-04-15)")
    print("Any difference = features for that fight depend on future data = LEAK.")
    print("=" * 78)

    print("\n── Run 1: FULL pipeline ──")
    t0 = time.time()
    full_df = run_full_pipeline(max_date=None)
    print(f"  done in {time.time()-t0:.0f}s — {len(full_df)} rows")

    print("\n── Run 2: TRUNCATED pipeline ──")
    t0 = time.time()
    trunc_df = run_full_pipeline(max_date=TARGET_DATE)
    print(f"  done in {time.time()-t0:.0f}s — {len(trunc_df)} rows")

    # Compare: pick fights with DATE close to target_date that exist in BOTH
    full_subset = full_df[full_df["DATE"] <= TARGET_DATE].copy()
    common = full_subset.merge(
        trunc_df[["DATE", "jfighter", "opp_jfighter"]],
        on=["DATE", "jfighter", "opp_jfighter"], how="inner")
    print(f"\n── Comparing {len(common)} fights present in BOTH runs ──")

    # Numeric columns to compare
    numeric_cols = [c for c in full_df.columns
                    if pd.api.types.is_numeric_dtype(full_df[c])
                    and c not in ("win", "weightindex", "scheduled_rounds")]

    full_mat = full_subset.merge(
        trunc_df, on=["DATE", "jfighter", "opp_jfighter"],
        suffixes=("_full", "_trunc"), how="inner")

    leaks = {}
    print(f"\n{'column':<55s} {'#diff':>6s}  {'max_abs_diff':>13s}  {'mean_abs_diff':>13s}")
    print("-" * 95)
    for col in numeric_cols:
        full_col = f"{col}_full"; trunc_col = f"{col}_trunc"
        if full_col not in full_mat.columns or trunc_col not in full_mat.columns:
            continue
        a = full_mat[full_col].values; b = full_mat[trunc_col].values
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() == 0: continue
        diff = a[finite] - b[finite]
        n_diff = int((np.abs(diff) > 1e-9).sum())
        if n_diff > 0:
            leaks[col] = {"n_diff": n_diff,
                          "max_abs": float(np.abs(diff).max()),
                          "mean_abs": float(np.abs(diff).mean())}
            print(f"  {col:<53s} {n_diff:>6d}  {leaks[col]['max_abs']:>12.6f}   {leaks[col]['mean_abs']:>12.6f}")

    print()
    if leaks:
        print(f"❌ LEAKAGE DETECTED in {len(leaks)} feature columns")
        print(f"   These features for fights ≤ {TARGET_DATE.date()} change when")
        print(f"   future fights are removed from the pipeline. NOT zero-leakage.")
    else:
        print(f"✅ NO LEAKAGE DETECTED — features for fights ≤ {TARGET_DATE.date()}")
        print(f"   are bit-identical regardless of future data presence.")

    # Save results
    import json
    Path("results").mkdir(exist_ok=True)
    Path("results/leakage_empirical_test.json").write_text(
        json.dumps({"target_date": str(TARGET_DATE.date()),
                    "n_compared_fights": int(len(full_mat)),
                    "leaky_columns": leaks},
                    indent=2))
    print(f"\n✓ Saved results/leakage_empirical_test.json")
    return leaks


if __name__ == "__main__":
    main()
