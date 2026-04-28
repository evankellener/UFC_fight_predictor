"""Diagnostic: which specific fights differ between full and truncated runs?

Picks a small column (ko_per_sig_str_land) and writes the differing fights
with their dates, fighters, and underlying smoothed values. Goal: identify
the pattern (early-career fights? specific weight classes? edge case in a
ratio?).
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


def build_through_step6(max_date=None):
    """Run pipeline through Step 6 only (not 7-9, no compute_wc_priors).
    Returns the per-fight clean df with smoothed/decayed/opp values."""
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


def main():
    TARGET = pd.Timestamp("2024-04-15")
    print("=" * 78)
    print(f"DIAGNOSTIC — find which fights differ between full and truncated runs")
    print(f"Target date: {TARGET.date()}")
    print(f"Comparing Step 1-6 outputs (per-fight clean — no compute_wc_priors)")
    print("=" * 78)

    print("\n── Run 1: FULL ──")
    t0 = time.time()
    df_full, _ = build_through_step6(None)
    print(f"  done {time.time()-t0:.0f}s, {len(df_full):,} rows")

    print("\n── Run 2: TRUNCATED ──")
    t0 = time.time()
    df_trunc, _ = build_through_step6(TARGET)
    print(f"  done {time.time()-t0:.0f}s, {len(df_trunc):,} rows")

    # Restrict full to fights ≤ target_date, then merge on (jevent, jbout, jfighter)
    full_sub = df_full[df_full["DATE"] <= TARGET].copy()
    keys = ["jevent", "jbout", "jfighter"]

    # Pick a few columns of interest
    cols_to_check = [
        "ctrl_smooth", "ctrl_ratio", "ctrl_pm",
        "ko_smooth", "ko_ratio", "ko_per_sig_str_land",
        "head_per_sig_str_land", "dist_per_sig_str_land",
        "rev_smooth", "rev_ratio",
        "sub_att_smooth", "sub_att_ratio",
        "ctrl_ratio_dec_avg", "ko_per_sig_str_land_dec_avg",
    ]

    merged = full_sub.merge(df_trunc, on=keys, suffixes=("_full", "_trunc"), how="inner")
    print(f"\n── Comparing {len(merged):,} fights present in BOTH runs (Step 1-6 outputs) ──")

    print(f"\n{'column':<35s} {'#diff':>6s}  {'max_abs_diff':>13s}  {'mean_abs_diff':>13s}")
    print("-" * 75)
    leaks = {}
    for c in cols_to_check:
        cf = f"{c}_full"; ct = f"{c}_trunc"
        if cf not in merged.columns or ct not in merged.columns:
            continue
        a = merged[cf].values; b = merged[ct].values
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() == 0: continue
        diff = a[finite] - b[finite]
        n_diff = int((np.abs(diff) > 1e-9).sum())
        if n_diff > 0:
            leaks[c] = (n_diff, float(np.abs(diff).max()), float(np.abs(diff).mean()))
            print(f"  {c:<33s} {n_diff:>6d}  {leaks[c][1]:>12.6f}   {leaks[c][2]:>12.6f}")

    # For one specific leaky column at the BASE level (not dec_avg), inspect the rows
    base_col = None
    for c in ("ctrl_smooth", "ko_smooth", "rev_smooth", "sub_att_smooth"):
        if c in leaks:
            base_col = c; break
    print()
    if base_col:
        print(f"── Inspecting leaky base column '{base_col}' ──")
        cf = f"{base_col}_full"; ct = f"{base_col}_trunc"
        diff_mask = (np.abs(merged[cf] - merged[ct]) > 1e-9) & merged[cf].notna() & merged[ct].notna()
        differing = merged[diff_mask].copy()
        differing["abs_diff"] = (differing[cf] - differing[ct]).abs()
        differing = differing.sort_values("abs_diff", ascending=False)
        cols_show = ["DATE_full", "jevent", "jbout", "jfighter", "weightindex_full", cf, ct, "abs_diff"]
        cols_show = [c for c in cols_show if c in differing.columns]
        print(f"\nTop 15 differing rows for {base_col}:")
        print(differing[cols_show].head(15).to_string())
        # Date distribution
        if "DATE_full" in differing.columns:
            differing["year"] = pd.to_datetime(differing["DATE_full"]).dt.year
            year_dist = differing.groupby("year").size().reset_index(name="n_diffs")
            print(f"\nYear distribution of differing rows for {base_col}:")
            print(year_dist.to_string(index=False))
        # WC distribution
        if "weightindex_full" in differing.columns:
            wc_dist = differing.groupby("weightindex_full").size().reset_index(name="n_diffs")
            print(f"\nWeightindex distribution of differing rows for {base_col}:")
            print(wc_dist.to_string(index=False))
    else:
        print("✓ No leaks at the BASE smoothed-stat level. Leaks are in derived ratios.")
        # Show the derived ratio leaks
        for c in ("ctrl_ratio", "ko_per_sig_str_land", "head_per_sig_str_land"):
            if c in leaks:
                cf = f"{c}_full"; ct = f"{c}_trunc"
                diff_mask = (np.abs(merged[cf] - merged[ct]) > 1e-9) & merged[cf].notna() & merged[ct].notna()
                differing = merged[diff_mask].copy()
                if len(differing) == 0: continue
                differing["abs_diff"] = (differing[cf] - differing[ct]).abs()
                differing = differing.sort_values("abs_diff", ascending=False)
                print(f"\n── Top 10 differing rows for derived ratio '{c}' ──")
                show_cols = ["DATE_full", "jfighter", "weightindex_full",
                             f"sig_str_land_smooth_full", f"sig_str_land_smooth_trunc",
                             cf, ct, "abs_diff"]
                show_cols = [x for x in show_cols if x in differing.columns]
                print(differing[show_cols].head(10).to_string())


if __name__ == "__main__":
    main()
