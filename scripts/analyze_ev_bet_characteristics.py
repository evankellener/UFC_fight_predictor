"""Tier 1 diagnostic — slice the 168 +EV bets 6 ways to find the edge source.

Shows per-slice n, win rate, mean odds, ROI (flat $1), bootstrap 95% CI.
Every slice should sum to ~168 (with ± 1-2 due to ties/missing data).

Slices:
  A. Favorite (vegas implied > 50%) vs Underdog
  B. Heavy favorites (>70%) / moderate (55-70%) / close (50-55%) / dogs (<50%)
  C. Model confidence on pick: p >= 0.65 / 0.55-0.65 / 0.50-0.55
  D. Edge magnitude: 1-5pp / 5-10pp / 10pp+
  E. Weight class tier (flyweight/lightweight/welter/heavyweight grouping)
  F. Event type (PPV UFC 100s / Fight Night / Apex-only)

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§11):
  Same as run_bet_sizing — uses final production model predictions.
  Sliced analysis is POST-HOC on the test-set predictions; we're describing
  *where* the edge lives, not tuning a strategy on test (no hyperparam
  selection).
"""
import json, sqlite3, sys, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats as scistats

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
from run_bet_sizing import (load_final_model_df, apply_filter, attach_vegas,
                             build_wf_predictions)

DT = Path("data/tmp")
DB = "data/sqlite_db/slim_scrapper.db"
RNG = np.random.default_rng(42)

WC_NAMES = {1: "W.Straw", 2: "W.Fly", 3: "W.Bantam", 4: "W.Feather",
            5: "Flyweight", 6: "Bantam", 7: "Feather", 8: "Light",
            9: "Welter", 10: "Middle", 11: "LHW", 12: "HW"}


def bootstrap_roi(profits, n_boot=1000):
    if len(profits) < 2: return 0.0, np.nan, np.nan
    rois = [RNG.choice(profits, len(profits), replace=True).mean()
            for _ in range(n_boot)]
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(profits.mean()), float(lo), float(hi)


def ttest_p(profits):
    if len(profits) < 2: return np.nan
    t, p_two = scistats.ttest_1samp(profits, 0.0)
    return float(p_two / 2 if t > 0 else 1 - p_two / 2)


def slice_report(df_slice, label):
    """Print formatted row: n, win%, avg dec odds, mean edge, ROI, CI, p."""
    n = len(df_slice)
    if n == 0:
        print(f"  {label:<32s}  n=0   (no bets)")
        return None
    wins = (df_slice["profit"] > 0).sum()
    win_rate = wins / n
    avg_dec = df_slice["dec_pick"].mean()
    avg_edge = df_slice["edge"].mean() * 100
    profits = df_slice["profit"].values
    roi, lo, hi = bootstrap_roi(profits)
    p = ttest_p(profits)
    sig = "✓" if p < 0.05 else ""
    print(f"  {label:<32s}  n={n:>3d}  win%={win_rate*100:>5.1f}  "
          f"odds={avg_dec:>4.2f}  edge={avg_edge:>4.1f}pp  "
          f"ROI={roi*100:>+6.2f}%  CI=[{lo*100:>+6.2f}%, {hi*100:>+6.2f}%]  "
          f"p={p:>5.3f} {sig}")
    return dict(label=label, n=n, win_rate=win_rate, avg_dec=avg_dec,
                avg_edge_pp=avg_edge, roi=roi, ci_lo=lo, ci_hi=hi, p_val=p)


def main():
    print("="*70)
    print("TIER 1 DIAGNOSTIC — where does the +EV edge come from?")
    print("="*70)

    print("\nBuilding final model predictions + Vegas attachment...")
    df = load_final_model_df()
    df = apply_filter(df)
    wf = build_wf_predictions(df)
    tv = attach_vegas(wf[["DATE", "jbout", "jfighter"]].drop_duplicates())
    wf = wf.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                      "dec_odds_f1", "dec_odds_f2"]],
                  on=["DATE", "jbout", "jfighter"], how="left")
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    wfv = wf[wf["p_vegas_f1"].notna()].copy()

    # Build bet metadata
    p_m = wfv["p_model"].values
    p_v = wfv["p_vegas_f1"].values
    pick_f1 = p_m >= 0.5
    edge_on_pick = np.where(pick_f1, p_m - p_v, (1 - p_m) - (1 - p_v))
    wfv["edge"] = edge_on_pick
    wfv["p_pick"] = np.where(pick_f1, p_m, 1 - p_m)
    wfv["dec_pick"] = np.where(pick_f1, wfv["dec_odds_f1"].values,
                                wfv["dec_odds_f2"].values)
    y_f1 = wfv["win"].astype(int).values
    wfv["won_pick"] = (np.where(pick_f1, y_f1, 1 - y_f1) == 1).astype(int)
    wfv["implied_p_pick"] = 1.0 / wfv["dec_pick"]     # with vig
    wfv["vegas_p_pick"] = np.where(pick_f1, p_v, 1 - p_v)  # devigged

    bets = wfv[wfv["edge"] > 0].copy().reset_index(drop=True)
    bets["profit"] = np.where(bets["won_pick"] == 1,
                                bets["dec_pick"] - 1, -1.0)
    print(f"\nTotal +EV bets: {len(bets)}")
    print(f"Overall: win%={bets['won_pick'].mean()*100:.1f}  "
          f"avg dec={bets['dec_pick'].mean():.2f}  "
          f"avg edge={bets['edge'].mean()*100:.1f}pp  "
          f"ROI={bets['profit'].mean()*100:+.2f}%")

    results = {}

    # ── A. Favorite vs Underdog (vegas-defined) ───────────────────
    print("\n── A. FAVORITE vs UNDERDOG (by Vegas devigged probability) ──")
    fav = bets[bets["vegas_p_pick"] >= 0.5].copy()
    dog = bets[bets["vegas_p_pick"] < 0.5].copy()
    results["fav"] = slice_report(fav, "Favorites (vegas ≥50%)")
    results["dog"] = slice_report(dog, "Underdogs (vegas <50%)")

    # ── B. Favorite depth ─────────────────────────────────────────
    print("\n── B. FAVORITE DEPTH (finer buckets) ──")
    heavy = bets[bets["vegas_p_pick"] >= 0.70]
    moderate = bets[(bets["vegas_p_pick"] >= 0.55) & (bets["vegas_p_pick"] < 0.70)]
    close = bets[(bets["vegas_p_pick"] >= 0.50) & (bets["vegas_p_pick"] < 0.55)]
    dogs_b = bets[bets["vegas_p_pick"] < 0.50]
    results["heavy_fav"] = slice_report(heavy, "Heavy favorites (≥70%)")
    results["moderate_fav"] = slice_report(moderate, "Moderate favorites (55-70%)")
    results["close_fav"] = slice_report(close, "Close favorites (50-55%)")
    results["dogs_b"] = slice_report(dogs_b, "Underdogs (<50%)")

    # ── C. Model confidence on pick ───────────────────────────────
    print("\n── C. MODEL CONFIDENCE on its pick ──")
    hi_conf = bets[bets["p_pick"] >= 0.65]
    mid_conf = bets[(bets["p_pick"] >= 0.55) & (bets["p_pick"] < 0.65)]
    lo_conf = bets[(bets["p_pick"] >= 0.50) & (bets["p_pick"] < 0.55)]
    results["conf_high"] = slice_report(hi_conf, "High model conf (p≥0.65)")
    results["conf_mid"] = slice_report(mid_conf, "Mid model conf (0.55-0.65)")
    results["conf_low"] = slice_report(lo_conf, "Low model conf (0.50-0.55)")

    # ── D. Edge magnitude ─────────────────────────────────────────
    print("\n── D. EDGE MAGNITUDE (model_p − vegas_devigged_p) ──")
    small_e = bets[(bets["edge"] > 0) & (bets["edge"] < 0.05)]
    mid_e = bets[(bets["edge"] >= 0.05) & (bets["edge"] < 0.10)]
    big_e = bets[bets["edge"] >= 0.10]
    results["edge_small"] = slice_report(small_e, "Small edge (1-5pp)")
    results["edge_mid"] = slice_report(mid_e, "Mid edge (5-10pp)")
    results["edge_big"] = slice_report(big_e, "Big edge (≥10pp)")

    # ── E. Weight class ─────────────────────────────────────────
    print("\n── E. WEIGHT CLASS ──")
    if "weightindex" in bets.columns:
        # Group: light divs (1-8), welter/middle (9-10), light-heavy+ (11-12)
        light = bets[bets["weightindex"].isin([1,2,3,4,5,6,7,8])]
        mid = bets[bets["weightindex"].isin([9,10])]
        heavy = bets[bets["weightindex"].isin([11,12])]
        results["wc_light"] = slice_report(light, "Fly/Bantam/Feather/Light (1-8)")
        results["wc_mid"] = slice_report(mid, "Welter + Middle (9-10)")
        results["wc_heavy"] = slice_report(heavy, "LHW + HW (11-12)")

    # ── F. Event type (by jevent name) ───────────────────────────
    print("\n── F. EVENT TYPE (parsed from jevent) ──")
    def event_tier(j):
        j = str(j)
        if j.startswith("UFC") and j[3:4].isdigit():  # UFC 309, UFC285, etc.
            return "ppv"
        if "FightNight" in j.replace("Fight Night", "FightNight"):
            return "fight_night"
        return "other"
    bets["event_type"] = bets["jevent"].apply(event_tier)
    for et, label in [("ppv", "PPV (UFC 100s / numbered)"),
                        ("fight_night", "Fight Night"),
                        ("other", "Other (TUF/Apex/etc)")]:
        sl = bets[bets["event_type"] == et]
        results[f"evt_{et}"] = slice_report(sl, label)

    # ── G. Favorite × Model Confidence interaction ────────────────
    print("\n── G. FAVORITE × MODEL CONF (does heavy+confident win?) ──")
    hfa_hi = bets[(bets["vegas_p_pick"] >= 0.65) & (bets["p_pick"] >= 0.65)]
    hfa_lo = bets[(bets["vegas_p_pick"] >= 0.65) & (bets["p_pick"] < 0.65)]
    dog_hi = bets[(bets["vegas_p_pick"] < 0.50) & (bets["p_pick"] >= 0.55)]
    results["fav_conf"] = slice_report(hfa_hi,
                                         "Heavy-fav (≥65%) + HI model conf (≥65%)")
    results["fav_noconf"] = slice_report(hfa_lo,
                                          "Heavy-fav (≥65%) + LO model conf (<65%)")
    results["dog_conf"] = slice_report(dog_hi,
                                         "Underdog + model-picks-dog (p≥0.55)")

    # ── Save & summary ───────────────────────────────────────────
    (DT / "ev_bet_slice_analysis.json").write_text(
        json.dumps({k: v for k, v in results.items() if v is not None},
                   indent=2, default=str))

    print("\n" + "="*70)
    print("TOP SLICES (sorted by ROI, min n=20)")
    print("="*70)
    rows = [r for r in results.values() if r is not None and r["n"] >= 20]
    rows.sort(key=lambda r: r["roi"], reverse=True)
    for r in rows[:8]:
        sig = "✓" if r["p_val"] < 0.05 else " "
        print(f"  {sig} {r['label']:<36s}  n={r['n']:>3d}  "
              f"win%={r['win_rate']*100:>5.1f}  ROI={r['roi']*100:>+6.2f}%  "
              f"p={r['p_val']:.3f}")

    print(f"\nSaved to {DT/'ev_bet_slice_analysis.json'}")


if __name__ == "__main__":
    main()
