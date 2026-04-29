"""Re-run Tier 1 strategy-confidence tests on beta-CV-OOF calibrated predictions.

Inputs:
  - results/train_calib_compare_v2_predictions.parquet  (p_B_beta column)

Tests (same as scripts/strategy_confidence_tests.py — see that audit):
  1a. Monte Carlo ROI under Vegas-true null  (PRIMARY)
  1b. Label-shuffle permutation              (DIAGNOSTIC)
  2.  Pre-registered held-out                (cutoff 2025-10-01)

Multiple-comparison disclosure: beta-CV-OOF was picked by ROI from a
14-cell sweep on this same test set. Bonferroni-adjusted threshold
p<0.0036 is reported alongside raw p so the user can judge.

Audit: docs/audits/strategy_confidence_beta_cvoof.md
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")

EPS                = 1e-9
HELDOUT_START      = pd.Timestamp("2025-10-01")
PERMUTATIONS       = 10_000
SEED               = 42
EV_THRESH          = 0.0
EDGE_5_PP          = 5.0
EDGE_10_PP         = 10.0
N_CELLS_SWEPT      = 14            # 7 calibrators × 2 strategies
BONFERRONI_CUT     = 0.05 / N_CELLS_SWEPT


def load_predictions_with_vegas(prediction_col: str = "p_B_beta") -> pd.DataFrame:
    pred_path = Path("results/train_calib_compare_v2_predictions.parquet")
    if not pred_path.exists():
        print(f"❌ Missing {pred_path}. Run train_calib_compare_v2.py first.")
        sys.exit(1)
    preds = pd.read_parquet(pred_path)
    preds["DATE"] = pd.to_datetime(preds["DATE"])
    if prediction_col not in preds.columns:
        print(f"❌ Column '{prediction_col}' not in parquet. Available: "
              f"{[c for c in preds.columns if c.startswith('p_')]}")
        sys.exit(1)
    preds = preds.rename(columns={prediction_col: "p_pred"})

    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    keys = preds[["DATE", "jbout", "jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    merged = preds.merge(
        tv[["DATE", "jbout", "jfighter", "p_vegas_f1", "dec_odds_f1", "dec_odds_f2"]],
        on=["DATE", "jbout", "jfighter"], how="left",
    )
    matched = merged[merged["p_vegas_f1"].notna()].copy()

    matched["pick_a"]       = (matched["p_pred"] >= 0.5).astype(int)
    matched["dec_pick"]     = np.where(matched["pick_a"]==1,
                                       matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_model_pick"] = np.where(matched["pick_a"]==1,
                                       matched["p_pred"], 1 - matched["p_pred"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"]==1,
                                       matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    matched["edge_pp"] = (matched["p_model_pick"] - matched["p_vegas_pick"]) * 100
    matched["ev"]      = matched["p_model_pick"] * matched["dec_pick"] - 1
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["pnl"] = np.where(matched["won_pick"]==1, matched["dec_pick"] - 1, -1.0)
    return matched


def filter_strategy(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if   name == "all_picks":   return df
    elif name == "ev_positive": return df[df["ev"] > EV_THRESH]
    elif name == "edge_5pp":    return df[(df["ev"] > EV_THRESH) & (df["edge_pp"] >= EDGE_5_PP)]
    elif name == "edge_10pp":   return df[(df["ev"] > EV_THRESH) & (df["edge_pp"] >= EDGE_10_PP)]
    raise ValueError(name)


STRATEGIES = ["all_picks", "ev_positive", "edge_5pp", "edge_10pp"]


def mc_pvalue_under_vegas(d: pd.DataFrame, n_sim: int, seed: int) -> dict:
    if len(d) == 0:
        return {"n": 0, "observed_roi_pct": None, "mc_p": None,
                "mc_mean_roi_pct": None, "mc_std_roi_pct": None}
    rng = np.random.default_rng(seed)
    won = d["won_pick"].values.astype(np.int8)
    dec = d["dec_pick"].values.astype(np.float64)
    p_v = d["p_vegas_pick"].values.astype(np.float64)
    n   = len(won)
    observed_roi = (np.where(won == 1, dec - 1.0, -1.0)).mean()
    sims = (rng.random((n_sim, n)) < p_v[None, :]).astype(np.int8)
    sim_pnl  = np.where(sims == 1, dec - 1.0, -1.0)
    sim_rois = sim_pnl.mean(axis=1)
    p_value  = float(((sim_rois >= observed_roi).sum() + 1) / (n_sim + 1))
    return {"n": int(n),
            "observed_roi_pct": float(observed_roi * 100),
            "mc_p":             p_value,
            "mc_mean_roi_pct":  float(sim_rois.mean() * 100),
            "mc_std_roi_pct":   float(sim_rois.std()  * 100)}


def label_shuffle_pvalue(d: pd.DataFrame, n_perm: int, seed: int) -> dict:
    if len(d) == 0:
        return {"n": 0, "perm_p": None,
                "perm_mean_roi_pct": None, "perm_std_roi_pct": None}
    rng = np.random.default_rng(seed)
    won = d["won_pick"].values.astype(np.int8)
    dec = d["dec_pick"].values.astype(np.float64)
    n   = len(won)
    observed_roi = (np.where(won == 1, dec - 1.0, -1.0)).mean()
    perms = np.empty((n_perm, n), dtype=np.int8)
    for i in range(n_perm):
        perms[i] = rng.permutation(won)
    perm_pnl  = np.where(perms == 1, dec - 1.0, -1.0)
    perm_rois = perm_pnl.mean(axis=1)
    p_value   = float(((perm_rois >= observed_roi).sum() + 1) / (n_perm + 1))
    return {"n": int(n),
            "perm_p":            p_value,
            "perm_mean_roi_pct": float(perm_rois.mean() * 100),
            "perm_std_roi_pct":  float(perm_rois.std()  * 100)}


def preregistered_heldout(df: pd.DataFrame) -> dict:
    val = df[df["DATE"] <  HELDOUT_START].copy()
    hot = df[df["DATE"] >= HELDOUT_START].copy()
    assert val["DATE"].max() < HELDOUT_START <= hot["DATE"].min()

    print(f"\n  validation slice : {len(val)} fights "
          f"({val['DATE'].min().date()} → {val['DATE'].max().date()})")
    print(f"  held-out slice   : {len(hot)} fights "
          f"({hot['DATE'].min().date()} → {hot['DATE'].max().date()})")

    val_roi = {}
    for s in STRATEGIES:
        sl = filter_strategy(val, s)
        val_roi[s] = {"n": int(len(sl)),
                      "win_pct": float(sl["won_pick"].mean()*100) if len(sl) else None,
                      "roi_pct": float(sl["pnl"].mean()*100)      if len(sl) else None}
    best = max(val_roi.keys(), key=lambda k: (val_roi[k]["roi_pct"] or -1e9))
    print(f"  validation winner: {best}  "
          f"(ROI {val_roi[best]['roi_pct']:+.2f}% on n={val_roi[best]['n']})")

    hot_sl  = filter_strategy(hot, best)
    n_hot   = len(hot_sl)
    obs_roi = float(hot_sl["pnl"].mean() * 100) if n_hot else None
    win_pct = float(hot_sl["won_pick"].mean() * 100) if n_hot else None
    ci_lo = ci_hi = None; mc_p = None
    if n_hot >= 10:
        rng = np.random.default_rng(SEED + 1)
        boots = []
        for _ in range(2_000):
            idx = rng.integers(0, n_hot, n_hot)
            boots.append(hot_sl["pnl"].values[idx].mean() * 100)
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        mc = mc_pvalue_under_vegas(hot_sl, PERMUTATIONS, SEED + 2)
        mc_p = mc["mc_p"]

    return {"heldout_start":      str(HELDOUT_START.date()),
            "val_per_strategy":   val_roi,
            "chosen_strategy":    best,
            "heldout_n":          int(n_hot),
            "heldout_win_pct":    win_pct,
            "heldout_roi_pct":    obs_roi,
            "heldout_roi_ci95":   [ci_lo, ci_hi] if ci_lo is not None else None,
            "heldout_mc_p":       mc_p}


def main():
    print("=" * 78)
    print("STRATEGY CONFIDENCE — Tier 1 on beta-CV-OOF calibrated predictions")
    print(f"Multiple-comparison context: beta-CV-OOF chosen from {N_CELLS_SWEPT}-cell sweep")
    print(f"  → Bonferroni-adjusted threshold for p<0.05: p < {BONFERRONI_CUT:.4f}")
    print("Audit: docs/audits/strategy_confidence_beta_cvoof.md")
    print("=" * 78)

    df = load_predictions_with_vegas("p_B_beta")
    print(f"\nLoaded {len(df)} Vegas-matched predictions  "
          f"({df['DATE'].min().date()} → {df['DATE'].max().date()})")

    # ── Test 1a: Vegas-MC ───────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"TEST 1a — Monte Carlo under Vegas-true null  ({PERMUTATIONS:,} sims)")
    print("─" * 60)
    print(f"  {'strategy':<12s}  {'n':>4s}  {'obs ROI':>9s}  "
          f"{'MC μ':>8s}  {'MC σ':>8s}  {'p (raw)':>9s}  {'verdict':>22s}")
    mc_results = {}
    for s in STRATEGIES:
        d = filter_strategy(df, s)
        r = mc_pvalue_under_vegas(d, PERMUTATIONS, SEED)
        mc_results[s] = r
        if r["n"] == 0:
            print(f"  {s:<12s}  (empty)"); continue
        if r["mc_p"] < BONFERRONI_CUT:
            verdict = "**REAL (Bonferroni)**"
        elif r["mc_p"] < 0.05:
            verdict = "raw-sig but multi-comp"
        elif r["mc_p"] < 0.10:
            verdict = "borderline"
        else:
            verdict = "noise"
        print(f"  {s:<12s}  {r['n']:>4d}  {r['observed_roi_pct']:>+8.2f}%  "
              f"{r['mc_mean_roi_pct']:>+7.2f}%  {r['mc_std_roi_pct']:>7.2f}%  "
              f"{r['mc_p']:>8.4f}   {verdict:>22s}")

    # ── Test 1b: label-shuffle (diagnostic) ─────────────────────────────
    print()
    print("─" * 60)
    print(f"TEST 1b — Label-shuffle within bet set (diagnostic)")
    print("─" * 60)
    print(f"  {'strategy':<12s}  {'n':>4s}  {'obs ROI':>9s}  "
          f"{'perm μ':>8s}  {'perm σ':>8s}  {'p-value':>9s}")
    perm_results = {}
    for s in STRATEGIES:
        d = filter_strategy(df, s)
        r = label_shuffle_pvalue(d, PERMUTATIONS, SEED + 100)
        perm_results[s] = r
        if r["n"] == 0:
            print(f"  {s:<12s}  (empty)"); continue
        obs = mc_results[s]["observed_roi_pct"]
        print(f"  {s:<12s}  {r['n']:>4d}  {obs:>+8.2f}%  "
              f"{r['perm_mean_roi_pct']:>+7.2f}%  {r['perm_std_roi_pct']:>7.2f}%  "
              f"{r['perm_p']:>8.4f}")

    # ── Test 2: pre-registered held-out ─────────────────────────────────
    print()
    print("─" * 60)
    print(f"TEST 2 — Pre-registered held-out  (cutoff = {HELDOUT_START.date()})")
    print("─" * 60)
    hot = preregistered_heldout(df)
    print(f"\n  → on held-out, '{hot['chosen_strategy']}' produced:")
    print(f"      n          = {hot['heldout_n']}")
    if hot["heldout_n"] > 0:
        print(f"      win rate   = {hot['heldout_win_pct']:.1f}%")
        print(f"      ROI        = {hot['heldout_roi_pct']:+.2f}%")
        if hot["heldout_roi_ci95"]:
            ci = hot["heldout_roi_ci95"]
            zero_in = "zero ∈ CI" if ci[0] <= 0 <= ci[1] else "zero ∉ CI"
            print(f"      95% CI     = [{ci[0]:+.2f}%, {ci[1]:+.2f}%]  ({zero_in})")
        if hot["heldout_mc_p"] is not None:
            print(f"      MC p (Vegas null) = {hot['heldout_mc_p']:.4f}")

    # ── Save ──
    out = {"prediction_column":   "p_B_beta",
           "n_cells_swept":       N_CELLS_SWEPT,
           "bonferroni_cut":      BONFERRONI_CUT,
           "n_predictions":       int(len(df)),
           "mc_vegas_null":       mc_results,
           "label_shuffle":       perm_results,
           "preregistered_heldout": hot}
    Path("results").mkdir(exist_ok=True)
    Path("results/strategy_confidence_beta_cvoof.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n✓ Saved results/strategy_confidence_beta_cvoof.json")

    # ── Side-by-side vs uncalibrated baseline ───────────────────────────
    base_path = Path("results/strategy_confidence.json")
    if base_path.exists():
        base = json.loads(base_path.read_text())
        print()
        print("=" * 78)
        print("Side-by-side: uncalibrated  vs  beta-CV-OOF")
        print("=" * 78)
        print(f"  {'strategy':<14s}  | {'raw n':>5s} {'raw ROI':>8s} {'raw p':>7s}  | "
              f"{'β  n':>5s} {'β  ROI':>8s} {'β  p':>7s}  | {'Δ ROI':>7s} {'Δ p':>7s}")
        print("  " + "-" * 90)
        for s in STRATEGIES:
            br = base["mc_vegas_null"].get(s, {})
            br_n   = br.get("n");                br_roi = br.get("observed_roi_pct")
            br_p   = br.get("mc_p")
            be     = mc_results.get(s, {})
            be_n   = be.get("n");                be_roi = be.get("observed_roi_pct")
            be_p   = be.get("mc_p")
            if None in (br_roi, be_roi, br_p, be_p): continue
            print(f"  {s:<14s}  | {br_n:>5d} {br_roi:>+7.2f}% {br_p:>7.4f}  | "
                  f"{be_n:>5d} {be_roi:>+7.2f}% {be_p:>7.4f}  | "
                  f"{be_roi-br_roi:>+6.2f}% {be_p-br_p:>+7.4f}")


if __name__ == "__main__":
    main()
