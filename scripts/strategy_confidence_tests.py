"""Tier 1 strategy-confidence tests on the frozen Elastic Net test predictions.

Inputs:
  - results/train_test_2016_2024_predictions.parquet  (391 frozen test
    predictions from train_test_split_2016_2024.py — 2024-10 → 2026-04)
  - data/tmp/odds_table.csv (or DB) via attach_vegas_rich

Tests:
  1. Monte Carlo ROI test (under Vegas-true null) + label-shuffle permutation
       (a) PRIMARY: Sample outcomes from Bernoulli(p_vegas_pick) — i.e. assume
           Vegas is correctly calibrated and the model has no edge. Recompute
           pnl 10,000 times. p-value = P(MC_ROI ≥ observed_ROI).
           This is the standard sports-betting confidence test: "does the
           model beat Vegas on this bet set?"
       (b) DIAGNOSTIC: shuffle the `won_pick` column within the bet set.
           This tells us whether the SELECTION among bet fights matters
           (independent of dec). Note: when picks concentrate on favorites,
           random shuffles can inflate the null (random wins on underdogs
           pay big), so this test is interpretive only.
       Run both for {all_picks, ev_positive, edge_5pp, edge_10pp}.

  2. Pre-registered held-out test
       Split chronologically at 2025-10-01:
         validation = 2024-10 → 2025-09  (~10mo)
         held-out   = 2025-10 → 2026-04  (~6mo)
       On validation: rank strategies by ROI, pick #1.
       On held-out:   evaluate ONCE, report ROI + bootstrap CI + perm p.

  3. Calibration plot
       Bin model `p_pred` and Vegas devigged `p_vegas` into deciles
       (0.05-wide bins from 0.50 to 0.95). For each bin, plot
       observed win rate vs predicted midpoint (reliability diagram).
       Brier score per side, ECE per side.

Outputs:
  - results/strategy_confidence.json
  - results/calibration.png

LEAKAGE_REFERENCE.md compliance:
  See docs/audits/strategy_confidence_tests.md.
  This script does NO training. Only post-hoc statistical tests on a
  frozen prediction set. Vegas odds attached after the fact.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")

# ── Pre-registered constants (defined BEFORE looking at any data) ────────
HELDOUT_START      = pd.Timestamp("2025-10-01")
PERMUTATIONS       = 10_000
SEED               = 42
EV_THRESH          = 0.0       # ev_positive: ev > 0
EDGE_5_PP          = 5.0
EDGE_10_PP         = 10.0
CALIB_BINS         = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.001])
EPS                = 1e-9


# ─── Helpers ─────────────────────────────────────────────────────────────

def load_predictions_with_vegas() -> pd.DataFrame:
    """Frozen predictions + Vegas devigged probs and decimal payouts."""
    pred_path = Path("results/train_test_2016_2024_predictions.parquet")
    if not pred_path.exists():
        print(f"❌ Missing {pred_path}. Run train_test_split_2016_2024.py first.")
        sys.exit(1)
    preds = pd.read_parquet(pred_path)
    preds["DATE"] = pd.to_datetime(preds["DATE"])

    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    keys = preds[["DATE", "jbout", "jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    merged = preds.merge(
        tv[["DATE", "jbout", "jfighter", "p_vegas_f1", "dec_odds_f1", "dec_odds_f2"]],
        on=["DATE", "jbout", "jfighter"], how="left",
    )
    matched = merged[merged["p_vegas_f1"].notna()].copy()

    # Resolve which side the model picked, then compute edge / EV / pnl
    matched["pick_a"]       = (matched["p_pred"] >= 0.5).astype(int)
    matched["dec_pick"]     = np.where(matched["pick_a"]==1,
                                       matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_model_pick"] = np.where(matched["pick_a"]==1,
                                       matched["p_pred"], 1 - matched["p_pred"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"]==1,
                                       matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    matched["edge_pp"]      = (matched["p_model_pick"] - matched["p_vegas_pick"]) * 100
    matched["ev"]           = matched["p_model_pick"] * matched["dec_pick"] - 1
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["pnl"] = np.where(matched["won_pick"]==1, matched["dec_pick"] - 1, -1.0)
    return matched


def filter_strategy(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if   name == "all_picks":   return df
    elif name == "ev_positive": return df[df["ev"] > EV_THRESH]
    elif name == "edge_5pp":    return df[(df["ev"] > EV_THRESH) & (df["edge_pp"] >= EDGE_5_PP)]
    elif name == "edge_10pp":   return df[(df["ev"] > EV_THRESH) & (df["edge_pp"] >= EDGE_10_PP)]
    else: raise ValueError(name)


STRATEGIES = ["all_picks", "ev_positive", "edge_5pp", "edge_10pp"]


# ── Test 1: Monte Carlo ROI under Vegas-true null + label-shuffle ────────

def mc_pvalue_under_vegas(d: pd.DataFrame, n_sim: int, seed: int) -> dict:
    """Monte Carlo: assume Vegas is the TRUE probability for each fight.
    Sample wins ~ Bernoulli(p_vegas_pick), recompute pnl, repeat n_sim times.
    Return one-sided p-value = P(sim_ROI >= observed_ROI).

    This tests whether the model's strategy beats a Vegas-believer who bets
    on the same fights. It is the standard sports-betting confidence test.
    """
    if len(d) == 0:
        return {"n": 0, "observed_roi_pct": None, "mc_p": None,
                "mc_mean_roi_pct": None, "mc_std_roi_pct": None}
    rng = np.random.default_rng(seed)
    won = d["won_pick"].values.astype(np.int8)
    dec = d["dec_pick"].values.astype(np.float64)
    p_v = d["p_vegas_pick"].values.astype(np.float64)
    n   = len(won)

    observed_roi = (np.where(won == 1, dec - 1.0, -1.0)).mean()

    # Sample (n_sim × n) Bernoulli outcomes
    sims = (rng.random((n_sim, n)) < p_v[None, :]).astype(np.int8)
    sim_pnl  = np.where(sims == 1, dec - 1.0, -1.0)
    sim_rois = sim_pnl.mean(axis=1)
    p_value  = float(((sim_rois >= observed_roi).sum() + 1) / (n_sim + 1))
    return {
        "n":                int(n),
        "observed_roi_pct": float(observed_roi * 100),
        "mc_p":             p_value,
        "mc_mean_roi_pct":  float(sim_rois.mean() * 100),
        "mc_std_roi_pct":   float(sim_rois.std()  * 100),
    }


def label_shuffle_pvalue(d: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """Diagnostic permutation: shuffle won_pick within the bet set, holding
    win-count fixed. p = P(perm_ROI >= observed_ROI).

    INTERPRETIVE: under this null, mean(perm_ROI) = win_rate*mean(dec-1) -
    (1-win_rate). When picks concentrate on favorites, perm mean >> 0 even
    though the model has skill, because random shuffles randomly assign
    wins to high-dec rows. So a high p here doesn't necessarily mean the
    model is bad — it means selection of HIGH-EV bets WITHIN the bet set
    is no better than random. Use the Vegas-MC test (mc_pvalue_under_vegas)
    as primary."""
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
    return {
        "n":                 int(n),
        "perm_p":            p_value,
        "perm_mean_roi_pct": float(perm_rois.mean() * 100),
        "perm_std_roi_pct":  float(perm_rois.std()  * 100),
    }


# ── Test 2: Pre-registered held-out ──────────────────────────────────────

def preregistered_heldout(df: pd.DataFrame) -> dict:
    val = df[df["DATE"] <  HELDOUT_START].copy()
    hot = df[df["DATE"] >= HELDOUT_START].copy()
    assert val["DATE"].max() < HELDOUT_START <= hot["DATE"].min(), \
        "Validation/held-out boundary violated"

    print(f"\n  validation slice : {len(val)} fights "
          f"({val['DATE'].min().date()} → {val['DATE'].max().date()})")
    print(f"  held-out slice   : {len(hot)} fights "
          f"({hot['DATE'].min().date()} → {hot['DATE'].max().date()})")

    # Rank strategies by validation ROI; pick the best.
    val_roi = {}
    for s in STRATEGIES:
        sl = filter_strategy(val, s)
        val_roi[s] = {
            "n":       int(len(sl)),
            "win_pct": float(sl["won_pick"].mean()*100) if len(sl) else None,
            "roi_pct": float(sl["pnl"].mean()*100)      if len(sl) else None,
        }
    best = max(val_roi.keys(), key=lambda k: (val_roi[k]["roi_pct"] or -1e9))
    print(f"  validation winner: {best}  "
          f"(ROI {val_roi[best]['roi_pct']:+.2f}% on n={val_roi[best]['n']})")

    # Evaluate that ONE strategy ONCE on held-out.
    hot_sl  = filter_strategy(hot, best)
    n_hot   = len(hot_sl)
    obs_roi = float(hot_sl["pnl"].mean() * 100) if n_hot else None
    win_pct = float(hot_sl["won_pick"].mean() * 100) if n_hot else None
    # Bootstrap CI on held-out ROI
    ci_lo = ci_hi = None
    mc_p  = None
    if n_hot >= 10:
        rng = np.random.default_rng(SEED + 1)
        boots = []
        for _ in range(2_000):
            idx = rng.integers(0, n_hot, n_hot)
            boots.append(hot_sl["pnl"].values[idx].mean() * 100)
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        # Vegas-MC p-value on held-out
        mc = mc_pvalue_under_vegas(hot_sl, PERMUTATIONS, SEED + 2)
        mc_p = mc["mc_p"]

    return {
        "heldout_start":      str(HELDOUT_START.date()),
        "val_per_strategy":   val_roi,
        "chosen_strategy":    best,
        "heldout_n":          int(n_hot),
        "heldout_win_pct":    win_pct,
        "heldout_roi_pct":    obs_roi,
        "heldout_roi_ci95":   [ci_lo, ci_hi] if ci_lo is not None else None,
        "heldout_mc_p":       mc_p,
    }


# ── Test 3: Calibration plot ─────────────────────────────────────────────

def calibration_table(probs: np.ndarray, outcomes: np.ndarray) -> pd.DataFrame:
    """Bin probs by CALIB_BINS, return per-bin n, mean predicted prob,
    observed win rate, Brier and Wilson 95% CI on observed."""
    probs = np.clip(probs, 0.02, 0.98)
    bin_idx = np.digitize(probs, CALIB_BINS) - 1
    rows = []
    for i in range(len(CALIB_BINS) - 1):
        lo, hi = CALIB_BINS[i], CALIB_BINS[i+1]
        sel = bin_idx == i
        n = int(sel.sum())
        if n == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "mean_pred": None, "obs_rate": None,
                         "ci_lo": None, "ci_hi": None})
            continue
        mp = float(probs[sel].mean())
        wr = float(outcomes[sel].mean())
        # Wilson score interval
        z = 1.96
        denom = 1 + z*z/n
        center = (wr + z*z/(2*n)) / denom
        margin = z * np.sqrt(wr*(1-wr)/n + z*z/(4*n*n)) / denom
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": n,
                     "mean_pred": mp, "obs_rate": wr,
                     "ci_lo": float(center - margin),
                     "ci_hi": float(center + margin)})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    tot = table["n"].sum()
    if tot == 0: return float("nan")
    e = 0.0
    for _, r in table.iterrows():
        if r["n"] == 0: continue
        e += (r["n"] / tot) * abs(r["mean_pred"] - r["obs_rate"])
    return float(e)


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("STRATEGY CONFIDENCE — Tier 1")
    print("Audit: docs/audits/strategy_confidence_tests.md")
    print("=" * 78)

    df = load_predictions_with_vegas()
    print(f"\nLoaded {len(df)} Vegas-matched predictions  "
          f"({df['DATE'].min().date()} → {df['DATE'].max().date()})")

    # ── Test 1a: Monte Carlo ROI under Vegas-true null (PRIMARY) ────────
    print()
    print("─" * 60)
    print(f"TEST 1a (PRIMARY) — Monte Carlo ROI under Vegas-true null")
    print(f"           wins ~ Bernoulli(p_vegas_pick), {PERMUTATIONS:,} sims")
    print("─" * 60)
    print(f"  {'strategy':<12s}  {'n':>4s}  {'obs ROI':>9s}  "
          f"{'MC μ':>8s}  {'MC σ':>8s}  {'p-value':>9s}")
    mc_results = {}
    for s in STRATEGIES:
        d = filter_strategy(df, s)
        r = mc_pvalue_under_vegas(d, PERMUTATIONS, SEED)
        mc_results[s] = r
        if r["n"] == 0:
            print(f"  {s:<12s}  (empty)"); continue
        sig = "***" if r["mc_p"] < 0.01 else ("**" if r["mc_p"] < 0.05 else
              ("*" if r["mc_p"] < 0.10 else ""))
        print(f"  {s:<12s}  {r['n']:>4d}  {r['observed_roi_pct']:>+8.2f}%  "
              f"{r['mc_mean_roi_pct']:>+7.2f}%  {r['mc_std_roi_pct']:>7.2f}%  "
              f"{r['mc_p']:>8.4f} {sig}")

    # ── Test 1b: Label-shuffle permutation (DIAGNOSTIC) ─────────────────
    print()
    print("─" * 60)
    print(f"TEST 1b (DIAGNOSTIC) — Label-shuffle within bet set")
    print("           (interpretive only; null inflated by underdog payouts)")
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

    # ── Test 2: Pre-registered held-out ─────────────────────────────────
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
            sig = "***" if hot["heldout_mc_p"] < 0.01 else \
                  ("**" if hot["heldout_mc_p"] < 0.05 else
                  ("*" if hot["heldout_mc_p"] < 0.10 else ""))
            print(f"      MC p (Vegas null) = {hot['heldout_mc_p']:.4f} {sig}")

    # ── Test 3: Calibration plot ────────────────────────────────────────
    print()
    print("─" * 60)
    print("TEST 3 — Calibration  (model vs Vegas devigged)")
    print("─" * 60)
    # Use side-of-pick probabilities (so we always look at win-of-pick rate)
    p_model = df["p_model_pick"].values
    p_vegas = df["p_vegas_pick"].values
    y_pick  = df["won_pick"].values

    tbl_model = calibration_table(p_model, y_pick)
    tbl_vegas = calibration_table(p_vegas, y_pick)
    ece_model = expected_calibration_error(tbl_model)
    ece_vegas = expected_calibration_error(tbl_vegas)
    brier_model = brier(p_model, y_pick)
    brier_vegas = brier(p_vegas, y_pick)

    def _print_tbl(name, tbl):
        print(f"\n  ── {name} ──")
        print(f"  {'bin':<14s} {'n':>5s} {'pred':>6s} {'obs':>6s} {'95% CI':>17s}")
        for _, r in tbl.iterrows():
            n = int(r["n"])
            if n == 0:
                print(f"  [{r['bin_lo']:.2f},{r['bin_hi']:.2f})    0     -      -            -")
                continue
            print(f"  [{r['bin_lo']:.2f},{r['bin_hi']:.2f}) {n:>5d}  "
                  f"{r['mean_pred']:.3f}  {r['obs_rate']:.3f}  "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    _print_tbl("model",  tbl_model)
    _print_tbl("vegas",  tbl_vegas)
    print(f"\n  ECE   model={ece_model:.4f}   vegas={ece_vegas:.4f}")
    print(f"  Brier model={brier_model:.4f}   vegas={brier_vegas:.4f}")

    # ── Plot reliability diagram ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for tbl, color, name, ece in [(tbl_model, "#2563eb", "model", ece_model),
                                  (tbl_vegas, "#dc2626", "vegas", ece_vegas)]:
        sub = tbl[tbl["n"] > 0]
        if len(sub) == 0: continue
        ax.errorbar(sub["mean_pred"], sub["obs_rate"],
                    yerr=[sub["obs_rate"] - sub["ci_lo"],
                          sub["ci_hi"] - sub["obs_rate"]],
                    fmt="o-", color=color, linewidth=2, markersize=7,
                    label=f"{name} (ECE={ece:.3f})")
        # n on each point
        for _, r in sub.iterrows():
            ax.annotate(f"n={r['n']}",
                        xy=(r["mean_pred"], r["obs_rate"]),
                        xytext=(5, -10), textcoords="offset points",
                        fontsize=7, color=color)
    ax.set_xlim(0.45, 1.0); ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted probability (of pick winning)")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration — clean Elastic Net vs Vegas devigged\n"
                 f"({len(df)} test fights, 2024-10 → 2026-04)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    out_png = Path("results/calibration.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\n✓ Saved {out_png}")

    # ── Save JSON ────────────────────────────────────────────────────────
    out = {
        "n_predictions":         int(len(df)),
        "mc_vegas_null":         mc_results,
        "label_shuffle":         perm_results,
        "preregistered_heldout": hot,
        "calibration": {
            "model": {"ece": ece_model, "brier": brier_model,
                      "table": tbl_model.to_dict("records")},
            "vegas": {"ece": ece_vegas, "brier": brier_vegas,
                      "table": tbl_vegas.to_dict("records")},
        },
    }
    Path("results/strategy_confidence.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"✓ Saved results/strategy_confidence.json")

    # ── Verdict summary ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    for s, r in mc_results.items():
        if r.get("mc_p") is None: continue
        verdict = ("REAL signal" if r["mc_p"] < 0.05 else
                   "borderline"  if r["mc_p"] < 0.10 else
                   "noise")
        print(f"  Test 1a {s:<12s}: MC p={r['mc_p']:.3f}  → {verdict}")
    if hot.get("heldout_mc_p") is not None:
        verdict = ("REAL signal" if hot['heldout_mc_p'] < 0.05 else
                   "borderline"  if hot['heldout_mc_p'] < 0.10 else
                   "noise")
        print(f"  Test 2  heldout({hot['chosen_strategy']:<10s}): MC p="
              f"{hot['heldout_mc_p']:.3f}  → {verdict}")
    print(f"  Test 3  ECE: model={ece_model:.3f}, vegas={ece_vegas:.3f}  "
          f"→ {'model better calibrated' if ece_model < ece_vegas else 'vegas better calibrated'}")


if __name__ == "__main__":
    main()
