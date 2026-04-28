"""Find strategies whose per-fold ROI is TRENDING UP across folds.

Hypothesis: if a strategy gets better as folds advance (2024→2026), maybe
the trend continues into the future. Test by fitting linear slope across
4 fold ROIs, and rank.

Caveat: 4 data points is brutally small for trend detection. A "positive
slope" can easily appear by chance. We compute a permutation p-value to
estimate how likely the observed slope is under the null (random fold
ordering) — but with only 24 permutations of 4 points, this is a sanity
check, not a rigorous test.
"""
import json, sys, warnings
from pathlib import Path
from itertools import combinations, permutations
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

CACHE = Path("results/parlay_predictions.parquet")
if not CACHE.exists():
    print("Run parlay_strategy_eval.py first."); sys.exit(1)
matched = pd.read_parquet(CACHE)
FOLD_ORDER = ["fold_1", "fold_2", "fold_3", "fold_4"]


def make_parlays(df, n_legs, picker="top_k_edge", edge_min=None, p_min=None, top_k=None):
    pos = df[df["ev"] > 0].copy()
    if edge_min is not None: pos = pos[pos["edge"] >= edge_min]
    if p_min is not None:    pos = pos[pos["p_pick"] >= p_min]
    parlays = []
    k = top_k if top_k is not None else n_legs
    for date, grp in pos.groupby("DATE"):
        g = grp.copy()
        if picker == "top_k_edge":
            g = g.sort_values("edge", ascending=False).head(k)
        elif picker == "top_k_prob":
            g = g.sort_values("p_pick", ascending=False).head(k)
        if len(g) < n_legs: continue
        for combo in combinations(g.itertuples(index=False), n_legs):
            combined_odds = float(np.prod([c.dec_odds_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            parlays.append(dict(combined_odds=combined_odds, won=won,
                                fold=str(combo[0].fold)))
    return parlays


def make_straights(df, edge_min=None, p_min=None):
    pos = df[df["ev"] > 0].copy()
    if edge_min is not None: pos = pos[pos["edge"] >= edge_min]
    if p_min is not None:    pos = pos[pos["p_pick"] >= p_min]
    return [dict(combined_odds=r.dec_odds_pick, won=int(r.won_pick), fold=str(r.fold))
            for r in pos.itertuples()]


def per_fold_roi(parlays):
    out = {}
    for f in FOLD_ORDER:
        ps = [p for p in parlays if p["fold"] == f]
        if not ps: out[f] = (0, None); continue
        pnl = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in ps])
        out[f] = (len(ps), float(pnl.mean() * 100))
    return out


def trend_stats(roi_per_fold):
    """Linear slope across folds. Permutation p-value (24 perms of 4 indices)."""
    rois = [r for n, r in roi_per_fold.values() if r is not None]
    if len(rois) < 4: return None
    x = np.arange(len(rois), dtype=float)
    y = np.array(rois, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted_first = intercept + slope * 0
    fitted_last  = intercept + slope * (len(x) - 1)

    # Permutation test: shuffle fold order; how often is |slope| as large?
    perms = list(permutations(range(len(x))))
    obs = abs(slope)
    bigger_or_equal = sum(1 for perm in perms
                          if abs(np.polyfit(x, y[list(perm)], 1)[0]) >= obs)
    p_two_sided = bigger_or_equal / len(perms)

    monotone_up = all(rois[i] <= rois[i+1] for i in range(len(rois)-1))
    return dict(slope=round(slope, 2), intercept=round(intercept, 2),
                first_fitted=round(fitted_first, 2), last_fitted=round(fitted_last, 2),
                monotone_up=monotone_up, perm_p=round(p_two_sided, 3))


def evaluate(label, parlays):
    n_total = len(parlays)
    if n_total == 0: return None
    pnl = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in parlays])
    pooled = float(pnl.mean() * 100)
    rpf = per_fold_roi(parlays)
    ts = trend_stats(rpf)
    return dict(label=label, n_total=n_total, pooled_roi=round(pooled, 2),
                fold_roi={f: rpf[f][1] for f in FOLD_ORDER},
                fold_n={f: rpf[f][0] for f in FOLD_ORDER},
                trend=ts)


# Strategies — straight bets + parlays
strategies = [
    ("STRAIGHT  | all +EV",                    "straight", dict()),
    ("STRAIGHT  | edge≥5pp",                   "straight", dict(edge_min=0.05)),
    ("STRAIGHT  | p≥0.65",                     "straight", dict(p_min=0.65)),
    # 2-leg parlays
    ("PARLAY-2  | all +EV pairs",              "parlay",   dict(n_legs=2, top_k=99)),
    ("PARLAY-2  | top-2 by edge",              "parlay",   dict(n_legs=2, top_k=2, picker="top_k_edge")),
    ("PARLAY-2  | top-2 by prob",              "parlay",   dict(n_legs=2, top_k=2, picker="top_k_prob")),
    ("PARLAY-2  | edge≥5pp, all pairs",        "parlay",   dict(n_legs=2, top_k=99, edge_min=0.05)),
    ("PARLAY-2  | edge≥5pp, top-2 by edge",    "parlay",   dict(n_legs=2, top_k=2, edge_min=0.05, picker="top_k_edge")),
    ("PARLAY-2  | p≥0.65, all pairs",          "parlay",   dict(n_legs=2, top_k=99, p_min=0.65, picker="top_k_prob")),
    ("PARLAY-2  | p≥0.65, top-2 by prob",      "parlay",   dict(n_legs=2, top_k=2, p_min=0.65, picker="top_k_prob")),
    # 3-leg parlays
    ("PARLAY-3  | top-3 by edge",              "parlay",   dict(n_legs=3, top_k=3, picker="top_k_edge")),
    ("PARLAY-3  | top-3 by prob",              "parlay",   dict(n_legs=3, top_k=3, picker="top_k_prob")),
    ("PARLAY-3  | top-3 edge≥5pp",             "parlay",   dict(n_legs=3, top_k=3, edge_min=0.05, picker="top_k_edge")),
    # 4-leg parlays
    ("PARLAY-4  | top-4 by edge",              "parlay",   dict(n_legs=4, top_k=4, picker="top_k_edge")),
    ("PARLAY-4  | top-4 edge≥5pp",             "parlay",   dict(n_legs=4, top_k=4, edge_min=0.05, picker="top_k_edge")),
]

results = []
for label, kind, kw in strategies:
    if kind == "straight":
        parlays = make_straights(matched, **kw)
    else:
        parlays = make_parlays(matched, **kw)
    r = evaluate(label, parlays)
    if r is not None: results.append(r)


# ── Print sorted by trend slope (largest positive first) ──────────────────
print("=" * 130)
print(f"{'strategy':<42s}  {'n':>4s}  {'pooled':>8s}    {'F1':>7s}  {'F2':>7s}  {'F3':>7s}  {'F4':>7s}    "
      f"{'slope':>7s}  {'mono↑':>5s}  {'perm_p':>6s}")
print("=" * 130)
sorted_by_slope = sorted(results, key=lambda r: -(r["trend"]["slope"] if r["trend"] else -999))
for r in sorted_by_slope:
    fr = r["fold_roi"]
    fn = r["fold_n"]
    t = r["trend"] or {}
    f_strs = []
    for f in FOLD_ORDER:
        if fr[f] is None: f_strs.append("   nan ")
        else:             f_strs.append(f"{fr[f]:>+6.1f}%")
    mono = "Y" if t.get("monotone_up") else " "
    print(f"  {r['label']:<42s}  {r['n_total']:>4d}  {r['pooled_roi']:>+7.2f}%    "
          f"{f_strs[0]} {f_strs[1]} {f_strs[2]} {f_strs[3]}    "
          f"{t.get('slope', 0):>+6.2f}  {mono:>5s}  {t.get('perm_p', 0):>6.3f}")

print()
print("KEY:")
print("  pooled  = ROI across all 4 folds (point estimate)")
print("  F1..F4  = per-fold ROI (fold_1 = 2024-04→2024-10, fold_4 = 2025-10→2026-04)")
print("  slope   = linear regression slope of ROI ~ fold_index (pp per fold)")
print("  mono↑   = Y if ROI is strictly non-decreasing across all 4 folds")
print("  perm_p  = permutation p-value (24 reorderings of fold labels)")
print("            < 0.10 = trend probably not chance")
print("            > 0.30 = trend looks like noise")
print()

# Highlight strategies with positive slope AND most-recent fold not negative
print("=" * 80)
print("CANDIDATES WITH UPWARD TREND (slope > 0) AND fold_4 ROI > 0:")
print("=" * 80)
candidates = [r for r in results
              if r["trend"] and r["trend"]["slope"] > 0
              and r["fold_roi"]["fold_4"] is not None
              and r["fold_roi"]["fold_4"] > 0]
candidates.sort(key=lambda r: -(r["fold_roi"]["fold_4"] or 0))
for r in candidates:
    fr = r["fold_roi"]; t = r["trend"]
    print(f"  {r['label']}")
    print(f"    pooled={r['pooled_roi']:+.2f}%  slope={t['slope']:+.2f}pp/fold  "
          f"perm_p={t['perm_p']:.3f}  fold_4={fr['fold_4']:+.2f}%")

# Save
out = Path("results/parlay_trend_analysis.json")
out.write_text(json.dumps(results, indent=2))
print(f"\n✓ Saved {out}")
