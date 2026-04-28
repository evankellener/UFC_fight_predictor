"""Validate parlay strategy ROIs with bootstrap CIs + theoretical EV.

Reads the cached predictions from parlay_strategy_eval.py and runs each
strategy with:
  - Realized pooled ROI
  - Bootstrap 95% CI on ROI (10,000 resamples of parlays with replacement)
  - Theoretical model-implied EV (multiply leg probabilities × multiplied odds)
  - Per-fold breakdown to check if one fold drives the result
  - Realized vs theoretical gap (positive gap = sample variance got lucky)
"""
import json, sys, warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

CACHE = Path("results/parlay_predictions.parquet")
if not CACHE.exists():
    print("Run parlay_strategy_eval.py first to populate predictions cache.")
    sys.exit(1)

matched = pd.read_parquet(CACHE)
print(f"Loaded {len(matched)} predictions from cache")
print(f"  Folds: {sorted(matched['fold'].unique())}")
print()


def make_parlays(df, n_legs, picker="top_k_edge", edge_min=None, p_min=None, top_k=None):
    """Return list of dicts, one per parlay: combined_odds, model_p, won, fold."""
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
            combined_p    = float(np.prod([c.p_pick for c in combo]))
            combined_p_v  = float(np.prod([c.p_vegas_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            parlays.append(dict(combined_odds=combined_odds,
                                combined_p=combined_p,
                                combined_p_vegas=combined_p_v,
                                won=won, fold=str(combo[0].fold), date=str(combo[0].DATE)))
    return parlays


def evaluate(parlays, label, n_boot=10000, seed=42):
    if not parlays:
        return dict(label=label, n=0)
    pnl = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in parlays])
    p_model    = np.array([p["combined_p"] for p in parlays])
    p_vegas    = np.array([p["combined_p_vegas"] for p in parlays])
    odds       = np.array([p["combined_odds"] for p in parlays])
    won        = np.array([p["won"] for p in parlays])
    folds      = [p["fold"] for p in parlays]

    realized_roi = pnl.mean() * 100
    theoretical_ev_model = (p_model * odds - 1).mean() * 100  # model-believed EV
    theoretical_ev_vegas = (p_vegas * odds - 1).mean() * 100  # devigged Vegas EV (should be ≈0)
    hit_rate = won.mean() * 100

    # Bootstrap ROI
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(pnl)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot[i] = pnl[idx].mean() * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Per-fold breakdown
    fold_breakdown = {}
    for f in sorted(set(folds)):
        mask = np.array([fl == f for fl in folds])
        if mask.sum() == 0: continue
        fold_breakdown[f] = dict(n=int(mask.sum()),
                                 roi=float(pnl[mask].mean() * 100),
                                 hits=int(won[mask].sum()))

    return dict(label=label, n=int(n), realized_roi=round(realized_roi, 2),
                ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                ci_crosses_zero=bool(lo < 0 < hi),
                hit_rate=round(hit_rate, 2),
                theoretical_ev_model=round(theoretical_ev_model, 2),
                theoretical_ev_vegas=round(theoretical_ev_vegas, 2),
                avg_combined_odds=round(odds.mean(), 2),
                fold_breakdown=fold_breakdown)


# ── Strategies to validate ────────────────────────────────────────────────
strategies = [
    # 2-leg
    ("S2  | all +EV pairs/card",         dict(n_legs=2, picker="top_k_edge", top_k=99)),
    ("S3  | top-2 by edge/card",         dict(n_legs=2, picker="top_k_edge", top_k=2)),
    ("S4  | top-2 by p_pick/card",       dict(n_legs=2, picker="top_k_prob", top_k=2)),
    ("S6  | edge≥5pp, all pairs/card",   dict(n_legs=2, picker="top_k_edge", top_k=99, edge_min=0.05)),
    ("S6b | edge≥5pp, top-2 by edge",    dict(n_legs=2, picker="top_k_edge", top_k=2, edge_min=0.05)),
    ("S7  | p≥0.65, all pairs/card",     dict(n_legs=2, picker="top_k_prob", top_k=99, p_min=0.65)),
    ("S7b | p≥0.65, top-2 by prob",      dict(n_legs=2, picker="top_k_prob", top_k=2, p_min=0.65)),
    # 3-leg
    ("S5  | top-3 by edge",              dict(n_legs=3, picker="top_k_edge", top_k=3)),
    ("S5b | top-3 by prob",              dict(n_legs=3, picker="top_k_prob", top_k=3)),
    ("S5c | top-3 edge≥5pp",             dict(n_legs=3, picker="top_k_edge", top_k=3, edge_min=0.05)),
    # 4-leg
    ("S5d | top-4 by edge",              dict(n_legs=4, picker="top_k_edge", top_k=4)),
    ("S5e | top-4 by prob",              dict(n_legs=4, picker="top_k_prob", top_k=4)),
    ("S5f | top-4 edge≥5pp",             dict(n_legs=4, picker="top_k_edge", top_k=4, edge_min=0.05)),
]

print("=" * 120)
print(f"{'strategy':<35s}  {'n':>4s}  {'ROI':>8s}  {'95% CI':>17s}  {'hits%':>6s}  "
      f"{'EV_mod':>7s}  {'EV_veg':>7s}  {'avg_dec':>7s}  cross_0")
print("=" * 120)
results = []
for label, kw in strategies:
    parlays = make_parlays(matched, **kw)
    r = evaluate(parlays, label)
    results.append(r)
    if r["n"] == 0:
        print(f"  {label:<35s}  (no parlays)"); continue
    cross = "✓" if r["ci_crosses_zero"] else " "
    print(f"  {label:<35s}  {r['n']:>4d}  {r['realized_roi']:>+7.2f}%  "
          f"[{r['ci_lo']:>+6.1f}, {r['ci_hi']:>+6.1f}]  "
          f"{r['hit_rate']:>5.1f}%  {r['theoretical_ev_model']:>+6.2f}%  "
          f"{r['theoretical_ev_vegas']:>+6.2f}%  {r['avg_combined_odds']:>6.2f}   {cross}")

print()
print("KEY:")
print("  ROI         = realized pooled ROI per parlay (% of stake)")
print("  95% CI      = bootstrap 95% confidence interval (10,000 resamples)")
print("  cross_0     = ✓ if CI crosses zero (statistically indistinguishable from 0% ROI)")
print("  EV_mod      = theoretical EV assuming model probabilities are correct")
print("  EV_veg      = theoretical EV assuming Vegas devigged probs are correct (≈0 means")
print("                Vegas prices the parlay fairly; >0 = even Vegas-fair would profit)")
print("  avg_dec     = average combined decimal odds (1.0 = 'even'; payout multiplier on win)")
print()
print("Per-fold breakdown for top strategies:")
for r in sorted(results, key=lambda x: -x.get("realized_roi", -999))[:5]:
    if r["n"] == 0: continue
    print(f"\n  {r['label']}  (n={r['n']}, ROI {r['realized_roi']:+.2f}%):")
    for f, fb in r["fold_breakdown"].items():
        print(f"    {f}: n={fb['n']:>3d}  ROI={fb['roi']:>+8.2f}%  hits={fb['hits']:>3d}")

out = Path("results/parlay_strategy_validate.json")
out.write_text(json.dumps(results, indent=2))
print(f"\n✓ Saved {out}")
