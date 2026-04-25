"""Slice the parlay backtest by sex (men vs women).

Uses the cached 4-yr-training λ=1.20 predictions from
parlay_predictions_lambda120_8fold_4yr.parquet, joins on
ufc_fight_results.sex (1=women, 2=men), and computes pooled ROI separately
for each sex on:
  - STRAIGHT all +EV
  - STRAIGHT edge≥5pp
  - PARLAY-2 all +EV pairs (mixed-sex pairs OK if both qualify)
  - PARLAY-2 edge≥5pp top-2 by edge (the production strategy candidate)

Sample sizes will be small for women (~11% of fights), so the women's slice
will have very wide CIs. The point is to check sign and direction.
"""
import sys, sqlite3, warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

CACHE = Path("results/parlay_predictions_lambda120_8fold_4yr.parquet")
matched = pd.read_parquet(CACHE)
print(f"Loaded {len(matched)} predictions")

# Attach sex from ufc_fight_results
conn = sqlite3.connect("data/sqlite_db/sqlite_scrapper.db")
res = pd.read_sql("SELECT jevent, jbout, sex FROM ufc_fight_results", conn)
conn.close()
matched = matched.merge(res, on=["jevent", "jbout"], how="left")
matched["sex_label"] = matched["sex"].map({1.0: "Women", 2.0: "Men"})
print(f"Sex distribution in matched test fights:")
print(matched["sex_label"].value_counts(dropna=False).to_string())
print()


def metrics_pooled(parlays):
    if not parlays: return None
    pnl = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in parlays])
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        idx = rng.choice(len(parlays), len(parlays), replace=True)
        boots.append(np.array([(parlays[i]["combined_odds"]-1.0) if parlays[i]["won"] else -1.0
                               for i in idx]).mean() * 100)
    return dict(n=len(parlays), roi=float(pnl.mean()*100),
                hit=float(np.mean([p["won"] for p in parlays])*100),
                ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)))


def make_straights(df, edge_min=None):
    pos = df[df["ev"] > 0].copy()
    if edge_min is not None: pos = pos[pos["edge"] >= edge_min]
    return [dict(combined_odds=r.dec_odds_pick, won=int(r.won_pick))
            for r in pos.itertuples()]


def make_parlays_all_pairs(df, edge_min=None):
    pos = df[df["ev"] > 0].copy()
    if edge_min is not None: pos = pos[pos["edge"] >= edge_min]
    parlays = []
    for date, grp in pos.groupby("DATE"):
        if len(grp) < 2: continue
        for combo in combinations(grp.itertuples(index=False), 2):
            co = float(np.prod([c.dec_odds_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            parlays.append(dict(combined_odds=co, won=won))
    return parlays


def make_parlays_top2_edge(df, edge_min=0.05):
    pos = df[(df["ev"] > 0) & (df["edge"] >= edge_min)].copy()
    parlays = []
    for date, grp in pos.groupby("DATE"):
        g = grp.sort_values("edge", ascending=False).head(2)
        if len(g) < 2: continue
        for combo in combinations(g.itertuples(index=False), 2):
            co = float(np.prod([c.dec_odds_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            parlays.append(dict(combined_odds=co, won=won))
    return parlays


# Compute results per sex slice
def slice_subset(df, label):
    print(f"\n{'='*100}")
    print(f"SLICE: {label}  (n_test_fights={len(df)})")
    print(f"{'='*100}")
    print(f"{'strategy':<42s}  {'n':>5s}  {'ROI':>9s}  {'hit%':>6s}  {'95% CI':>17s}")
    print("-" * 100)
    for label_s, fn in [
        ("STRAIGHT  all +EV",                   lambda d: make_straights(d)),
        ("STRAIGHT  edge≥5pp",                  lambda d: make_straights(d, edge_min=0.05)),
        ("PARLAY-2  all +EV pairs/card",        lambda d: make_parlays_all_pairs(d)),
        ("PARLAY-2  edge≥5pp top-2 by edge",    lambda d: make_parlays_top2_edge(d, edge_min=0.05)),
    ]:
        parlays = fn(df)
        m = metrics_pooled(parlays)
        if m is None:
            print(f"  {label_s:<42s}  (no qualifying)"); continue
        ci = f"[{m['ci_lo']:>+5.1f},{m['ci_hi']:>+5.1f}]"
        print(f"  {label_s:<42s}  {m['n']:>5d}  {m['roi']:>+8.2f}%  {m['hit']:>5.1f}%  {ci:>17s}")


slice_subset(matched, "ALL fights (baseline)")
slice_subset(matched[matched["sex_label"] == "Men"], "MEN only")
slice_subset(matched[matched["sex_label"] == "Women"], "WOMEN only")
print()
print("=" * 100)
print("INTERPRETATION:")
print("  If women's-fight ROI > men's: tonight's recommendation (mixed-sex parlay) is reasonable")
print("  If women's-fight ROI ≈ men's: no edge difference; treat all fights equally")
print("  If women's-fight ROI < men's: model has weaker edge on women — be cautious tonight")
print("  CIs will be very wide on women's slice (small n) — sign matters more than magnitude")
