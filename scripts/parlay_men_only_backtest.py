"""Men-only parlay backtest at both λ=1.20 and λ=1.50, 4-yr training.

Filters cached predictions to sex=2 (men) and re-evaluates the strategies
to see if the men-only filter cleanly improves edge.
"""
import sys, sqlite3, warnings
from pathlib import Path
from itertools import combinations, permutations
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

# Use the 4-yr training cache (matches our preferred config)
CACHE_120 = Path("results/parlay_predictions_lambda120_8fold_4yr.parquet")
matched = pd.read_parquet(CACHE_120)

# Attach sex
conn = sqlite3.connect("data/sqlite_db/sqlite_scrapper.db")
res = pd.read_sql("SELECT jevent, jbout, sex FROM ufc_fight_results", conn)
conn.close()
matched = matched.merge(res, on=["jevent", "jbout"], how="left")
print(f"Loaded {len(matched)} predictions; {(matched['sex']==2).sum()} men, "
      f"{(matched['sex']==1).sum()} women, {matched['sex'].isna().sum()} unknown")

FOLD_ORDER = sorted(matched["fold"].unique())


def make_strategies(df, label):
    pos = df[df["ev"] > 0].copy()
    pos5 = df[(df["ev"] > 0) & (df["edge"] >= 0.05)].copy()
    out = {}
    out["STRAIGHT all +EV"] = [dict(combined_odds=r.dec_odds_pick, won=int(r.won_pick),
                                     fold=str(r.fold)) for r in pos.itertuples()]
    out["STRAIGHT edge≥5pp"] = [dict(combined_odds=r.dec_odds_pick, won=int(r.won_pick),
                                      fold=str(r.fold)) for r in pos5.itertuples()]
    pairs_all = []
    for date, grp in pos.groupby("DATE"):
        if len(grp) < 2: continue
        for combo in combinations(grp.itertuples(index=False), 2):
            co = float(np.prod([c.dec_odds_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            pairs_all.append(dict(combined_odds=co, won=won, fold=str(combo[0].fold)))
    out["PARLAY-2 all +EV pairs"] = pairs_all
    pairs_top2 = []
    for date, grp in pos5.groupby("DATE"):
        g = grp.sort_values("edge", ascending=False).head(2)
        if len(g) < 2: continue
        for combo in combinations(g.itertuples(index=False), 2):
            co = float(np.prod([c.dec_odds_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            pairs_top2.append(dict(combined_odds=co, won=won, fold=str(combo[0].fold)))
    out["PARLAY-2 edge≥5pp top-2 by edge"] = pairs_top2
    return out


def evaluate(parlays):
    if not parlays: return None
    pnl = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in parlays])
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        idx = rng.choice(len(parlays), len(parlays), replace=True)
        boots.append(np.array([(parlays[i]["combined_odds"]-1.0) if parlays[i]["won"] else -1.0
                               for i in idx]).mean() * 100)
    fold_rois = {}
    for f in FOLD_ORDER:
        ps = [p for p in parlays if p["fold"] == f]
        fold_rois[f] = float(np.mean([(p["combined_odds"]-1.0) if p["won"] else -1.0
                                       for p in ps]) * 100) if ps else None
    n_pos = sum(1 for v in fold_rois.values() if v is not None and v > 0)
    valid = [v for v in fold_rois.values() if v is not None]
    return dict(n=len(parlays), pooled=float(pnl.mean()*100),
                hit=float(np.mean([p["won"] for p in parlays])*100),
                ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)),
                n_pos_folds=n_pos, n_folds=len(valid),
                f8=fold_rois.get("fold_8"))


def report(label, df):
    print(f"\n{'='*100}")
    print(f"SLICE: {label}  (n_test={len(df)})")
    print(f"{'='*100}")
    print(f"{'strategy':<42s}  {'n':>5s}  {'ROI':>8s}  {'95% CI':>17s}  {'hit':>6s}  "
          f"{'F8':>7s}  {'pos':>5s}")
    print("-" * 100)
    for label_s, parlays in make_strategies(df, label).items():
        m = evaluate(parlays)
        if m is None:
            print(f"  {label_s:<42s}  (none)"); continue
        ci = f"[{m['ci_lo']:>+5.1f},{m['ci_hi']:>+5.1f}]"
        f8 = f"{m['f8']:>+6.1f}%" if m['f8'] is not None else "  nan "
        print(f"  {label_s:<42s}  {m['n']:>5d}  {m['pooled']:>+7.2f}%  {ci:>17s}  "
              f"{m['hit']:>5.1f}%  {f8}  {m['n_pos_folds']}/{m['n_folds']}")


report("ALL fights (baseline)", matched)
report("MEN only", matched[matched["sex"] == 2])
report("WOMEN only", matched[matched["sex"] == 1])

print()
print("=" * 100)
print("WHAT THIS TELLS YOU:")
print("  If MEN-only has higher pooled ROI + tighter CI than ALL: filter to men in production")
print("  If MEN-only n_parlays is too small: cost of filtering > gain in edge quality")
print("  Compare F8 (forward-looking) — that's the live-betting indicator")
