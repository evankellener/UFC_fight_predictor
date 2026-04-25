"""λ-sweep at 4-yr training across multiple strategies — confirm 1.5 holds.

For each λ ∈ {1.0, 1.2, 1.5, 2.0}, run 8 folds × 3-mo tests with 4-yr training,
and evaluate THREE strategies:
  - STRAIGHT all +EV
  - STRAIGHT edge≥5pp
  - PARLAY-2 all +EV pairs
  - PARLAY-2 edge≥5pp top-2 by edge (the candidate winner)

Reports pooled ROI + F8 (most recent fold) per strategy per λ.
Tells us whether λ=1.5 is universally better or just for the parlay strategy.
"""
import sys, json, warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features, flip_row_dataframe
from walk_forward_4fold import select_features, leakage_assertions
from build_walkforward_vegas_multi_threshold import attach_vegas_rich

EPS = 1e-6
TRAIN_YEARS = 4
LAMBDAS = [1.0, 1.2, 1.5, 2.0]

def build_folds():
    starts = pd.date_range("2024-04-01", "2026-01-01", freq="3MS")
    return [{"name": f"fold_{i}",
             "train_start": (s - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y-%m-%d"),
             "train_end":   s.strftime("%Y-%m-%d"),
             "test_start":  s.strftime("%Y-%m-%d"),
             "test_end":   (s + pd.DateOffset(months=3)).strftime("%Y-%m-%d")}
            for i, s in enumerate(starts, 1)]
FOLDS = build_folds()
FOLD_ORDER = [f["name"] for f in FOLDS]

def temp_cal(p, y):
    p = np.clip(p, EPS, 1 - EPS); logit = np.log(p / (1 - p))
    def nll(T):
        if T <= 0: return 1e9
        pc = 1 / (1 + np.exp(-logit / T)); pc = np.clip(pc, EPS, 1 - EPS)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()
    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)

def apply_temp(p, T):
    p = np.clip(p, EPS, 1 - EPS); return 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))

def fit_predict(train, test, feats, lam):
    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    train_end = test["DATE"].min()
    w = np.exp(-lam * (train_end - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)
    X_tr_orig = sc.transform(imp.transform(train[usable]))
    p_tr = lr.predict_proba(X_tr_orig)[:, 1]
    T = temp_cal(p_tr, train["win"].astype(int).values)
    return apply_temp(lr.predict_proba(sc.transform(imp.transform(test[usable])))[:, 1], T)

def collect(df, feats, lam):
    parts = []
    for fold in FOLDS:
        ts, te = pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"])
        s, e = pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
        train = df[(df["DATE"] >= ts) & (df["DATE"] < te)].copy()
        test  = df[(df["DATE"] >= s) & (df["DATE"] < e)].copy()
        if len(test) == 0: continue
        leakage_assertions(train, test, fold)
        p = fit_predict(train, test, feats, lam)
        test = test.copy(); test["p_model"] = p; test["fold"] = fold["name"]
        parts.append(test)
    pred = pd.concat(parts, ignore_index=True)
    keys = pred[["DATE", "jbout", "jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    merged = pred.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                             "dec_odds_f1", "dec_odds_f2"]],
                        on=["DATE", "jbout", "jfighter"], how="left")
    merged = merged.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    matched = merged[merged["p_vegas_f1"].notna()].copy()
    matched["pick_a"] = (matched["p_model"] >= 0.5).astype(int)
    matched["dec_odds_pick"] = np.where(matched["pick_a"]==1, matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_pick"] = np.where(matched["pick_a"]==1, matched["p_model"], 1 - matched["p_model"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"]==1, matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["edge"] = matched["p_pick"] - matched["p_vegas_pick"]
    matched["ev"]   = matched["p_pick"] * matched["dec_odds_pick"] - 1.0
    return matched.drop_duplicates(subset=["DATE", "jbout"]).reset_index(drop=True)

def make_straights(df, edge_min=None):
    pos = df[df["ev"] > 0].copy()
    if edge_min is not None: pos = pos[pos["edge"] >= edge_min]
    return [dict(combined_odds=r.dec_odds_pick, won=int(r.won_pick), fold=str(r.fold))
            for r in pos.itertuples()]

def make_parlays(df, n_legs=2, edge_min=None, top_k=None):
    pos = df[df["ev"] > 0].copy()
    if edge_min is not None: pos = pos[pos["edge"] >= edge_min]
    out = []
    k = top_k if top_k else 999
    for date, grp in pos.groupby("DATE"):
        g = grp.sort_values("edge", ascending=False).head(k)
        if len(g) < n_legs: continue
        for combo in combinations(g.itertuples(index=False), n_legs):
            co = float(np.prod([c.dec_odds_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            out.append(dict(combined_odds=co, won=won, fold=str(combo[0].fold)))
    return out

def evaluate(parlays):
    if not parlays: return None
    pnl = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in parlays])
    fold_rois = {}
    for f in FOLD_ORDER:
        ps = [p for p in parlays if p["fold"] == f]
        fold_rois[f] = float(np.mean([(p["combined_odds"]-1.0) if p["won"] else -1.0
                                       for p in ps]) * 100) if ps else None
    return dict(n=len(parlays), pooled=float(pnl.mean()*100), folds=fold_rois)


def main():
    print("Loading base...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)

    rows = []
    for lam in LAMBDAS:
        print(f"\n=== λ={lam}, 4-yr training ===")
        matched = collect(df, feats, lam)
        for label, fn in [
            ("STRAIGHT all +EV",                   lambda d: make_straights(d)),
            ("STRAIGHT edge≥5pp",                  lambda d: make_straights(d, edge_min=0.05)),
            ("PARLAY-2 all +EV pairs",             lambda d: make_parlays(d, 2)),
            ("PARLAY-2 edge≥5pp top-2 by edge",    lambda d: make_parlays(d, 2, edge_min=0.05, top_k=2)),
        ]:
            parlays = fn(matched)
            m = evaluate(parlays)
            if not m: continue
            f8 = m["folds"].get("fold_8")
            n_pos = sum(1 for v in m["folds"].values() if v is not None and v > 0)
            n_total = sum(1 for v in m["folds"].values() if v is not None)
            print(f"  {label:<40s}  n={m['n']:>4d}  pooled={m['pooled']:>+7.2f}%  "
                  f"F8={f8 if f8 is None else f'{f8:>+6.2f}%'}  pos={n_pos}/{n_total}")
            rows.append(dict(lam=lam, strategy=label, **m))

    print()
    print("=" * 110)
    print("PIVOTED VIEW — Pooled ROI by (strategy × λ):")
    print("=" * 110)
    print(f"{'strategy':<42s}  " + "  ".join(f"λ={l}".rjust(10) for l in LAMBDAS))
    print("-" * 110)
    strategies = list(dict.fromkeys([r["strategy"] for r in rows]))
    for s in strategies:
        cells = []
        for lam in LAMBDAS:
            r = next((x for x in rows if x["lam"]==lam and x["strategy"]==s), None)
            cells.append(f"{r['pooled']:>+7.2f}%" if r else "    -    ")
        print(f"  {s:<42s}  " + "  ".join(c.rjust(10) for c in cells))

    print()
    print(f"{'strategy':<42s}  " + "  ".join(f"F8 λ={l}".rjust(10) for l in LAMBDAS))
    print("-" * 110)
    for s in strategies:
        cells = []
        for lam in LAMBDAS:
            r = next((x for x in rows if x["lam"]==lam and x["strategy"]==s), None)
            f8 = r["folds"].get("fold_8") if r else None
            cells.append(f"{f8:>+7.2f}%" if f8 is not None else "    -    ")
        print(f"  {s:<42s}  " + "  ".join(c.rjust(10) for c in cells))

    out = Path("results/lambda_sweep_all_strategies.json")
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
