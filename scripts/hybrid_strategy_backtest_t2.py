"""Hybrid strategy backtest at THRESHOLD=2 prior UFC fights (vs default 3).

Same strategies as hybrid_strategy_backtest.py but lowers the prior-fight
threshold from 3 to 2 — meaning fighters with as few as 2 UFC fights are
included in BOTH training and test. More data per fold but each pick has
weaker average edge per finding_threshold_matters.md (t=2: +9.7% borderline;
t=3: +16.4% p=0.007).

Compare these results against threshold=3 baseline to see if the extra data
volume offsets the weaker per-pick edge.
"""
import sys, json, sqlite3, warnings
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
LAM = 1.50
TRAIN_YEARS = 4
THRESHOLD = 2   # ← CHANGED from 3

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
    p = np.clip(p, EPS, 1-EPS); logit = np.log(p/(1-p))
    def nll(T):
        if T <= 0: return 1e9
        pc = 1/(1+np.exp(-logit/T)); pc = np.clip(pc, EPS, 1-EPS)
        return -(y*np.log(pc) + (1-y)*np.log(1-pc)).mean()
    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)


def fit_predict(train, test, feats):
    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    train_end = test["DATE"].min()
    w = np.exp(-LAM * (train_end - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)
    p_tr = lr.predict_proba(sc.transform(imp.transform(train[usable])))[:, 1]
    T = temp_cal(p_tr, train["win"].astype(int).values)
    p_raw = lr.predict_proba(sc.transform(imp.transform(test[usable])))[:, 1]
    p = np.clip(p_raw, EPS, 1-EPS); lg = np.log(p/(1-p))
    return 1/(1+np.exp(-lg/T))


def collect_matched():
    print(f"Generating λ=1.50 / 4-yr predictions, THRESHOLD={THRESHOLD}, 8 folds...")
    base = load_base_both_elos()
    df = apply_threshold(base, THRESHOLD)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)
    print(f"  Filtered df has {len(df)} rows (threshold={THRESHOLD})")
    parts = []
    for fold in FOLDS:
        ts, te = pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"])
        s, e = pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
        train = df[(df["DATE"] >= ts) & (df["DATE"] < te)].copy()
        test  = df[(df["DATE"] >= s) & (df["DATE"] < e)].copy()
        if len(test) == 0: continue
        leakage_assertions(train, test, fold)
        p = fit_predict(train, test, feats)
        test = test.copy(); test["p_model"] = p; test["fold"] = fold["name"]
        parts.append(test)
        print(f"  {fold['name']}  train={len(train)}  test={len(test)}")
    pred = pd.concat(parts, ignore_index=True)
    keys = pred[["DATE","jbout","jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    merged = pred.merge(tv[["DATE","jbout","jfighter","p_vegas_f1","dec_odds_f1","dec_odds_f2"]],
                        on=["DATE","jbout","jfighter"], how="left")
    merged = merged.drop_duplicates(subset=["DATE","jbout","jfighter"]).reset_index(drop=True)
    matched = merged[merged["p_vegas_f1"].notna()].copy()
    matched["pick_a"] = (matched["p_model"] >= 0.5).astype(int)
    matched["dec_odds_pick"] = np.where(matched["pick_a"]==1, matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_pick"] = np.where(matched["pick_a"]==1, matched["p_model"], 1 - matched["p_model"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"]==1, matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["edge"] = matched["p_pick"] - matched["p_vegas_pick"]
    matched["ev"]   = matched["p_pick"] * matched["dec_odds_pick"] - 1.0
    matched = matched.drop_duplicates(subset=["DATE","jbout"]).reset_index(drop=True)
    conn = sqlite3.connect("data/sqlite_db/sqlite_scrapper.db")
    res = pd.read_sql("SELECT jevent, jbout, sex FROM ufc_fight_results", conn)
    conn.close()
    matched = matched.merge(res, on=["jevent","jbout"], how="left")
    return matched[matched["sex"]==2]  # men-only


def hybrid(matched, parlay_min, single_min, edge_min=0.05):
    pos = matched[(matched["ev"]>0) & (matched["edge"]>=edge_min)].copy()
    bets = []
    for date, grp in pos.groupby("DATE"):
        ranked = grp.sort_values("edge", ascending=False)
        if len(ranked) >= 2 and ranked.iloc[0]["edge"]>=parlay_min and ranked.iloc[1]["edge"]>=parlay_min:
            top2 = ranked.head(2)
            for combo in combinations(top2.itertuples(index=False), 2):
                bets.append(dict(type="parlay",
                                 odds=float(np.prod([c.dec_odds_pick for c in combo])),
                                 won=int(np.prod([c.won_pick for c in combo])),
                                 fold=str(combo[0].fold)))
        elif ranked.iloc[0]["edge"] >= single_min:
            r = ranked.iloc[0]
            bets.append(dict(type="single", odds=float(r["dec_odds_pick"]),
                             won=int(r["won_pick"]), fold=str(r["fold"])))
    return bets


def pure_parlay(matched, edge_threshold=0.05):
    pos = matched[(matched["ev"]>0) & (matched["edge"]>=edge_threshold)].copy()
    bets = []
    for date, grp in pos.groupby("DATE"):
        top2 = grp.sort_values("edge", ascending=False).head(2)
        if len(top2) < 2: continue
        for combo in combinations(top2.itertuples(index=False), 2):
            bets.append(dict(type="parlay",
                             odds=float(np.prod([c.dec_odds_pick for c in combo])),
                             won=int(np.prod([c.won_pick for c in combo])),
                             fold=str(combo[0].fold)))
    return bets


def pure_singles(matched, edge_threshold=0.05):
    pos = matched[(matched["ev"]>0) & (matched["edge"]>=edge_threshold)].copy()
    return [dict(type="single", odds=float(r.dec_odds_pick),
                 won=int(r.won_pick), fold=str(r.fold)) for r in pos.itertuples()]


def evaluate(bets):
    if not bets: return None
    pnl = np.array([(b["odds"]-1.0) if b["won"] else -1.0 for b in bets])
    rng = np.random.default_rng(42); boots = []
    for _ in range(2000):
        idx = rng.choice(len(bets), len(bets), replace=True)
        boots.append(np.array([(bets[i]["odds"]-1.0) if bets[i]["won"] else -1.0
                               for i in idx]).mean() * 100)
    fold_rois = {}
    for f in FOLD_ORDER:
        bb = [b for b in bets if b.get("fold") == f]
        fold_rois[f] = float(np.mean([(b["odds"]-1.0) if b["won"] else -1.0 for b in bb]) * 100) if bb else None
    n_pos = sum(1 for v in fold_rois.values() if v is not None and v > 0)
    n_total = sum(1 for v in fold_rois.values() if v is not None)
    n_p = sum(1 for b in bets if b["type"]=="parlay")
    return dict(n=len(bets), n_parlay=n_p, n_single=len(bets)-n_p,
                pooled=float(pnl.mean()*100),
                hit=float(np.mean([b["won"] for b in bets])*100),
                ci_lo=float(np.percentile(boots,2.5)),
                ci_hi=float(np.percentile(boots,97.5)),
                f8=fold_rois.get("fold_8"),
                pos_folds=f"{n_pos}/{n_total}")


def main():
    cache = Path(f"results/hybrid_matched_lambda150_men_t{THRESHOLD}.parquet")
    if cache.exists():
        matched = pd.read_parquet(cache)
        print(f"Loaded {len(matched)} cached predictions")
    else:
        matched = collect_matched()
        matched.to_parquet(cache)
        print(f"Cached {len(matched)} predictions")

    print()
    strategies = [
        ("STRAIGHT edge≥5pp",                 pure_singles(matched, 0.05)),
        ("STRAIGHT edge≥10pp",                pure_singles(matched, 0.10)),
        ("PARLAY-2 edge≥5pp top-2",           pure_parlay(matched, 0.05)),
        ("PARLAY-2 edge≥10pp top-2 ★",        pure_parlay(matched, 0.10)),
        ("H1: parlay≥10pp / single≥10pp",     hybrid(matched, 0.10, 0.10)),
        ("H2: parlay≥10pp / single≥5pp",      hybrid(matched, 0.10, 0.05)),
        ("H3: parlay≥7pp / single≥10pp",      hybrid(matched, 0.07, 0.10)),
        ("H4: parlay≥5pp / single≥10pp",      hybrid(matched, 0.05, 0.10)),
        ("H5: parlay≥7pp / single≥7pp",       hybrid(matched, 0.07, 0.07)),
        ("H6: parlay≥10pp / single≥7pp",      hybrid(matched, 0.10, 0.07)),
    ]

    print("=" * 145)
    print(f"THRESHOLD={THRESHOLD} (≥{THRESHOLD} prior UFC fights both sides)")
    print("=" * 145)
    print(f"{'strategy':<46s}  {'n':>4s}  {'P|S':>7s}  {'pooled':>9s}  "
          f"{'95% CI':>17s}  {'hit%':>5s}  {'F8':>7s}  {'pos_f':>5s}")
    print("-" * 145)
    rows = []
    for label, bets in strategies:
        m = evaluate(bets)
        if m is None:
            print(f"  {label:<46s}  (no bets)"); continue
        ci = f"[{m['ci_lo']:>+5.1f},{m['ci_hi']:>+5.1f}]"
        f8 = f"{m['f8']:>+5.1f}%" if m['f8'] is not None else "  nan "
        ps = f"{m['n_parlay']}|{m['n_single']}"
        print(f"  {label:<46s}  {m['n']:>4d}  {ps:>7s}  {m['pooled']:>+8.2f}%  "
              f"{ci:>17s}  {m['hit']:>4.1f}%  {f8}  {m['pos_folds']:>5s}")
        rows.append(dict(label=label, **m))

    # Compare to threshold=3 cached results if available
    t3_path = Path("results/hybrid_strategy_backtest.json")
    if t3_path.exists():
        t3 = {r["label"]: r for r in json.loads(t3_path.read_text())}
        print()
        print("=" * 110)
        print(f"COMPARISON: threshold=2 vs threshold=3")
        print("=" * 110)
        print(f"{'strategy':<46s}  {'t=3 pooled':>11s}  {'t=2 pooled':>11s}  {'Δ':>9s}  {'t=3 n':>6s}  {'t=2 n':>6s}")
        print("-" * 110)
        for r in rows:
            label = r["label"].replace(" ★", "").strip()
            for k, v in t3.items():
                if k.replace(" ★", "").strip().startswith(label[:30]):
                    delta = r["pooled"] - v["pooled"]
                    print(f"  {label:<46s}  {v['pooled']:>+9.2f}%  {r['pooled']:>+9.2f}%  "
                          f"{delta:>+7.2f}pp  {v['n']:>6d}  {r['n']:>6d}")
                    break

    Path(f"results/hybrid_strategy_backtest_t{THRESHOLD}.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n✓ Saved results/hybrid_strategy_backtest_t{THRESHOLD}.json")


if __name__ == "__main__":
    main()
