"""Parlay strategy evaluation using LR with λ=1.20 ALONE (no ensemble).

Hypothesis: λ=1.20 alone shows flat-to-rising per-fold ROI (handles era-drift),
while ensemble inherits λ=0.13's decay. So parlays built on λ=1.20 predictions
should show more stable / forward-looking ROI than ensemble parlays.

Same methodology as parlay_strategy_eval.py except prediction = single LR
trained at λ=1.20 (calibrated). Same walk-forward folds, same +EV bet logic,
same parlay grouping (per-event combinations).
"""
import sys, json, time, warnings
from pathlib import Path
from itertools import combinations, permutations
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
from walk_forward_4fold import FOLDS, select_features, leakage_assertions
from build_walkforward_vegas_multi_threshold import attach_vegas_rich

LAM = 1.20
EPS = 1e-6
FOLD_ORDER = ["fold_1", "fold_2", "fold_3", "fold_4"]


def temp_cal(p_train, y_train):
    p_train = np.clip(p_train, EPS, 1 - EPS)
    logit = np.log(p_train / (1 - p_train))
    def nll(T):
        if T <= 0: return 1e9
        pc = 1 / (1 + np.exp(-logit / T))
        pc = np.clip(pc, EPS, 1 - EPS)
        return -(y_train * np.log(pc) + (1 - y_train) * np.log(1 - pc)).mean()
    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)


def apply_temp(p, T):
    p = np.clip(p, EPS, 1 - EPS); logit = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-logit / T))


def fit_and_predict(train, test, feats, lam):
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
    Xte = sc.transform(imp.transform(test[usable]))
    p_raw = lr.predict_proba(Xte)[:, 1]
    return apply_temp(p_raw, T)


def collect():
    print("Loading...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)
    all_test = []
    for fold in FOLDS:
        train_start = pd.Timestamp(fold["train_start"]); train_end = pd.Timestamp(fold["train_end"])
        test_start  = pd.Timestamp(fold["test_start"]);  test_end  = pd.Timestamp(fold["test_end"])
        train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
        test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
        leakage_assertions(train, test, fold)
        p = fit_and_predict(train, test, feats, LAM)
        test = test.copy(); test["p_model"] = p; test["fold"] = fold["name"]
        all_test.append(test)
        print(f"  {fold['name']}: {len(test)} predictions")
    pred = pd.concat(all_test, ignore_index=True)
    keys = pred[["DATE", "jbout", "jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    merged = pred.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                             "dec_odds_f1", "dec_odds_f2"]],
                        on=["DATE", "jbout", "jfighter"], how="left")
    merged = merged.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    matched = merged[merged["p_vegas_f1"].notna()].copy()
    matched["pick_a"] = (matched["p_model"] >= 0.5).astype(int)
    matched["dec_odds_pick"] = np.where(matched["pick_a"] == 1,
                                         matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_pick"] = np.where(matched["pick_a"] == 1,
                                  matched["p_model"], 1 - matched["p_model"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"] == 1,
                                        matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"] == 1, y == 1, y == 0).astype(int)
    matched["edge"] = matched["p_pick"] - matched["p_vegas_pick"]
    matched["ev"]   = matched["p_pick"] * matched["dec_odds_pick"] - 1.0
    matched = matched.drop_duplicates(subset=["DATE", "jbout"]).reset_index(drop=True)
    print(f"  Deduplicated to one row per bout: {len(matched)}")
    return matched


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
    rois = [r for n, r in roi_per_fold.values() if r is not None]
    if len(rois) < 4: return None
    x = np.arange(len(rois), dtype=float)
    y = np.array(rois, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    perms = list(permutations(range(len(x))))
    obs = abs(slope)
    p_two = sum(1 for perm in perms
                if abs(np.polyfit(x, y[list(perm)], 1)[0]) >= obs) / len(perms)
    monotone_up = all(rois[i] <= rois[i+1] for i in range(len(rois)-1))
    return dict(slope=round(slope, 2), monotone_up=monotone_up, perm_p=round(p_two, 3))


def evaluate(label, parlays):
    n_total = len(parlays)
    if n_total == 0: return None
    pnl = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in parlays])
    pooled = float(pnl.mean() * 100)
    rpf = per_fold_roi(parlays)
    ts = trend_stats(rpf)
    return dict(label=label, n_total=n_total, pooled_roi=round(pooled, 2),
                fold_roi={f: rpf[f][1] for f in FOLD_ORDER},
                fold_n={f: rpf[f][0] for f in FOLD_ORDER}, trend=ts)


def main():
    cache = Path("results/parlay_predictions_lambda120.parquet")
    if cache.exists():
        print(f"Loading cached predictions from {cache}")
        matched = pd.read_parquet(cache)
    else:
        matched = collect()
        matched.to_parquet(cache)
        print(f"Saved predictions to {cache}")

    print(f"\nTotal +EV picks: {(matched['ev']>0).sum()}")
    print(f"Events with ≥2 +EV picks: "
          f"{(matched[matched['ev']>0].groupby('DATE').size() >= 2).sum()}")

    strategies = [
        ("STRAIGHT  | all +EV",                    "straight", dict()),
        ("STRAIGHT  | edge≥5pp",                   "straight", dict(edge_min=0.05)),
        ("STRAIGHT  | p≥0.65",                     "straight", dict(p_min=0.65)),
        ("PARLAY-2  | all +EV pairs",              "parlay",   dict(n_legs=2, top_k=99)),
        ("PARLAY-2  | top-2 by edge",              "parlay",   dict(n_legs=2, top_k=2, picker="top_k_edge")),
        ("PARLAY-2  | top-2 by prob",              "parlay",   dict(n_legs=2, top_k=2, picker="top_k_prob")),
        ("PARLAY-2  | edge≥5pp, all pairs",        "parlay",   dict(n_legs=2, top_k=99, edge_min=0.05)),
        ("PARLAY-2  | edge≥5pp, top-2 by edge",    "parlay",   dict(n_legs=2, top_k=2, edge_min=0.05, picker="top_k_edge")),
        ("PARLAY-2  | p≥0.65, all pairs",          "parlay",   dict(n_legs=2, top_k=99, p_min=0.65, picker="top_k_prob")),
        ("PARLAY-2  | p≥0.65, top-2 by prob",      "parlay",   dict(n_legs=2, top_k=2, p_min=0.65, picker="top_k_prob")),
        ("PARLAY-3  | top-3 by edge",              "parlay",   dict(n_legs=3, top_k=3, picker="top_k_edge")),
        ("PARLAY-3  | top-3 by prob",              "parlay",   dict(n_legs=3, top_k=3, picker="top_k_prob")),
        ("PARLAY-3  | top-3 edge≥5pp",             "parlay",   dict(n_legs=3, top_k=3, edge_min=0.05, picker="top_k_edge")),
        ("PARLAY-4  | top-4 by edge",              "parlay",   dict(n_legs=4, top_k=4, picker="top_k_edge")),
        ("PARLAY-4  | top-4 edge≥5pp",             "parlay",   dict(n_legs=4, top_k=4, edge_min=0.05, picker="top_k_edge")),
    ]
    results = []
    for label, kind, kw in strategies:
        if kind == "straight": parlays = make_straights(matched, **kw)
        else:                  parlays = make_parlays(matched, **kw)
        r = evaluate(label, parlays)
        if r is not None: results.append(r)

    print()
    print("=" * 130)
    print(f"{'strategy':<42s}  {'n':>4s}  {'pooled':>8s}    {'F1':>7s}  {'F2':>7s}  {'F3':>7s}  {'F4':>7s}    "
          f"{'slope':>7s}  {'mono↑':>5s}  {'perm_p':>6s}")
    print("=" * 130)
    sorted_by_slope = sorted(results, key=lambda r: -(r["trend"]["slope"] if r["trend"] else -999))
    for r in sorted_by_slope:
        fr = r["fold_roi"]; t = r["trend"] or {}
        f_strs = []
        for f in FOLD_ORDER:
            if fr[f] is None: f_strs.append("   nan ")
            else:             f_strs.append(f"{fr[f]:>+6.1f}%")
        mono = "Y" if t.get("monotone_up") else " "
        print(f"  {r['label']:<42s}  {r['n_total']:>4d}  {r['pooled_roi']:>+7.2f}%    "
              f"{f_strs[0]} {f_strs[1]} {f_strs[2]} {f_strs[3]}    "
              f"{t.get('slope', 0):>+6.2f}  {mono:>5s}  {t.get('perm_p', 0):>6.3f}")

    print()
    print("CANDIDATES TRENDING UP (slope > 0) WITH POSITIVE FOLD_4:")
    print("=" * 80)
    cands = [r for r in results
             if r["trend"] and r["trend"]["slope"] > 0
             and r["fold_roi"]["fold_4"] is not None
             and r["fold_roi"]["fold_4"] > 0]
    cands.sort(key=lambda r: -r["fold_roi"]["fold_4"])
    if not cands:
        print("  (none)")
    for r in cands:
        fr = r["fold_roi"]; t = r["trend"]
        print(f"  {r['label']}")
        print(f"    pooled={r['pooled_roi']:+.2f}%  slope={t['slope']:+.2f}pp/fold  "
              f"perm_p={t['perm_p']:.3f}  fold_4={fr['fold_4']:+.2f}%  n={r['n_total']}")

    out = Path("results/parlay_lambda120_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
