"""Option C: 8 folds × 3-month tests with 4-YEAR training windows.

Same as parlay_lambda120_8fold.py but TRAIN_YEARS = 4 instead of 7.
Consecutive folds now share ~85% of training data (vs ~99% with 7-yr)
so the per-fold ROIs are more statistically independent.

If the trend / no-trend pattern survives this independence check, it's a
real result. If a "trend" was an artifact of overlapping training data,
shrinking the overlap will break it.

Test fights are IDENTICAL to the 7-yr run (same 2024-04 → 2026-04 windows),
so no new odds scraping required.
"""
import sys, json, warnings
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
from walk_forward_4fold import select_features, leakage_assertions
from build_walkforward_vegas_multi_threshold import attach_vegas_rich

LAM = 1.20
EPS = 1e-6
TRAIN_YEARS = 4   # ← CHANGED from 7

def build_folds():
    starts = pd.date_range("2024-04-01", "2026-01-01", freq="3MS")
    folds = []
    for i, s in enumerate(starts, 1):
        ts = s
        te = s + pd.DateOffset(months=3)
        train_start = ts - pd.DateOffset(years=TRAIN_YEARS)
        folds.append({"name": f"fold_{i}",
                      "train_start": train_start.strftime("%Y-%m-%d"),
                      "train_end":   ts.strftime("%Y-%m-%d"),
                      "test_start":  ts.strftime("%Y-%m-%d"),
                      "test_end":    te.strftime("%Y-%m-%d")})
    return folds

FOLDS_8 = build_folds()
FOLD_ORDER = [f["name"] for f in FOLDS_8]


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
    print(f"8 folds × 3mo tests, {TRAIN_YEARS}-yr training (Option C: more fold independence)")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)
    all_test = []
    for fold in FOLDS_8:
        train_start = pd.Timestamp(fold["train_start"]); train_end = pd.Timestamp(fold["train_end"])
        test_start  = pd.Timestamp(fold["test_start"]);  test_end  = pd.Timestamp(fold["test_end"])
        train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
        test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
        if len(test) == 0: continue
        leakage_assertions(train, test, fold)
        p = fit_and_predict(train, test, feats, LAM)
        test = test.copy(); test["p_model"] = p; test["fold"] = fold["name"]
        all_test.append(test)
        print(f"  {fold['name']}  train {fold['train_start']}→{fold['train_end']} "
              f"({len(train)} fights)  test {fold['test_start']}→{fold['test_end']} ({len(test)})")
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
    print(f"  Total bouts after dedup + Vegas match: {len(matched)}")
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
    obs = abs(slope)
    perms = list(permutations(range(len(x))))
    bigger = sum(1 for perm in perms
                 if abs(np.polyfit(x, y[list(perm)], 1)[0]) >= obs)
    perm_p = bigger / len(perms)
    rng = np.random.default_rng(42)
    boot_slopes = []
    for _ in range(1000):
        idx = rng.choice(len(x), len(x), replace=True)
        if len(set(idx)) < 2: continue
        bs, _ = np.polyfit(x[idx], y[idx], 1)
        boot_slopes.append(bs)
    boot_slopes = np.array(boot_slopes)
    slo, shi = np.percentile(boot_slopes, [2.5, 97.5]) if len(boot_slopes) else (np.nan, np.nan)
    monotone_up = all(rois[i] <= rois[i+1] for i in range(len(rois)-1))
    n_pos = sum(1 for r in rois if r > 0)
    return dict(slope=round(slope, 2),
                slope_ci_lo=round(float(slo), 2), slope_ci_hi=round(float(shi), 2),
                slope_ci_above_zero=bool(slo > 0),
                monotone_up=monotone_up, perm_p=round(perm_p, 4),
                n_folds=len(rois), n_pos_folds=n_pos)


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
    cache = Path(f"results/parlay_predictions_lambda120_8fold_4yr.parquet")
    if cache.exists():
        print(f"Loading cached predictions from {cache}")
        matched = pd.read_parquet(cache)
    else:
        matched = collect()
        matched.to_parquet(cache)
        print(f"Saved predictions to {cache}")

    print(f"\nTotal +EV picks: {(matched['ev']>0).sum()}")

    strategies = [
        ("STRAIGHT  | all +EV",                    "straight", dict()),
        ("STRAIGHT  | edge≥5pp",                   "straight", dict(edge_min=0.05)),
        ("STRAIGHT  | p≥0.65",                     "straight", dict(p_min=0.65)),
        ("PARLAY-2  | all +EV pairs",              "parlay",   dict(n_legs=2, top_k=99)),
        ("PARLAY-2  | top-2 by edge",              "parlay",   dict(n_legs=2, top_k=2, picker="top_k_edge")),
        ("PARLAY-2  | top-2 by prob",              "parlay",   dict(n_legs=2, top_k=2, picker="top_k_prob")),
        ("PARLAY-2  | edge≥5pp, all pairs",        "parlay",   dict(n_legs=2, top_k=99, edge_min=0.05)),
        ("PARLAY-2  | edge≥5pp, top-2 by edge",    "parlay",   dict(n_legs=2, top_k=2, edge_min=0.05, picker="top_k_edge")),
        ("PARLAY-2  | p≥0.65, top-2 by prob",      "parlay",   dict(n_legs=2, top_k=2, p_min=0.65, picker="top_k_prob")),
        ("PARLAY-3  | top-3 by edge",              "parlay",   dict(n_legs=3, top_k=3, picker="top_k_edge")),
        ("PARLAY-3  | top-3 edge≥5pp",             "parlay",   dict(n_legs=3, top_k=3, edge_min=0.05, picker="top_k_edge")),
    ]
    results = []
    for label, kind, kw in strategies:
        if kind == "straight": parlays = make_straights(matched, **kw)
        else:                  parlays = make_parlays(matched, **kw)
        r = evaluate(label, parlays)
        if r is not None: results.append(r)

    print()
    print("=" * 165)
    hdr = f"{'strategy':<42s}  {'n':>4s}  {'pooled':>8s}    "
    hdr += "  ".join(f"{f.replace('fold_','F'):>6s}" for f in FOLD_ORDER)
    hdr += f"    {'slope':>7s}  {'CI_slope':>15s}  {'>0?':>3s}  {'mono↑':>5s}  {'perm_p':>7s}  {'pos_f':>5s}"
    print(hdr)
    print("=" * 165)
    for r in sorted(results, key=lambda r: -(r["trend"]["slope"] if r["trend"] else -999)):
        fr = r["fold_roi"]; t = r["trend"] or {}
        f_strs = []
        for f in FOLD_ORDER:
            if fr[f] is None: f_strs.append("  nan ")
            else:             f_strs.append(f"{fr[f]:>+5.0f}%")
        mono = "Y" if t.get("monotone_up") else " "
        ci_above = "Y" if t.get("slope_ci_above_zero") else " "
        ci_str = f"[{t.get('slope_ci_lo',0):>+5.1f},{t.get('slope_ci_hi',0):>+5.1f}]"
        n_pos = t.get("n_pos_folds", 0); n_folds = t.get("n_folds", 0)
        print(f"  {r['label']:<42s}  {r['n_total']:>4d}  {r['pooled_roi']:>+7.2f}%    "
              f"{'  '.join(f_strs)}    {t.get('slope',0):>+6.2f}  {ci_str:>15s}  "
              f"{ci_above:>3s}  {mono:>5s}  {t.get('perm_p',0):>7.4f}  {n_pos}/{n_folds}")

    out = Path("results/parlay_lambda120_8fold_4yr_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
