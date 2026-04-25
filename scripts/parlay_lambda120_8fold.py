"""λ=1.20 parlay analysis on 8 folds × 3-month test windows.

Folds (each train uses 7-year window ending at test_start):
  fold_1  test 2024-04 → 2024-07
  fold_2  test 2024-07 → 2024-10
  fold_3  test 2024-10 → 2025-01
  fold_4  test 2025-01 → 2025-04
  fold_5  test 2025-04 → 2025-07
  fold_6  test 2025-07 → 2025-10
  fold_7  test 2025-10 → 2026-01
  fold_8  test 2026-01 → 2026-04 (includes today)

Same leakage guards: per-fold imputer/scaler/calibrator refit, symmetric
doubled training, recency-weighted (λ=1.20), Vegas attached after predictions.

Trend analysis:
  - Linear slope across 8 folds
  - Permutation p-value (40,320 reorderings)
  - Bootstrap 95% CI on slope (1,000 resamples of fold-RFI pairs)
  - Per-fold breakdown
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
from walk_forward_4fold import select_features, leakage_assertions
from build_walkforward_vegas_multi_threshold import attach_vegas_rich

LAM = 1.20
EPS = 1e-6
TRAIN_YEARS = 7

# Build 8 folds × 3-month test windows
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
    print("Loading + 8-fold predictions (λ=1.20 only)...")
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
        if len(test) == 0:
            print(f"  {fold['name']}: NO test fights — skipping"); continue
        leakage_assertions(train, test, fold)
        p = fit_and_predict(train, test, feats, LAM)
        test = test.copy(); test["p_model"] = p; test["fold"] = fold["name"]
        all_test.append(test)
        print(f"  {fold['name']}  test {fold['test_start']}→{fold['test_end']}  n={len(test)}")
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
    ns   = [n for n, r in roi_per_fold.values() if r is not None]
    if len(rois) < 4: return None
    x = np.arange(len(rois), dtype=float)
    y = np.array(rois, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    # Permutation test: shuffle fold-ROI labels keeping x fixed
    obs = abs(slope)
    if len(rois) <= 8:
        perms = list(permutations(range(len(x))))
        bigger = sum(1 for perm in perms
                     if abs(np.polyfit(x, y[list(perm)], 1)[0]) >= obs)
        perm_p = bigger / len(perms)
    else:
        rng = np.random.default_rng(42)
        bigger = sum(1 for _ in range(20000)
                     if abs(np.polyfit(x, rng.permutation(y), 1)[0]) >= obs)
        perm_p = bigger / 20000

    # Bootstrap CI on slope (resample fold pairs with replacement)
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
    n_pos_folds = sum(1 for r in rois if r > 0)
    return dict(slope=round(slope, 2), intercept=round(intercept, 2),
                slope_ci_lo=round(float(slo), 2), slope_ci_hi=round(float(shi), 2),
                slope_ci_above_zero=bool(slo > 0),
                monotone_up=monotone_up, perm_p=round(perm_p, 4),
                n_folds=len(rois), n_pos_folds=n_pos_folds)


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
    cache = Path("results/parlay_predictions_lambda120_8fold.parquet")
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
    sorted_by_slope = sorted(results, key=lambda r: -(r["trend"]["slope"] if r["trend"] else -999))
    for r in sorted_by_slope:
        fr = r["fold_roi"]; t = r["trend"] or {}
        f_strs = []
        for f in FOLD_ORDER:
            if fr[f] is None: f_strs.append("  nan ")
            else:             f_strs.append(f"{fr[f]:>+5.0f}%")
        mono = "Y" if t.get("monotone_up") else " "
        ci_above = "Y" if t.get("slope_ci_above_zero") else " "
        ci_str = f"[{t.get('slope_ci_lo',0):>+5.1f},{t.get('slope_ci_hi',0):>+5.1f}]"
        n_pos = t.get("n_pos_folds", 0)
        n_folds = t.get("n_folds", 0)
        print(f"  {r['label']:<42s}  {r['n_total']:>4d}  {r['pooled_roi']:>+7.2f}%    "
              f"{'  '.join(f_strs)}    {t.get('slope',0):>+6.2f}  {ci_str:>15s}  "
              f"{ci_above:>3s}  {mono:>5s}  {t.get('perm_p',0):>7.4f}  {n_pos}/{n_folds}")

    print()
    print("KEY:")
    print("  CI_slope        = bootstrap 95% CI on slope (1,000 resamples)")
    print("  >0?             = Y if entire slope CI is above zero (statistically rising)")
    print("  perm_p          = permutation p-value (40,320 perms for 8 folds)")
    print("  pos_f           = number of folds with positive ROI / total folds")

    # Top candidates
    print()
    print("TOP CANDIDATES — slope > 0, perm_p < 0.20, ≥6 of 8 folds positive:")
    print("=" * 80)
    cands = [r for r in results
             if r["trend"] and r["trend"]["slope"] > 0
             and r["trend"]["perm_p"] < 0.20
             and (r["trend"]["n_pos_folds"] / r["trend"]["n_folds"]) >= 0.75]
    cands.sort(key=lambda r: r["trend"]["perm_p"])
    if not cands:
        print("  (none meet the bar — see strict-trend section below)")
    for r in cands:
        fr = r["fold_roi"]; t = r["trend"]
        print(f"  {r['label']}")
        print(f"    pooled={r['pooled_roi']:+.2f}%  slope={t['slope']:+.2f}pp/fold  "
              f"perm_p={t['perm_p']:.4f}  pos_folds={t['n_pos_folds']}/{t['n_folds']}  "
              f"n={r['n_total']}")

    out = Path("results/parlay_lambda120_8fold_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
