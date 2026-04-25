"""λ-sweep on 4-yr training, evaluated on PARLAY-2 edge≥5pp top-2 by edge.

Test λ ∈ {0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0} with 4-yr training window.
For each λ, run 8 folds × 3-mo tests, build the winning parlay strategy,
and report:
  - Pooled ROI
  - Per-fold ROI (8 numbers)
  - Slope across folds + perm_p
  - F8 (most recent) ROI specifically
  - Number of folds positive

Find the λ that gives the best risk-adjusted forward-looking performance.
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

EPS = 1e-6
TRAIN_YEARS = 4
LAMBDAS = [0.50, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00]


def build_folds():
    starts = pd.date_range("2024-04-01", "2026-01-01", freq="3MS")
    out = []
    for i, s in enumerate(starts, 1):
        ts = s; te = s + pd.DateOffset(months=3)
        out.append({"name": f"fold_{i}",
                    "train_start": (ts - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y-%m-%d"),
                    "train_end":   ts.strftime("%Y-%m-%d"),
                    "test_start":  ts.strftime("%Y-%m-%d"),
                    "test_end":    te.strftime("%Y-%m-%d")})
    return out


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


def collect_for_lambda(df, feats, lam):
    parts = []
    for fold in FOLDS:
        train_start = pd.Timestamp(fold["train_start"]); train_end = pd.Timestamp(fold["train_end"])
        test_start  = pd.Timestamp(fold["test_start"]);  test_end  = pd.Timestamp(fold["test_end"])
        train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
        test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
        if len(test) == 0: continue
        leakage_assertions(train, test, fold)
        p = fit_predict(train, test, feats, lam)
        test = test.copy(); test["p_model"] = p; test["fold"] = fold["name"]
        parts.append(test)
    pred = pd.concat(parts, ignore_index=True)
    return pred


def make_parlay_strategy(matched, n_legs=2, edge_min=0.05):
    """PARLAY-2 edge≥5pp top-2 by edge per card — the candidate winner."""
    pos = matched[(matched["ev"] > 0) & (matched["edge"] >= edge_min)].copy()
    parlays = []
    for date, grp in pos.groupby("DATE"):
        g = grp.sort_values("edge", ascending=False).head(n_legs)
        if len(g) < n_legs: continue
        for combo in combinations(g.itertuples(index=False), n_legs):
            combined_odds = float(np.prod([c.dec_odds_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            parlays.append(dict(combined_odds=combined_odds, won=won, fold=str(combo[0].fold)))
    return parlays


def metrics_for_lambda(parlays):
    if not parlays: return None
    pnl = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in parlays])
    pooled = float(pnl.mean() * 100)
    fold_rois = []
    fold_n = []
    for f in FOLD_ORDER:
        ps = [p for p in parlays if p["fold"] == f]
        if not ps:
            fold_rois.append(None); fold_n.append(0); continue
        fp = np.array([(p["combined_odds"] - 1.0) if p["won"] else -1.0 for p in ps])
        fold_rois.append(float(fp.mean() * 100)); fold_n.append(len(ps))
    valid = [r for r in fold_rois if r is not None]
    n_pos = sum(1 for r in valid if r > 0)
    x = np.arange(len(valid), dtype=float); y = np.array(valid, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    perms = list(permutations(range(len(x))))
    obs = abs(slope)
    perm_p = sum(1 for perm in perms
                 if abs(np.polyfit(x, y[list(perm)], 1)[0]) >= obs) / len(perms)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        idx = rng.choice(len(parlays), len(parlays), replace=True)
        boot.append(np.array([(parlays[i]["combined_odds"]-1.0) if parlays[i]["won"] else -1.0
                              for i in idx]).mean() * 100)
    boot = np.array(boot)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=len(parlays), pooled=round(pooled, 2),
                fold_rois=[round(r, 1) if r is not None else None for r in fold_rois],
                fold_n=fold_n, n_pos_folds=n_pos, n_folds=len(valid),
                slope=round(float(slope), 2), perm_p=round(perm_p, 4),
                roi_ci_lo=round(float(ci_lo), 2), roi_ci_hi=round(float(ci_hi), 2),
                f8_roi=fold_rois[-1] if fold_rois else None)


def main():
    print(f"λ-sweep on 4-yr training, evaluating PARLAY-2 edge≥5pp top-2 by edge")
    print(f"λ ∈ {LAMBDAS}")
    print()
    print("Loading base...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)

    rows = []
    for lam in LAMBDAS:
        print(f"\nFitting λ={lam}...")
        pred = collect_for_lambda(df, feats, lam)
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
        matched = matched.drop_duplicates(subset=["DATE", "jbout"]).reset_index(drop=True)
        parlays = make_parlay_strategy(matched)
        m = metrics_for_lambda(parlays)
        if m is None: continue
        m["lambda"] = lam
        rows.append(m)
        print(f"  λ={lam}  n={m['n']}  pooled={m['pooled']:+.2f}%  pos={m['n_pos_folds']}/{m['n_folds']}  "
              f"F8={m['f8_roi']}  slope={m['slope']}  perm_p={m['perm_p']}")

    print()
    print("=" * 145)
    print(f"{'λ':>5s}  {'n':>4s}  {'pooled':>8s}  {'CI':>17s}  "
          + "  ".join(f"F{i}".rjust(6) for i in range(1, 9))
          + f"  {'slope':>6s}  {'perm_p':>7s}  {'pos':>5s}")
    print("=" * 145)
    for r in rows:
        ci = f"[{r['roi_ci_lo']:>+5.1f},{r['roi_ci_hi']:>+5.1f}]"
        f_strs = []
        for v in r["fold_rois"]:
            if v is None: f_strs.append("  nan ")
            else:         f_strs.append(f"{v:>+5.0f}%")
        print(f"  {r['lambda']:>3.2f}  {r['n']:>4d}  {r['pooled']:>+7.2f}%  {ci:>17s}  "
              f"{'  '.join(f_strs)}  {r['slope']:>+5.2f}  {r['perm_p']:>7.4f}  "
              f"{r['n_pos_folds']}/{r['n_folds']}")

    out = Path("results/parlay_lambda_sweep_4yr.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
