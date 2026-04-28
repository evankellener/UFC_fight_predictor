"""Kelly sizing backtest on the shipped parlay strategy.

Strategy: PARLAY-2 edge≥5pp top-2 by edge per card, MEN-ONLY, λ=1.50, 4-yr training.

Generates predictions per fold, builds parlays in chronological order, then
simulates a $1,000 starting bankroll under multiple sizing rules:
  - Flat 1% / 2% / 5%
  - Fractional Kelly capped (¼K, ½K, 1K) at 5% / 10%

For each rule reports:
  - Final bankroll mean + percentiles (across 500 random orderings of the
    historical parlays — handles path-dependence)
  - Max drawdown (worst peak-to-trough %)
  - Sharpe (mean return / std dev)
  - Bust rate: % of orderings where bankroll ever drops below $100
"""
import sys, json, warnings, sqlite3
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
LAM = 1.50; TRAIN_YEARS = 4

def build_folds():
    starts = pd.date_range("2024-04-01", "2026-01-01", freq="3MS")
    return [{"name": f"fold_{i}",
             "train_start": (s - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y-%m-%d"),
             "train_end":   s.strftime("%Y-%m-%d"),
             "test_start":  s.strftime("%Y-%m-%d"),
             "test_end":   (s + pd.DateOffset(months=3)).strftime("%Y-%m-%d")}
            for i, s in enumerate(starts, 1)]
FOLDS = build_folds()


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


def collect_parlays():
    print("Generating λ=1.50 / 4-yr predictions across 8 folds...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)
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
    # Attach sex
    conn = sqlite3.connect("data/sqlite_db/sqlite_scrapper.db")
    res = pd.read_sql("SELECT jevent, jbout, sex FROM ufc_fight_results", conn)
    conn.close()
    matched = matched.merge(res, on=["jevent","jbout"], how="left")
    print(f"  {len(matched)} bouts after vegas+sex match")

    # Apply strategy: men-only, +EV, edge≥5pp, top-2 by edge per card
    pos = matched[(matched["sex"]==2) & (matched["ev"]>0) & (matched["edge"]>=0.05)].copy()
    parlays = []
    for date, grp in pos.groupby("DATE"):
        g = grp.sort_values("edge", ascending=False).head(2)
        if len(g) < 2: continue
        for combo in combinations(g.itertuples(index=False), 2):
            co  = float(np.prod([c.dec_odds_pick for c in combo]))
            cp  = float(np.prod([c.p_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            parlays.append(dict(date=str(combo[0].DATE), combined_odds=co,
                                combined_p=cp, won=won))
    parlays.sort(key=lambda x: x["date"])
    print(f"  {len(parlays)} parlays in chronological order")
    return parlays


def kelly_frac(p, b):
    return max(0.0, (p*(b+1) - 1) / b) if b > 0 else 0.0


def simulate(parlays, sizing_fn, start=1000.0, ruin_floor=100.0):
    """Sequentially apply sizing rule. Returns final bankroll, max drawdown,
    bust flag, return list."""
    bk = start; peak = start; max_dd = 0.0; busted = False; returns = []
    for p in parlays:
        b = p["combined_odds"] - 1
        stake = sizing_fn(bk, p["combined_p"], b)
        if stake <= 0:
            returns.append(0.0); continue
        if p["won"]:
            bk += stake * b
        else:
            bk -= stake
        if bk < ruin_floor: busted = True
        peak = max(peak, bk)
        if peak > 0:
            dd = (peak - bk) / peak
            if dd > max_dd: max_dd = dd
        returns.append((stake * b if p["won"] else -stake) / max(start, 1))
    return dict(final=bk, max_dd=max_dd, busted=busted,
                ret_mean=float(np.mean(returns)) if returns else 0.0,
                ret_std=float(np.std(returns)) if returns else 0.0)


def sizing_rules():
    return {
        "Flat 1%":           lambda bk,p,b: bk * 0.01,
        "Flat 2%":           lambda bk,p,b: bk * 0.02,
        "Flat 5%":           lambda bk,p,b: bk * 0.05,
        "¼ Kelly cap 5%":    lambda bk,p,b: bk * min(kelly_frac(p,b)*0.25, 0.05),
        "½ Kelly cap 5%":    lambda bk,p,b: bk * min(kelly_frac(p,b)*0.50, 0.05),
        "½ Kelly cap 10%":   lambda bk,p,b: bk * min(kelly_frac(p,b)*0.50, 0.10),
        "Full Kelly cap 10%":lambda bk,p,b: bk * min(kelly_frac(p,b),       0.10),
        "Full Kelly cap 25%":lambda bk,p,b: bk * min(kelly_frac(p,b),       0.25),
    }


def main():
    cache = Path("results/parlay_lambda150_men_parlays.json")
    if cache.exists():
        parlays = json.loads(cache.read_text())
        print(f"Loaded {len(parlays)} cached parlays from {cache}")
    else:
        parlays = collect_parlays()
        cache.write_text(json.dumps(parlays, indent=2))
        print(f"Cached parlays to {cache}")

    print()
    print(f"=" * 110)
    print(f"Kelly sizing simulation on PARLAY-2 edge≥5pp top-2 by edge, MEN-ONLY, λ=1.50, 4-yr")
    print(f"  Start bankroll: $1,000  ·  Bust floor: $100  ·  n_parlays: {len(parlays)}")
    print(f"  Bootstrap: 500 random orderings of the historical parlays")
    print(f"=" * 110)
    rng = np.random.default_rng(42)
    rows = []
    for label, fn in sizing_rules().items():
        finals = []; max_dds = []; busts = 0
        # Chronological order (the "true" path)
        chron = simulate(parlays, fn)
        # Bootstrap 500 random orderings (path dependence)
        for _ in range(500):
            shuffled = list(parlays)
            rng.shuffle(shuffled)
            r = simulate(shuffled, fn)
            finals.append(r["final"]); max_dds.append(r["max_dd"])
            if r["busted"]: busts += 1
        finals = np.array(finals); max_dds = np.array(max_dds)
        rows.append(dict(label=label,
                         chron_final=round(chron["final"], 0),
                         chron_max_dd=round(chron["max_dd"]*100, 1),
                         med=round(float(np.median(finals)), 0),
                         p10=round(float(np.percentile(finals, 10)), 0),
                         p90=round(float(np.percentile(finals, 90)), 0),
                         worst_dd=round(float(np.max(max_dds)*100), 1),
                         med_dd=round(float(np.median(max_dds)*100), 1),
                         bust_rate=round(busts / 500 * 100, 1)))

    # Pretty print
    print()
    print(f"{'sizing':<22s}  {'chron $':>9s}  {'med $':>9s}  {'p10 $':>9s}  {'p90 $':>9s}  "
          f"{'med DD':>7s}  {'worst DD':>9s}  {'bust%':>6s}")
    print("-" * 110)
    for r in rows:
        print(f"  {r['label']:<22s}  {r['chron_final']:>9,.0f}  {r['med']:>9,.0f}  "
              f"{r['p10']:>9,.0f}  {r['p90']:>9,.0f}  {r['med_dd']:>5.1f}%  "
              f"{r['worst_dd']:>7.1f}%  {r['bust_rate']:>5.1f}%")

    print()
    print("KEY:")
    print("  chron $ = final bankroll on the actual chronological order of parlays")
    print("  med/p10/p90 $ = median / 10th-pctile / 90th-pctile final bankroll")
    print("                  across 500 bootstrap reorderings")
    print("  med DD  = median worst peak-to-trough drawdown")
    print("  worst DD = max worst drawdown across all 500 paths")
    print("  bust%   = % of paths where bankroll ever fell below $100 (ruin)")
    print()
    print("Pick the rule that maximizes p10 final bankroll (worst-decile outcome)")
    print("WHILE keeping bust% low (<5%). That's risk-adjusted optimal.")

    Path("results/kelly_parlay_backtest.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
