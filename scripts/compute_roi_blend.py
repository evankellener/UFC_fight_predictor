"""ROI analysis on LR×0.8 + XGB×0.2 blend vs pure LR — does better calibration pay off?

Same 8 strategies, same filtered test set, same odds handling as
`compute_roi.py`. Only difference: the probability source.

Rationale: the nonlinear ablation (scripts/run_nonlinear_ablation.py) showed
LR×0.8 + XGB×0.2 improves log loss by −0.0017 and Brier by −0.0005 vs LR alone,
with accuracy change within the ±4pp noise floor at n=434. This tests whether
those calibration gains translate to real betting edge.

Feature stack: the final 196-feature production stack (MMA-AI + Elo + Tier 1c
recency + Tier 2b style Elos).

Leakage guardrails (LEAKAGE_REFERENCE.md):
  §1  Temporal split preserved. §3 Filter frozen (threshold=3, strict methods).
  §4  LR scaler/imputer fit on train-only; XGB fit on train-only imputed raw.
  §6  LR params (C=0.05, l1=0.5) + XGB params (d=3, 400 trees, lr=0.05) are
      FROZEN from prior experiments. NOT tuned on this 339-row ROI subset.
  §7  Vegas odds are evaluation-only — never a model feature. Both LR and XGB
      are trained on the same 195 base features, neither sees odds.
  §8a Same odds pre-processing: reject |American|<100, clip prob to [0.02, 0.98].
  §10 Single run, single report per model.
"""
import json, sqlite3, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from scipy import stats
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from elo_feature import compute_elo

DT = Path("data/tmp")
DB = "data/sqlite_db/slim_scrapper.db"
SCRAPER_DB = "data/sqlite_db/sqlite_scrapper.db"
TEST_START = pd.Timestamp("2024-05-04")
TEST_END   = pd.Timestamp("2025-11-08")
TRAIN_START = pd.Timestamp("2016-01-01")
FILTER_THRESHOLD = 3

ELO_COLS = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
            "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
ELO_PARAMS = dict(K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
                  decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
                  logistic_scale=449.205,
                  opp_quality_k=True, sliding_k=True, upset_momentum=True,
                  champ_mult=1.2)
TIER_1C = ["win_streak_entering_diff", "coming_off_loss_diff", "fights_last_12m_diff"]
STYLE_COLS = ["striking_elo_diff", "grappling_elo_diff"]
RNG = np.random.default_rng(42)
BLEND_W_LR = 0.8  # 80% LR + 20% XGB


# ── odds helpers (identical to compute_roi.py) ─────────────────────────────
def american_to_prob(o):
    if pd.isna(o) or abs(o) < 100: return np.nan
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def american_to_decimal(o):
    if pd.isna(o) or abs(o) < 100: return np.nan
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


def apply_filter(df):
    conn = sqlite3.connect(SCRAPER_DB)
    hist = pd.read_sql("SELECT w.jfighter, e.DATE FROM ufc_winlossko w "
                       "JOIN ufc_event_details e ON e.jevent=w.jevent", conn)
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fd = {f: grp["DATE"].values for f, grp in
          hist.sort_values(["jfighter", "DATE"]).groupby("jfighter")}

    def prior(j, d):
        dates = fd.get(j, np.array([], dtype="datetime64[ns]"))
        return int((dates < np.datetime64(d)).sum()) if len(dates) else 0

    df = df.copy()
    df["f1_priors"] = df.apply(lambda r: prior(r["jfighter"], r["DATE"]), axis=1)
    df["f2_priors"] = df.apply(lambda r: prior(r["opp_jfighter"], r["DATE"]), axis=1)
    df = df[(df["f1_priors"] >= FILTER_THRESHOLD) & (df["f2_priors"] >= FILTER_THRESHOLD)]
    res = pd.read_sql("SELECT jevent, jbout, METHOD FROM ufc_fight_results", conn)
    res["METHOD_norm"] = res["METHOD"].str.lower().fillna("")
    conn.close()
    df = df.merge(res[["jevent", "jbout", "METHOD_norm"]], on=["jevent", "jbout"], how="left")
    unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
    m = df["METHOD_norm"].apply(lambda x: any(u in str(x) for u in unwanted)
                                 if pd.notna(x) else False)
    df = df[~m]; df = df[df["DATE"] >= TRAIN_START]
    return df.reset_index(drop=True)


def merge_all_layers(df):
    bouts = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    ed, *_ = compute_elo(bouts, **ELO_PARAMS)
    keep = ["jbout", "DATE", "f1", "f2"] + ELO_COLS
    ed = ed[keep].copy(); ed["DATE"] = pd.to_datetime(ed["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(ed, on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] == df["f2"]
    for c in ELO_COLS:
        df.loc[flip, c] = (1 - df.loc[flip, c]) if c == "elo_win_prob" else -df.loc[flip, c]
    df.drop(columns=["f1", "f2"], inplace=True, errors="ignore")
    for c in ELO_COLS:
        df[c] = df[c].fillna(0.5 if c == "elo_win_prob" else 0.0)
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    df = df.merge(rs[["DATE", "jbout", "jfighter"] + TIER_1C],
                  on=["DATE", "jbout", "jfighter"], how="left")
    for c in TIER_1C: df[c] = df[c].fillna(0)
    se = pd.read_csv(DT / "style_elo_features.csv", parse_dates=["DATE"])
    df = df.merge(se, on=["DATE", "jbout", "jfighter"], how="left")
    for c in STYLE_COLS: df[c] = df[c].fillna(0.0)
    return df


def attach_vegas(test):
    conn = sqlite3.connect(DB)
    odds = pd.read_sql("SELECT * FROM ufc_fight_odds", conn, parse_dates=["DATE"])
    conn.close()
    bad = ((odds["avg_odds_f1"].abs() < 100) | (odds["avg_odds_f2"].abs() < 100)
           | odds["avg_odds_f1"].isna() | odds["avg_odds_f2"].isna())
    if bad.any():
        print(f"  Dropping {bad.sum()} odds rows with invalid American odds")
        odds = odds[~bad]
    odds["p_raw_f1"] = odds["avg_odds_f1"].apply(american_to_prob)
    odds["p_raw_f2"] = odds["avg_odds_f2"].apply(american_to_prob)
    odds["dec_f1"] = odds["avg_odds_f1"].apply(american_to_decimal)
    odds["dec_f2"] = odds["avg_odds_f2"].apply(american_to_decimal)
    vig = odds["p_raw_f1"] + odds["p_raw_f2"]
    odds["p_f1_devig"] = odds["p_raw_f1"] / vig
    odds["p_f2_devig"] = odds["p_raw_f2"] / vig

    m = test.merge(odds[["jbout", "jfighter", "p_f1_devig", "p_f2_devig",
                         "dec_f1", "dec_f2"]],
                   on=["jbout"], how="left", suffixes=("", "_odds"))
    flip = m["jfighter"] != m["jfighter_odds"]
    m["p_vegas_f1"] = np.where(flip, m["p_f2_devig"], m["p_f1_devig"])
    m["dec_odds_f1"] = np.where(flip, m["dec_f2"], m["dec_f1"])
    m["dec_odds_f2"] = np.where(flip, m["dec_f1"], m["dec_f2"])
    return m.drop(columns=["jfighter_odds", "p_f1_devig", "p_f2_devig", "dec_f1", "dec_f2"])


# ── ROI machinery (identical to compute_roi.py) ────────────────────────────
def strategy_profit(df, mask, pick_f1):
    sel = df[mask].copy(); pf1 = pick_f1[mask]
    y_f1 = sel["win"].astype(int).values
    dec = np.where(pf1, sel["dec_odds_f1"].values, sel["dec_odds_f2"].values)
    correct = np.where(pf1, y_f1, 1 - y_f1)
    return np.where(correct == 1, dec - 1, -1.0)


def bootstrap_ci(profits, reps=1000):
    if len(profits) == 0: return (np.nan, np.nan, np.nan)
    rois = [RNG.choice(profits, len(profits), replace=True).mean() for _ in range(reps)]
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(np.mean(profits)), float(lo), float(hi)


def ttest_p(profits):
    if len(profits) < 2: return np.nan
    t, p_two = stats.ttest_1samp(profits, 0.0)
    return float(p_two / 2 if t > 0 else 1 - p_two / 2)


def run_roi(sub, p_model, label):
    """Run 8 strategies against p_model; return results dict + printed table rows."""
    p_m = p_model
    p_v = sub["p_vegas_f1"].values
    y_f1 = sub["win"].astype(int).values
    model_pick_f1 = p_m >= 0.5
    vegas_pick_f1 = p_v >= 0.5
    edge_f1 = p_m - p_v; edge_f2 = (1 - p_m) - (1 - p_v)
    edge_on_pick = np.where(model_pick_f1, edge_f1, edge_f2)
    p_fav = np.maximum(p_v, 1 - p_v)
    agree = (model_pick_f1 == vegas_pick_f1)

    strategies = {
        "A. All picks":                       np.ones(len(sub), dtype=bool),
        "B. AGREE":                           agree,
        "C. DISAGREE":                        ~agree,
        "D. +EV (edge > 0)":                  edge_on_pick > 0,
        "E. Edge >= 5pp":                     edge_on_pick >= 0.05,
        "F. Edge >= 10pp":                    edge_on_pick >= 0.10,
        "G. AGREE & p_fav >= 0.65":           agree & (p_fav >= 0.65),
        "H. DISAGREE & edge >= 5pp":          (~agree) & (edge_on_pick >= 0.05),
    }

    out_rows = []
    for name, mask in strategies.items():
        mask = np.asarray(mask)
        if mask.sum() == 0:
            out_rows.append(dict(strategy=name, n=0)); continue
        profits = strategy_profit(sub, mask, model_pick_f1)
        wr = float((profits > 0).mean())
        roi, lo, hi = bootstrap_ci(profits)
        pval = ttest_p(profits)
        out_rows.append(dict(strategy=name, n=int(mask.sum()),
                             win_rate=wr, roi=float(roi),
                             ci_lo=lo, ci_hi=hi, p=pval))
    return out_rows


def main():
    print("="*70)
    print("STEP 1: Load, filter, merge all feature layers")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    df = merge_all_layers(df)
    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,}   Test: {len(test):,}")

    feats = [c for c in df.columns if (c.endswith("_diff") or c in
             ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    print(f"  Feature count: {len(usable)}")

    # ── Attach Vegas to test ────────────────────────────────────────────
    test_v = attach_vegas(test)
    sub = test_v[test_v["p_vegas_f1"].notna()].copy().reset_index(drop=True)
    print(f"  Test rows with Vegas odds: {len(sub)}")

    # ── Train LR and XGB on train, predict on sub ───────────────────────
    imp = SimpleImputer(strategy="median")
    X_tr_raw = imp.fit_transform(train[usable]); X_sub_raw = imp.transform(sub[usable])
    sc = StandardScaler()
    X_tr_lr = sc.fit_transform(X_tr_raw); X_sub_lr = sc.transform(X_sub_raw)
    y_tr = train["win"].astype(int).values
    w_tr = np.exp(-0.13 * (TEST_START - train["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr_lr, y_tr, sample_weight=w_tr)
    p_lr = lr.predict_proba(X_sub_lr)[:, 1]

    xgb = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                        min_child_weight=10, tree_method="hist",
                        eval_metric="logloss", random_state=42)
    xgb.fit(X_tr_raw, y_tr, sample_weight=w_tr)
    p_xgb = xgb.predict_proba(X_sub_raw)[:, 1]

    p_blend = BLEND_W_LR * p_lr + (1 - BLEND_W_LR) * p_xgb

    # Sanity: accuracy on the Vegas-matched subset
    y_sub = sub["win"].astype(int).values
    print(f"\n  Model accuracy on matched subset:")
    for name, p in [("LR alone", p_lr), ("XGB alone", p_xgb),
                    (f"Blend (LR×{BLEND_W_LR}+XGB×{1-BLEND_W_LR:.1f})", p_blend)]:
        from sklearn.metrics import accuracy_score, log_loss
        pc = np.clip(p, 0.02, 0.98)
        print(f"    {name:<40s}  acc={accuracy_score(y_sub, (pc>=0.5).astype(int))*100:5.2f}%  "
              f"ll={log_loss(y_sub, pc):.4f}")

    # ── Run ROI on LR alone vs blend ──────────────────────────────────
    print("\n" + "="*90)
    print(f"ROI: pure LR vs LR×{BLEND_W_LR}+XGB×{1-BLEND_W_LR:.1f} blend")
    print("  Same 8 strategies, same fights. Which is better for betting?")
    print("="*90)

    results_lr = run_roi(sub, p_lr, "LR")
    results_bl = run_roi(sub, p_blend, "Blend")

    print(f"\n{'Strategy':<28s}  {'':>4s} |  "
          f"{'LR n':>4s}  {'LR win%':>7s}  {'LR ROI':>7s}  {'LR p':>5s}  |  "
          f"{'Blend n':>7s}  {'Blend win%':>10s}  {'Blend ROI':>9s}  {'Blend p':>7s}  | "
          f"{'ΔROI':>7s}")
    print("-" * 140)
    for r_lr, r_bl in zip(results_lr, results_bl):
        if r_lr.get("n", 0) == 0 and r_bl.get("n", 0) == 0:
            continue
        dri = (r_bl["roi"] - r_lr["roi"]) * 100 if (r_lr.get("n") and r_bl.get("n")) else np.nan
        print(f"{r_lr['strategy']:<28s}  {'':>4s} |  "
              f"{r_lr['n']:>4d}  {r_lr['win_rate']*100:>6.1f}%  {r_lr['roi']*100:>+6.2f}%  {r_lr['p']:>5.3f}  |  "
              f"{r_bl['n']:>7d}  {r_bl['win_rate']*100:>9.1f}%  {r_bl['roi']*100:>+8.2f}%  {r_bl['p']:>7.3f}  | "
              f"{dri:>+6.2f}pp")

    print("\nKey comparison (Strategy D, +EV):")
    dlr = [r for r in results_lr if r["strategy"].startswith("D")][0]
    dbl = [r for r in results_bl if r["strategy"].startswith("D")][0]
    print(f"  Pure LR:   n={dlr['n']}  ROI={dlr['roi']*100:+.2f}%  "
          f"CI=[{dlr['ci_lo']*100:+.2f}%, {dlr['ci_hi']*100:+.2f}%]  p={dlr['p']:.3f}")
    print(f"  Blend:     n={dbl['n']}  ROI={dbl['roi']*100:+.2f}%  "
          f"CI=[{dbl['ci_lo']*100:+.2f}%, {dbl['ci_hi']*100:+.2f}%]  p={dbl['p']:.3f}")

    (DT / "roi_blend_vs_lr.json").write_text(json.dumps(
        {"lr": results_lr, "blend": results_bl, "blend_w_lr": BLEND_W_LR}, indent=2
    ))
    print(f"\nSaved to {DT/'roi_blend_vs_lr.json'}")


if __name__ == "__main__":
    main()
