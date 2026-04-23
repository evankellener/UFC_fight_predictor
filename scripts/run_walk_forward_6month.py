"""3-fold walk-forward with ~6-month retrain cadence.

Tests whether a more conservative retraining interval (6 months vs 8-fold's
~2.25 months) gives better ROI. Hypothesis from `run_walk_forward_8fold.py`
result: shorter cadences caused calibration drift on Fold 6 (-72% ROI fold),
tanking aggregate ROI from +14.36% (single-shot, yearly) to -11.90%.

Fold structure (over MMA-AI's 18-month test window):
  Fold 1: 2024-05-04 → 2024-11-04  (~6 months)
  Fold 2: 2024-11-04 → 2025-05-04  (~6 months)
  Fold 3: 2025-05-04 → 2025-11-08  (~6 months)

Per fold: LR retrained on prior 8 years of data (sliding window), same features
as the production 196-feature stack.

Also fixes the `attach_vegas` duplicate-row bug from the 8-fold script:
dedups odds by (jbout, jfighter) BEFORE merge to prevent multi-row explosions.

Leakage guardrails (LEAKAGE_REFERENCE.md):
  §1  Temporal walk-forward, no shuffle
  §3  Features built with d<fight_date
  §4  Imputer/scaler refit per fold on train-only
  §6  LR hyperparams frozen (C=0.05, l1=0.5)
  §7  Vegas odds evaluation-only
  §10 Single run, single report
"""
import json, sqlite3, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, log_loss,
                             brier_score_loss, roc_auc_score)
from scipy import stats as scistats

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from elo_feature import compute_elo

DT = Path("data/tmp")
DB = "data/sqlite_db/slim_scrapper.db"
SCRAPER_DB = "data/sqlite_db/sqlite_scrapper.db"

TEST_FIRST = pd.Timestamp("2024-05-04")
TEST_LAST  = pd.Timestamp("2025-11-08")
N_FOLDS = 3                           # ~6 months per fold
TRAIN_YEARS = 8
TRAIN_ERA_FLOOR = pd.Timestamp("2016-01-01")
FILTER_THRESHOLD = 3
LAM = 0.13

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
    df = df[~m]
    # Deduplicate to prevent downstream merge explosions (§new hygiene rule)
    df = df.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return df


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
    # Dedup again after all merges
    df = df.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return df


def attach_vegas(test):
    """FIXED vs 8-fold version: dedup odds on (DATE, jbout, jfighter) BEFORE merge
    to prevent row explosion. 8-fold script had this bug causing fold 6 to show
    n_total=104 when only 61 test fights existed.
    """
    conn = sqlite3.connect(DB)
    odds = pd.read_sql("SELECT * FROM ufc_fight_odds", conn, parse_dates=["DATE"])
    conn.close()
    bad = ((odds["avg_odds_f1"].abs() < 100) | (odds["avg_odds_f2"].abs() < 100)
           | odds["avg_odds_f1"].isna() | odds["avg_odds_f2"].isna())
    odds = odds[~bad]
    # DEDUP: keep one odds row per (DATE, jbout, jfighter)
    odds = odds.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

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
    m = m.drop(columns=["jfighter_odds", "p_f1_devig", "p_f2_devig", "dec_f1", "dec_f2"])
    # POST-merge dedup as an extra safeguard
    m = m.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return m


def metrics(y, p):
    p_clip = np.clip(p, 0.02, 0.98)
    pred = (p_clip >= 0.5).astype(int)
    return dict(acc=float(accuracy_score(y, pred)),
                ll=float(log_loss(y, p_clip)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p_clip)))


def train_lr(train, usable):
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(train[usable])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    y = train["win"].astype(int).values
    ref = train["DATE"].max()
    w = np.exp(-LAM * (ref - train["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs, y, sample_weight=w)
    return lr, imp, sc


def predict(test, lr, imp, sc, usable):
    X = imp.transform(test[usable])
    Xs = sc.transform(X)
    return lr.predict_proba(Xs)[:, 1]


def main():
    print("="*70)
    print("3-fold walk-forward: ~6 months per fold, LR retrained each fold")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    df = merge_all_layers(df)

    feats = [c for c in df.columns if (c.endswith("_diff") or c in
             ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    print(f"Total fights after filter: {len(df):,}")

    # ── Build 3 folds × ~6 months ─────────────────────────────────
    span = (TEST_LAST - TEST_FIRST).days
    folds = []
    for i in range(N_FOLDS):
        fs = TEST_FIRST + pd.Timedelta(days=int(round(i * span / N_FOLDS)))
        fe = TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span / N_FOLDS))) \
             if i < N_FOLDS-1 else TEST_LAST
        folds.append((fs, fe))
    print(f"\nFolds:")
    for i, (fs, fe) in enumerate(folds, 1):
        print(f"  Fold {i}: {fs.date()} → {fe.date()}  ({(fe-fs).days} days)")

    # ── Attach Vegas to all test ───────────────────────────────────
    all_test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()
    test_v = attach_vegas(all_test)
    print(f"\nTest fights in window: {len(all_test)}  "
          f"matched w/ Vegas: {test_v['p_vegas_f1'].notna().sum()}")

    # ── Walk-forward ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("Per-fold results")
    print("="*70)
    per_fold = []
    all_rows = []
    for i, (fs, fe) in enumerate(folds, 1):
        train_start = max(TRAIN_ERA_FLOOR, fs - pd.DateOffset(years=TRAIN_YEARS))
        tr = df[(df["DATE"] >= train_start) & (df["DATE"] < fs)].copy()
        te = df[(df["DATE"] >= fs) & (df["DATE"] < fe)].copy() if i < N_FOLDS \
             else df[(df["DATE"] >= fs) & (df["DATE"] <= fe)].copy()
        if len(te) == 0: continue

        usable = [c for c in feats if c in tr.columns and tr[c].std() > 1e-8]
        lr, imp, sc = train_lr(tr, usable)
        p_te = predict(te, lr, imp, sc, usable)
        y_te = te["win"].astype(int).values
        m = metrics(y_te, p_te)
        per_fold.append(dict(fold=i, train_start=str(train_start.date()),
                             test_start=str(fs.date()), test_end=str(fe.date()),
                             n_train=len(tr), n_test=len(te), **m))
        te_c = te.copy(); te_c["p_model"] = p_te; te_c["fold"] = i
        all_rows.append(te_c)
        print(f"  Fold {i}: train {train_start.date()}→{fs.date()} "
              f"({len(tr):,})  test →{fe.date()} ({len(te):>3})  "
              f"acc={m['acc']*100:.2f}%  ll={m['ll']:.4f}  "
              f"auc={m['auc']:.4f}  brier={m['brier']:.4f}")

    wf = pd.concat(all_rows, ignore_index=True)
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    print(f"\nPooled predictions: {len(wf)}")

    # Pooled metrics
    y = wf["win"].astype(int).values
    p = wf["p_model"].values
    pm = metrics(y, p)
    print(f"POOLED: acc={pm['acc']*100:.2f}%  ll={pm['ll']:.4f}  "
          f"auc={pm['auc']:.4f}  brier={pm['brier']:.4f}")

    # Compare to baselines
    print(f"\nComparison on same test window:")
    print(f"  Single-shot LR (prior best): 70.97% / 0.5914 / 0.7528 / 0.2019  (n=434)")
    print(f"  8-fold walk-forward:         71.89% / 0.5901 / 0.7502 / 0.2015  (n=434)")
    print(f"  3-fold (6mo) walk-forward:   {pm['acc']*100:.2f}% / {pm['ll']:.4f} / "
          f"{pm['auc']:.4f} / {pm['brier']:.4f}  (n={len(y)})")

    # ── ROI ───────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("ROI — Strategy D (+EV where model_p > Vegas_devig)")
    print("="*90)
    tv_keys = test_v[["DATE", "jbout", "jfighter", "p_vegas_f1",
                      "dec_odds_f1", "dec_odds_f2"]]
    wf = wf.merge(tv_keys, on=["DATE", "jbout", "jfighter"], how="left")
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    wf_v = wf[wf["p_vegas_f1"].notna()].copy()
    print(f"Vegas-matched: {len(wf_v)}/{len(wf)}")

    p_m = wf_v["p_model"].values
    p_v = wf_v["p_vegas_f1"].values
    pick_f1 = p_m >= 0.5
    edge_on_pick = np.where(pick_f1, p_m - p_v, (1 - p_m) - (1 - p_v))
    mask_d = edge_on_pick > 0

    print(f"\n{'Fold':>4s}  {'n':>4s}  {'n_ev':>4s}  {'win%':>5s}  {'ROI':>7s}")
    for fi in sorted(wf_v["fold"].unique()):
        fm = (wf_v["fold"] == fi).values & mask_d
        if fm.sum() == 0: continue
        sub = wf_v[fm]
        y_fl = sub["win"].astype(int).values
        pf1 = (sub["p_model"].values >= 0.5)
        dec = np.where(pf1, sub["dec_odds_f1"].values, sub["dec_odds_f2"].values)
        correct = np.where(pf1, y_fl, 1 - y_fl)
        profits = np.where(correct == 1, dec - 1, -1.0)
        n_total = (wf_v["fold"] == fi).sum()
        print(f"  {int(fi):>2d}   {n_total:>4d}  {int(fm.sum()):>4d}  "
              f"{(profits > 0).mean()*100:>4.1f}%  {profits.mean()*100:>+6.2f}%")

    total_bets = wf_v[mask_d].copy()
    y_all = total_bets["win"].astype(int).values
    pf1_all = (total_bets["p_model"].values >= 0.5)
    dec_all = np.where(pf1_all, total_bets["dec_odds_f1"].values,
                       total_bets["dec_odds_f2"].values)
    correct_all = np.where(pf1_all, y_all, 1 - y_all)
    profits_all = np.where(correct_all == 1, dec_all - 1, -1.0)
    rois_boot = [RNG.choice(profits_all, len(profits_all), replace=True).mean()
                 for _ in range(1000)]
    lo, hi = np.percentile(rois_boot, [2.5, 97.5])
    t, p_two = scistats.ttest_1samp(profits_all, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2

    print(f"\nAGGREGATE Strategy D: n={len(total_bets)}  "
          f"win_rate={(profits_all > 0).mean()*100:.1f}%  "
          f"ROI={profits_all.mean()*100:+.2f}%  "
          f"CI=[{lo*100:+.2f}%, {hi*100:+.2f}%]  p={p_one:.3f}")
    print(f"\nComparison:")
    print(f"  Single-shot (yearly retrain):   ROI +14.36%  CI=[+1.56%, +27.85%]  p=0.023  (n=173)")
    print(f"  8-fold (~2.25mo retrain):       ROI -11.90%  CI=[-22.45%, +0.13%]  p=0.981  (n=250 — buggy)")
    print(f"  3-fold (6mo retrain):           ROI {profits_all.mean()*100:+.2f}%  "
          f"CI=[{lo*100:+.2f}%, {hi*100:+.2f}%]  p={p_one:.3f}  (n={len(total_bets)})")

    out = dict(
        config={"n_folds": N_FOLDS, "fold_months": 6,
                "test_first": str(TEST_FIRST.date()),
                "test_last": str(TEST_LAST.date())},
        per_fold=per_fold,
        pooled_metrics=pm,
        roi={"n": len(total_bets),
             "win_rate": float((profits_all > 0).mean()),
             "roi": float(profits_all.mean()),
             "ci_lo": float(lo), "ci_hi": float(hi), "p": float(p_one)},
    )
    (DT / "walk_forward_6month.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {DT/'walk_forward_6month.json'}")


if __name__ == "__main__":
    main()
