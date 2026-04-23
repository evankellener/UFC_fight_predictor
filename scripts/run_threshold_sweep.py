"""Threshold sweep: threshold ∈ {1, 2, 3} for min prior UFC fights per fighter.

For each threshold, re-runs the full pipeline on our MMA-AI test window:
  1. Single-shot LR (train once, test on the whole window)
  2. 3-fold 6-month walk-forward (production cadence)
  3. Vegas head-to-head on matched subset
  4. Strategy D (+EV) ROI with bootstrap CI + t-test

Purpose: understand how much the threshold=3 choice (which matched MMA-AI's 411
test fights) is helping vs relaxing to include 2-prior or 1-prior fighters.

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§11):
  All prior scripts' guardrails inherited; threshold is a study-design constant,
  not a hyperparameter tuned on test.
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
TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_ERA_FLOOR = pd.Timestamp("2016-01-01")
TRAIN_YEARS = 8
N_FOLDS = 3   # 6-month cadence
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


def load_base():
    """Load features + prior counts + method filter — apply threshold later."""
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    conn = sqlite3.connect(SCRAPER_DB)
    hist = pd.read_sql("SELECT w.jfighter, e.DATE FROM ufc_winlossko w "
                       "JOIN ufc_event_details e ON e.jevent=w.jevent", conn)
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fd = {f: grp["DATE"].values for f, grp in
          hist.sort_values(["jfighter", "DATE"]).groupby("jfighter")}

    def prior(j, d):
        dates = fd.get(j, np.array([], dtype="datetime64[ns]"))
        return int((dates < np.datetime64(d)).sum()) if len(dates) else 0

    df["f1_priors"] = df.apply(lambda r: prior(r["jfighter"], r["DATE"]), axis=1)
    df["f2_priors"] = df.apply(lambda r: prior(r["opp_jfighter"], r["DATE"]), axis=1)

    res = pd.read_sql("SELECT jevent, jbout, METHOD FROM ufc_fight_results", conn)
    res["METHOD_norm"] = res["METHOD"].str.lower().fillna("")
    conn.close()
    df = df.merge(res[["jevent", "jbout", "METHOD_norm"]], on=["jevent", "jbout"], how="left")
    unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
    m = df["METHOD_norm"].apply(lambda x: any(u in str(x) for u in unwanted)
                                 if pd.notna(x) else False)
    df = df[~m]
    df = df.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    # Merge Elo
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
    df = df.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return df


def apply_threshold(df, threshold):
    d = df[(df["f1_priors"] >= threshold) & (df["f2_priors"] >= threshold)].copy()
    d = d[d["DATE"] >= TRAIN_START].reset_index(drop=True)
    return d


def attach_vegas(test):
    conn = sqlite3.connect(DB)
    odds = pd.read_sql("SELECT * FROM ufc_fight_odds", conn, parse_dates=["DATE"])
    conn.close()
    bad = ((odds["avg_odds_f1"].abs() < 100) | (odds["avg_odds_f2"].abs() < 100)
           | odds["avg_odds_f1"].isna() | odds["avg_odds_f2"].isna())
    odds = odds[~bad]
    odds = odds.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    odds["p_raw_f1"] = odds["avg_odds_f1"].apply(american_to_prob)
    odds["p_raw_f2"] = odds["avg_odds_f2"].apply(american_to_prob)
    odds["dec_f1"] = odds["avg_odds_f1"].apply(american_to_decimal)
    odds["dec_f2"] = odds["avg_odds_f2"].apply(american_to_decimal)
    vig = odds["p_raw_f1"] + odds["p_raw_f2"]
    odds["p_f1_devig"] = odds["p_raw_f1"] / vig
    odds["p_f2_devig"] = odds["p_raw_f2"] / vig
    m = test.merge(odds[["jbout", "jfighter", "p_f1_devig", "p_f2_devig",
                         "dec_f1", "dec_f2"]], on=["jbout"], how="left",
                   suffixes=("", "_odds"))
    flip = m["jfighter"] != m["jfighter_odds"]
    m["p_vegas_f1"] = np.where(flip, m["p_f2_devig"], m["p_f1_devig"])
    m["dec_odds_f1"] = np.where(flip, m["dec_f2"], m["dec_f1"])
    m["dec_odds_f2"] = np.where(flip, m["dec_f1"], m["dec_f2"])
    m = m.drop(columns=["jfighter_odds", "p_f1_devig", "p_f2_devig", "dec_f1", "dec_f2"])
    m = m.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return m


def metrics(y, p):
    p_clip = np.clip(p, 0.02, 0.98)
    pred = (p_clip >= 0.5).astype(int)
    return dict(acc=float(accuracy_score(y, pred)),
                ll=float(log_loss(y, p_clip)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p_clip)))


def train_lr(train, usable, ref_date=None):
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(train[usable])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    y = train["win"].astype(int).values
    ref = ref_date if ref_date is not None else train["DATE"].max()
    w = np.exp(-LAM * (ref - train["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs, y, sample_weight=w)
    return lr, imp, sc


def predict(test, lr, imp, sc, usable):
    X = imp.transform(test[usable])
    Xs = sc.transform(X)
    return lr.predict_proba(Xs)[:, 1]


def pick_features(df):
    feats = [c for c in df.columns if (c.endswith("_diff") or c in
             ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    return feats


def run_single_shot(df):
    train = df[df["DATE"] < TEST_FIRST].copy()
    test  = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()
    feats = pick_features(df)
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    lr, imp, sc = train_lr(train, usable, ref_date=TEST_FIRST)
    p = predict(test, lr, imp, sc, usable)
    y = test["win"].astype(int).values
    m = metrics(y, p)
    test_out = test.copy(); test_out["p_model"] = p
    return m, test_out, len(train), len(test)


def run_6mo_wf(df):
    span = (TEST_LAST - TEST_FIRST).days
    folds = [(TEST_FIRST + pd.Timedelta(days=int(round(i * span / N_FOLDS))),
              TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span / N_FOLDS))) if i < N_FOLDS-1 else TEST_LAST)
             for i in range(N_FOLDS)]
    feats = pick_features(df)
    all_rows, per_fold = [], []
    for i, (fs, fe) in enumerate(folds, 1):
        train_start = max(TRAIN_ERA_FLOOR, fs - pd.DateOffset(years=TRAIN_YEARS))
        tr = df[(df["DATE"] >= train_start) & (df["DATE"] < fs)].copy()
        te = df[(df["DATE"] >= fs) & (df["DATE"] < fe)].copy() if i < N_FOLDS \
             else df[(df["DATE"] >= fs) & (df["DATE"] <= fe)].copy()
        if len(te) == 0: continue
        usable = [c for c in feats if c in tr.columns and tr[c].std() > 1e-8]
        lr, imp, sc = train_lr(tr, usable)
        p_te = predict(te, lr, imp, sc, usable)
        per_fold.append(dict(fold=i, n_train=len(tr), n_test=len(te),
                              **metrics(te["win"].astype(int).values, p_te)))
        te_c = te.copy(); te_c["p_model"] = p_te; te_c["fold"] = i
        all_rows.append(te_c)
    wf = pd.concat(all_rows, ignore_index=True)
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    pooled = metrics(wf["win"].astype(int).values, wf["p_model"].values)
    return pooled, per_fold, wf


def run_roi(wf):
    test_v = attach_vegas(wf[["DATE", "jbout", "jfighter"]].drop_duplicates())
    wf = wf.merge(test_v[["DATE", "jbout", "jfighter", "p_vegas_f1",
                          "dec_odds_f1", "dec_odds_f2"]],
                  on=["DATE", "jbout", "jfighter"], how="left")
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    wf_v = wf[wf["p_vegas_f1"].notna()].copy()
    n_match = len(wf_v)

    p_m = wf_v["p_model"].values
    p_v = wf_v["p_vegas_f1"].values
    pick_f1 = p_m >= 0.5
    edge_on_pick = np.where(pick_f1, p_m - p_v, (1 - p_m) - (1 - p_v))
    mask_d = edge_on_pick > 0
    sub = wf_v[mask_d].copy()
    y = sub["win"].astype(int).values
    pf1 = (sub["p_model"].values >= 0.5)
    dec = np.where(pf1, sub["dec_odds_f1"].values, sub["dec_odds_f2"].values)
    correct = np.where(pf1, y, 1 - y)
    profits = np.where(correct == 1, dec - 1, -1.0)
    if len(profits) < 2:
        return dict(n=len(profits), n_match=n_match, win_rate=np.nan,
                    roi=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    rois_boot = [RNG.choice(profits, len(profits), replace=True).mean()
                 for _ in range(1000)]
    lo, hi = np.percentile(rois_boot, [2.5, 97.5])
    t, p_two = scistats.ttest_1samp(profits, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return dict(n=len(profits), n_match=n_match,
                win_rate=float((profits > 0).mean()),
                roi=float(profits.mean()),
                ci_lo=float(lo), ci_hi=float(hi), p=float(p_one))


def main():
    print("Loading base features + prior counts...")
    base = load_base()
    print(f"Base: {len(base):,} fights after method filter")
    print(f"  f1_priors distribution: 0={sum(base['f1_priors']==0):4d}  "
          f"1={sum(base['f1_priors']==1):4d}  2={sum(base['f1_priors']==2):4d}  "
          f"3+={sum(base['f1_priors']>=3):4d}")

    results = {}
    for threshold in [1, 2, 3]:
        print("\n" + "="*80)
        print(f"THRESHOLD = {threshold}  (both fighters need ≥{threshold} prior UFC fights)")
        print("="*80)

        df = apply_threshold(base, threshold)
        test_window = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)]
        print(f"  Total fights (post-filter): {len(df):,}")
        print(f"  In test window: {len(test_window)}")

        # Single-shot
        m_ss, test_ss, n_tr_ss, n_te_ss = run_single_shot(df)
        print(f"\n  Single-shot: train={n_tr_ss:,}  test={n_te_ss}")
        print(f"    acc={m_ss['acc']*100:.2f}%  ll={m_ss['ll']:.4f}  "
              f"auc={m_ss['auc']:.4f}  brier={m_ss['brier']:.4f}")

        # 6-month walk-forward
        m_wf, per_fold, wf_df = run_6mo_wf(df)
        print(f"\n  6-month WF pooled: n={len(wf_df)}")
        print(f"    acc={m_wf['acc']*100:.2f}%  ll={m_wf['ll']:.4f}  "
              f"auc={m_wf['auc']:.4f}  brier={m_wf['brier']:.4f}")
        print(f"    Per-fold acc: "
              + "  ".join(f"f{p['fold']}={p['acc']*100:.1f}%" for p in per_fold))

        # ROI on 6-month WF predictions
        roi = run_roi(wf_df)
        print(f"\n  Strategy D (+EV) on 6mo WF: "
              f"matched={roi['n_match']}  bets={roi['n']}")
        if not np.isnan(roi.get('roi', np.nan)):
            print(f"    ROI={roi['roi']*100:+.2f}%  "
                  f"CI=[{roi['ci_lo']*100:+.2f}%, {roi['ci_hi']*100:+.2f}%]  "
                  f"p={roi['p']:.3f}  win%={roi['win_rate']*100:.1f}")

        results[threshold] = dict(
            n_total=len(df), n_test=len(test_window),
            single_shot=m_ss, wf_pooled=m_wf,
            per_fold=per_fold, roi=roi,
        )

    # ── Summary tables ─────────────────────────────────────────────
    print("\n" + "="*80)
    print("SIDE-BY-SIDE SUMMARY")
    print("="*80)

    print("\nSingle-shot LR (train-once):")
    print(f"{'Threshold':>9s}  {'n_test':>6s}  {'Acc':>7s}  {'LogLoss':>8s}  "
          f"{'AUC':>7s}  {'Brier':>7s}")
    for t in [1, 2, 3]:
        m = results[t]["single_shot"]
        print(f"{t:>9d}  {results[t]['n_test']:>6d}  {m['acc']*100:>6.2f}%  "
              f"{m['ll']:>8.4f}  {m['auc']:>7.4f}  {m['brier']:>7.4f}")

    print("\n6-month walk-forward pooled:")
    print(f"{'Threshold':>9s}  {'n_test':>6s}  {'Acc':>7s}  {'LogLoss':>8s}  "
          f"{'AUC':>7s}  {'Brier':>7s}")
    for t in [1, 2, 3]:
        m = results[t]["wf_pooled"]
        print(f"{t:>9d}  {results[t]['n_test']:>6d}  {m['acc']*100:>6.2f}%  "
              f"{m['ll']:>8.4f}  {m['auc']:>7.4f}  {m['brier']:>7.4f}")

    print("\nROI (Strategy D, +EV, on 6mo WF predictions):")
    print(f"{'Threshold':>9s}  {'n_match':>7s}  {'n_bets':>6s}  {'Win%':>5s}  "
          f"{'ROI':>7s}  {'CI_lo':>7s}  {'CI_hi':>7s}  {'p':>5s}")
    for t in [1, 2, 3]:
        r = results[t]["roi"]
        def fmt(v, d=2, p=False):
            if np.isnan(v): return "   -"
            return f"{v*100:+.{d}f}%" if p else f"{v:.{d}f}"
        print(f"{t:>9d}  {r['n_match']:>7d}  {r['n']:>6d}  "
              f"{r['win_rate']*100 if not np.isnan(r['win_rate']) else 0:>4.1f}%  "
              f"{fmt(r['roi'], p=True):>7s}  {fmt(r['ci_lo'], p=True):>7s}  "
              f"{fmt(r['ci_hi'], p=True):>7s}  {r['p']:>5.3f}")

    (DT / "threshold_sweep.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved to {DT/'threshold_sweep.json'}")


if __name__ == "__main__":
    main()
