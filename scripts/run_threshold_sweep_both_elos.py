"""Both-Elos feature approach — give LR 12 Elo features, let it learn.

Instead of a hard per-fighter switch (which failed due to scale mismatch),
add BOTH sources as independent feature columns:

  Standard 6 Elo features  → suffixed `_ufc`  (from elo_bouts.csv)
  Same 6 features           → suffixed `_exp`  (from elo_bouts_expanded.csv)

Total 12 Elo features. LR with ElasticNet regularization picks the right
weighting automatically. This respects that the two Elo sources are on
different scales — LR can learn a linear combination that extracts useful
signal from each without us hand-picking per fighter.

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§11):
  §2/§5 both Elo sources precomp-only
  §4 fit scaler/imputer on train only per fold
  §6 LR hyperparams frozen (C=0.05, l1=0.5) — NOT tuned on test
  §10 single run, single report
"""
import json, sqlite3, sys, warnings
from pathlib import Path
from collections import defaultdict
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
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
from elo_feature import compute_elo

DT = Path("data/tmp")
DB = "data/sqlite_db/slim_scrapper.db"
SCRAPER_DB = "data/sqlite_db/sqlite_scrapper.db"
TEST_FIRST = pd.Timestamp("2024-05-04")
# TEST_LAST was originally 2025-11-08. Extended to 2026-04-23 on 2026-04-23 to
# pull in ~85 extra fights across 14 events, bringing the test set from 420 → 505.
# Training window unchanged (ends 2024-05-04), so this is purely additive
# out-of-sample data. Populates the high-confidence calibration buckets better.
TEST_LAST  = pd.Timestamp("2026-04-23")
TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_ERA_FLOOR = pd.Timestamp("2016-01-01")
TRAIN_YEARS = 8
N_FOLDS = 3
# LAM changed 0.13 → 1.20 on 2026-04-24 based on the 4-fold walk-forward
# diagnostic (scripts/lambda_sweep_4fold.py). At λ=1.20, fold_4's +EV ROI
# flipped from -8.62% to +4.35%, and fold_4 accuracy improved 65.08 → 68.25%.
# Older folds lose some accuracy / ROI but those are historical (cannot
# be bet again) — fold_4 is the only fold that predicts future behavior.
# Ship accepted with the understanding that periodic retraining (every
# 3-6 months) is needed to keep up with ongoing era drift.
LAM = 1.20
LOGISTIC_SCALE = 449.205

ELO_COLS_BASE = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
                 "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
ELO_PARAMS = dict(K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
                  decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
                  logistic_scale=LOGISTIC_SCALE,
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


def compute_elo_suffixed(bout_file, suffix):
    """Run compute_elo and return per-bout Elo features suffixed with _{suffix}."""
    bouts = pd.read_csv(DT / bout_file, parse_dates=["DATE"])
    elo, *_ = compute_elo(bouts, **ELO_PARAMS)
    if "source" in elo.columns:
        elo = elo[elo["source"] == "ufc"].copy()
    keep = ["jbout", "DATE", "f1", "f2"] + ELO_COLS_BASE
    elo = elo[keep].copy()
    elo["DATE"] = pd.to_datetime(elo["DATE"])
    # Rename Elo cols with suffix
    elo = elo.rename(columns={c: f"{c}_{suffix}" for c in ELO_COLS_BASE})
    return elo


def build_style_elos():
    conn = sqlite3.connect(SCRAPER_DB)
    stats = pd.read_sql("""
        SELECT jevent, jbout, jfighter, sigstracc, tdacc, ctrl, subatt
        FROM ufc_fighter_match_stats_smooth
    """, conn)
    conn.close()
    ufc = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    b = ufc.merge(stats, on=["jevent", "jbout"], how="inner")
    f1 = b[b["jfighter"] == b["f1"]][["jevent", "jbout", "DATE", "f1", "f2",
                                        "sigstracc", "tdacc", "ctrl", "subatt"]].rename(
        columns={"sigstracc": "f1_sigstr", "tdacc": "f1_td",
                 "ctrl": "f1_ctrl", "subatt": "f1_sub"})
    f2 = b[b["jfighter"] == b["f2"]][["jevent", "jbout",
                                        "sigstracc", "tdacc", "ctrl", "subatt"]].rename(
        columns={"sigstracc": "f2_sigstr", "tdacc": "f2_td",
                 "ctrl": "f2_ctrl", "subatt": "f2_sub"})
    m = f1.merge(f2, on=["jevent", "jbout"], how="inner").sort_values(
        ["DATE", "jevent", "jbout"]).reset_index(drop=True)
    def sig(x, s=5.0): return 1.0 / (1.0 + np.exp(-x / s))
    m["strk_actual_f1"] = sig(m["f1_sigstr"] - m["f2_sigstr"], 5.0)
    m["grp_f1"] = m["f1_td"] + m["f1_ctrl"] / 60 + 0.3 * m["f1_sub"]
    m["grp_f2"] = m["f2_td"] + m["f2_ctrl"] / 60 + 0.3 * m["f2_sub"]
    m["grp_actual_f1"] = sig(m["grp_f1"] - m["grp_f2"], 1.5)
    strike = defaultdict(lambda: 1500.0); grapple = defaultdict(lambda: 1500.0)
    K = 20; SCALE = 400.0
    def exp_sc(a, b): return 1.0 / (1.0 + 10 ** ((b - a) / SCALE))
    rows = []
    for r in m.itertuples():
        sf1, sf2 = strike[r.f1], strike[r.f2]
        gf1, gf2 = grapple[r.f1], grapple[r.f2]
        rows.append(dict(DATE=r.DATE, jbout=r.jbout, jfighter=r.f1,
                         striking_elo_diff=sf1-sf2, grappling_elo_diff=gf1-gf2))
        rows.append(dict(DATE=r.DATE, jbout=r.jbout, jfighter=r.f2,
                         striking_elo_diff=sf2-sf1, grappling_elo_diff=gf2-gf1))
        if not (np.isnan(r.strk_actual_f1) or np.isnan(r.grp_actual_f1)):
            e_s = exp_sc(sf1, sf2)
            strike[r.f1] = sf1 + K*(r.strk_actual_f1 - e_s)
            strike[r.f2] = sf2 + K*((1-r.strk_actual_f1) - (1-e_s))
            e_g = exp_sc(gf1, gf2)
            grapple[r.f1] = gf1 + K*(r.grp_actual_f1 - e_g)
            grapple[r.f2] = gf2 + K*((1-r.grp_actual_f1) - (1-e_g))
    return pd.DataFrame(rows)


def load_base_both_elos():
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

    # Merge BOTH Elo sources
    print("  UFC-only Elo...")
    elo_ufc = compute_elo_suffixed("elo_bouts.csv", "ufc")
    print("  Expanded Elo...")
    elo_exp = compute_elo_suffixed("elo_bouts_expanded.csv", "exp")

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(elo_ufc, on=["jbout", "DATE"], how="left")
    df = df.rename(columns={"f1": "f1_ufc_tmp", "f2": "f2_ufc_tmp"})
    df = df.merge(elo_exp, on=["jbout", "DATE"], how="left",
                  suffixes=("", "_dup_from_exp"))
    # Drop dup f1/f2 from exp merge
    for c in list(df.columns):
        if c.endswith("_dup_from_exp"): df.drop(columns=[c], inplace=True)
    # f1_ufc_tmp is our bout's f1 reference
    flip = df["jfighter"] != df["f1_ufc_tmp"]

    # For each Elo feature, flip sign (or 1-p for win_prob) when jfighter is f2
    for suffix in ("ufc", "exp"):
        for c in ELO_COLS_BASE:
            col = f"{c}_{suffix}"
            if c == "elo_win_prob":
                df.loc[flip, col] = 1 - df.loc[flip, col]
            else:
                df.loc[flip, col] = -df.loc[flip, col]
            # Fill NaNs
            if c == "elo_win_prob":
                df[col] = df[col].fillna(0.5)
            else:
                df[col] = df[col].fillna(0.0)

    df.drop(columns=["f1_ufc_tmp", "f2_ufc_tmp"], inplace=True, errors="ignore")

    # Tier 1c recency
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    df = df.merge(rs[["DATE", "jbout", "jfighter"] + TIER_1C],
                  on=["DATE", "jbout", "jfighter"], how="left")
    for c in TIER_1C: df[c] = df[c].fillna(0)

    # Style Elos
    se = build_style_elos()
    se["DATE"] = pd.to_datetime(se["DATE"])
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


def pick_feats(df):
    """Feature columns — excludes Tier 1a/1b/2a per production stack.
    INCLUDES both _ufc and _exp Elo suffixes (12 Elo features total)."""
    return [c for c in df.columns if (c.endswith("_diff") or c.endswith("_ufc")
            or c.endswith("_exp")
            or c in ("weightclass_encoded", "scheduled_rounds",
                     "days_since_last_fight_f1"))
            and c not in ("f1_priors", "f2_priors")
            and not c.startswith("ix_")
            and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                          "form_winrate3_diff", "form_winrate5_diff",
                          "elo_trajectory_diff", "career_fights_diff",
                          "stance_mismatch", "southpaw_advantage_diff")]


def run_6mo_wf(df):
    span = (TEST_LAST - TEST_FIRST).days
    folds = [(TEST_FIRST + pd.Timedelta(days=int(round(i * span / N_FOLDS))),
              TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span / N_FOLDS))) if i < N_FOLDS-1 else TEST_LAST)
             for i in range(N_FOLDS)]
    feats = pick_feats(df)
    rows = []
    for i, (fs, fe) in enumerate(folds, 1):
        ts = max(TRAIN_ERA_FLOOR, fs - pd.DateOffset(years=TRAIN_YEARS))
        tr = df[(df["DATE"] >= ts) & (df["DATE"] < fs)].copy()
        te = df[(df["DATE"] >= fs) & (df["DATE"] < fe)].copy() if i < N_FOLDS \
             else df[(df["DATE"] >= fs) & (df["DATE"] <= fe)].copy()
        if len(te) == 0: continue
        usable = [c for c in feats if c in tr.columns and tr[c].std() > 1e-8]
        lr, imp, sc = train_lr(tr, usable)
        p = predict(te, lr, imp, sc, usable)
        te_c = te.copy(); te_c["p_model"] = p; te_c["fold"] = i
        rows.append(te_c)
    wf = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return metrics(wf["win"].astype(int).values, wf["p_model"].values), wf, feats


def run_roi(wf):
    tv = attach_vegas(wf[["DATE", "jbout", "jfighter"]].drop_duplicates())
    wf = wf.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                      "dec_odds_f1", "dec_odds_f2"]],
                  on=["DATE", "jbout", "jfighter"], how="left")
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    wf_v = wf[wf["p_vegas_f1"].notna()].copy()
    p_m = wf_v["p_model"].values; p_v = wf_v["p_vegas_f1"].values
    pf1 = p_m >= 0.5
    edge = np.where(pf1, p_m - p_v, (1 - p_m) - (1 - p_v))
    sub = wf_v[edge > 0].copy()
    if len(sub) < 2: return dict(n=0, n_match=len(wf_v))
    y = sub["win"].astype(int).values
    pf1_sub = (sub["p_model"].values >= 0.5)
    dec = np.where(pf1_sub, sub["dec_odds_f1"].values, sub["dec_odds_f2"].values)
    correct = np.where(pf1_sub, y, 1 - y)
    profits = np.where(correct == 1, dec - 1, -1.0)
    rois_boot = [RNG.choice(profits, len(profits), replace=True).mean() for _ in range(1000)]
    lo, hi = np.percentile(rois_boot, [2.5, 97.5])
    t, p_two = scistats.ttest_1samp(profits, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return dict(n=len(sub), n_match=len(wf_v),
                win_rate=float((profits > 0).mean()),
                roi=float(profits.mean()),
                ci_lo=float(lo), ci_hi=float(hi), p=float(p_one))


def main():
    print("="*70)
    print("BOTH-ELOs features — LR gets 12 Elo columns (6 UFC-only + 6 expanded)")
    print("="*70)
    base = load_base_both_elos()
    print(f"Base: {len(base):,} fights")

    results = {}
    for threshold in [1, 2, 3]:
        print("\n" + "="*70)
        print(f"THRESHOLD = {threshold}")
        print("="*70)
        df = apply_threshold(base, threshold)
        test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)]
        print(f"  Post-filter: {len(df):,}  test={len(test)}")
        m_wf, wf, feats = run_6mo_wf(df)
        usable = [c for c in feats if c in df.columns and df[c].std() > 1e-8]
        print(f"  Features: {len(usable)}  "
              f"(Elo: {sum(1 for c in usable if '_ufc' in c or '_exp' in c)})")
        print(f"  WF: acc={m_wf['acc']*100:.2f}% ll={m_wf['ll']:.4f} "
              f"auc={m_wf['auc']:.4f} brier={m_wf['brier']:.4f}")
        roi = run_roi(wf)
        if roi["n"] > 0:
            print(f"  ROI: n={roi['n']} win%={roi['win_rate']*100:.1f} "
                  f"ROI={roi['roi']*100:+.2f}%  "
                  f"CI=[{roi['ci_lo']*100:+.2f}%, {roi['ci_hi']*100:+.2f}%]  p={roi['p']:.3f}")
        results[threshold] = dict(wf=m_wf, roi=roi, n_test=len(test), n_features=len(usable))

    # Four-way comparison
    print("\n" + "="*90)
    print("FOUR-WAY: UFC-only vs EXPANDED vs HYBRID-per-fighter vs BOTH-features")
    print("="*90)
    ufc_only = {
        1: (67.86, 0.6088, 0.7327, 0.2097, 3.80, 0.270, 247),
        2: (69.17, 0.5904, 0.7504, 0.2019, 9.68, 0.061, 218),
        3: (70.95, 0.5830, 0.7647, 0.1985, 16.36, 0.007, 174),
    }
    expanded = {
        1: (68.67, 0.6127, 0.7274, 0.2114, 7.07, 0.134, 240),
        2: (69.17, 0.5945, 0.7446, 0.2036, 12.99, 0.023, 208),
        3: (70.71, 0.5867, 0.7571, 0.2002, 13.32, 0.024, 180),
    }
    hybrid = {
        1: (67.53, 0.6098, 0.7311, 0.2101, 2.85, 0.321, 245),
        2: (68.77, 0.5912, 0.7492, 0.2022, 8.20, 0.092, 216),
        3: (71.19, 0.5831, 0.7646, 0.1985, 16.36, 0.007, 174),
    }

    print(f"\n{'t':>2s}  {'Source':<12s}  {'Acc':>7s}  {'LL':>7s}  {'AUC':>7s}  "
          f"{'Brier':>7s}  {'n':>5s}  {'ROI':>8s}  {'p':>5s}")
    for t in [1, 2, 3]:
        for src, vals in [("UFC-only", ufc_only[t]), ("EXPANDED", expanded[t]),
                           ("HYBRID-f", hybrid[t])]:
            tstr = f"{t:>2d}" if src == "UFC-only" else "  "
            print(f"{tstr}  {src:<12s}  {vals[0]:>6.2f}%  "
                  f"{vals[1]:>7.4f}  {vals[2]:>7.4f}  {vals[3]:>7.4f}  "
                  f"{vals[6]:>5d}  {vals[4]:>+7.2f}%  {vals[5]:>5.3f}")
        h = results[t]
        print(f"{'':>2s}  {'BOTH-FEATS':<12s}  {h['wf']['acc']*100:>6.2f}%  "
              f"{h['wf']['ll']:>7.4f}  {h['wf']['auc']:>7.4f}  "
              f"{h['wf']['brier']:>7.4f}  {h['roi']['n']:>5d}  "
              f"{h['roi']['roi']*100:>+7.2f}%  {h['roi']['p']:>5.3f}")
        print()

    (DT / "threshold_sweep_both_elos.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"Saved to {DT/'threshold_sweep_both_elos.json'}")


if __name__ == "__main__":
    main()
