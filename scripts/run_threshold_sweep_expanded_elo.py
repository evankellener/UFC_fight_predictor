"""Threshold sweep with pre-UFC fight history in Elo initialization.

Same logic as `scripts/run_threshold_sweep.py` but points the Elo computation at
`data/tmp/elo_bouts_expanded.csv` (8,500 UFC + 2,438 non-UFC bouts) instead of
`elo_bouts.csv` (UFC-only). Also rebuilds style Elos on the expanded set.

Hypothesis: expanded Elo gives debutants and 1-2-fight UFC fighters real Elo
priors from their pre-UFC careers → better predictions at threshold=1 and
threshold=2 specifically (where Elo quality matters most).

Leakage guardrails:
  §2/§5  Non-UFC bouts only affect ratings BEFORE the UFC fight. Precomp-only
         invariant preserved: each bout uses strictly earlier bouts.
  Non-UFC bouts are warm-up only — never appear as test targets (test fights
  come from the UFC-only mmaai_features.csv).
  §10    Same 196-feature stack, same LR hyperparams, same test window, same
         filter thresholds — only the Elo CSV source differs.
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
TEST_LAST  = pd.Timestamp("2025-11-08")
TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_ERA_FLOOR = pd.Timestamp("2016-01-01")
TRAIN_YEARS = 8
N_FOLDS = 3
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

# ── Which Elo bout source to use ──────────────────────────────────────────
ELO_BOUTS_FILE = DT / "elo_bouts_expanded.csv"   # <<<<  KEY DIFFERENCE
# (run_threshold_sweep.py uses DT/"elo_bouts.csv" — UFC-only)


def american_to_prob(o):
    if pd.isna(o) or abs(o) < 100: return np.nan
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def american_to_decimal(o):
    if pd.isna(o) or abs(o) < 100: return np.nan
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


def build_style_elos_from_ufc_stats():
    """Rebuild style Elos on the UFC-only subset (non-UFC has no per-fight stats)."""
    conn = sqlite3.connect(SCRAPER_DB)
    stats = pd.read_sql("""
        SELECT jevent, jbout, jfighter, sigstracc, tdacc, ctrl, subatt
        FROM ufc_fighter_match_stats_smooth
    """, conn)
    conn.close()
    # Only use UFC bouts for style Elo (non-UFC has no per-fight stats)
    ufc = pd.read_csv(ELO_BOUTS_FILE, parse_dates=["DATE"])
    ufc = ufc[ufc["source"] == "ufc"].copy() if "source" in ufc.columns else ufc
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

    strike = defaultdict(lambda: 1500.0)
    grapple = defaultdict(lambda: 1500.0)
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


def load_base():
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

    # MAIN ELO (using EXPANDED bouts)
    bouts = pd.read_csv(ELO_BOUTS_FILE, parse_dates=["DATE"])
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

    # Tier 1c recency
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    df = df.merge(rs[["DATE", "jbout", "jfighter"] + TIER_1C],
                  on=["DATE", "jbout", "jfighter"], how="left")
    for c in TIER_1C: df[c] = df[c].fillna(0)

    # Style Elos (rebuilt from expanded-ufc-subset stats)
    se = build_style_elos_from_ufc_stats()
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
    return [c for c in df.columns if (c.endswith("_diff") or c in
            ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
            and c not in ("f1_priors", "f2_priors")
            and not c.startswith("ix_")
            and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                          "form_winrate3_diff", "form_winrate5_diff",
                          "elo_trajectory_diff", "career_fights_diff",
                          "stance_mismatch", "southpaw_advantage_diff")]


def run_single_shot(df):
    tr = df[df["DATE"] < TEST_FIRST].copy()
    te = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()
    feats = pick_feats(df)
    usable = [c for c in feats if c in tr.columns and tr[c].std() > 1e-8]
    lr, imp, sc = train_lr(tr, usable, ref_date=TEST_FIRST)
    p = predict(te, lr, imp, sc, usable)
    return metrics(te["win"].astype(int).values, p), len(tr), len(te)


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
    return metrics(wf["win"].astype(int).values, wf["p_model"].values), wf


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
    if len(sub) < 2:
        return dict(n=len(sub), n_match=len(wf_v), roi=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, p=np.nan, win_rate=np.nan)
    y = sub["win"].astype(int).values
    pf1_sub = (sub["p_model"].values >= 0.5)
    dec = np.where(pf1_sub, sub["dec_odds_f1"].values, sub["dec_odds_f2"].values)
    correct = np.where(pf1_sub, y, 1 - y)
    profits = np.where(correct == 1, dec - 1, -1.0)
    rois_boot = [RNG.choice(profits, len(profits), replace=True).mean()
                 for _ in range(1000)]
    lo, hi = np.percentile(rois_boot, [2.5, 97.5])
    t, p_two = scistats.ttest_1samp(profits, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return dict(n=len(sub), n_match=len(wf_v),
                win_rate=float((profits > 0).mean()),
                roi=float(profits.mean()),
                ci_lo=float(lo), ci_hi=float(hi), p=float(p_one))


def main():
    print("Loading base with EXPANDED Elo (pre-UFC + UFC)...")
    base = load_base()
    print(f"Base: {len(base):,} fights\n")

    results = {}
    for threshold in [1, 2, 3]:
        print("="*80)
        print(f"THRESHOLD = {threshold}")
        print("="*80)
        df = apply_threshold(base, threshold)
        test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)]
        print(f"  Post-filter: {len(df):,}  test={len(test)}")

        m_ss, n_tr, n_te = run_single_shot(df)
        print(f"  Single-shot: acc={m_ss['acc']*100:.2f}% ll={m_ss['ll']:.4f} "
              f"auc={m_ss['auc']:.4f} brier={m_ss['brier']:.4f}")
        m_wf, wf = run_6mo_wf(df)
        print(f"  6mo WF:      acc={m_wf['acc']*100:.2f}% ll={m_wf['ll']:.4f} "
              f"auc={m_wf['auc']:.4f} brier={m_wf['brier']:.4f}")
        roi = run_roi(wf)
        if not np.isnan(roi.get("roi", np.nan)):
            print(f"  ROI (D):     n={roi['n']} win%={roi['win_rate']*100:.1f} "
                  f"ROI={roi['roi']*100:+.2f}%  "
                  f"CI=[{roi['ci_lo']*100:+.2f}%, {roi['ci_hi']*100:+.2f}%]  p={roi['p']:.3f}")
        results[threshold] = dict(single_shot=m_ss, wf=m_wf, roi=roi,
                                    n_test=len(test))

    # ── Summary + compare to UFC-only prior ──
    print("\n" + "="*80)
    print("EXPANDED vs UFC-ONLY — side by side")
    print("="*80)

    prior = {
        1: dict(ss=(67.69, 0.6083, 0.7367, 0.2094),
                wf=(67.86, 0.6088, 0.7327, 0.2097),
                roi=(3.80, -8.12, 16.13, 0.270, 247)),
        2: dict(ss=(70.16, 0.5930, 0.7519, 0.2027),
                wf=(69.17, 0.5904, 0.7504, 0.2019),
                roi=(9.68, -2.44, 21.88, 0.061, 218)),
        3: dict(ss=(70.48, 0.5841, 0.7690, 0.1987),
                wf=(70.95, 0.5830, 0.7647, 0.1985),
                roi=(16.36, 3.32, 29.91, 0.007, 174)),
    }

    print(f"\n{'Threshold':>9s}  {'Source':<12s}  {'SS Acc':>7s}  {'WF Acc':>7s}  "
          f"{'WF LL':>7s}  {'WF AUC':>7s}  {'WF Brier':>8s}  {'ROI':>7s}  {'p':>5s}")
    for t in [1, 2, 3]:
        # Prior (UFC only)
        pri = prior[t]
        print(f"{t:>9d}  {'UFC-only':<12s}  {pri['ss'][0]:>6.2f}%  {pri['wf'][0]:>6.2f}%  "
              f"{pri['wf'][1]:>7.4f}  {pri['wf'][2]:>7.4f}  {pri['wf'][3]:>8.4f}  "
              f"{pri['roi'][0]:>+6.2f}%  {pri['roi'][3]:>5.3f}")
        r = results[t]
        print(f"{'':>9s}  {'EXPANDED':<12s}  {r['single_shot']['acc']*100:>6.2f}%  "
              f"{r['wf']['acc']*100:>6.2f}%  {r['wf']['ll']:>7.4f}  "
              f"{r['wf']['auc']:>7.4f}  {r['wf']['brier']:>8.4f}  "
              f"{r['roi']['roi']*100:>+6.2f}%  {r['roi']['p']:>5.3f}")
        # Deltas
        d_acc = r['wf']['acc']*100 - pri['wf'][0]
        d_roi = r['roi']['roi']*100 - pri['roi'][0]
        print(f"{'':>9s}  {'Δ':<12s}  {'':>7s}  {d_acc:>+6.2f}pp  "
              f"{pri['wf'][1]-r['wf']['ll']:>+7.4f}  "
              f"{r['wf']['auc']-pri['wf'][2]:>+7.4f}  "
              f"{pri['wf'][3]-r['wf']['brier']:>+8.4f}  "
              f"{d_roi:>+6.2f}pp")
        print()

    (DT / "threshold_sweep_expanded_elo.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"Saved to {DT/'threshold_sweep_expanded_elo.json'}")


if __name__ == "__main__":
    main()
