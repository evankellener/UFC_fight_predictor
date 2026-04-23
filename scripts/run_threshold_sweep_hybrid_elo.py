"""Hybrid Elo — per-fighter source selection based on UFC experience.

For each fighter at each fight:
  - If UFC prior count >= HYBRID_CUTOFF (default 3):
      use UFC-only Elo (their UFC history is rich; pre-UFC is noise)
  - Else:
      use expanded Elo (they need the pre-UFC prior to escape BASE_ELO=1500)

Builds per-fight hybrid `precomp_elo_f1/f2` (and peak, momentum, avg_opp,
consist) by selecting per-fighter from each source run. Recomputes
`precomp_elo_diff` and `elo_win_prob` from the hybrid values.

Hypothesis: hybrid should (a) retain +16% ROI at threshold=3 (UFC-only
veterans keep their sharp rating) AND (b) lift threshold=1,2 ROI
(rookies get informative non-UFC prior).

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§11):
  §2/§5 both Elo sources are precomp-only by construction
  §3 UFC priors count uses strict d<fight_date
  §6 HYBRID_CUTOFF=3 is NOT tuned on this test window — it's the same
     threshold we already use in production (finding_threshold_matters.md)
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
TEST_LAST  = pd.Timestamp("2025-11-08")
TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_ERA_FLOOR = pd.Timestamp("2016-01-01")
TRAIN_YEARS = 8
N_FOLDS = 3
LAM = 0.13
HYBRID_CUTOFF = 3    # <3 UFC priors → use expanded Elo; ≥3 → use UFC-only

ELO_PARAMS = dict(K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
                  decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
                  logistic_scale=449.205,
                  opp_quality_k=True, sliding_k=True, upset_momentum=True,
                  champ_mult=1.2)
LOGISTIC_SCALE = 449.205  # matches ELO_PARAMS.logistic_scale
TIER_1C = ["win_streak_entering_diff", "coming_off_loss_diff", "fights_last_12m_diff"]
STYLE_COLS = ["striking_elo_diff", "grappling_elo_diff"]
RNG = np.random.default_rng(42)

# Per-fighter Elo columns we need from each source
PER_FIGHTER = ["precomp_elo", "elo_momentum", "peak_elo", "avg_opp_elo", "elo_consist"]


def american_to_prob(o):
    if pd.isna(o) or abs(o) < 100: return np.nan
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def american_to_decimal(o):
    if pd.isna(o) or abs(o) < 100: return np.nan
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


def compute_both_elos():
    """Run compute_elo twice (UFC-only vs expanded), return per-bout per-fighter
    Elo values so we can select per-fighter later."""
    print("  Computing UFC-only Elo...")
    ufc = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    elo_ufc, *_ = compute_elo(ufc, **ELO_PARAMS)

    print("  Computing EXPANDED Elo (UFC + pre-UFC)...")
    exp = pd.read_csv(DT / "elo_bouts_expanded.csv", parse_dates=["DATE"])
    elo_exp, *_ = compute_elo(exp, **ELO_PARAMS)
    # Keep only UFC bouts from expanded result (non-UFC bouts aren't test targets)
    if "source" in elo_exp.columns:
        elo_exp = elo_exp[elo_exp["source"] == "ufc"].copy()

    keep = ["jbout", "DATE", "f1", "f2"] + [f"{c}_f1" for c in PER_FIGHTER] + \
           [f"{c}_f2" for c in PER_FIGHTER]
    elo_ufc = elo_ufc[keep].copy()
    elo_exp = elo_exp[keep].copy()
    for d in (elo_ufc, elo_exp):
        d["DATE"] = pd.to_datetime(d["DATE"])

    # Merge on (jbout, DATE); suffix "_ufc" vs "_exp"
    elo_ufc = elo_ufc.rename(columns={c: f"{c}_ufc" for c in elo_ufc.columns
                                        if c not in ("jbout", "DATE", "f1", "f2")})
    elo_exp = elo_exp.rename(columns={c: f"{c}_exp" for c in elo_exp.columns
                                        if c not in ("jbout", "DATE", "f1", "f2")})
    both = elo_ufc.merge(elo_exp[["jbout", "DATE"] +
                                 [c for c in elo_exp.columns if c.endswith("_exp")]],
                         on=["jbout", "DATE"], how="left")
    print(f"  Merged: {len(both)} UFC bouts × "
          f"{len([c for c in both.columns if c.endswith('_ufc') or c.endswith('_exp')])} Elo cols")
    return both


def build_style_elos():
    """Style Elos on UFC-only (non-UFC has no per-fight stats)."""
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


def load_base_hybrid():
    """Load mmaai_features + merge in HYBRID Elo per fighter."""
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    # UFC prior counts
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

    # ── HYBRID ELO ─────────────────────────────────────────────────
    both = compute_both_elos()
    df["DATE"] = pd.to_datetime(df["DATE"])
    both["DATE"] = pd.to_datetime(both["DATE"])
    df = df.merge(both, on=["jbout", "DATE"], how="left")

    # Per-fight hybrid: each fighter gets UFC or EXP Elo based on their UFC priors.
    # Our mmaai row has jfighter (red corner = f1 in pipeline) and opp_jfighter
    # (blue corner = f2). both df has f1/f2 columns (alphabetical or bout order).
    # Check if jfighter matches f1 or f2; compute priors for both positions.
    jf_is_f1 = df["jfighter"] == df["f1"]
    # jfighter's priors:
    priors_of_jf = df["f1_priors"]   # jfighter is "fighter1" in mmaai context — we already computed this
    priors_of_opp = df["f2_priors"]

    # For each of the 5 Elo features, build a hybrid per fighter:
    # Fighter in position f1 (from both): if their priors >= cutoff use _ufc, else _exp
    # But we don't know which POSITION each fighter holds in `both` directly — we have
    # df["f1"] (bout's f1) and df["jfighter"]. We need priors of bout-position-f1
    # and bout-position-f2 to select the right Elo source.
    # If df["jfighter"] == df["f1"]: bout_f1 is jfighter → bout_f1's priors = priors_of_jf
    # If df["jfighter"] == df["f2"]: bout_f1 is opp_jfighter → bout_f1's priors = priors_of_opp
    priors_bout_f1 = np.where(jf_is_f1, priors_of_jf, priors_of_opp)
    priors_bout_f2 = np.where(jf_is_f1, priors_of_opp, priors_of_jf)

    use_ufc_f1 = priors_bout_f1 >= HYBRID_CUTOFF
    use_ufc_f2 = priors_bout_f2 >= HYBRID_CUTOFF

    # Build hybrid per-fighter Elos (bout-position f1, f2)
    for feat in PER_FIGHTER:
        f1_ufc = df[f"{feat}_f1_ufc"].values
        f1_exp = df[f"{feat}_f1_exp"].values
        f2_ufc = df[f"{feat}_f2_ufc"].values
        f2_exp = df[f"{feat}_f2_exp"].values
        df[f"{feat}_f1_hybrid"] = np.where(use_ufc_f1, f1_ufc, f1_exp)
        df[f"{feat}_f2_hybrid"] = np.where(use_ufc_f2, f2_ufc, f2_exp)
        # Diff (f1 - f2 in bout ordering)
        df[f"{feat}_diff_bout"] = df[f"{feat}_f1_hybrid"] - df[f"{feat}_f2_hybrid"]

    # Convert to jfighter-oriented diffs (flip sign if jfighter == f2)
    flip = ~jf_is_f1
    # precomp_elo_diff
    df["precomp_elo_diff"] = np.where(flip, -df["precomp_elo_diff_bout"], df["precomp_elo_diff_bout"])
    df["elo_momentum_diff"] = np.where(flip, -df["elo_momentum_diff_bout"], df["elo_momentum_diff_bout"])
    df["peak_elo_diff"] = np.where(flip, -df["peak_elo_diff_bout"], df["peak_elo_diff_bout"])
    df["avg_opp_elo_diff"] = np.where(flip, -df["avg_opp_elo_diff_bout"], df["avg_opp_elo_diff_bout"])
    df["elo_consist_diff"] = np.where(flip, -df["elo_consist_diff_bout"], df["elo_consist_diff_bout"])
    # elo_win_prob from hybrid diff (logistic)
    df["elo_win_prob"] = 1.0 / (1.0 + 10 ** (-df["precomp_elo_diff"] / LOGISTIC_SCALE))

    # Fill NaNs
    for c in ["precomp_elo_diff", "elo_momentum_diff", "peak_elo_diff",
              "avg_opp_elo_diff", "elo_consist_diff"]:
        df[c] = df[c].fillna(0.0)
    df["elo_win_prob"] = df["elo_win_prob"].fillna(0.5)

    # Drop the intermediate columns to keep df clean
    drop_cols = [c for c in df.columns if (c.endswith("_ufc") or c.endswith("_exp")
                 or c.endswith("_hybrid") or c.endswith("_bout"))
                 and c not in ("f1", "f2")]
    df.drop(columns=drop_cols + ["f1", "f2"], inplace=True, errors="ignore")

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
    return [c for c in df.columns if (c.endswith("_diff") or c in
            ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1",
             "elo_win_prob"))
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
    print(f"HYBRID ELO — cutoff = {HYBRID_CUTOFF} UFC priors")
    print(f"  < {HYBRID_CUTOFF} priors → use EXPANDED (UFC + pre-UFC) Elo")
    print(f"  >= {HYBRID_CUTOFF} priors → use UFC-ONLY Elo")
    print("="*70)

    print("\nBuilding hybrid feature matrix...")
    base = load_base_hybrid()
    print(f"Base: {len(base):,} fights\n")

    results = {}
    for threshold in [1, 2, 3]:
        print("="*70)
        print(f"THRESHOLD = {threshold}")
        print("="*70)
        df = apply_threshold(base, threshold)
        test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)]
        print(f"  Post-filter: {len(df):,}  test={len(test)}")

        m_wf, wf = run_6mo_wf(df)
        print(f"  6mo WF: acc={m_wf['acc']*100:.2f}% ll={m_wf['ll']:.4f} "
              f"auc={m_wf['auc']:.4f} brier={m_wf['brier']:.4f}")
        roi = run_roi(wf)
        print(f"  ROI: n={roi['n']} win%={roi['win_rate']*100:.1f} "
              f"ROI={roi['roi']*100:+.2f}%  "
              f"CI=[{roi['ci_lo']*100:+.2f}%, {roi['ci_hi']*100:+.2f}%]  p={roi['p']:.3f}")
        results[threshold] = dict(wf=m_wf, roi=roi, n_test=len(test))

    # ── Three-way comparison ────────────────────────────────────
    print("\n" + "="*90)
    print("THREE-WAY: UFC-only vs EXPANDED vs HYBRID")
    print("="*90)

    ufc_only = {
        1: dict(wf=(67.86, 0.6088, 0.7327, 0.2097), roi=(3.80, -8.12, 16.13, 0.270, 247)),
        2: dict(wf=(69.17, 0.5904, 0.7504, 0.2019), roi=(9.68, -2.44, 21.88, 0.061, 218)),
        3: dict(wf=(70.95, 0.5830, 0.7647, 0.1985), roi=(16.36, 3.32, 29.91, 0.007, 174)),
    }
    expanded = {
        1: dict(wf=(68.67, 0.6127, 0.7274, 0.2114), roi=(7.07, -4.91, 19.17, 0.134, 240)),
        2: dict(wf=(69.17, 0.5945, 0.7446, 0.2036), roi=(12.99, 0.21, 25.27, 0.023, 208)),
        3: dict(wf=(70.71, 0.5867, 0.7571, 0.2002), roi=(13.32, -0.75, 25.99, 0.024, 180)),
    }

    print(f"\n{'t':>2s}  {'Source':<10s}  {'WF Acc':>7s}  {'WF LL':>7s}  "
          f"{'WF AUC':>7s}  {'Brier':>7s}  {'n_bets':>6s}  {'ROI':>8s}  {'p':>5s}")
    for t in [1, 2, 3]:
        u = ufc_only[t]; e = expanded[t]; h = results[t]
        print(f"{t:>2d}  {'UFC-only':<10s}  {u['wf'][0]:>6.2f}%  {u['wf'][1]:>7.4f}  "
              f"{u['wf'][2]:>7.4f}  {u['wf'][3]:>7.4f}  {u['roi'][4]:>6d}  "
              f"{u['roi'][0]:>+7.2f}%  {u['roi'][3]:>5.3f}")
        print(f"{'':>2s}  {'EXPANDED':<10s}  {e['wf'][0]:>6.2f}%  {e['wf'][1]:>7.4f}  "
              f"{e['wf'][2]:>7.4f}  {e['wf'][3]:>7.4f}  {e['roi'][4]:>6d}  "
              f"{e['roi'][0]:>+7.2f}%  {e['roi'][3]:>5.3f}")
        print(f"{'':>2s}  {'HYBRID':<10s}  {h['wf']['acc']*100:>6.2f}%  "
              f"{h['wf']['ll']:>7.4f}  {h['wf']['auc']:>7.4f}  {h['wf']['brier']:>7.4f}  "
              f"{h['roi']['n']:>6d}  {h['roi']['roi']*100:>+7.2f}%  {h['roi']['p']:>5.3f}")
        print()

    (DT / "threshold_sweep_hybrid_elo.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"Saved to {DT/'threshold_sweep_hybrid_elo.json'}")


if __name__ == "__main__":
    main()
