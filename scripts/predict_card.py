"""predict_card.py — Inline predictions for upcoming UFC fights.

Edit MATCHUPS and EVENT_DATE below, then run:
  python scripts/predict_card.py

Trains inline on all available data (~60s). No saved artifacts needed.
Audit: docs/audits/predict_card.md
"""
import sys, warnings, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")

import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

import mma_ai_pipeline as mma
from elo_feature import compute_elo, BASE_ELO
from run_threshold_sweep_both_elos import (
    load_base_both_elos, apply_threshold, ELO_PARAMS, LOGISTIC_SCALE, DT
)
from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features, wc_state
from walk_forward_4fold import select_features
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ── CONFIG — must match master_validation.py ──────────────────────────────────
C, L1_RATIO, LAM = 0.05, 0.5, 1.20
THRESHOLD        = 3
TRAIN_START      = "2016-01-01"
NOVEL_FEATS      = ["home_advantage_diff", "card_position_norm_career_diff"]
MENS_WCS         = {5, 6, 7, 8, 9, 10, 11, 12}
BW_CODE          = 10

# ── MATCHUPS — edit this list ─────────────────────────────────────────────────
# (fighter1_jid, fighter2_jid, odds_f1_american, odds_f2_american,
#  weightclass_idx, scheduled_rounds)
#
# fighter_jid: CamelCase no spaces. E.g. "IslamMakhachev", "CharlesOliveira"
# weightclass:  5=WSW 6=BW 7=FW 8=LW 9=WW 10=MW 11=LHW 12=HW
# odds: American. Favorite = negative (-280), underdog = positive (+230)
EVENT_DATE = "2026-05-03"
MATCHUPS = [
    ("IslamMakhachev", "ArmanTsarukyan",   -450, +340, 8, 5),
    ("AlexandrePantoja", "SteveCasse",     -600, +420, 5, 5),
    ("GilbertBurns",  "SeanBrady",         -160, +130, 9, 3),
]

# ─────────────────────────────────────────────────────────────────────────────

def american_to_decimal(o):
    return 1 + 100/abs(o) if o < 0 else 1 + o/100

def american_to_prob(o):
    return abs(o)/(abs(o)+100) if o < 0 else 100/(o+100)

def devig(p1, p2):
    t = p1 + p2; return p1/t, p2/t


def build_style_elo_ratings():
    """Run style Elo and return final (strike_ratings, grapple_ratings) dicts."""
    conn = sqlite3.connect(str(elo_feature.DB_PATH))
    stats = pd.read_sql(
        "SELECT jevent, jbout, jfighter, sigstracc, tdacc, ctrl, subatt "
        "FROM ufc_fighter_match_stats_smooth", conn)
    conn.close()
    ufc = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    b = ufc.merge(stats, on=["jevent", "jbout"], how="inner")
    f1 = b[b["jfighter"]==b["f1"]][["jevent","jbout","DATE","f1","f2",
        "sigstracc","tdacc","ctrl","subatt"]].rename(columns={
        "sigstracc":"f1_sigstr","tdacc":"f1_td","ctrl":"f1_ctrl","subatt":"f1_sub"})
    f2 = b[b["jfighter"]==b["f2"]][["jevent","jbout",
        "sigstracc","tdacc","ctrl","subatt"]].rename(columns={
        "sigstracc":"f2_sigstr","tdacc":"f2_td","ctrl":"f2_ctrl","subatt":"f2_sub"})
    m = f1.merge(f2, on=["jevent","jbout"],how="inner").sort_values(
        ["DATE","jevent","jbout"]).reset_index(drop=True)
    def sig(x, s=5.0): return 1.0/(1.0+np.exp(-x/s))
    m["strk_a"] = sig(m["f1_sigstr"]-m["f2_sigstr"], 5.0)
    m["grp_f1"] = m["f1_td"] + m["f1_ctrl"]/60 + 0.3*m["f1_sub"]
    m["grp_f2"] = m["f2_td"] + m["f2_ctrl"]/60 + 0.3*m["f2_sub"]
    m["grp_a"]  = sig(m["grp_f1"]-m["grp_f2"], 1.5)
    strike = defaultdict(lambda: 1500.0); grapple = defaultdict(lambda: 1500.0)
    K=20; SCALE=400.0
    def exp_sc(a,b): return 1.0/(1.0+10**((b-a)/SCALE))
    for r in m.itertuples():
        sf1,sf2 = strike[r.f1], strike[r.f2]
        gf1,gf2 = grapple[r.f1], grapple[r.f2]
        if not (np.isnan(r.strk_a) or np.isnan(r.grp_a)):
            e_s=exp_sc(sf1,sf2); strike[r.f1]=sf1+K*(r.strk_a-e_s); strike[r.f2]=sf2+K*((1-r.strk_a)-(1-e_s))
            e_g=exp_sc(gf1,gf2); grapple[r.f1]=gf1+K*(r.grp_a-e_g); grapple[r.f2]=gf2+K*((1-r.grp_a)-(1-e_g))
    return dict(strike), dict(grapple)


def elo_diff_feats(jf1, jf2, ratings, extra, suffix):
    """Compute the 6 Elo diff features for jf1 vs jf2."""
    e1 = ratings.get(jf1, BASE_ELO); e2 = ratings.get(jf2, BASE_ELO)
    diff = e1 - e2
    win_prob = 1.0 / (1.0 + 10.0**(-diff/LOGISTIC_SCALE))
    h1 = extra["elo_3ago"].get(jf1, []);  h2 = extra["elo_3ago"].get(jf2, [])
    mom1 = float(e1-h1[0]) if len(h1)>=4 else np.nan
    mom2 = float(e2-h2[0]) if len(h2)>=4 else np.nan
    pk1  = extra["peak_elo"].get(jf1, BASE_ELO)
    pk2  = extra["peak_elo"].get(jf2, BASE_ELO)
    oq1  = extra["opp_elos"].get(jf1, []); oq2 = extra["opp_elos"].get(jf2, [])
    ao1  = float(np.mean(oq1)) if oq1 else BASE_ELO
    ao2  = float(np.mean(oq2)) if oq2 else BASE_ELO
    c1   = float(np.std(h1)) if len(h1)>=2 else np.nan
    c2   = float(np.std(h2)) if len(h2)>=2 else np.nan
    return {
        f"precomp_elo_diff_{suffix}": diff,
        f"elo_win_prob_{suffix}":     win_prob,
        f"elo_momentum_diff_{suffix}": (mom1-mom2) if not (np.isnan(mom1) or np.isnan(mom2)) else np.nan,
        f"peak_elo_diff_{suffix}":    pk1-pk2,
        f"avg_opp_elo_diff_{suffix}": ao1-ao2,
        f"elo_consist_diff_{suffix}": (c1-c2) if not (np.isnan(c1) or np.isnan(c2)) else np.nan,
    }


def build_matchup_row(jf1, jf2, wc, rounds, event_ts,
                      fighter_stats, feat_cols,
                      ratings_ufc, extra_ufc,
                      ratings_exp, extra_exp,
                      strike_r, grapple_r,
                      wc_hist):
    """Build one feature row dict for jf1 vs jf2."""
    s1 = fighter_stats.get(jf1, {}); s2 = fighter_stats.get(jf2, {})
    row = {}

    # MMA-AI diff features (fighter_stats stores per-fighter values)
    for feat in feat_cols:
        col = f"{feat}_diff" if not feat.endswith("_diff") else feat
        v1 = s1.get(feat, 0.0); v2 = s2.get(feat, 0.0)
        row[col] = float(v1) - float(v2)

    # Override days_since_last_fight with fresh calculation from event date
    date1 = s1.get("DATE", event_ts); date2 = s2.get("DATE", event_ts)
    dsf1 = max((event_ts - date1).days, 0)
    dsf2 = max((event_ts - date2).days, 0)
    row["days_since_last_fight_diff"] = float(dsf1 - dsf2)
    row["days_since_last_fight_f1"]   = float(dsf1)

    # Special ratio features (not simple diffs)
    age1 = max(float(s1.get("age", 28.0)), 18)
    age2 = max(float(s2.get("age", 28.0)), 18)
    row["age_ratio_diff"]   = (age1/age2) - 1.0
    reach1 = float(s1.get("REACH", 70.0)) or 70.0
    reach2 = float(s2.get("REACH", 70.0)) or 70.0
    row["reach_ratio_diff"] = (reach1/max(reach2, 1.0)) - 1.0

    # Non-diff metadata features
    row["weightclass_encoded"] = float(wc)
    row["scheduled_rounds"]    = float(rounds)

    # Elo features (UFC-only and expanded)
    row.update(elo_diff_feats(jf1, jf2, ratings_ufc, extra_ufc, "ufc"))
    row.update(elo_diff_feats(jf1, jf2, ratings_exp, extra_exp, "exp"))

    # Style Elos
    row["striking_elo_diff"]  = strike_r.get(jf1, 1500.0)  - strike_r.get(jf2, 1500.0)
    row["grappling_elo_diff"] = grapple_r.get(jf1, 1500.0) - grapple_r.get(jf2, 1500.0)

    # WC history features
    st1 = wc_state(jf1, event_ts, wc, wc_hist)
    st2 = wc_state(jf2, event_ts, wc, wc_hist)
    row["wc_native_winrate_diff"]  = st1["winrate"]  - st2["winrate"]
    row["wc_native_fights_diff"]   = st1["fights"]   - st2["fights"]
    row["wc_native_ko_rate_diff"]  = st1["ko_rate"]  - st2["ko_rate"]
    row["days_since_this_wc_diff"] = st1["days_since"] - st2["days_since"]
    row["cross_division_flag"]     = float(
        (st1["modal_wc"] not in (0, wc)) or (st2["modal_wc"] not in (0, wc)))

    # Market features — default 0 (home_advantage requires nationality lookup,
    # card_position requires knowing fight order on the card)
    for f in NOVEL_FEATS:
        row.setdefault(f, 0.0)

    return row


def main():
    event_ts = pd.Timestamp(EVENT_DATE)
    print("="*72)
    print(f"  PREDICT CARD — {EVENT_DATE}")
    print("="*72)

    # ── Step 1: MMA-AI pipeline (steps 1-6) ────────────────────────────────
    print("\n[1/5] MMA-AI pipeline (steps 1–6)...")
    df_full = mma.load_base_data()
    df_full = mma.beta_binomial_smooth(df_full)
    df_full = mma.poisson_gamma_smooth(df_full)
    df_full = mma.compute_derived_features(df_full)
    stat_cols = sorted(set(c for c in df_full.columns if (
        c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
        c.endswith("_ratio") or c.endswith("_per_ctrl") or
        c in ["ko_smooth","win_smooth","decision_smooth","sub_land_smooth",
              "sub_land_rate","ctrl_pm","ko_per_sig_str_land","td_per_sig_str_att",
              "ground_per_ctrl","dist_per_sig_str_land","head_per_sig_str_land",
              "rev_per_ctrlopp","sig_str_land_ratio","ko_ratio","sub_att_ratio",
              "ctrl_ratio","ground_land_per_ctrl","td_land_per_ctrl"]
    ) and c in df_full.columns and not c.startswith("opp_") and not c.endswith("_raw")))
    df_full = mma.compute_decayed_averages(df_full, stat_cols, mma.V7_CONFIG["decay_lambda"])
    df_full = mma.compute_opponent_history(df_full, stat_cols, mma.V7_CONFIG["decay_lambda"])

    priors = mma.compute_wc_priors(df_full, stat_cols)
    df_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    _, fighter_stats, feat_cols = mma.assemble_features(
        df_adj, stat_cols, mma.V7_CONFIG["decay_lambda"], return_individuals=True)
    print(f"  Fighter stats: {len(fighter_stats)} fighters")

    # ── Step 2: Load training dataframe + train model ───────────────────────
    print("\n[2/5] Loading training data and fitting model...")
    base = load_base_both_elos()
    df   = apply_threshold(base, THRESHOLD)
    df   = add_wc_features(df, load_wc_history_from_db())
    feats_baseline = select_features(df)

    mf   = pd.read_csv(DT.parent / "tmp" / "market_features_clean.csv", parse_dates=["DATE"])
    avail = [c for c in NOVEL_FEATS if c in mf.columns]
    df   = df.merge(mf[["DATE","jbout","jfighter"]+avail], on=["DATE","jbout","jfighter"], how="left")
    usable = [c for c in feats_baseline + [n for n in avail if n not in feats_baseline]
              if c in df.columns and df[c].std() > 1e-8]

    train = df[df["DATE"] >= TRAIN_START].copy()
    from retrain_lr_symmetric import flip_row_dataframe
    td   = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    imp  = SimpleImputer(strategy="median")
    sc   = StandardScaler()
    Xtr  = sc.fit_transform(imp.fit_transform(td[usable]))
    w    = np.exp(-LAM * (event_ts - td["DATE"]).dt.days.values / 365.25)
    lr   = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1_RATIO,
                               solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    n_act = int((np.abs(lr.coef_[0]) > 1e-8).sum())
    print(f"  Trained on {len(train):,} fights, {n_act} active features")

    # ── Step 3: Elo ratings ─────────────────────────────────────────────────
    print("\n[3/5] Computing Elo ratings...")
    bouts_ufc = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    _, ratings_ufc, _, extra_ufc = compute_elo(bouts_ufc, **ELO_PARAMS)
    bouts_exp = pd.read_csv(DT / "elo_bouts_expanded.csv", parse_dates=["DATE"])
    _, ratings_exp, _, extra_exp = compute_elo(bouts_exp, **ELO_PARAMS)
    strike_r, grapple_r = build_style_elo_ratings()
    print(f"  UFC Elo: {len(ratings_ufc)} fighters  |  Expanded: {len(ratings_exp)}")

    # ── Step 4: WC history ──────────────────────────────────────────────────
    print("\n[4/5] Loading WC history...")
    wc_hist = load_wc_history_from_db()

    # ── Step 5: Predict ─────────────────────────────────────────────────────
    print("\n[5/5] Predicting...\n")
    print(f"{'Fighter':>22}  vs  {'Fighter':<22}  {'p_model':>8}  {'p_vegas':>8}  {'edge':>6}  {'EV':>7}  {'rec':>6}")
    print("-"*90)

    for jf1, jf2, odds1, odds2, wc, rounds in MATCHUPS:
        if jf1 not in fighter_stats:
            print(f"  ⚠  {jf1} not found in fighter_stats — skipping"); continue
        if jf2 not in fighter_stats:
            print(f"  ⚠  {jf2} not found in fighter_stats — skipping"); continue

        row = build_matchup_row(jf1, jf2, wc, rounds, event_ts,
                                fighter_stats, feat_cols,
                                ratings_ufc, extra_ufc,
                                ratings_exp, extra_exp,
                                strike_r, grapple_r, wc_hist)

        x_row = pd.DataFrame([row], columns=usable).reindex(columns=usable).fillna(np.nan)
        x_imp = imp.transform(x_row)
        x_sc  = sc.transform(x_imp)
        p1    = float(lr.predict_proba(x_sc)[0, 1])

        p_v1_raw = american_to_prob(odds1)
        p_v2_raw = american_to_prob(odds2)
        p_v1, _  = devig(p_v1_raw, p_v2_raw)
        dec1     = american_to_decimal(odds1)
        ev       = p1 * dec1 - 1.0

        bet = wc in MENS_WCS and wc != BW_CODE and ev > 0
        tag = "✅ BET" if bet else ("skip" if ev <= 0 else "⚠ BW")
        print(f"  {jf1:>22}  vs  {jf2:<22}  {p1:>7.1%}  {p_v1:>7.1%}  {(p1-p_v1)*100:>+5.1f}pp  {ev:>+6.1%}  {tag}")

    print()


if __name__ == "__main__":
    main()
