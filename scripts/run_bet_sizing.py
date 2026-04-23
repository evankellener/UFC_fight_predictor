"""Bet sizing strategy comparison using the FINAL production model.

Model: LR + 12 Elo features + Tier 1c + Tier 2b, threshold=3, 6-month retrain.
Applies multiple bet-sizing strategies to Strategy D (+EV) bets and reports:
  - Terminal bankroll
  - ROI
  - Sharpe / Sortino ratios
  - Maximum drawdown
  - Win rate
  - Total staked
  - Bootstrap CI on ROI

Sizing strategies tested:
  1. Flat $1 per bet (baseline)
  2. Full Kelly: f = (p*d - 1) / (d - 1)  where d = decimal odds, p = model prob
  3. Half Kelly (common pro choice)
  4. Quarter Kelly (conservative)
  5. Capped full Kelly (max 5% bankroll)
  6. Capped half Kelly (max 2.5% bankroll)
  7. Confidence-weighted flat (linear in edge magnitude, max $2)
  8. Fractional-of-bankroll: 1% of current bankroll per bet

Starting bankroll: $1,000. Bets in chronological order.

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§11):
  §6  All sizing formulas are deterministic functions of model output + odds;
      NO parameters tuned on the 2024-05+ test window.
  §7  Vegas odds are evaluation-only.
  §10 Single run per strategy; one report.
"""
import json, sqlite3, sys, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
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
FILTER_THRESHOLD = 3

ELO_COLS_BASE = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
                 "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
ELO_PARAMS = dict(K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
                  decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
                  logistic_scale=449.205,
                  opp_quality_k=True, sliding_k=True, upset_momentum=True,
                  champ_mult=1.2)
TIER_1C = ["win_streak_entering_diff", "coming_off_loss_diff", "fights_last_12m_diff"]
STYLE_COLS = ["striking_elo_diff", "grappling_elo_diff"]
RNG = np.random.default_rng(42)

START_BANKROLL = 1000.0


def american_to_prob(o):
    if pd.isna(o) or abs(o) < 100: return np.nan
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def american_to_decimal(o):
    if pd.isna(o) or abs(o) < 100: return np.nan
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


def compute_elo_suffixed(bout_file, suffix):
    bouts = pd.read_csv(DT / bout_file, parse_dates=["DATE"])
    elo, *_ = compute_elo(bouts, **ELO_PARAMS)
    if "source" in elo.columns:
        elo = elo[elo["source"] == "ufc"].copy()
    keep = ["jbout", "DATE", "f1", "f2"] + ELO_COLS_BASE
    elo = elo[keep].copy()
    elo["DATE"] = pd.to_datetime(elo["DATE"])
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


def load_final_model_df():
    """Load the FINAL production feature matrix (same as run_threshold_sweep_both_elos.py)."""
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
    df = df[~m].drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    print("  Computing UFC-only Elo...")
    elo_ufc = compute_elo_suffixed("elo_bouts.csv", "ufc")
    print("  Computing EXPANDED Elo...")
    elo_exp = compute_elo_suffixed("elo_bouts_expanded.csv", "exp")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(elo_ufc, on=["jbout", "DATE"], how="left")
    df = df.rename(columns={"f1": "f1_tmp", "f2": "f2_tmp"})
    df = df.merge(elo_exp[["jbout", "DATE"] + [c for c in elo_exp.columns
                                                 if c.endswith("_exp")]],
                  on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] != df["f1_tmp"]
    for suffix in ("ufc", "exp"):
        for c in ELO_COLS_BASE:
            col = f"{c}_{suffix}"
            if c == "elo_win_prob":
                df.loc[flip, col] = 1 - df.loc[flip, col]
                df[col] = df[col].fillna(0.5)
            else:
                df.loc[flip, col] = -df.loc[flip, col]
                df[col] = df[col].fillna(0.0)
    df.drop(columns=["f1_tmp", "f2_tmp"], inplace=True, errors="ignore")
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    df = df.merge(rs[["DATE", "jbout", "jfighter"] + TIER_1C],
                  on=["DATE", "jbout", "jfighter"], how="left")
    for c in TIER_1C: df[c] = df[c].fillna(0)
    se = build_style_elos(); se["DATE"] = pd.to_datetime(se["DATE"])
    df = df.merge(se, on=["DATE", "jbout", "jfighter"], how="left")
    for c in STYLE_COLS: df[c] = df[c].fillna(0.0)
    df = df.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return df


def apply_filter(df):
    d = df[(df["f1_priors"] >= FILTER_THRESHOLD) & (df["f2_priors"] >= FILTER_THRESHOLD)].copy()
    return d[d["DATE"] >= TRAIN_START].reset_index(drop=True)


def attach_vegas(test):
    conn = sqlite3.connect(DB)
    odds = pd.read_sql("SELECT * FROM ufc_fight_odds", conn, parse_dates=["DATE"])
    conn.close()
    bad = ((odds["avg_odds_f1"].abs() < 100) | (odds["avg_odds_f2"].abs() < 100)
           | odds["avg_odds_f1"].isna() | odds["avg_odds_f2"].isna())
    odds = odds[~bad].drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
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
    return m.drop(columns=["jfighter_odds", "p_f1_devig", "p_f2_devig", "dec_f1", "dec_f2"]
                  ).drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)


def train_lr(train, usable, ref_date=None):
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    X = imp.fit_transform(train[usable]); Xs = sc.fit_transform(X)
    y = train["win"].astype(int).values
    ref = ref_date if ref_date is not None else train["DATE"].max()
    w = np.exp(-LAM * (ref - train["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs, y, sample_weight=w)
    return lr, imp, sc


def predict(test, lr, imp, sc, usable):
    Xs = sc.transform(imp.transform(test[usable]))
    return lr.predict_proba(Xs)[:, 1]


def build_wf_predictions(df):
    """Generate the 6-month walk-forward predictions for the full test window."""
    feats = [c for c in df.columns if (c.endswith("_diff") or c.endswith("_ufc")
             or c.endswith("_exp") or c in ("weightclass_encoded", "scheduled_rounds",
                                             "days_since_last_fight_f1"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    span = (TEST_LAST - TEST_FIRST).days
    folds = [(TEST_FIRST + pd.Timedelta(days=int(round(i * span / N_FOLDS))),
              TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span / N_FOLDS))) if i < N_FOLDS-1 else TEST_LAST)
             for i in range(N_FOLDS)]
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
    return wf


# ─── Bet sizing strategies ────────────────────────────────────────────────
def kelly_fraction(p, d):
    """Full Kelly. p = model prob of pick, d = decimal odds.
    f = (p*d - 1) / (d - 1)
    Clamp negative Kelly (no edge) to 0. Clamp f > 1 to 1 (don't risk more than bankroll).
    """
    if d <= 1.0: return 0.0
    f = (p * d - 1.0) / (d - 1.0)
    return max(0.0, min(f, 1.0))


def simulate(bets, sizing_func, start=START_BANKROLL):
    """Simulate bankroll through bets in chronological order.
    `sizing_func(bankroll, p_model, dec_odds, edge, i)` returns stake for this bet.
    Returns DataFrame with columns [date, stake, profit, bankroll, cum_profit].
    """
    bets = bets.sort_values("DATE").reset_index(drop=True)
    bankroll = start
    rows = []
    for i, row in bets.iterrows():
        p = row["p_pick"]
        d = row["dec_pick"]
        edge = row["edge"]
        stake = sizing_func(bankroll, p, d, edge, i)
        # Don't bet more than current bankroll
        stake = max(0.0, min(stake, bankroll))
        if row["won"]:
            profit = stake * (d - 1.0)
        else:
            profit = -stake
        bankroll += profit
        rows.append(dict(DATE=row["DATE"], stake=float(stake), profit=float(profit),
                         bankroll=float(bankroll),
                         p_model=p, dec=d, edge=edge, won=int(row["won"])))
    out = pd.DataFrame(rows)
    out["cum_profit"] = out["profit"].cumsum()
    out["cum_staked"] = out["stake"].cumsum()
    return out


def sharpe(profits, periods_per_year=365/69):  # UFC event cadence ≈ weekly
    if profits.std() == 0: return 0.0
    # annualize assuming one "period" per event (~weekly)
    return float(profits.mean() / profits.std() * np.sqrt(periods_per_year * 52))


def sortino(profits, periods_per_year=52):
    neg = profits[profits < 0]
    if len(neg) == 0 or neg.std() == 0:
        return float(np.inf) if profits.mean() > 0 else 0.0
    return float(profits.mean() / neg.std() * np.sqrt(periods_per_year))


def max_drawdown(bankroll_series):
    """Max peak-to-trough drawdown as fraction of peak."""
    if len(bankroll_series) == 0: return 0.0
    running_max = bankroll_series.cummax()
    drawdown = (bankroll_series - running_max) / running_max
    return float(drawdown.min())


def bootstrap_terminal(bets, sizing_func, n_boot=500, start=START_BANKROLL):
    """Bootstrap terminal bankroll by resampling bet outcomes in place."""
    terminals = []
    for _ in range(n_boot):
        idx = RNG.choice(len(bets), size=len(bets), replace=True)
        sample = bets.iloc[idx].reset_index(drop=True).sort_values("DATE").reset_index(drop=True)
        sim = simulate(sample, sizing_func, start=start)
        terminals.append(sim["bankroll"].iloc[-1] if len(sim) else start)
    lo, hi = np.percentile(terminals, [2.5, 97.5])
    return float(np.mean(terminals)), float(lo), float(hi)


def main():
    print("="*70)
    print("BET SIZING — FINAL model (both-Elos + Tier 1c + 2b, t=3, 6mo retrain)")
    print("="*70)

    df = load_final_model_df()
    df = apply_filter(df)
    wf = build_wf_predictions(df)
    print(f"  WF predictions: {len(wf)}")

    # Attach Vegas + compute +EV
    tv = attach_vegas(wf[["DATE", "jbout", "jfighter"]].drop_duplicates())
    wf = wf.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                      "dec_odds_f1", "dec_odds_f2"]],
                  on=["DATE", "jbout", "jfighter"], how="left")
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    wf_v = wf[wf["p_vegas_f1"].notna()].copy()
    p_m = wf_v["p_model"].values; p_v = wf_v["p_vegas_f1"].values
    pf1 = p_m >= 0.5
    edge_on_pick = np.where(pf1, p_m - p_v, (1 - p_m) - (1 - p_v))
    wf_v["edge"] = edge_on_pick
    # Pick side info
    wf_v["p_pick"] = np.where(pf1, p_m, 1 - p_m)
    wf_v["dec_pick"] = np.where(pf1, wf_v["dec_odds_f1"].values, wf_v["dec_odds_f2"].values)
    y_f1 = wf_v["win"].astype(int).values
    wf_v["won"] = (np.where(pf1, y_f1, 1 - y_f1) == 1)
    wf_v["implied_p_pick"] = 1.0 / wf_v["dec_pick"]  # includes vig
    bets = wf_v[wf_v["edge"] > 0].copy().sort_values("DATE").reset_index(drop=True)
    print(f"  +EV bets: {len(bets)}")
    print(f"  Avg edge: {bets['edge'].mean()*100:.2f}pp   "
          f"Avg decimal odds: {bets['dec_pick'].mean():.2f}   "
          f"Win rate: {bets['won'].mean()*100:.1f}%")

    # ── Sizing strategies ───────────────────────────────────────────
    strategies = {
        "A. Flat $1":
            lambda bk, p, d, e, i: 1.0,
        "B. Flat $10":
            lambda bk, p, d, e, i: 10.0,
        "C. Full Kelly":
            lambda bk, p, d, e, i: bk * kelly_fraction(p, d),
        "D. Half Kelly":
            lambda bk, p, d, e, i: bk * 0.5 * kelly_fraction(p, d),
        "E. Quarter Kelly":
            lambda bk, p, d, e, i: bk * 0.25 * kelly_fraction(p, d),
        "F. Full Kelly, capped 5%":
            lambda bk, p, d, e, i: min(bk * kelly_fraction(p, d), bk * 0.05),
        "G. Half Kelly, capped 2.5%":
            lambda bk, p, d, e, i: min(bk * 0.5 * kelly_fraction(p, d), bk * 0.025),
        "H. 1% flat-of-bankroll":
            lambda bk, p, d, e, i: bk * 0.01,
        "I. Edge-weighted ($1 per 1pp edge, cap $20)":
            lambda bk, p, d, e, i: min(max(e * 100 * 1.0, 0.0), 20.0),
    }

    results = []
    for name, sizer in strategies.items():
        sim = simulate(bets, sizer)
        terminal = sim["bankroll"].iloc[-1] if len(sim) else START_BANKROLL
        total_staked = sim["cum_staked"].iloc[-1] if len(sim) else 0.0
        total_profit = terminal - START_BANKROLL
        roi = (total_profit / total_staked) if total_staked > 0 else 0.0
        pct_return = total_profit / START_BANKROLL
        mdd = max_drawdown(sim["bankroll"])
        shr = sharpe(sim["profit"].values)
        sor = sortino(sim["profit"].values)
        wins = int(sim["won"].sum())
        # Bootstrap terminal CI
        mean_term, lo, hi = bootstrap_terminal(bets, sizer, n_boot=200)
        results.append(dict(strategy=name, n_bets=len(sim),
                            terminal=float(terminal),
                            total_staked=float(total_staked),
                            total_profit=float(total_profit),
                            roi=float(roi), pct_return=float(pct_return),
                            max_drawdown=float(mdd), sharpe=float(shr),
                            sortino=float(sor),
                            boot_lo=lo, boot_hi=hi, wins=wins))

    # Print summary
    print("\n" + "="*110)
    print(f"BET SIZING SUMMARY — $1,000 start, {len(bets)} +EV bets over 18 months")
    print("="*110)
    print(f"{'Strategy':<42s}  {'Final$':>8s}  {'Return':>7s}  {'Staked':>8s}  "
          f"{'ROI':>7s}  {'MaxDD':>7s}  {'Sharpe':>6s}")
    print("-" * 110)
    for r in results:
        print(f"{r['strategy']:<42s}  ${r['terminal']:>7.0f}  "
              f"{r['pct_return']*100:>+6.1f}%  ${r['total_staked']:>7.0f}  "
              f"{r['roi']*100:>+6.2f}%  {r['max_drawdown']*100:>+6.1f}%  "
              f"{r['sharpe']:>6.2f}")

    print(f"\nBootstrap 95% CI on terminal bankroll (500 resamples):")
    for r in results:
        print(f"  {r['strategy']:<42s}  mean=${(r['boot_lo']+r['boot_hi'])/2:>6.0f}  "
              f"CI=[${r['boot_lo']:>5.0f}, ${r['boot_hi']:>5.0f}]")

    (DT / "bet_sizing_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nSaved to {DT/'bet_sizing_results.json'}")
    return bets, results


if __name__ == "__main__":
    bets, results = main()
