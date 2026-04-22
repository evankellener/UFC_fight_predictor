"""Compute betting ROI for model predictions vs. Vegas on matched test fights.

Strategies evaluated:
  [A] All model picks       — bet $1 on whoever model predicts (all N fights)
  [B] Model agrees w/ Vegas — bet only when model pick == Vegas pick
  [C] Model disagrees       — bet our side only when model pick != Vegas pick
  [D] +EV only              — bet when model_p > vegas_devig_p  (any edge > 0)
  [E] Edge ≥ 5pp            — bet when model_p − vegas_devig_p ≥ 0.05
  [F] Edge ≥ 10pp           — bet when model_p − vegas_devig_p ≥ 0.10
  [G] CLV-style: AGREE AND p_fav ≥ 0.65  — per memory `finding_clv_signal_confirmed.md`

For each strategy, reports:
  - n_bets (how many fights qualified)
  - win_rate
  - ROI = total_profit / total_wagered (flat $1 bets)
  - Bootstrap 95% CI on ROI (1000 resamples)
  - Permutation-test p-value (H0: strategy has no edge vs coin-flip picks)

IMPORTANT per memory `finding_roi_is_noise.md`:
  - Prior edge_5pp test: +9% ROI headline but p=0.12 (not significant)
  - ROI claims require significance, not just a positive point estimate

Leakage guardrails (per LEAKAGE_REFERENCE.md):
  §1 Temporal split preserved (same 348-fight test window).
  §6 Thresholds (5pp, 10pp, 0.65) FROZEN from memory — NOT tuned on this window.
  §7 Vegas odds used at evaluation only; never as model features.
  §10 Strategies defined BEFORE looking at results. Single run, single report.
"""
import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

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
ELO_PARAMS = dict(
    K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
    decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
    logistic_scale=449.205,
    opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2,
)
RNG = np.random.default_rng(42)


# ─── odds / probability helpers ─────────────────────────────────────────────
def american_to_prob(o):
    # Valid American odds: |o| >= 100. Anything smaller is a scraper artifact
    # (odds like -1, -40, +34 are not real bookmaker lines and produce
    # nonsensical decimals when converted).
    if pd.isna(o) or abs(o) < 100:
        return np.nan
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def american_to_decimal(o):
    if pd.isna(o) or abs(o) < 100:
        return np.nan
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


# ─── filter + Elo (same as compare_to_vegas.py) ─────────────────────────────
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
    m = df["METHOD_norm"].apply(lambda m: any(u in str(m) for u in unwanted)
                                 if pd.notna(m) else False)
    df = df[~m]
    df = df[df["DATE"] >= TRAIN_START]
    return df.reset_index(drop=True)


def build_elo(df):
    bouts = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    elo_df, *_ = compute_elo(bouts, **ELO_PARAMS)
    keep = ["jbout", "DATE", "f1", "f2"] + ELO_COLS
    elo_df = elo_df[keep].copy()
    elo_df["DATE"] = pd.to_datetime(elo_df["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(elo_df, on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] == df["f2"]
    for c in ELO_COLS:
        if c == "elo_win_prob":
            df.loc[flip, c] = 1 - df.loc[flip, c]
        else:
            df.loc[flip, c] = -df.loc[flip, c]
    df.drop(columns=["f1", "f2"], inplace=True, errors="ignore")
    for c in ELO_COLS:
        df[c] = df[c].fillna(0.5 if c == "elo_win_prob" else 0.0)
    return df


def attach_vegas(test):
    conn = sqlite3.connect(DB)
    odds = pd.read_sql("SELECT * FROM ufc_fight_odds", conn, parse_dates=["DATE"])
    conn.close()
    # Reject scraper-error rows: valid American odds have |value| >= 100.
    bad = ((odds["avg_odds_f1"].abs() < 100) | (odds["avg_odds_f2"].abs() < 100)
           | odds["avg_odds_f1"].isna() | odds["avg_odds_f2"].isna())
    if bad.any():
        print(f"  Dropping {bad.sum()} odds rows with invalid American odds (|o| < 100)")
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
    # Vegas probability/odds for THIS test row's fighter1:
    m["p_vegas_f1"] = np.where(flip, m["p_f2_devig"], m["p_f1_devig"])
    m["p_vegas_f2"] = np.where(flip, m["p_f1_devig"], m["p_f2_devig"])
    m["dec_odds_f1"] = np.where(flip, m["dec_f2"], m["dec_f1"])
    m["dec_odds_f2"] = np.where(flip, m["dec_f1"], m["dec_f2"])
    return m.drop(columns=["jfighter_odds", "p_f1_devig", "p_f2_devig",
                            "dec_f1", "dec_f2"])


def train_model(train, test, feats, label="win", lam=0.13):
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train[usable])
    X_te = imp.transform(test[usable])
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    y_tr = train[label].astype(int).values
    w = np.exp(-lam * (TEST_START - train["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr, y_tr, sample_weight=w)
    return lr.predict_proba(X_te)[:, 1]


# ─── ROI + significance ─────────────────────────────────────────────────────
def strategy_profit(df, strategy_mask, pick_f1):
    """
    For each fight selected by strategy_mask, bet $1 on pick_f1 (True = fighter1).
    Profit on a win = dec_odds − 1; loss = −1.
    Returns array of per-bet profits.
    """
    sel = df[strategy_mask].copy()
    pf1 = pick_f1[strategy_mask]
    y_f1 = sel["win"].astype(int).values      # 1 if fighter1 won
    dec = np.where(pf1, sel["dec_odds_f1"].values, sel["dec_odds_f2"].values)
    correct = np.where(pf1, y_f1, 1 - y_f1)
    return np.where(correct == 1, dec - 1, -1.0)


def bootstrap_ci(profits, reps=1000):
    if len(profits) == 0:
        return (np.nan, np.nan, np.nan)
    rois = []
    for _ in range(reps):
        s = RNG.choice(profits, size=len(profits), replace=True)
        rois.append(s.mean())  # ROI with $1 flat bets = mean profit
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(np.mean(profits)), float(lo), float(hi)


def ttest_p(profits):
    """One-sample t-test: H0 ROI=0 vs H1 ROI>0. One-sided p-value.
    Directly interpretable, no permutation-shuffle ambiguity.
    """
    if len(profits) < 2: return np.nan
    from scipy import stats
    t, p_two = stats.ttest_1samp(profits, popmean=0.0)
    # one-sided: if mean > 0, p = p_two/2; else 1 - p_two/2
    return float(p_two / 2 if t > 0 else 1 - p_two / 2)


def fmt(x, d=3):
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:+.{d}f}"


def main():
    print("="*70)
    print("STEP 1: Build features + train LR+Elo + attach Vegas")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    df = build_elo(df)
    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,}   Test: {len(test):,}")

    base_feats = [c for c in df.columns if (c.endswith("_diff") or c in
                  ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
                  and c not in ELO_COLS]
    elo_feats = base_feats + ELO_COLS

    test_v = attach_vegas(test)
    sub = test_v[test_v["p_vegas_f1"].notna()].copy().reset_index(drop=True)
    print(f"  With Vegas odds: {len(sub)}")

    # Train on FULL train (pre-test-date), predict on sub
    train = train.loc[:, ~train.columns.duplicated()]
    p_lr_el = train_model(train, sub, elo_feats)
    sub["p_model_f1"] = p_lr_el

    # Verify alignment: accuracy should match our earlier comparison (~69.5%)
    pred_f1_is_f1 = sub["p_model_f1"] >= 0.5
    print(f"  Model accuracy on Vegas-matched subset: "
          f"{accuracy_score(sub['win'].astype(int), pred_f1_is_f1.astype(int)):.4f}")

    # ─── Define strategies ────────────────────────────────────────────
    p_m = sub["p_model_f1"].values
    p_v = sub["p_vegas_f1"].values
    y_f1 = sub["win"].astype(int).values
    # Pick for each fight: True = bet fighter1, False = bet fighter2
    model_pick_f1 = p_m >= 0.5
    vegas_pick_f1 = p_v >= 0.5
    model_edge_f1 = p_m - p_v                     # + = edge on f1
    model_edge_f2 = (1 - p_m) - (1 - p_v)         # + = edge on f2
    # "Edge on the side we'd bet (model pick)":
    edge_on_pick = np.where(model_pick_f1, model_edge_f1, model_edge_f2)
    # Favorite probability (larger of p_vegas_f1, p_vegas_f2):
    p_fav = np.maximum(p_v, 1 - p_v)

    agree = (model_pick_f1 == vegas_pick_f1)

    strategies = {
        "A. All picks (baseline)":            (np.ones(len(sub), dtype=bool),  model_pick_f1),
        "B. AGREE with Vegas":                (agree,                           model_pick_f1),
        "C. DISAGREE with Vegas":             (~agree,                          model_pick_f1),
        "D. +EV (edge > 0)":                  (edge_on_pick > 0,                model_pick_f1),
        "E. Edge >= 5pp":                     (edge_on_pick >= 0.05,            model_pick_f1),
        "F. Edge >= 10pp":                    (edge_on_pick >= 0.10,            model_pick_f1),
        "G. AGREE & p_fav >= 0.65":           (agree & (p_fav >= 0.65),         model_pick_f1),
        "H. DISAGREE & edge on pick >= 5pp":  ((~agree) & (edge_on_pick >= 0.05), model_pick_f1),
    }

    # ─── Evaluate ───────────────────────────────────────────────────
    print("\n" + "="*88)
    print(f"ROI — LR+Elo on {len(sub)} Vegas-matched fights, 2024-05-04 → 2025-11-08")
    print("  Flat $1 bets. Decimal odds from avg book (pre-vig-removal = actual payout).")
    print("="*88)
    hdr = f"{'Strategy':<36s}  {'n':>3s}  {'win%':>5s}  {'ROI':>7s}  {'95% CI':>20s}  {'p':>5s}"
    print(hdr)
    print("-" * len(hdr))

    out_rows = []
    for name, (mask, pick) in strategies.items():
        mask = np.asarray(mask)
        pick = np.asarray(pick)
        n = int(mask.sum())
        if n == 0:
            print(f"{name:<36s}  {n:>3d}  {'-':>5s}  {'-':>7s}  {'-':>20s}  {'-':>5s}")
            out_rows.append(dict(strategy=name, n=0))
            continue
        profits = strategy_profit(sub, mask, pick)
        wr = (profits > 0).mean() * 100
        roi_mean, lo, hi = bootstrap_ci(profits)
        pval = ttest_p(profits)
        print(f"{name:<36s}  {n:>3d}  {wr:>4.1f}%  {roi_mean*100:>+6.2f}%  "
              f"[{lo*100:>+5.2f}%,{hi*100:>+6.2f}%]  {pval:>5.3f}")
        out_rows.append(dict(strategy=name, n=n, win_rate=float(wr/100),
                             roi=float(roi_mean), ci_lo=float(lo), ci_hi=float(hi),
                             perm_p=float(pval)))

    print("\nNotes:")
    print("  • ROI = mean profit per $1 bet. Positive = profitable. Must account for vig (~-5% baseline).")
    print("  • 95% CI from 1000-sample bootstrap. If CI crosses zero, no statistical edge.")
    print("  • p = one-sided t-test vs H0 (ROI=0). p < 0.05 = significant edge.")
    print("  • Thresholds (5pp, 10pp, 0.65) are FROZEN per memory `finding_clv_signal_confirmed.md`.")

    (DT / "roi_results.json").write_text(json.dumps(out_rows, indent=2, default=str))
    print(f"\nSaved to {DT/'roi_results.json'}")


if __name__ == "__main__":
    main()
