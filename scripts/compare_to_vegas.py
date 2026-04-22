"""Compare our models (MMA-AI replica, LR+Elo) to Vegas on the exact same 422 fights.

Uses American odds from ufc_fight_odds, devigs them to get clean market
probabilities, then evaluates Vegas on acc/log-loss/AUC/Brier on the same
subset our models see.

IMPORTANT: Vegas odds are NEVER used as model features (per LEAKAGE_REFERENCE.md §7).
This script only uses them at evaluation time, which is exactly what §7 permits.
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
from catboost import CatBoostClassifier

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


def metrics(y, p):
    """Compute metrics with probability clipping for log loss robustness.
    Applies PROB_CLIP = 0.02 cap (i.e., clip to [0.02, 0.98]) before log loss
    and Brier. This matches standard ML practice (sklearn default eps is 1e-15
    which is pathologically sensitive to bad probabilities).
    """
    p_clipped = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    pred = (p_clipped >= 0.5).astype(int)
    return dict(
        acc=float(accuracy_score(y, pred)),
        ll=float(log_loss(y, p_clipped)),
        auc=float(roc_auc_score(y, p)),  # AUC is ranking-based, no clipping needed
        brier=float(brier_score_loss(y, p_clipped)),
    )


def american_to_prob(odds):
    """Convert American odds to raw (NOT devigged) implied probability.
    0.0 American odds is invalid data from the scraper — return NaN.
    """
    if pd.isna(odds) or odds == 0.0:
        return np.nan
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


# Probability clip for log-loss robustness. Real bookmakers don't price past ±5000
# (implied prob ~0.98). Clipping prevents single bad-data points from destroying
# log loss. Brier is naturally robust and doesn't need clipping.
PROB_CLIP = 0.02


def apply_filter(df):
    conn = sqlite3.connect(SCRAPER_DB)
    hist = pd.read_sql("""
        SELECT w.jfighter, e.DATE FROM ufc_winlossko w
        JOIN ufc_event_details e ON e.jevent = w.jevent
    """, conn)
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fighter_dates = {f: grp["DATE"].values
                     for f, grp in hist.sort_values(["jfighter", "DATE"]).groupby("jfighter")}

    def prior_count(j, d):
        dates = fighter_dates.get(j, np.array([], dtype="datetime64[ns]"))
        return int((dates < np.datetime64(d)).sum()) if len(dates) else 0

    df = df.copy()
    df["f1_priors"] = df.apply(lambda r: prior_count(r["jfighter"], r["DATE"]), axis=1)
    df["f2_priors"] = df.apply(lambda r: prior_count(r["opp_jfighter"], r["DATE"]), axis=1)
    df = df[(df["f1_priors"] >= FILTER_THRESHOLD) & (df["f2_priors"] >= FILTER_THRESHOLD)]

    results = pd.read_sql("SELECT jevent, jbout, METHOD FROM ufc_fight_results", conn)
    results["METHOD_norm"] = results["METHOD"].str.lower().fillna("")
    conn.close()
    df = df.merge(results[["jevent", "jbout", "METHOD_norm"]], on=["jevent", "jbout"], how="left")
    unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
    mask = df["METHOD_norm"].apply(
        lambda m: any(u in str(m) for u in unwanted) if pd.notna(m) else False
    )
    df = df[~mask]
    df = df[df["DATE"] >= TRAIN_START]
    return df.reset_index(drop=True)


def load_vegas(test_df):
    """Load and devig Vegas odds for the test set; align to fighter1."""
    conn = sqlite3.connect(DB)
    odds = pd.read_sql("SELECT * FROM ufc_fight_odds", conn, parse_dates=["DATE"])
    conn.close()
    # Row has (jbout, jfighter=odds-f1, opp_jfighter=odds-f2, avg_odds_f1, avg_odds_f2)
    # Drop rows where either side has 0.0 (scraper bug — invalid American odds).
    bad = (odds["avg_odds_f1"] == 0.0) | (odds["avg_odds_f2"] == 0.0)
    if bad.any():
        print(f"  Dropping {bad.sum()} odds rows with 0.0 American odds (scraper bug)")
        odds = odds[~bad]
    odds["p_raw_f1"] = odds["avg_odds_f1"].apply(american_to_prob)
    odds["p_raw_f2"] = odds["avg_odds_f2"].apply(american_to_prob)
    vig = odds["p_raw_f1"] + odds["p_raw_f2"]
    odds["p_f1_devig"] = odds["p_raw_f1"] / vig
    odds["p_f2_devig"] = odds["p_raw_f2"] / vig

    # Align to test_df's fighter1 (test_df.jfighter)
    # odds row has: jfighter (=their "f1"), opp_jfighter, p_f1_devig, p_f2_devig
    merged = test_df.merge(
        odds[["jbout", "jfighter", "p_f1_devig", "p_f2_devig"]],
        on=["jbout"], how="left", suffixes=("", "_oddsf1"),
    )
    # If test_df.jfighter == odds.jfighter (odds' f1), use p_f1_devig.
    # Else swap to p_f2_devig (odds' f2 matches our f1).
    flip = merged["jfighter"] != merged["jfighter_oddsf1"]
    merged["p_vegas_f1"] = np.where(flip, merged["p_f2_devig"], merged["p_f1_devig"])
    merged = merged.drop(columns=["jfighter_oddsf1", "p_f1_devig", "p_f2_devig"])
    return merged


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


def train_and_predict_lr(train, test, feature_cols, label="win", add_elo=False):
    usable = [c for c in feature_cols if c in train.columns and train[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median")
    X_tr_raw = imp.fit_transform(train[usable])
    X_te_raw = imp.transform(test[usable])
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr_raw)
    X_te = sc.transform(X_te_raw)
    y_tr = train[label].astype(int).values
    w_tr = np.exp(-0.13 * (TEST_START - train["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr, y_tr, sample_weight=w_tr)
    return lr.predict_proba(X_te)[:, 1]


def main():
    print("="*70)
    print("STEP 1: Load + filter + build Elo")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    df = build_elo(df)

    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,}   Test: {len(test):,}")

    # ── STEP 2: Get Vegas for test set ─────────────────────────────
    print("\n" + "="*70)
    print("STEP 2: Load Vegas odds + devig")
    print("="*70)
    test_v = load_vegas(test)
    has_odds = test_v["p_vegas_f1"].notna()
    print(f"  Test fights with Vegas odds: {has_odds.sum()} / {len(test_v)}")
    sub = test_v[has_odds].copy().reset_index(drop=True)
    print(f"  Working subset: {len(sub)} fights")

    # ── STEP 3: Train our models on FULL train, predict on sub ─────
    print("\n" + "="*70)
    print("STEP 3: Train LR (no Elo) and LR (+Elo) on train, predict on subset")
    print("="*70)
    base_feats = [c for c in df.columns if c.endswith("_diff") or c in
                  ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1")
                  and c not in ELO_COLS]
    base_feats = [c for c in base_feats if c not in ELO_COLS]
    elo_feats = base_feats + ELO_COLS

    p_lr_no = train_and_predict_lr(train, sub, base_feats)
    p_lr_el = train_and_predict_lr(train, sub, elo_feats)

    # Also blend: LR+Elo + CatBoost
    y_tr = train["win"].astype(int).values
    w_tr = np.exp(-0.13 * (TEST_START - train["DATE"]).dt.days.values / 365.25)
    usable_cb = [c for c in elo_feats if c in train.columns and train[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median")
    X_tr_cb = imp.fit_transform(train[usable_cb])
    X_te_cb = imp.transform(sub[usable_cb])
    cb = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                            l2_leaf_reg=3.0, subsample=0.8, random_seed=42,
                            verbose=False, bootstrap_type="Bernoulli")
    cb.fit(X_tr_cb, y_tr, sample_weight=w_tr)
    p_cb_el = cb.predict_proba(X_te_cb)[:, 1]
    p_blend_el = 0.5 * p_lr_el + 0.5 * p_cb_el

    y_te = sub["win"].astype(int).values
    p_vegas = sub["p_vegas_f1"].values

    # ── STEP 4: Head-to-head on same subset ────────────────────────
    print("\n" + "="*70)
    print(f"STEP 4: Head-to-head — SAME {len(sub)} fights")
    print("="*70)
    rows = [
        ("Vegas (closing-ish odds, devigged)", p_vegas),
        ("LR (no Elo)",                        p_lr_no),
        ("LR + Elo",                           p_lr_el),
        ("LR+CatBoost+Elo blend",              p_blend_el),
    ]
    print(f"{'Model':<40s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    results = {}
    for name, p in rows:
        m = metrics(y_te, p)
        results[name] = m
        print(f"{name:<40s}  {m['acc']*100:>6.2f}%  {m['ll']:>8.4f}  "
              f"{m['auc']:>7.4f}  {m['brier']:>7.4f}")

    # Deltas vs Vegas
    print("\nΔ vs Vegas (positive acc/AUC/brier-drop = we beat Vegas):")
    v = results["Vegas (closing-ish odds, devigged)"]
    for name, p in rows[1:]:
        m = results[name]
        print(f"  {name:<38s}  Δacc={100*(m['acc']-v['acc']):+.2f}pp  "
              f"Δll={m['ll']-v['ll']:+.4f}  Δauc={m['auc']-v['auc']:+.4f}  "
              f"Δbrier={m['brier']-v['brier']:+.4f}")

    # Agreement / disagreement
    vegas_pick = (p_vegas >= 0.5).astype(int)
    ours_pick = (p_lr_el >= 0.5).astype(int)
    agree = (vegas_pick == ours_pick)
    print(f"\n  Agreement rate (LR+Elo vs Vegas): {agree.mean()*100:.1f}% ({agree.sum()}/{len(sub)})")
    vegas_right = (vegas_pick == y_te)
    ours_right = (ours_pick == y_te)
    both_right = (vegas_right & ours_right).sum()
    both_wrong = ((~vegas_right) & (~ours_right)).sum()
    vegas_only = (vegas_right & ~ours_right).sum()
    ours_only = (~vegas_right & ours_right).sum()
    print(f"  Both right : {both_right}   Both wrong : {both_wrong}")
    print(f"  Vegas only right : {vegas_only}   Our model only right : {ours_only}")
    print(f"  Net disagreement edge: {ours_only - vegas_only} fights in our favor")

    out = {
        "n": int(len(sub)),
        "test_window": [str(TEST_START.date()), str(TEST_END.date())],
        "metrics": {k: v for k, v in results.items()},
        "agreement_rate": float(agree.mean()),
        "both_right": int(both_right),
        "both_wrong": int(both_wrong),
        "vegas_only": int(vegas_only),
        "ours_only": int(ours_only),
    }
    (DT / "vegas_comparison.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {DT/'vegas_comparison.json'}")


if __name__ == "__main__":
    main()
