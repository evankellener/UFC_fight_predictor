"""Tier 1 + Tier 2 feature ablation on top of MMA-AI replica + Elo.

Baseline: LR + 6 Elo features → 70.38% / 0.5921 / 0.7567 / 0.2024 on 422 fights.

Layers added on top, each as an independent ablation AND stacked cumulatively:

  Tier 1a interactions (computed inline, no external script):
    ix_age_x_elo_diff          = age_diff × precomp_elo_diff
    ix_elo_x_streak_diff       = precomp_elo_diff × win_streak_entering_diff
    ix_elo_x_layoff_diff       = precomp_elo_diff × days_since_last_fight_diff
    ix_elo_x_rounds            = precomp_elo_diff × (scheduled_rounds − 3)

  Tier 1b SoS/form (from scripts/build_sos_form_features.py):
    sos_last3_diff, sos_last5_diff, sos_trajectory_diff,
    form_winrate3_diff, form_winrate5_diff, elo_trajectory_diff,
    career_fights_diff

  Tier 1c recency (from scripts/build_recency_stance.py):
    win_streak_entering_diff, coming_off_loss_diff, fights_last_12m_diff

  Tier 2 stance (from scripts/build_recency_stance.py):
    stance_mismatch, southpaw_advantage_diff

Leakage guardrails (LEAKAGE_REFERENCE.md):
  §1   temporal split, no shuffle; 422 test fights
  §2   all source features already shifted/pre-fight (audited in sub-scripts)
  §3   external feature builders verified d < fight_date
  §4   scaler/imputer fit on train-only
  §5   Elo precomp, params frozen (deployed config)
  §6   LR hyperparams frozen across all ablations
  §10  single run per config
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

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from elo_feature import compute_elo

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"
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
    pred = (p >= 0.5).astype(int)
    return dict(
        acc=float(accuracy_score(y, pred)),
        ll=float(log_loss(y, p)),
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
    )


def apply_filter(df):
    conn = sqlite3.connect(DB)
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


def merge_elo(df):
    bouts = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    ed, *_ = compute_elo(bouts, **ELO_PARAMS)
    keep = ["jbout", "DATE", "f1", "f2"] + ELO_COLS
    ed = ed[keep].copy()
    ed["DATE"] = pd.to_datetime(ed["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(ed, on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] == df["f2"]
    for c in ELO_COLS:
        df.loc[flip, c] = (1 - df.loc[flip, c]) if c == "elo_win_prob" else -df.loc[flip, c]
    df.drop(columns=["f1", "f2"], inplace=True, errors="ignore")
    for c in ELO_COLS:
        df[c] = df[c].fillna(0.5 if c == "elo_win_prob" else 0.0)
    return df


def add_interactions(df):
    """Tier 1a: Elo × context interactions.
    §5 safe — precomp_elo_diff is pre-fight; age/layoff/streak/rounds are all pre-fight.
    """
    df = df.copy()
    # Make sure required inputs exist
    df["ix_age_x_elo_diff"]     = df.get("age_diff", 0) * df["precomp_elo_diff"]
    # win_streak_entering_diff is added in add_recency; here we create a fallback
    # (handled later when recency features are merged)
    df["ix_elo_x_layoff_diff"]  = df["precomp_elo_diff"] * df.get("days_since_last_fight_diff", 0)
    # scheduled_rounds centered at 3 (non-championship baseline)
    df["ix_elo_x_rounds"]       = df["precomp_elo_diff"] * (df.get("scheduled_rounds", 3) - 3)
    return df


def merge_sos_form(df):
    """Tier 1b: SoS + form features."""
    sf = pd.read_csv(DT / "sos_form_features.csv", parse_dates=["DATE"])
    cols = [c for c in sf.columns if c not in ("DATE", "jbout", "jfighter")]
    return df.merge(sf, on=["DATE", "jbout", "jfighter"], how="left")


def merge_recency(df):
    """Tier 1c + Tier 2: recency/streak + stance."""
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    df = df.merge(rs, on=["DATE", "jbout", "jfighter"], how="left")
    # Now that streak is present, add ix_elo_x_streak
    df["ix_elo_x_streak_diff"] = (df["precomp_elo_diff"]
                                   * df["win_streak_entering_diff"].fillna(0))
    return df


def train_eval(train, test, feats, label="win"):
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train[usable])
    X_te = imp.transform(test[usable])
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    y_tr = train[label].astype(int).values
    y_te = test[label].astype(int).values
    w = np.exp(-0.13 * (TEST_START - train["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr, y_tr, sample_weight=w)
    p = lr.predict_proba(X_te)[:, 1]
    return metrics(y_te, p), lr, usable


def main():
    print("="*70)
    print("STEP 1: Load features, apply filter, merge Elo + all feature layers")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    df = merge_elo(df)
    df = add_interactions(df)      # Tier 1a (most interactions)
    df = merge_sos_form(df)        # Tier 1b
    df = merge_recency(df)         # Tier 1c + Tier 2; also adds ix_elo_x_streak_diff

    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,}   Test: {len(test):,}")

    # ── Feature group definitions ───────────────────────────────────
    BASE_MMA = [c for c in df.columns if (c.endswith("_diff") or c in
                ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
                and c not in ELO_COLS
                and not c.startswith("ix_")
                and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                              "form_winrate3_diff", "form_winrate5_diff",
                              "elo_trajectory_diff", "career_fights_diff",
                              "win_streak_entering_diff", "coming_off_loss_diff",
                              "fights_last_12m_diff", "stance_mismatch",
                              "southpaw_advantage_diff")]
    TIER_1A = ["ix_age_x_elo_diff", "ix_elo_x_layoff_diff", "ix_elo_x_rounds",
               "ix_elo_x_streak_diff"]
    TIER_1B = ["sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
               "form_winrate3_diff", "form_winrate5_diff",
               "elo_trajectory_diff", "career_fights_diff"]
    TIER_1C = ["win_streak_entering_diff", "coming_off_loss_diff", "fights_last_12m_diff"]
    TIER_2A = ["stance_mismatch", "southpaw_advantage_diff"]

    print(f"  BASE_MMA features: {len(BASE_MMA)}")
    print(f"  ELO_COLS: {len(ELO_COLS)}   Tier1a: {len(TIER_1A)}   "
          f"Tier1b: {len(TIER_1B)}   Tier1c: {len(TIER_1C)}   Tier2a: {len(TIER_2A)}")

    # ── Configs ─────────────────────────────────────────────────────
    configs = [
        ("1. MMA-AI only (baseline)",        BASE_MMA),
        ("2. + Elo (prior best)",            BASE_MMA + ELO_COLS),
        ("3. + Elo + Tier 1a interactions",  BASE_MMA + ELO_COLS + TIER_1A),
        ("4. + Elo + Tier 1b SoS/form",      BASE_MMA + ELO_COLS + TIER_1B),
        ("5. + Elo + Tier 1c recency",       BASE_MMA + ELO_COLS + TIER_1C),
        ("6. + Elo + ALL Tier 1 (a+b+c)",    BASE_MMA + ELO_COLS + TIER_1A + TIER_1B + TIER_1C),
        ("7. + Elo + Tier 1 + Tier 2 stance", BASE_MMA + ELO_COLS + TIER_1A + TIER_1B + TIER_1C + TIER_2A),
    ]

    print("\n" + "="*94)
    print("Tier 1+2 ablation on MMA-AI + Elo replica (422 test fights)")
    print("="*94)
    print(f"{'Config':<42s}  {'feats':>5s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    print("-" * 94)

    results = []
    baseline_elo_metrics = None
    for name, feats in configs:
        m, lr, used = train_eval(train, test, feats)
        print(f"{name:<42s}  {len(used):>5d}  {m['acc']*100:>6.2f}%  {m['ll']:>8.4f}  "
              f"{m['auc']:>7.4f}  {m['brier']:>7.4f}")
        results.append(dict(name=name, n_feats=len(used), **m))
        if "Elo (prior best)" in name:
            baseline_elo_metrics = m

    # Delta vs Elo-only baseline
    print("\nΔ vs 'MMA-AI + Elo' (row 2) — positive = improvement:")
    bm = baseline_elo_metrics
    for r in results[2:]:
        dacc = (r["acc"] - bm["acc"]) * 100
        dll  = bm["ll"] - r["ll"]
        dauc = r["auc"] - bm["auc"]
        dbr  = bm["brier"] - r["brier"]
        print(f"  {r['name']:<40s}  Δacc={dacc:+5.2f}pp  Δll={dll:+.4f}  "
              f"Δauc={dauc:+.4f}  Δbrier={dbr:+.4f}")

    # Save
    (DT / "tier12_ablation.json").write_text(json.dumps(
        {"configs": results,
         "baseline_elo": bm,
         "feature_counts": {
             "BASE_MMA": len(BASE_MMA), "ELO": len(ELO_COLS),
             "TIER_1A": len(TIER_1A), "TIER_1B": len(TIER_1B),
             "TIER_1C": len(TIER_1C), "TIER_2A": len(TIER_2A),
         }}, indent=2))
    print(f"\nSaved to {DT/'tier12_ablation.json'}")


if __name__ == "__main__":
    main()
