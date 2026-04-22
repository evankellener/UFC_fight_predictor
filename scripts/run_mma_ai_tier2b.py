"""Tier 2b — adds striking + grappling Elo on top of Tier 1+2a best config.

Extends `run_mma_ai_tier12.py` with the style-specific Elos built by
`build_style_elos.py`. Measures incremental lift over the current best
(LR + Elo + all Tier 1 + Tier 2a stance = 70.62% / 0.5893 / 0.7597 / 0.2012).

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§5):
  - style Elos are precomp (rating BEFORE the fight), confirmed in build_style_elos.py
  - same filter (threshold=3, strict methods), same 422 test fights
  - scaler/imputer fit on train-only (§4)
  - LR hyperparams frozen (§6)
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
STYLE_COLS = ["striking_elo_diff", "grappling_elo_diff"]


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(acc=float(accuracy_score(y, pred)), ll=float(log_loss(y, p)),
                auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)))


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
    m = df["METHOD_norm"].apply(lambda x: any(u in str(x) for u in unwanted)
                                 if pd.notna(x) else False)
    df = df[~m]
    df = df[df["DATE"] >= TRAIN_START]
    return df.reset_index(drop=True)


def merge_elo(df):
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
    return df


def merge_style(df):
    """Tier 2b — style-specific Elos (already per-(DATE,jbout,jfighter), no flip needed)."""
    se = pd.read_csv(DT / "style_elo_features.csv", parse_dates=["DATE"])
    df = df.merge(se, on=["DATE", "jbout", "jfighter"], how="left")
    for c in STYLE_COLS:
        df[c] = df[c].fillna(0.0)
    return df


def train_eval(train, test, feats):
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train[usable]); X_te = imp.transform(test[usable])
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    y_tr = train["win"].astype(int).values; y_te = test["win"].astype(int).values
    w = np.exp(-0.13 * (TEST_START - train["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr, y_tr, sample_weight=w)
    return metrics(y_te, lr.predict_proba(X_te)[:, 1]), len(usable)


def main():
    print("="*70)
    print("STEP 1: Load features, apply filter, merge all layers + style Elos")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    df = merge_elo(df)

    # Tier 1 feature merges
    sf = pd.read_csv(DT / "sos_form_features.csv", parse_dates=["DATE"])
    df = df.merge(sf, on=["DATE", "jbout", "jfighter"], how="left")
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    df = df.merge(rs, on=["DATE", "jbout", "jfighter"], how="left")
    df["ix_age_x_elo_diff"] = df.get("age_diff", 0) * df["precomp_elo_diff"]
    df["ix_elo_x_layoff_diff"] = df["precomp_elo_diff"] * df.get("days_since_last_fight_diff", 0)
    df["ix_elo_x_rounds"] = df["precomp_elo_diff"] * (df.get("scheduled_rounds", 3) - 3)
    df["ix_elo_x_streak_diff"] = df["precomp_elo_diff"] * df["win_streak_entering_diff"].fillna(0)
    df = merge_style(df)

    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,}   Test: {len(test):,}")

    # ── Feature groups ──
    BASE_MMA = [c for c in df.columns if (c.endswith("_diff") or c in
                ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
                and c not in ELO_COLS and c not in STYLE_COLS
                and not c.startswith("ix_")
                and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                              "form_winrate3_diff", "form_winrate5_diff",
                              "elo_trajectory_diff", "career_fights_diff",
                              "win_streak_entering_diff", "coming_off_loss_diff",
                              "fights_last_12m_diff", "stance_mismatch",
                              "southpaw_advantage_diff")]
    TIER_1A = ["ix_age_x_elo_diff", "ix_elo_x_layoff_diff", "ix_elo_x_rounds", "ix_elo_x_streak_diff"]
    TIER_1B = ["sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
               "form_winrate3_diff", "form_winrate5_diff",
               "elo_trajectory_diff", "career_fights_diff"]
    TIER_1C = ["win_streak_entering_diff", "coming_off_loss_diff", "fights_last_12m_diff"]
    TIER_2A = ["stance_mismatch", "southpaw_advantage_diff"]
    TIER_2B = STYLE_COLS

    configs = [
        ("0. MMA-AI only",                            BASE_MMA),
        ("1. + Elo",                                  BASE_MMA + ELO_COLS),
        ("2. + Elo + Tier 1c recency (prior best acc)", BASE_MMA + ELO_COLS + TIER_1C),
        ("3. + Elo + ALL Tier 1 + Tier 2a stance",    BASE_MMA + ELO_COLS + TIER_1A + TIER_1B + TIER_1C + TIER_2A),
        ("4. + Tier 2b STYLE Elo only (to current best)", BASE_MMA + ELO_COLS + TIER_1A + TIER_1B + TIER_1C + TIER_2A + TIER_2B),
        ("5. + Elo + Tier 1c + Tier 2b STYLE",        BASE_MMA + ELO_COLS + TIER_1C + TIER_2B),
        ("6. Minimal: Elo + Tier 1c + style Elo",     BASE_MMA + ELO_COLS + TIER_1C + TIER_2B),  # same as 5, marker
    ]

    print("\n" + "="*96)
    print("Tier 2b ablation — adds striking/grappling Elo on 422 test fights")
    print("="*96)
    print(f"{'Config':<48s}  {'feats':>5s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    print("-" * 96)
    res = []
    for name, feats in configs:
        m, nf = train_eval(train, test, feats)
        print(f"{name:<48s}  {nf:>5d}  {m['acc']*100:>6.2f}%  {m['ll']:>8.4f}  "
              f"{m['auc']:>7.4f}  {m['brier']:>7.4f}")
        res.append(dict(name=name, n_feats=nf, **m))

    # Delta vs config 3 (prior best with all Tier 1 + 2a)
    print("\nΔ vs config 3 (current prior best = Elo + all Tier 1 + Tier 2a stance):")
    base = res[3]
    for r in res[4:5]:
        print(f"  {r['name']:<46s}  Δacc={(r['acc']-base['acc'])*100:+5.2f}pp  "
              f"Δll={base['ll']-r['ll']:+.4f}  Δauc={r['auc']-base['auc']:+.4f}  "
              f"Δbrier={base['brier']-r['brier']:+.4f}")

    (DT / "tier2b_ablation.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved to {DT/'tier2b_ablation.json'}")


if __name__ == "__main__":
    main()
