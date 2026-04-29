"""Retrain the deploy LR on ALL available data using master_validation CONFIG.

Saves updated lr.pkl / lr_scaler.pkl / lr_imputer.pkl / feat_cols.json to
app/models/blend_v2/  (overwrites the 2024-05-04-cutoff model in place).

Changes from the original build_predictor_v2_artifacts.py::train_final_lr():
  • TEST_FIRST  2024-05-04 → 2026-05-01  (train on all data through Apr 2026)
  • TRAIN_LAM   0.13       → 1.20         (master_validation recency weight)

All other hyperparams match master_validation CONFIG:
  C=0.05, l1_ratio=0.5, threshold=3, TRAIN_START=2016-01-01

Leakage: there is no hold-out — we are training on all data for deployment.
  The model's out-of-sample validity is proven by master_validation.py's
  STRONG PASS result (3/3 folds, p=0.007, +19.03% ROI).
"""
import json, pickle, sqlite3, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "app")

import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

DT  = Path("data/tmp")
OUT = Path("app/models/deploy_v1")   # separate dir — blend_v2 stays intact for web app
OUT.mkdir(parents=True, exist_ok=True)

# ── Master-validation CONFIG (must match master_validation.py) ──────────────
TEST_FIRST        = pd.Timestamp("2026-05-01")  # train on everything before this
TRAIN_START       = pd.Timestamp("2016-01-01")
FILTER_THRESHOLD  = 3
TRAIN_LAM         = 1.20   # recency weight (master_val λ); NOT feature-decay λ=0.13
SCRAPER_DB        = str(elo_feature.DB_PATH)


def retrain():
    print("="*70)
    print("RETRAIN DEPLOY LR — master_validation config, train < 2026-05-01")
    print("="*70)
    print(f"  Cutoff:     {TEST_FIRST.date()}")
    print(f"  Train λ:    {TRAIN_LAM}")
    print(f"  C / l1:     0.05 / 0.5")
    print(f"  Threshold:  {FILTER_THRESHOLD}")

    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])

    from run_threshold_sweep_both_elos import (
        compute_elo_suffixed as _elo_suf,
        build_style_elos as _style_elos,
        apply_threshold, TIER_1C, STYLE_COLS, ELO_COLS_BASE as _ecb,
    )

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
    df = df.merge(res[["jevent", "jbout", "METHOD_norm"]],
                  on=["jevent", "jbout"], how="left")
    unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
    m = df["METHOD_norm"].apply(
        lambda x: any(u in str(x) for u in unwanted) if pd.notna(x) else False)
    df = df[~m].drop_duplicates(
        subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    print("\n  Merging Elos...")
    elo_ufc = _elo_suf("elo_bouts.csv", "ufc")
    elo_exp = _elo_suf("elo_bouts_expanded.csv", "exp")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(elo_ufc, on=["jbout", "DATE"], how="left")
    df = df.rename(columns={"f1": "f1_tmp", "f2": "f2_tmp"})
    df = df.merge(elo_exp[["jbout", "DATE"] + [c for c in elo_exp.columns
                                                if c.endswith("_exp")]],
                  on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] != df["f1_tmp"]
    for suffix in ("ufc", "exp"):
        for c in _ecb:
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
    for c in TIER_1C:
        df[c] = df[c].fillna(0)

    se = _style_elos()
    se["DATE"] = pd.to_datetime(se["DATE"])
    df = df.merge(se, on=["DATE", "jbout", "jfighter"], how="left")
    for c in STYLE_COLS:
        df[c] = df[c].fillna(0.0)

    df = df.drop_duplicates(
        subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    # Filter threshold=3 + train slice (ALL data through April 2026)
    df = apply_threshold(df, FILTER_THRESHOLD)
    df = df[df["DATE"] >= TRAIN_START].copy()
    train = df[df["DATE"] < TEST_FIRST].copy()
    print(f"  Train rows: {len(train):,}  "
          f"({train['DATE'].min().date()} → {train['DATE'].max().date()})")

    feats = [c for c in df.columns if (c.endswith("_diff") or c.endswith("_ufc")
             or c.endswith("_exp") or c in ("weightclass_encoded", "scheduled_rounds",
                                            "days_since_last_fight_f1"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    print(f"  Features:   {len(usable)}")

    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(train[usable])
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    y   = train["win"].astype(int).values

    # Recency weight using master_val LAM=1.20 anchored at TEST_FIRST
    w = np.exp(-TRAIN_LAM * (TEST_FIRST - train["DATE"]).dt.days.values / 365.25)
    print(f"  Recency weight range: [{w.min():.4f}, {w.max():.1f}]")

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xs, y, sample_weight=w)
    n_act = int((np.abs(lr.coef_[0]) > 1e-8).sum())
    print(f"  Active features: {n_act} / {len(usable)}")

    with open(OUT / "lr.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(OUT / "lr_scaler.pkl", "wb") as f:
        pickle.dump(sc, f)
    with open(OUT / "lr_imputer.pkl", "wb") as f:
        pickle.dump(imp, f)
    (OUT / "feat_cols.json").write_text(json.dumps(usable, indent=2))

    # Also copy Elo, mma_history, bios from blend_v2 (those artifacts don't
    # depend on the training cutoff — they include all fight history)
    import shutil
    for fname in ["fighter_mma_history.parquet", "fighter_elo_ufc.json",
                  "fighter_elo_exp.json", "fighter_style_elo.json",
                  "fighter_bios.json", "fighter_recent_form.json",
                  "fighter_wc_history.json", "fighter_winlossko.json"]:
        src = Path("app/models/blend_v2") / fname
        if src.exists():
            shutil.copy2(src, OUT / fname)

    print(f"\n  ✓ Saved lr.pkl / lr_scaler.pkl / lr_imputer.pkl / feat_cols.json")
    print(f"    ✓ Copied per-fighter artifacts from blend_v2 → {OUT}")
    print("\n  Deploy model ready.  Use scripts/predict_card.py for predictions.")


if __name__ == "__main__":
    retrain()
