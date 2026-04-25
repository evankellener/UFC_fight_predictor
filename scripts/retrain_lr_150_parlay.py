"""Train the λ=1.50 LR for the parlay-strategy-specific predictor.

Background: per λ-sweep across strategies (8 folds × 3-mo, 4-yr training):
  - λ=1.20 (production) is best for STRAIGHT bets
  - λ=1.50 is best for PARLAY-2 edge≥5pp top-2 by edge
    (pooled +36.11% vs +27.25% at λ=1.20; F8 +48% vs +9%)

So we train a SECOND LR at λ=1.50 specifically for the parlay strategy,
saved alongside the production lr.pkl. PredictorV2.parlay_predict() will
use this model; PredictorV2.predict() (the general API) keeps using lr.pkl
at λ=1.20.

Saves to app/models/blend_v2/:
  lr_150.pkl            — LR trained at λ=1.50 (parlay use)
  lr_scaler_150.pkl     — scaler fit on doubled training rows
  lr_imputer_150.pkl    — imputer fit on doubled training rows
  calibrator_150.pkl    — temperature calibrator fit on undoubled train preds

Same leakage guards as retrain_lr_symmetric:
  - imputer/scaler fit on TRAIN ONLY
  - calibrator fit on undoubled train predictions only
  - 4-yr training window, same TRAIN_FIRST → TEST_FIRST as production
  - feature selection identical to retrain_lr_symmetric
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, TEST_FIRST
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features, flip_row_dataframe, TRAIN_FIRST,
)

OUT = Path("app/models/blend_v2")
LAM_PARLAY = 1.50


def fit_temp_cal(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    def nll(T):
        if T <= 0: return 1e9
        pc = 1 / (1 + np.exp(-logit / T))
        pc = np.clip(pc, 1e-6, 1 - 1e-6)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()
    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)


def main():
    print(f"Training parlay-strategy LR at λ={LAM_PARLAY}, {(TEST_FIRST - TRAIN_FIRST).days/365.25:.1f}yr training")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    train = df[(df["DATE"] >= TRAIN_FIRST) & (df["DATE"] < TEST_FIRST)].copy()
    print(f"  Train window: {TRAIN_FIRST.date()} → {TEST_FIRST.date()}  ({len(train)} fights)")

    feats = [c for c in df.columns if (c.endswith("_diff") or c.endswith("_ufc")
             or c.endswith("_exp") or c in ("weightclass_encoded", "scheduled_rounds",
                                             "days_since_last_fight_f1",
                                             "cross_division_flag"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]

    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    w = np.exp(-LAM_PARLAY * (TEST_FIRST - train_d["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    X_tr_orig = sc.transform(imp.transform(train[usable]))
    p_tr = lr.predict_proba(X_tr_orig)[:, 1]
    T = fit_temp_cal(p_tr, train["win"].astype(int).values)
    print(f"  T (parlay calibrator) = {T:.4f}")

    pickle.dump(lr,  open(OUT / "lr_150.pkl", "wb"))
    pickle.dump(sc,  open(OUT / "lr_scaler_150.pkl", "wb"))
    pickle.dump(imp, open(OUT / "lr_imputer_150.pkl", "wb"))
    pickle.dump({"method": "temperature", "params": {"T": T}, "n_train": len(train),
                 "lambda": LAM_PARLAY, "train_years": 4,
                 "use_case": "parlay_strategy_only"},
                open(OUT / "calibrator_150.pkl", "wb"))

    n_active = int((np.abs(lr.coef_[0]) > 1e-8).sum())
    print(f"  Features: {len(usable)} ({n_active} active after ElasticNet)")
    print(f"\n✓ Saved parlay model to {OUT}/lr_150.pkl + companions")


if __name__ == "__main__":
    main()
