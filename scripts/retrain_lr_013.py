"""Train the λ=0.13 LR companion model for the production ensemble.

The main production LR (lr.pkl) is trained at λ=1.20 (recent-era bias).
This trains a second LR at λ=0.13 (mild recency weighting — trusts older
data more). PredictorV2 loads both and averages calibrated probabilities.

Saves to app/models/blend_v2/:
  lr_013.pkl            — LR trained at λ=0.13
  lr_scaler_013.pkl     — scaler (same training rows so same scaler, but kept separate for safety)
  lr_imputer_013.pkl    — imputer (same logic)
  calibrator_013.pkl    — temperature calibrator fit on train predictions

Leakage: same guards as retrain_lr_symmetric (strict train/test date split,
imputer/scaler/calibrator fit on train only).
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
    load_wc_history_from_db, add_wc_features, flip_row_dataframe,
)

OUT = Path("app/models/blend_v2")
LAM_COMPANION = 0.13


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
    print(f"Training companion LR at λ={LAM_COMPANION} for ensemble...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    train = df[df["DATE"] < TEST_FIRST].copy()

    # Feature selection — matches retrain_lr_symmetric
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
    w = np.exp(-LAM_COMPANION * (TEST_FIRST - train_d["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    # Fit temperature calibrator on undoubled training predictions
    X_tr_orig = sc.transform(imp.transform(train[usable]))
    p_tr = lr.predict_proba(X_tr_orig)[:, 1]
    T = fit_temp_cal(p_tr, train["win"].astype(int).values)
    print(f"  T (companion calibrator) = {T:.4f}")

    # Save
    pickle.dump(lr,  open(OUT / "lr_013.pkl", "wb"))
    pickle.dump(sc,  open(OUT / "lr_scaler_013.pkl", "wb"))
    pickle.dump(imp, open(OUT / "lr_imputer_013.pkl", "wb"))
    pickle.dump({"method": "temperature", "params": {"T": T}, "n_train": len(train),
                 "lambda": LAM_COMPANION},
                open(OUT / "calibrator_013.pkl", "wb"))

    n_active = int((np.abs(lr.coef_[0]) > 1e-8).sum())
    print(f"  Features: {len(usable)} ({n_active} active after ElasticNet)")
    print(f"\n✓ Saved companion artifacts to {OUT}")


if __name__ == "__main__":
    main()
