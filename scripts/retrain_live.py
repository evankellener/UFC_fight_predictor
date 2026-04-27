"""Train production LIVE models on ALL data through TODAY.

The standard `retrain_lr_symmetric.py`, `retrain_lr_013.py`, and
`retrain_lr_150_parlay.py` train through TEST_FIRST (2024-05-04) so that
walk-forward backtests can honestly evaluate the model on the held-out
2024-05 → today fights. That's correct for measuring model quality, but
leaves ~2 years of fights unused at inference time.

This script trains 3 parallel "live" models that use ALL available data
through today (no TEST_FIRST cutoff). Saves artifacts with `_live` suffix
so they don't clobber the validation-cutoff models. PredictorV2 will prefer
the `_live` artifacts at inference if present.

Models trained:
  lr_live.pkl              — main, λ=1.20, 4-yr training ending TODAY
  lr_013_live.pkl          — companion, λ=0.13, 4-yr training ending TODAY
  lr_150_live.pkl          — parlay strategy, λ=1.50, 4-yr training ending TODAY

Each with their own scaler/imputer/calibrator. Calibrator metadata records
the train_through date so PredictorV2 can warn if the live model is stale.

Run weekly (or after each event card) to keep predictions fresh.
"""
import sys, json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features, flip_row_dataframe,
)

OUT = Path("app/models/blend_v2")
TRAIN_YEARS = 4
THRESHOLD = 3
EPS = 1e-6


def fit_temp_cal(p, y):
    p = np.clip(p, EPS, 1 - EPS); logit = np.log(p / (1 - p))
    def nll(T):
        if T <= 0: return 1e9
        pc = 1 / (1 + np.exp(-logit / T))
        pc = np.clip(pc, EPS, 1 - EPS)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()
    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)


def train_one(train, feats, lam, label, save_prefix):
    """Fit one LR + scaler + imputer + calibrator on `train` using λ=lam.
    Saves with given suffix (e.g. 'lr_live.pkl', 'lr_150_live.pkl')."""
    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    # Use the canonical feature list from feat_cols.json (validation-trained)
    # so live + validation models share the same input dimensionality.
    # Without this, the live train data might filter out a few zero-std
    # features and produce a 195-feature model — but PredictorV2 always sends
    # 207 features (from feat_cols.json), which would dimension-mismatch.
    canon = json.loads((OUT / "feat_cols.json").read_text())
    usable = [c for c in canon if c in train_d.columns]
    # Replace any zero-std column with tiny noise (avoids scaler /0 = NaN)
    for c in usable:
        if train_d[c].std() < 1e-12:
            # constant column → leave it; StandardScaler handles 0-variance
            # by treating std as 1, so the column passes through unchanged.
            pass
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    train_anchor = train_d["DATE"].max()  # most recent fight in training set
    w = np.exp(-lam * (train_anchor - train_d["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    # Calibrator on undoubled-train predictions
    p_tr = lr.predict_proba(sc.transform(imp.transform(train[usable])))[:, 1]
    T = fit_temp_cal(p_tr, train["win"].astype(int).values)
    n_active = int((np.abs(lr.coef_[0]) > 1e-8).sum())

    pickle.dump(lr,  open(OUT / f"{save_prefix}.pkl", "wb"))
    pickle.dump(sc,  open(OUT / f"{save_prefix.replace('lr', 'lr_scaler', 1)}.pkl", "wb"))
    pickle.dump(imp, open(OUT / f"{save_prefix.replace('lr', 'lr_imputer', 1)}.pkl", "wb"))
    pickle.dump({
        "method": "temperature", "params": {"T": T},
        "n_train": len(train), "lambda": lam,
        "train_first": str(train["DATE"].min().date()),
        "train_through": str(train["DATE"].max().date()),
        "label": label,
    }, open(OUT / f"{save_prefix.replace('lr', 'calibrator', 1)}.pkl", "wb"))
    print(f"  ✓ {label}  λ={lam}  T={T:.4f}  features={len(usable)} ({n_active} active)")
    return T


def main():
    today = pd.Timestamp.today().normalize()
    train_first = today - pd.DateOffset(years=TRAIN_YEARS)

    print("=" * 76)
    print(f"LIVE retrain — all data through TODAY ({today.date()})")
    print(f"Training window: {train_first.date()} → {today.date()} ({TRAIN_YEARS}yr)")
    print(f"Threshold: ≥{THRESHOLD} prior UFC fights")
    print("=" * 76)

    base = load_base_both_elos()
    df = apply_threshold(base, THRESHOLD)
    df = add_wc_features(df, load_wc_history_from_db())
    train = df[(df["DATE"] >= train_first) & (df["DATE"] <= today)].copy()
    print(f"Train fights (single-orient): {len(train):,}  "
          f"(date range {train['DATE'].min().date()} → {train['DATE'].max().date()})")

    # Same feature selection as production retrain
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

    print()
    train_one(train, feats, lam=1.20, label="MAIN (general predictions)",     save_prefix="lr_live")
    train_one(train, feats, lam=0.13, label="COMPANION (ensemble partner)",   save_prefix="lr_013_live")
    train_one(train, feats, lam=1.50, label="PARLAY (strategy model)",        save_prefix="lr_150_live")

    print()
    print(f"✓ Saved 3 LIVE models to {OUT}/")
    print(f"  lr_live.pkl + lr_scaler_live.pkl + lr_imputer_live.pkl + calibrator_live.pkl")
    print(f"  lr_013_live.pkl + lr_scaler_013_live.pkl + ...")
    print(f"  lr_150_live.pkl + lr_scaler_150_live.pkl + ...")
    print()
    print("Run this script after each UFC event card so the next card's predictions")
    print("incorporate the latest results. Validation-cutoff models (lr.pkl etc.)")
    print("remain unchanged for honest walk-forward measurement.")


if __name__ == "__main__":
    main()
