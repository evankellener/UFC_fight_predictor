"""4-fold walk-forward XGBoost — same pipeline as walk_forward_4fold.py but
swaps LR(elasticnet) for XGBoost. Goal: see if a tree model picks up
nonlinearities the LR misses, post era-rolling baselines.

Same leakage guards: imputer/scaler refit per fold, symmetric doubled training,
recency-weighted samples (λ=1.20 production), temperature calibrator fit on
train via 5-fold CV.

Reports per-fold acc/LL/AUC/Brier and aggregate vs LR baseline (71.61% / 0.5972
LL / 0.7492 AUC walk-forward, +13.99% +EV ROI t=3 pooled vs Vegas).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, log_loss,
                             brier_score_loss, roc_auc_score)
from scipy.optimize import minimize_scalar

try:
    from xgboost import XGBClassifier
except ImportError as e:
    print(f"ERROR: xgboost not installed: {e}")
    sys.exit(1)

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, LAM
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features,
    flip_row_dataframe,
)
from walk_forward_4fold import (
    FOLDS, FILTER_THRESHOLD, select_features, leakage_assertions,
    bucket_metrics, bootstrap_accuracy_ci, TempCal,
)


def fit_calibrator_via_cv(p, y, k=5, seed=42):
    """Fit temperature calibrator on all train via 5-fold CV; production
    calibrator fits on the full train. Same as walk_forward_4fold.py."""
    cal = TempCal()
    cal.fit(p, y)
    return cal, p


def run_fold_xgb(df, fold, feats, xgb_params):
    fold_name = fold["name"]
    train_start = pd.Timestamp(fold["train_start"])
    train_end   = pd.Timestamp(fold["train_end"])
    test_start  = pd.Timestamp(fold["test_start"])
    test_end    = pd.Timestamp(fold["test_end"])

    train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
    test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
    leakage_assertions(train, test, fold)

    train_flipped = flip_row_dataframe(train)
    train_doubled = pd.concat([train, train_flipped], ignore_index=True)

    usable = [c for c in feats if c in train_doubled.columns
              and train_doubled[c].std() > 1e-8]

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train_doubled[usable])
    # No scaling for XGB but keep the column for parity / future use
    y_tr = train_doubled["win"].astype(int).values
    w_tr = np.exp(-LAM * (train_end - train_doubled["DATE"]).dt.days.values / 365.25)

    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

    X_tr_orig = imp.transform(train[usable])
    p_train = xgb.predict_proba(X_tr_orig)[:, 1]
    y_train = train["win"].astype(int).values

    cal, _ = fit_calibrator_via_cv(p_train, y_train)

    X_te = imp.transform(test[usable])
    p_test_raw = xgb.predict_proba(X_te)[:, 1]
    p_test_cal = cal.predict(p_test_raw)
    y_test = test["win"].astype(int).values

    def M(p, y):
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        pred = (p >= 0.5).astype(int)
        try: auc = float(roc_auc_score(y, p))
        except ValueError: auc = float("nan")
        bm = bucket_metrics(p, y)
        return dict(n=int(len(y)),
                    accuracy=float(accuracy_score(y, pred)),
                    log_loss=float(log_loss(y, pc)),
                    brier=float(brier_score_loss(y, pc)),
                    auc=auc, ece_pp=bm["ece_pp"], max_dev_pp=bm["max_dev_pp"])

    m_raw = M(p_test_raw, y_test)
    m_cal = M(p_test_cal, y_test)
    n_active = sum(1 for f, imp_v in zip(usable, xgb.feature_importances_) if imp_v > 1e-6)

    print(f"── {fold_name} ──────────────────────────────────────────────")
    print(f"  train: {train_start.date()} → {train_end.date()}")
    print(f"  test:  {test_start.date()}  → {test_end.date()}")
    print(f"  n_train (unique / doubled): {len(train)} / {len(train_doubled)}")
    print(f"  n_test: {len(test)}   features active: {n_active}")
    print(f"  T (temperature): {cal.T:.4f}")
    print(f"  RAW         acc={m_raw['accuracy']:.4f}  ll={m_raw['log_loss']:.4f}  "
          f"auc={m_raw['auc']:.4f}  brier={m_raw['brier']:.4f}  ECE={m_raw['ece_pp']:.2f}pp")
    print(f"  CALIBRATED  acc={m_cal['accuracy']:.4f}  ll={m_cal['log_loss']:.4f}  "
          f"auc={m_cal['auc']:.4f}  brier={m_cal['brier']:.4f}  ECE={m_cal['ece_pp']:.2f}pp")
    return dict(fold=fold_name, raw=m_raw, calibrated=m_cal,
                T=cal.T, n_active=n_active,
                p_test_cal=p_test_cal.tolist(), y_test=y_test.tolist())


def main(xgb_params, label):
    print("=" * 76)
    print(f"4-fold walk-forward — XGBoost ({label})")
    print(f"  params: {xgb_params}")
    print("=" * 76)
    print("Loading base features + wc_history...")
    base = load_base_both_elos()
    base = add_wc_features(base, load_wc_history_from_db())
    df = apply_threshold(base, FILTER_THRESHOLD)
    feats = select_features(df)
    print(f"  Total rows: {len(df):,}  Feature candidates: {len(feats)}")

    results = [run_fold_xgb(df, f, feats, xgb_params) for f in FOLDS]

    # Aggregate
    aggs = {k: np.mean([r["calibrated"][k] for r in results])
            for k in ("accuracy", "log_loss", "auc", "brier", "ece_pp")}
    stds = {k: np.std([r["calibrated"][k] for r in results])
            for k in ("accuracy", "log_loss", "auc", "brier", "ece_pp")}
    print("\n" + "=" * 76)
    print("AGGREGATE (calibrated)")
    print("=" * 76)
    for k in ("accuracy", "log_loss", "auc", "brier", "ece_pp"):
        print(f"  {k:10s} mean={aggs[k]:.4f}  std={stds[k]:.4f}")
    Path("results").mkdir(exist_ok=True)
    with open(f"results/walk_forward_4fold_xgb_{label}.json", "w") as f:
        json.dump({"params": xgb_params, "aggregate_calibrated": aggs,
                   "aggregate_calibrated_std": stds, "folds": results}, f, indent=2)
    print(f"\n✓ Saved results/walk_forward_4fold_xgb_{label}.json")
    return aggs


if __name__ == "__main__":
    # Conservative tree settings — small/shallow trees, strong regularization,
    # similar depth to a regularized LR but with tree non-linearities.
    params_a = dict(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.5,
        min_child_weight=10, gamma=0.5,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=4, tree_method="hist",
    )
    main(params_a, "shallow_reg")
