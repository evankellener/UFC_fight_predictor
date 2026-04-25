"""Sweep LR+XGB blend weights using fresh walk-forward predictions."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, LAM
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features, flip_row_dataframe,
)
from walk_forward_4fold import (
    FOLDS, FILTER_THRESHOLD, select_features, leakage_assertions, TempCal,
)

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.5, min_child_weight=10, gamma=0.5,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, n_jobs=4, tree_method="hist",
)


def run_fold(df, fold, feats):
    train_start = pd.Timestamp(fold["train_start"])
    train_end   = pd.Timestamp(fold["train_end"])
    test_start  = pd.Timestamp(fold["test_start"])
    test_end    = pd.Timestamp(fold["test_end"])
    train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
    test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
    leakage_assertions(train, test, fold)
    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train_d[usable])
    sc = StandardScaler(); Xs_tr = sc.fit_transform(X_tr)
    y_tr = train_d["win"].astype(int).values
    w_tr = np.exp(-LAM * (train_end - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs_tr, y_tr, sample_weight=w_tr)
    xgb = XGBClassifier(**XGB_PARAMS); xgb.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    X_tr_orig = imp.transform(train[usable]); Xs_tr_orig = sc.transform(X_tr_orig)
    p_tr_lr = lr.predict_proba(Xs_tr_orig)[:, 1]
    p_tr_xgb = xgb.predict_proba(X_tr_orig)[:, 1]
    y_train = train["win"].astype(int).values
    cal_lr = TempCal(); cal_lr.fit(p_tr_lr, y_train)
    cal_xgb = TempCal(); cal_xgb.fit(p_tr_xgb, y_train)
    X_te = imp.transform(test[usable]); Xs_te = sc.transform(X_te)
    p_lr  = cal_lr.predict( lr.predict_proba(Xs_te)[:, 1])
    p_xgb = cal_xgb.predict(xgb.predict_proba(X_te)[:, 1])
    y = test["win"].astype(int).values
    return p_lr, p_xgb, y


def metrics(p, y):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    try: auc = float(roc_auc_score(y, p))
    except ValueError: auc = float("nan")
    return dict(acc=accuracy_score(y, pred), ll=log_loss(y, pc),
                brier=brier_score_loss(y, pc), auc=auc, n=len(y))


def main():
    print("Loading...")
    base = load_base_both_elos()
    base = add_wc_features(base, load_wc_history_from_db())
    df = apply_threshold(base, FILTER_THRESHOLD)
    feats = select_features(df)
    print(f"  {len(df)} rows, {len(feats)} feature candidates")

    parts = [run_fold(df, f, feats) for f in FOLDS]
    p_lr_all  = np.concatenate([p[0] for p in parts])
    p_xgb_all = np.concatenate([p[1] for p in parts])
    y_all     = np.concatenate([p[2] for p in parts])
    print(f"  Pooled n={len(y_all)}")

    print()
    print(f"{'w_xgb':>6} {'acc':>8} {'ll':>8} {'auc':>8} {'brier':>8}")
    print("-" * 50)
    for w in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        p = (1 - w) * p_lr_all + w * p_xgb_all
        m = metrics(p, y_all)
        print(f"{w:>6.2f} {m['acc']:>8.4f} {m['ll']:>8.4f} {m['auc']:>8.4f} {m['brier']:>8.4f}")

    Path("results").mkdir(exist_ok=True)
    np.savez("results/wf_lr_xgb_predictions.npz",
             p_lr=p_lr_all, p_xgb=p_xgb_all, y=y_all)
    print("\n✓ Saved results/wf_lr_xgb_predictions.npz")


if __name__ == "__main__":
    main()
