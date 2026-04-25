"""LR+XGB blend vs Vegas at threshold=3 — produces +EV ROI curve.

Trains LR + XGB per fold, blends calibrated probabilities at multiple weights,
matches against Vegas devigged odds, computes +EV ROI per blend weight.
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
from build_walkforward_vegas_multi_threshold import attach_vegas_rich

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.5, min_child_weight=10, gamma=0.5,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, n_jobs=4, tree_method="hist",
)
WEIGHTS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def per_fold_predictions(df, fold, feats):
    train_start = pd.Timestamp(fold["train_start"]); train_end = pd.Timestamp(fold["train_end"])
    test_start  = pd.Timestamp(fold["test_start"]);  test_end  = pd.Timestamp(fold["test_end"])
    train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
    test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
    leakage_assertions(train, test, fold)
    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median"); X_tr = imp.fit_transform(train_d[usable])
    sc = StandardScaler(); Xs_tr = sc.fit_transform(X_tr)
    y_tr = train_d["win"].astype(int).values
    w_tr = np.exp(-LAM * (train_end - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs_tr, y_tr, sample_weight=w_tr)
    xgb = XGBClassifier(**XGB_PARAMS); xgb.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    X_tr_orig = imp.transform(train[usable]); Xs_tr_orig = sc.transform(X_tr_orig)
    p_tr_lr  = lr.predict_proba(Xs_tr_orig)[:, 1]
    p_tr_xgb = xgb.predict_proba(X_tr_orig)[:, 1]
    y_train = train["win"].astype(int).values
    cal_lr  = TempCal(); cal_lr.fit(p_tr_lr, y_train)
    cal_xgb = TempCal(); cal_xgb.fit(p_tr_xgb, y_train)
    X_te = imp.transform(test[usable]); Xs_te = sc.transform(X_te)
    p_lr  = cal_lr.predict( lr.predict_proba(Xs_te)[:, 1])
    p_xgb = cal_xgb.predict(xgb.predict_proba(X_te)[:, 1])
    test = test.copy()
    test["p_lr"]  = p_lr
    test["p_xgb"] = p_xgb
    return test


def vegas_attach(test):
    """Attach Vegas devigged odds (CSV-preferred, DB-fallback) — same as production."""
    keys = test[["DATE", "jbout", "jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    return test.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                          "dec_odds_f1", "dec_odds_f2"]],
                      on=["DATE", "jbout", "jfighter"], how="left")


def compute_roi(matched_df, weight):
    """For given blend weight, compute pooled +EV ROI matching production logic:
    bet only on the model's PICKED side (p>=0.5) and only when EV>0 there."""
    EPS = 1e-6
    p = (1 - weight) * matched_df["p_lr"].values + weight * matched_df["p_xgb"].values
    p = np.clip(p, EPS, 1 - EPS)
    y = matched_df["win"].astype(int).values
    do_f1 = matched_df["dec_odds_f1"].values
    do_f2 = matched_df["dec_odds_f2"].values
    pick_a = p >= 0.5
    dec_pick = np.where(pick_a, do_f1, do_f2)
    mpick    = np.where(pick_a, p, 1 - p)
    won_pick = np.where(pick_a, y == 1, y == 0)
    pnl_flat = np.where(won_pick, dec_pick - 1.0, -1.0)
    ev = mpick * dec_pick - 1.0
    pos_ev = ev > 0
    n_bets = int(pos_ev.sum())
    roi_ev = float(pnl_flat[pos_ev].mean() * 100) if n_bets > 0 else 0.0
    pred = (p >= 0.5).astype(int)
    return dict(weight=weight, n_bets=n_bets, roi_pct=roi_ev,
                roi_flat=float(pnl_flat.mean() * 100),
                acc_model=float(accuracy_score(y, pred)),
                ll_model=float(log_loss(y, p)),
                brier=float(brier_score_loss(y, p)),
                auc=float(roc_auc_score(y, p)))


def main():
    print("Loading...")
    base = load_base_both_elos()
    base = add_wc_features(base, load_wc_history_from_db())
    df = apply_threshold(base, FILTER_THRESHOLD)
    feats = select_features(df)
    print(f"  {len(df)} rows @ t={FILTER_THRESHOLD}, {len(feats)} feature candidates")

    parts = [per_fold_predictions(df, f, feats) for f in FOLDS]
    test_all = pd.concat(parts, ignore_index=True)
    print(f"  Test rows pooled: {len(test_all)}")

    matched = vegas_attach(test_all)
    matched = matched[matched["p_vegas_f1"].notna()].copy()
    print(f"  Matched against Vegas: {len(matched)}")

    print()
    print(f"{'w_xgb':>6} {'n_bets':>7} {'+EV_ROI':>9} {'flat_ROI':>9} {'acc':>8} {'ll':>8} {'auc':>8}")
    print("-" * 70)
    rows = []
    for w in WEIGHTS:
        r = compute_roi(matched, w)
        rows.append(r)
        print(f"{w:>6.2f} {r['n_bets']:>7d} {r['roi_pct']:>+8.2f}% {r['roi_flat']:>+8.2f}% "
              f"{r['acc_model']:>8.4f} {r['ll_model']:>8.4f} {r['auc']:>8.4f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/wf_lr_xgb_vegas_roi.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n✓ Saved results/wf_lr_xgb_vegas_roi.json")


if __name__ == "__main__":
    main()
