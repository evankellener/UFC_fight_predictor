"""LR + XGB walk-forward ensemble — averages calibrated probabilities.

Compares:
  - LR alone (production baseline at λ=1.20)
  - XGB alone (shallow regularized)
  - LR + XGB (50/50 calibrated avg)

Same fold definitions, same feature pipeline, same recency weighting.
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
from sklearn.metrics import (accuracy_score, log_loss,
                             brier_score_loss, roc_auc_score)
from xgboost import XGBClassifier

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, LAM
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features, flip_row_dataframe,
)
from walk_forward_4fold import (
    FOLDS, FILTER_THRESHOLD, select_features, leakage_assertions,
    bucket_metrics, TempCal,
)

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.5, min_child_weight=10, gamma=0.5,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, n_jobs=4, tree_method="hist",
)


def metrics(p, y):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    try: auc = float(roc_auc_score(y, p))
    except ValueError: auc = float("nan")
    bm = bucket_metrics(p, y)
    return dict(n=int(len(y)), acc=float(accuracy_score(y, pred)),
                ll=float(log_loss(y, pc)),
                brier=float(brier_score_loss(y, pc)),
                auc=auc, ece=bm["ece_pp"])


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
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    y_tr = train_d["win"].astype(int).values
    w_tr = np.exp(-LAM * (train_end - train_d["DATE"]).dt.days.values / 365.25)

    # LR
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs_tr, y_tr, sample_weight=w_tr)
    # XGB (no scaling)
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

    # Calibrators on undoubled train
    X_tr_orig = imp.transform(train[usable])
    Xs_tr_orig = sc.transform(X_tr_orig)
    p_tr_lr = lr.predict_proba(Xs_tr_orig)[:, 1]
    p_tr_xgb = xgb.predict_proba(X_tr_orig)[:, 1]
    y_train = train["win"].astype(int).values
    cal_lr = TempCal(); cal_lr.fit(p_tr_lr, y_train)
    cal_xgb = TempCal(); cal_xgb.fit(p_tr_xgb, y_train)

    # Test
    X_te = imp.transform(test[usable])
    Xs_te = sc.transform(X_te)
    p_te_lr_raw = lr.predict_proba(Xs_te)[:, 1]
    p_te_xgb_raw = xgb.predict_proba(X_te)[:, 1]
    p_te_lr = cal_lr.predict(p_te_lr_raw)
    p_te_xgb = cal_xgb.predict(p_te_xgb_raw)
    p_te_ens = 0.5 * p_te_lr + 0.5 * p_te_xgb
    y_test = test["win"].astype(int).values

    return dict(fold=fold["name"],
                lr=metrics(p_te_lr, y_test),
                xgb=metrics(p_te_xgb, y_test),
                ens=metrics(p_te_ens, y_test),
                p_lr=p_te_lr.tolist(), p_xgb=p_te_xgb.tolist(),
                p_ens=p_te_ens.tolist(), y=y_test.tolist())


def main():
    print("Loading...")
    base = load_base_both_elos()
    base = add_wc_features(base, load_wc_history_from_db())
    df = apply_threshold(base, FILTER_THRESHOLD)
    feats = select_features(df)

    res = [run_fold(df, f, feats) for f in FOLDS]
    print("\n" + "=" * 90)
    print(f"{'fold':<8} {'model':<6} {'n':>4} {'acc':>8} {'ll':>8} {'auc':>8} {'brier':>8} {'ece':>7}")
    print("-" * 90)
    for r in res:
        for k in ("lr", "xgb", "ens"):
            m = r[k]
            print(f"{r['fold']:<8} {k:<6} {m['n']:>4} {m['acc']:>8.4f} {m['ll']:>8.4f} "
                  f"{m['auc']:>8.4f} {m['brier']:>8.4f} {m['ece']:>6.2f}pp")

    print("\n" + "=" * 90)
    print("POOLED across folds (concatenate predictions)")
    print("=" * 90)
    for k in ("lr", "xgb", "ens"):
        ps = np.concatenate([np.array(r[f"p_{k}"]) for r in res])
        ys = np.concatenate([np.array(r["y"]) for r in res])
        m = metrics(ps, ys)
        print(f"  {k:<6}  acc={m['acc']:.4f}  ll={m['ll']:.4f}  auc={m['auc']:.4f}  "
              f"brier={m['brier']:.4f}  ece={m['ece']:.2f}pp  n={m['n']}")

    Path("results").mkdir(exist_ok=True)
    with open("results/wf_lr_xgb_ensemble.json", "w") as f:
        json.dump([{k: r[k] for k in ("fold","lr","xgb","ens")} for r in res], f, indent=2)
    print("\n✓ Saved results/wf_lr_xgb_ensemble.json")


if __name__ == "__main__":
    main()
