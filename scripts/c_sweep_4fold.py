"""ElasticNet C sweep across the 4 walk-forward folds.

Hypothesis: current C=0.05 aggressively shrinks coefficients toward 0,
which causes the LR to hedge probabilities toward 0.5 — visible as a
structural ~0.033 log-loss gap vs Vegas that era-rolling didn't close.
Looser regularization (higher C) might improve calibration without
hurting accuracy by much.

Trade-off: less regularization = more overfit risk. Watch LL on test,
not just training-set LL.

Leakage: each fold still refits imputer, scaler, LR from scratch.
C is a hyperparameter; picking the best C per fold would be overfitting
to test. Report fold-level for diagnostic insight, choose global best
by pooled LL/Brier for ship decision.
"""
import sys, json, time
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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, LAM
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features, flip_row_dataframe,
)
from walk_forward_4fold import FOLDS, select_features, leakage_assertions

# Range centered on current C=0.05, extending ~20x either direction
C_VALUES = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 3.00]


def run_one(df, fold, feats, C_val):
    train_start = pd.Timestamp(fold["train_start"])
    train_end   = pd.Timestamp(fold["train_end"])
    test_start  = pd.Timestamp(fold["test_start"])
    test_end    = pd.Timestamp(fold["test_end"])
    train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
    test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
    leakage_assertions(train, test, fold)

    train_doubled = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    usable = [c for c in feats if c in train_doubled.columns and train_doubled[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_doubled[usable]))
    ytr = train_doubled["win"].astype(int).values
    w = np.exp(-LAM * (train_end - train_doubled["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=C_val, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    Xte = sc.transform(imp.transform(test[usable]))
    p = lr.predict_proba(Xte)[:, 1]
    yte = test["win"].astype(int).values
    pc = np.clip(p, 1e-6, 1 - 1e-6)

    n_active = int((np.abs(lr.coef_[0]) > 1e-8).sum())
    return {
        "n":        int(len(yte)),
        "accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(yte, pc)),
        "brier":    float(brier_score_loss(yte, pc)),
        "auc":      float(roc_auc_score(yte, p)),
        "n_active": n_active,
        "n_total":  len(usable),
    }


def main():
    print("=" * 78)
    print("ElasticNet C sweep — 4 walk-forward folds × 7 C values")
    print("=" * 78)
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)
    print(f"Features: {len(feats)}\n")

    results = {}
    t0 = time.time()
    for fold in FOLDS:
        fn = fold["name"]
        results[fn] = {}
        print(f"── {fn}  test: {fold['test_start']} → {fold['test_end']}")
        for C in C_VALUES:
            m = run_one(df, fold, feats, C)
            results[fn][C] = m
            marker = "  ← current" if C == 0.05 else ""
            print(f"    C={C:<5.2f}  active={m['n_active']:>3d}/{m['n_total']}  "
                  f"acc={m['accuracy']*100:>5.2f}%  ll={m['log_loss']:.4f}  "
                  f"brier={m['brier']:.4f}  auc={m['auc']:.3f}{marker}")

    # Aggregate across folds
    print("\n" + "=" * 78)
    print("Mean metrics across 4 folds (lower LL = better calibration vs Vegas 0.568)")
    print("=" * 78)
    print(f"{'C':<6s} {'active_mean':>11s} {'acc_mean':>9s} {'ll_mean':>8s} {'brier_mean':>10s} {'auc_mean':>9s}")
    print("-" * 70)
    agg = {}
    for C in C_VALUES:
        accs = [results[fn][C]["accuracy"] for fn in results]
        lls  = [results[fn][C]["log_loss"] for fn in results]
        brs  = [results[fn][C]["brier"] for fn in results]
        aucs = [results[fn][C]["auc"] for fn in results]
        acts = [results[fn][C]["n_active"] for fn in results]
        agg[C] = {"acc_mean": np.mean(accs), "ll_mean": np.mean(lls),
                  "brier_mean": np.mean(brs), "auc_mean": np.mean(aucs),
                  "active_mean": np.mean(acts)}
        marker = "  ← current" if C == 0.05 else ""
        print(f"{C:<6.2f} {np.mean(acts):>11.1f} {np.mean(accs)*100:>8.2f}% "
              f"{np.mean(lls):>8.4f} {np.mean(brs):>10.4f} {np.mean(aucs):>9.4f}{marker}")

    # Winner by lowest LL (calibration focus)
    best_by_ll = min(agg.items(), key=lambda kv: kv[1]["ll_mean"])
    best_by_acc = max(agg.items(), key=lambda kv: kv[1]["acc_mean"])
    print(f"\n🏆 Best mean LL:  C={best_by_ll[0]}  ll={best_by_ll[1]['ll_mean']:.4f}")
    print(f"🏆 Best mean acc: C={best_by_acc[0]}  acc={best_by_acc[1]['acc_mean']*100:.2f}%")
    current = agg[0.05]
    print(f"\nCurrent (C=0.05): acc={current['acc_mean']*100:.2f}%  ll={current['ll_mean']:.4f}  brier={current['brier_mean']:.4f}")

    out = Path("results/c_sweep_4fold.json")
    out.write_text(json.dumps({
        "C_values": C_VALUES,
        "results":  {fn: {str(C): v for C, v in rv.items()} for fn, rv in results.items()},
        "aggregate": {str(C): v for C, v in agg.items()},
        "best_by_ll":  {"C": best_by_ll[0], **best_by_ll[1]},
        "best_by_acc": {"C": best_by_acc[0], **best_by_acc[1]},
        "elapsed_s":   time.time() - t0,
    }, indent=2, default=str))
    print(f"\n✓ Saved {out}  (elapsed {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
