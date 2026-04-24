"""λ (recency-decay) sweep across the 4 walk-forward folds.

Baseline λ=0.13 means a 7-year-old fight has weight exp(-0.13*7) ≈ 0.40.
Higher λ = more aggressive down-weighting of older fights.

If drift is "older training data drags coefficients toward stale patterns,"
higher λ should help, especially in fold 4. If accuracy is FLAT or
decreasing with λ, recency-weighting isn't the knob — drift is about
feature values themselves, pointing to era-rolling baselines (option B).

Leakage: sample weights are a property of train data only; test never sees
them. §1/§4/§6 compliance identical to walk_forward_4fold.py.
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

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features, flip_row_dataframe,
)
from walk_forward_4fold import FOLDS, select_features, leakage_assertions

LAMBDAS = [0.13, 0.25, 0.40, 0.60, 0.80, 1.20]


def run_one(df, fold, feats, lam):
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
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_doubled[usable]))
    ytr = train_doubled["win"].astype(int).values
    # Recency weight anchored at train_end — only λ varies
    w = np.exp(-lam * (train_end - train_doubled["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    Xte = sc.transform(imp.transform(test[usable]))
    p = lr.predict_proba(Xte)[:, 1]
    yte = test["win"].astype(int).values
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(yte)),
        "accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(yte, pc)),
        "brier":    float(brier_score_loss(yte, pc)),
        "auc":      float(roc_auc_score(yte, p)),
        # Also: effective sample size (sum of weights / max weight)
        "eff_n_pct": float(w.sum() / (len(w) * w.max()) * 100),
    }


def main():
    print("=" * 78)
    print("λ (recency-weight) sweep across 4 walk-forward folds")
    print("=" * 78)

    print("Loading base features + wc_history...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)
    print(f"  rows: {len(df):,}   features: {len(feats)}")

    results = {}  # {fold_name: {lam: metrics}}
    t0 = time.time()
    for fold in FOLDS:
        fold_name = fold["name"]
        results[fold_name] = {}
        print(f"\n── {fold_name}   test: {fold['test_start']} → {fold['test_end']}")
        for lam in LAMBDAS:
            m = run_one(df, fold, feats, lam)
            results[fold_name][lam] = m
            print(f"    λ={lam:<4.2f}  eff_n%={m['eff_n_pct']:>5.1f}%  "
                  f"acc={m['accuracy']*100:>5.2f}%  ll={m['log_loss']:.4f}  "
                  f"auc={m['auc']:.3f}  brier={m['brier']:.4f}")

    # Summary tables
    print("\n" + "=" * 78)
    print("Accuracy grid (rows=folds, cols=λ)")
    print("=" * 78)
    hdr = "fold     " + "  ".join(f"λ={l:<4.2f}" for l in LAMBDAS)
    print(hdr)
    print("-" * len(hdr))
    for fold_name in results:
        row_vals = [f"{results[fold_name][l]['accuracy']*100:>5.2f}%" for l in LAMBDAS]
        print(f"{fold_name}  " + "  ".join(row_vals))

    # Best λ per fold
    print("\n" + "=" * 78)
    print("Best λ per fold (by accuracy, then by log loss as tiebreaker)")
    print("=" * 78)
    best_lam_per_fold = {}
    for fold_name, by_lam in results.items():
        sorted_lams = sorted(by_lam.items(),
                             key=lambda kv: (-kv[1]["accuracy"], kv[1]["log_loss"]))
        best_lam, best_m = sorted_lams[0]
        baseline = by_lam[0.13]
        gain = (best_m["accuracy"] - baseline["accuracy"]) * 100
        best_lam_per_fold[fold_name] = best_lam
        print(f"  {fold_name}  best λ={best_lam:<4.2f}  "
              f"acc={best_m['accuracy']*100:>5.2f}%  "
              f"(baseline λ=0.13: {baseline['accuracy']*100:.2f}%, "
              f"gain: {gain:+.2f}pp)")

    # Aggregate (same λ across all folds — which single λ is best globally?)
    print("\n" + "=" * 78)
    print("Global λ choice (mean accuracy across 4 folds)")
    print("=" * 78)
    for lam in LAMBDAS:
        accs = [results[f][lam]["accuracy"] for f in results]
        lls = [results[f][lam]["log_loss"] for f in results]
        marker = "  ← current" if lam == 0.13 else ""
        print(f"  λ={lam:<4.2f}  mean_acc={np.mean(accs)*100:.2f}%  "
              f"std={np.std(accs)*100:.2f}pp  mean_ll={np.mean(lls):.4f}{marker}")

    # Save
    out = Path("results/lambda_sweep_4fold.json")
    out.write_text(json.dumps({
        "lambdas":  LAMBDAS,
        "folds":    [f["name"] for f in FOLDS],
        "results":  results,
        "best_lam_per_fold": best_lam_per_fold,
        "elapsed_s": time.time() - t0,
    }, indent=2, default=str))
    print(f"\n✓ Saved {out}  (elapsed {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
