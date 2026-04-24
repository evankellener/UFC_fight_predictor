"""Sweep era-rolling window size across walk-forward folds.

Currently era-rolling WC baselines use a 2-year window (manually chosen).
This tests 1yr / 2yr / 3yr / 4yr to see if a different window gives better
walk-forward metrics. Requires rebuilding features per window (~60s each).

If any window beats 2yr on pooled LL by >0.005 AND doesn't hurt accuracy,
ship it. Otherwise document.
"""
import sys, json, time, importlib, shutil
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

WINDOWS = [365, 730, 1095, 1460]   # 1yr, 2yr, 3yr, 4yr
BACKUP = Path("data/tmp/mmaai_features.csv.pre_window_sweep")


def rebuild_features_with_window(window_days):
    # Patch ERA_WINDOW_DAYS in mma_ai_pipeline + rebuild
    import mma_ai_pipeline
    mma_ai_pipeline.ERA_WINDOW_DAYS = window_days
    importlib.reload(mma_ai_pipeline)
    mma_ai_pipeline.ERA_WINDOW_DAYS = window_days  # reapply after reload
    from mma_ai_pipeline import build_features
    df = build_features(config_name="v7")
    df.to_csv("data/tmp/mmaai_features.csv", index=False)
    # Also refresh the downstream loaders
    for mod in list(sys.modules):
        if mod in ("run_threshold_sweep_both_elos", "retrain_lr_symmetric", "walk_forward_4fold"):
            del sys.modules[mod]


def evaluate_walk_forward():
    """Run 4-fold walk-forward at the current mmaai_features.csv, return
    pooled metrics (mean across folds)."""
    from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, LAM
    from retrain_lr_symmetric import (load_wc_history_from_db, add_wc_features,
                                        flip_row_dataframe)
    from walk_forward_4fold import FOLDS, select_features, leakage_assertions

    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)

    per_fold = []
    for fold in FOLDS:
        train_start = pd.Timestamp(fold["train_start"])
        train_end   = pd.Timestamp(fold["train_end"])
        test_start  = pd.Timestamp(fold["test_start"])
        test_end    = pd.Timestamp(fold["test_end"])
        train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
        test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
        leakage_assertions(train, test, fold)

        train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
        usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]

        imp = SimpleImputer(strategy="median"); sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
        ytr = train_d["win"].astype(int).values
        w = np.exp(-LAM * (train_end - train_d["DATE"]).dt.days.values / 365.25)
        lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                                solver="saga", max_iter=6000, random_state=42)
        lr.fit(Xtr, ytr, sample_weight=w)

        Xte = sc.transform(imp.transform(test[usable]))
        p = lr.predict_proba(Xte)[:, 1]
        yte = test["win"].astype(int).values
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        per_fold.append({
            "fold": fold["name"],
            "n":    int(len(yte)),
            "accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int))),
            "log_loss": float(log_loss(yte, pc)),
            "brier":    float(brier_score_loss(yte, pc)),
            "auc":      float(roc_auc_score(yte, p)),
        })
    return per_fold


def main():
    print("=" * 72)
    print("Era-rolling window-size sweep — 4 folds × 4 windows")
    print("=" * 72)
    # Back up current features so we can restore 2yr state at the end
    if not BACKUP.exists():
        shutil.copy("data/tmp/mmaai_features.csv", BACKUP)
        print(f"Backed up current features to {BACKUP}")

    results = {}
    t0 = time.time()
    for w in WINDOWS:
        print(f"\n── Window = {w} days ({w/365.25:.1f} years)")
        rebuild_features_with_window(w)
        folds = evaluate_walk_forward()
        accs = [f["accuracy"] for f in folds]
        lls  = [f["log_loss"] for f in folds]
        brs  = [f["brier"] for f in folds]
        results[str(w)] = {
            "per_fold":   folds,
            "acc_mean":   float(np.mean(accs)),
            "acc_std":    float(np.std(accs)),
            "ll_mean":    float(np.mean(lls)),
            "brier_mean": float(np.mean(brs)),
        }
        print(f"  acc_mean={np.mean(accs)*100:.2f}% (std {np.std(accs)*100:.2f}pp)  "
              f"ll_mean={np.mean(lls):.4f}  brier_mean={np.mean(brs):.4f}")

    # Restore the 2yr features (production setting)
    print(f"\nRestoring 2yr window features from backup...")
    rebuild_features_with_window(730)

    # Report
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'window':<10s} {'acc_mean':>9s} {'acc_std':>9s} {'ll_mean':>8s} {'brier':>8s}")
    print("-" * 55)
    for w in WINDOWS:
        r = results[str(w)]
        marker = "  ← current" if w == 730 else ""
        print(f"{w}d ({w//365}yr){'':<2s} {r['acc_mean']*100:>8.2f}% "
              f"{r['acc_std']*100:>8.2f}pp {r['ll_mean']:>8.4f} {r['brier_mean']:>8.4f}{marker}")

    out = Path("results/window_sweep_4fold.json")
    out.write_text(json.dumps({"windows_days": WINDOWS, "results": results,
                                "elapsed_s": time.time() - t0}, indent=2))
    print(f"\n✓ Saved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
