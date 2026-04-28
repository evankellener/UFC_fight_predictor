"""Re-optimize PG/BB τ hyperparameters for fold 4 only.

Purpose: diagnose whether τ staleness is a meaningful contributor to the
fold_4 accuracy degradation (65.1% vs fold_1's 76.4%, an 11.3pp drop).

Methodology:
- Fold 4 training window: 2018-10-01 → 2025-10-01 (7 years)
- Inner walk-forward: split training into [history, validation] where
  validation = last 12 months of training (2024-10 → 2025-10, 6mo × 2 folds).
  Optuna searches τ to minimize inner-validation log loss.
- Once best τ found, rebuild features, train symmetric LR on full training
  window, evaluate on fold 4's true test window (2025-10 → 2026-04).
- Compare against the baseline fold 4 result (τ held fixed): accuracy 65.08%,
  log loss 0.6639.

LEAKAGE_REFERENCE.md compliance:
- §1: Test window (2025-10 → 2026-04) is NEVER touched during τ search.
  Optuna only sees inner-validation folds from the training window.
- §4: Imputer + scaler re-fit per inner fold on training slice only.
- §6: τ are hyperparameters; tuning on inner validation, evaluating on held-out test.

Cost: ~20-30 min (20 trials × 60-90s feature rebuild each).
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, roc_auc_score

# ── Fold 4 config (matches walk_forward_4fold.py) ───────────────────────
FOLD4_TRAIN_START = pd.Timestamp("2018-10-01")
FOLD4_TRAIN_END   = pd.Timestamp("2025-10-01")
FOLD4_TEST_START  = pd.Timestamp("2025-10-01")
FOLD4_TEST_END    = pd.Timestamp("2026-04-24")

# Inner validation folds (within fold 4's training window) — for τ search
INNER_FOLDS = [
    (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-10-01")),  # 6 months
    (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-04-01")),  # 6 months
    (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-10-01")),  # 6 months
]

# Baseline for comparison (from walk_forward_4fold.py output)
BASELINE = {"accuracy": 0.6508, "log_loss": 0.6639, "auc": 0.6837,
            "brier": 0.2308, "ece_pp": 7.70}

# Search over BB_TAU_GLOBAL (the likely-most-impactful smoothing params)
# and a subset of PG_TAU_GLOBAL. Full τ space is ~30 scalars; we search
# a curated subset known to matter most for the LR's signal.
BB_RANGES = {
    "ko":       (2, 50),
    "win":      (5, 100),
    "decision": (2, 30),
    "sub_land": (2, 20),
    "ctrl":     (1, 10),
}
PG_SEARCH = {
    "sig_str_land": (0.1, 5.0),
    "head_land":    (0.1, 5.0),
    "body_land":    (0.5, 8.0),
    "td_land":      (3.0, 30.0),
    "sub_att":      (5.0, 50.0),
    "kd":           (2.0, 40.0),
}


def rebuild_and_evaluate(pg_tau_overrides, bb_tau_overrides, log_prefix=""):
    """Rebuild features with modified τ, then evaluate on inner folds.
    Returns mean inner-fold log loss (the Optuna objective)."""
    # Import inside to pick up mutated config
    from mma_ai_config import PG_TAU_GLOBAL, BB_TAU_GLOBAL
    PG_TAU_GLOBAL.update(pg_tau_overrides)
    BB_TAU_GLOBAL.update(bb_tau_overrides)

    # Force re-import of the feature pipeline to pick up new τ
    import importlib
    import mma_ai_pipeline
    importlib.reload(mma_ai_pipeline)
    from mma_ai_pipeline import build_features

    # Rebuild features (expensive — ~60-90s)
    df_feats = build_features(config_name="v7")

    # Write to the standard CSV path so downstream loaders pick it up
    feats_path = Path("data/tmp/mmaai_features.csv")
    df_feats.to_csv(feats_path, index=False)

    # Now load the base via the normal pipeline (which reads the CSV we just wrote)
    # Clear any cached module state
    for mod_name in list(sys.modules):
        if mod_name.startswith("run_threshold_sweep_both_elos") or mod_name == "retrain_lr_symmetric":
            del sys.modules[mod_name]

    from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, LAM
    from retrain_lr_symmetric import (
        load_wc_history_from_db, add_wc_features,
        flip_row_dataframe,
    )

    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    from walk_forward_4fold import select_features
    feats = select_features(df)

    # Restrict to fold 4 training window + inner validation
    train_full = df[(df["DATE"] >= FOLD4_TRAIN_START) & (df["DATE"] < FOLD4_TRAIN_END)].copy()

    # Inner eval: train on [train_start, inner_start), validate on [inner_start, inner_end)
    inner_lls = []
    for inner_start, inner_end in INNER_FOLDS:
        inner_train = train_full[train_full["DATE"] < inner_start].copy()
        inner_val   = train_full[(train_full["DATE"] >= inner_start) &
                                  (train_full["DATE"] < inner_end)].copy()
        if len(inner_val) == 0: continue

        inner_train_flipped = flip_row_dataframe(inner_train)
        inner_train_doubled = pd.concat([inner_train, inner_train_flipped], ignore_index=True)
        usable = [c for c in feats if c in inner_train_doubled.columns
                  and inner_train_doubled[c].std() > 1e-8]

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(inner_train_doubled[usable]))
        ytr = inner_train_doubled["win"].astype(int).values
        w   = np.exp(-LAM * (inner_start - inner_train_doubled["DATE"]).dt.days.values / 365.25)

        lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                                solver="saga", max_iter=4000, random_state=42)
        lr.fit(Xtr, ytr, sample_weight=w)

        Xv = sc.transform(imp.transform(inner_val[usable]))
        pv = lr.predict_proba(Xv)[:, 1]
        yv = inner_val["win"].astype(int).values
        inner_lls.append(float(log_loss(yv, np.clip(pv, 1e-6, 1-1e-6))))

    mean_ll = float(np.mean(inner_lls)) if inner_lls else 1e9
    return mean_ll, df, feats


def objective_factory():
    best_state = {"trial": 0}
    t0 = time.time()
    def objective(trial):
        best_state["trial"] += 1
        pg_tau = {k: trial.suggest_float(f"pg_{k}", *rng, log=True)
                  for k, rng in PG_SEARCH.items()}
        bb_tau = {k: trial.suggest_float(f"bb_{k}", *rng, log=True)
                  for k, rng in BB_RANGES.items()}
        try:
            mean_ll, _, _ = rebuild_and_evaluate(pg_tau, bb_tau)
        except Exception as e:
            print(f"  trial {trial.number}: FAILED — {type(e).__name__}: {str(e)[:80]}")
            return 1.0
        elapsed = time.time() - t0
        print(f"  trial {trial.number:>2d}: mean_inner_ll = {mean_ll:.4f}   "
              f"(t={elapsed:.0f}s)")
        return mean_ll
    return objective


def main():
    print("=" * 76)
    print("τ re-optimization for fold 4")
    print("=" * 76)
    print(f"Fold 4 train: {FOLD4_TRAIN_START.date()} → {FOLD4_TRAIN_END.date()}")
    print(f"Fold 4 test:  {FOLD4_TEST_START.date()} → {FOLD4_TEST_END.date()}")
    print(f"Inner validation folds (for τ search): {len(INNER_FOLDS)}")
    print(f"Search space: {len(PG_SEARCH)} PG + {len(BB_RANGES)} BB = "
          f"{len(PG_SEARCH) + len(BB_RANGES)} hyperparameters")

    # Step 1: Baseline — re-evaluate with CURRENT τ to anchor the comparison
    # (This is the value we computed in walk_forward_4fold.py for fold 4.)
    print("\n── Baseline (τ fixed at current mma_ai_config.py values) ──")
    print(f"  accuracy={BASELINE['accuracy']:.4f}  ll={BASELINE['log_loss']:.4f}  "
          f"auc={BASELINE['auc']:.4f}  brier={BASELINE['brier']:.4f}  "
          f"ECE={BASELINE['ece_pp']:.2f}pp")

    # Step 2: Optuna search
    print("\n── Optuna search (20 trials, ~60-90s each — expect 20-30 min) ──")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_factory(), n_trials=20, show_progress_bar=False)

    best = study.best_params
    best_ll = study.best_value
    print(f"\n✓ Best inner-validation log loss: {best_ll:.4f}")
    print(f"✓ Best params:")
    for k, v in sorted(best.items()):
        print(f"    {k} = {v:.4f}")

    # Step 3: Rebuild with best τ + evaluate on the held-out fold 4 test set
    pg_best = {k.removeprefix("pg_"): v for k, v in best.items() if k.startswith("pg_")}
    bb_best = {k.removeprefix("bb_"): v for k, v in best.items() if k.startswith("bb_")}

    print("\n── Rebuilding features with best τ + evaluating on fold 4 TEST ──")
    _, df, feats = rebuild_and_evaluate(pg_best, bb_best)

    # Train on FULL fold 4 training window, evaluate on test
    from run_threshold_sweep_both_elos import LAM
    from retrain_lr_symmetric import flip_row_dataframe

    train = df[(df["DATE"] >= FOLD4_TRAIN_START) & (df["DATE"] < FOLD4_TRAIN_END)].copy()
    test  = df[(df["DATE"] >= FOLD4_TEST_START) & (df["DATE"] < FOLD4_TEST_END)].copy()

    # Symmetric LR
    train_flipped = flip_row_dataframe(train)
    train_doubled = pd.concat([train, train_flipped], ignore_index=True)
    usable = [c for c in feats if c in train_doubled.columns and train_doubled[c].std() > 1e-8]

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_doubled[usable]))
    ytr = train_doubled["win"].astype(int).values
    w = np.exp(-LAM * (FOLD4_TRAIN_END - train_doubled["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    Xte = sc.transform(imp.transform(test[usable]))
    p = lr.predict_proba(Xte)[:, 1]
    yte = test["win"].astype(int).values
    pc = np.clip(p, 1e-6, 1 - 1e-6)

    # Metrics (raw, no calibrator — we want apples-to-apples vs baseline raw)
    retuned = {
        "accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(yte, pc)),
        "auc":      float(roc_auc_score(yte, p)),
        "brier":    float(brier_score_loss(yte, pc)),
        "n_test":   int(len(yte)),
    }

    # Bucket ECE
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5) == (yte == 1)).astype(int)
    ece = 0.0
    for lo in np.arange(0.5, 1.0, 0.05):
        m = (conf >= lo) & (conf < lo + 0.05)
        if m.sum() == 0: continue
        ece += m.sum()/len(yte) * abs(conf[m].mean() - correct[m].mean())
    retuned["ece_pp"] = float(ece * 100)

    print("\n" + "=" * 76)
    print("RESULTS — baseline (fixed τ) vs re-optimized τ on fold 4 TEST SET")
    print("=" * 76)
    print(f"{'metric':<12s}  {'baseline':>10s}  {'re-optim':>10s}  {'delta':>10s}")
    print("-" * 48)
    for name, field in [("accuracy", "accuracy"), ("log_loss", "log_loss"),
                        ("auc", "auc"), ("brier", "brier"), ("ece_pp", "ece_pp")]:
        b = BASELINE[field]
        r = retuned[field]
        d = r - b
        print(f"  {name:<10s}  {b:>10.4f}  {r:>10.4f}  {d:>+10.4f}")

    # Save results
    out = Path("results/tau_reoptimize_fold4.json")
    out.write_text(json.dumps({
        "baseline": BASELINE,
        "retuned":  retuned,
        "delta":    {k: retuned[k] - BASELINE[k] for k in retuned if k in BASELINE},
        "best_params": best,
        "best_inner_ll": best_ll,
        "n_trials": 20,
        "methodology": "Optuna TPE, 20 trials, 6 PG + 5 BB τ values searched",
        "fold4_test_range": [str(FOLD4_TEST_START.date()), str(FOLD4_TEST_END.date())],
    }, indent=2, default=str))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
