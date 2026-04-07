"""
Tau Optimizer — Bayesian optimization of PG/BB smoothing parameters + LR hyperparams.

Uses walk-forward validation (NOT the test set) to avoid data leakage.

Validation: 2023-01-01 to 2024-05-01 (held out from training)
Test set:   2024-05-01+ (never seen during optimization)

Usage:
    python3 optimize_taus.py                # 100 trials
    python3 optimize_taus.py --trials 30    # Quick run
    python3 optimize_taus.py --resume       # Resume from saved study

Saves results incrementally to data/tmp/tau_optimized.json.
"""

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import functools
print = functools.partial(print, flush=True)

RESULTS_PATH = Path(__file__).parent.parent / "data" / "tmp" / "tau_optimized.json"
STUDY_DB = f"sqlite:///{Path(__file__).parent.parent / 'data' / 'tmp' / 'tau_optuna_study_v2.db'}"

# Walk-forward validation folds — train on everything BEFORE train_end,
# evaluate on [eval_start, eval_end). Test set (2024-05+) is never touched.
FOLDS = [
    ("2020-01-01", "2020-01-01", "2021-01-01"),
    ("2021-01-01", "2021-01-01", "2022-01-01"),
    ("2022-01-01", "2022-01-01", "2023-01-01"),
    ("2023-01-01", "2023-01-01", "2024-01-01"),
    ("2024-01-01", "2024-01-01", "2024-05-01"),
]

# Fitness weights (same as optimize_pipeline.py)
W_LL  = 0.35
W_BS  = 0.30
W_AUC = 0.25
W_ACC = 0.10

# Current best taus (baseline to beat)
BASELINE_PG = {
    "sig_str_land": 2.8, "sig_str_att": 0.175,
    "head_land": 0.8, "head_att": 3.2,
    "body_land": 1.25, "body_att": 10.0,
    "leg_land": 4.2, "leg_att": 1.05,
    "td_land": 28.0, "td_att": 1.75,
    "sub_att": 48.0, "rev": 168.0,
    "dist_land": 1.4, "dist_att": 1.4,
    "clinch_land": 10.0, "clinch_att": 0.625,
    "ground_land": 0.625, "ground_att": 10.0,
    "kd": 10.0,
}
BASELINE_BB = {"ko": 7, "win": 75, "decision": 6, "sub_land": 9, "ctrl": 2}

# Search ranges (log-uniform for most, centered around current values)
PG_RANGES = {
    "sig_str_land": (0.1, 10.0), "sig_str_att": (0.05, 5.0),
    "head_land": (0.1, 5.0), "head_att": (0.1, 10.0),
    "body_land": (0.3, 8.0), "body_att": (1.0, 30.0),
    "leg_land": (0.5, 15.0), "leg_att": (0.2, 5.0),
    "td_land": (3.0, 60.0), "td_att": (0.3, 10.0),
    "sub_att": (5.0, 100.0), "rev": (20.0, 300.0),
    "dist_land": (0.2, 8.0), "dist_att": (0.2, 8.0),
    "clinch_land": (1.0, 30.0), "clinch_att": (0.1, 5.0),
    "ground_land": (0.1, 5.0), "ground_att": (1.0, 30.0),
    "kd": (2.0, 40.0),
}
BB_RANGES = {
    "ko": (2, 30), "win": (10, 100), "decision": (2, 30),
    "sub_land": (2, 25), "ctrl": (1, 10),
}


def build_and_evaluate(pg_tau, bb_tau, lr_C=0.1, lr_l1=0.4, recency_lambda=0.347):
    """Build pipeline with given taus, evaluate via walk-forward CV on validation folds."""
    from mma_ai_pipeline import build_features
    from mma_ai_config import PG_TAU_GLOBAL, BB_TAU_GLOBAL

    # Set taus
    PG_TAU_GLOBAL.update(pg_tau)
    BB_TAU_GLOBAL.update(bb_tau)

    # Build features (suppressed output)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features("jan26")

    feat_cols = [c for c in df.columns if c.endswith("_diff")]
    for c in ["weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"]:
        if c in df.columns:
            feat_cols.append(c)

    for c in feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100)

    # Walk-forward CV
    fold_scores = []
    for train_end, eval_start, eval_end in FOLDS:
        train = df[df["DATE"] < train_end]
        val = df[(df["DATE"] >= eval_start) & (df["DATE"] < eval_end)]

        if len(train) < 100 or len(val) < 20:
            continue

        max_date = pd.to_datetime(train_end)
        days_before = (max_date - train["DATE"]).dt.days.clip(lower=0)
        weights = np.exp(-recency_lambda * days_before / 365.25)
        weights = weights / weights.mean()

        imp = SimpleImputer(strategy="median")
        Xtr = pd.DataFrame(imp.fit_transform(train[feat_cols]), columns=feat_cols)
        Xval = pd.DataFrame(imp.transform(val[feat_cols]), columns=feat_cols)

        pipe = Pipeline([
            ("sc", RobustScaler()),
            ("clf", LogisticRegression(C=lr_C, solver="saga", max_iter=3000,
                                       penalty="elasticnet", l1_ratio=lr_l1,
                                       random_state=42)),
        ])
        pipe.fit(Xtr, train["win"].values, clf__sample_weight=weights.values)
        proba = pipe.predict_proba(Xval)[:, 1]
        y = val["win"].values

        fold_scores.append({
            "log_loss": log_loss(y, proba),
            "brier": brier_score_loss(y, proba),
            "auc": roc_auc_score(y, proba),
            "accuracy": accuracy_score(y, (proba >= 0.5).astype(int)),
        })

    if not fold_scores:
        return -999.0, {}

    avg = {k: float(np.mean([s[k] for s in fold_scores])) for k in fold_scores[0]}
    fitness = -W_LL * avg["log_loss"] - W_BS * avg["brier"] + W_AUC * avg["auc"] + W_ACC * avg["accuracy"]
    return float(fitness), avg


def objective(trial):
    """Optuna objective: maximize walk-forward fitness."""
    pg_tau = {}
    for stat, (lo, hi) in PG_RANGES.items():
        pg_tau[stat] = trial.suggest_float(f"pg_{stat}", lo, hi, log=True)

    bb_tau = {}
    for stat, (lo, hi) in BB_RANGES.items():
        bb_tau[stat] = trial.suggest_int(f"bb_{stat}", lo, hi)

    # Also optimize LR hyperparams jointly
    lr_C = trial.suggest_float("lr_C", 0.01, 1.0, log=True)
    lr_l1 = trial.suggest_float("lr_l1", 0.1, 0.9)
    recency_lambda = trial.suggest_float("recency_lambda", 0.05, 0.8)

    fitness, avg = build_and_evaluate(pg_tau, bb_tau, lr_C, lr_l1, recency_lambda)

    for k, v in avg.items():
        trial.set_user_attr(k, v)

    return fitness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("Installing optuna...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
        import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Baseline
    print("Running baseline (walk-forward CV)...")
    t0 = time.time()
    base_fitness, base_avg = build_and_evaluate(BASELINE_PG, BASELINE_BB)
    base_time = time.time() - t0
    print(f"  Baseline fitness: {base_fitness:.4f}")
    print(f"    LL={base_avg['log_loss']:.4f}  Acc={base_avg['accuracy']:.1%}  "
          f"AUC={base_avg['auc']:.4f}  Brier={base_avg['brier']:.4f}")
    print(f"  ({base_time:.0f}s per trial)")
    print(f"  Estimated time: ~{base_time * args.trials / 60:.0f} min for {args.trials} trials")

    # Create/load study
    study = optuna.create_study(
        study_name="tau_optimization_v2",
        storage=STUDY_DB,
        direction="maximize",  # maximize fitness
        load_if_exists=args.resume,
    )

    # Seed with baseline params
    if len(study.trials) == 0:
        baseline_params = {f"pg_{k}": v for k, v in BASELINE_PG.items()}
        baseline_params.update({f"bb_{k}": v for k, v in BASELINE_BB.items()})
        baseline_params["lr_C"] = 0.1
        baseline_params["lr_l1"] = 0.4
        baseline_params["recency_lambda"] = 0.347
        study.enqueue_trial(baseline_params)

    def callback(study, trial):
        if trial.value >= study.best_value - 0.0001:
            # Save best
            best = study.best_trial
            pg = {k.replace("pg_", ""): best.params[k]
                  for k in best.params if k.startswith("pg_")}
            bb = {k.replace("bb_", ""): int(best.params[k])
                  for k in best.params if k.startswith("bb_")}
            save_data = {
                "pg_tau": pg,
                "bb_tau": bb,
                "lr_C": best.params["lr_C"],
                "lr_l1": best.params["lr_l1"],
                "recency_lambda": best.params["recency_lambda"],
                "fitness": best.value,
                "avg_log_loss": best.user_attrs.get("log_loss", 0),
                "avg_accuracy": best.user_attrs.get("accuracy", 0),
                "avg_auc": best.user_attrs.get("auc", 0),
                "avg_brier": best.user_attrs.get("brier", 0),
                "trial": best.number,
                "total_trials": len(study.trials),
                "method": "walk_forward_cv_v2",
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(save_data, f, indent=2)

        n = len(study.trials)
        attrs = trial.user_attrs
        best_f = study.best_value
        marker = " *" if trial.value >= best_f - 0.0001 else ""
        print(f"  [{n}/{args.trials}] fitness={trial.value:.4f} "
              f"LL={attrs.get('log_loss', 0):.4f} Acc={attrs.get('accuracy', 0):.1%} "
              f"AUC={attrs.get('auc', 0):.4f} (best={best_f:.4f}){marker}")

    print(f"\nStarting optimization ({args.trials} trials, walk-forward CV)...")
    study.optimize(objective, n_trials=args.trials, callbacks=[callback])

    # Final report
    best = study.best_trial
    pg_best = {k.replace("pg_", ""): best.params[k]
               for k in best.params if k.startswith("pg_")}
    bb_best = {k.replace("bb_", ""): int(best.params[k])
               for k in best.params if k.startswith("bb_")}

    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION COMPLETE — {len(study.trials)} trials")
    print(f"{'='*60}")
    print(f"  Baseline fitness: {base_fitness:.4f}")
    print(f"    LL={base_avg['log_loss']:.4f}  Acc={base_avg['accuracy']:.1%}")
    print(f"  Best fitness:     {best.value:.4f}")
    print(f"    LL={best.user_attrs.get('log_loss', 0):.4f}  "
          f"Acc={best.user_attrs.get('accuracy', 0):.1%}  "
          f"AUC={best.user_attrs.get('auc', 0):.4f}")
    print(f"  LR params: C={best.params['lr_C']:.4f}  "
          f"l1={best.params['lr_l1']:.3f}  decay={best.params['recency_lambda']:.3f}")
    print(f"\n  Best PG taus:")
    for k, v in sorted(pg_best.items()):
        old = BASELINE_PG.get(k, 0)
        change = f" ({v/old:.1f}x)" if old > 0 else ""
        print(f"    {k:<20} {v:>8.3f}{change}")
    print(f"\n  Best BB taus:")
    for k, v in sorted(bb_best.items()):
        old = BASELINE_BB.get(k, 0)
        print(f"    {k:<20} {v:>8d}  (was {old})")

    print(f"\n  Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
