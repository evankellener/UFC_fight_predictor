"""Compare calibration methods for the v2 LR on the 420-fight test set.

Methods tested:
  (1) Temperature scaling — 1 param, logit/T → sigmoid
  (2) Platt scaling — 2 params, LR on logit(p)
  (3) Isotonic regression — monotonic step function
  (4) Beta calibration (Kull et al. 2017) — 3 params, LR on [log(p), log(1-p)]

Evaluation via 5-fold CV: each fold's predictions are calibrated using
the other four folds, so no data point is calibrated on itself.

Winner is saved to app/models/blend_v2/calibrator.pkl as a simple dict:
  {"method": str, "params": ..., "apply": callable-via-pickle}

PredictorV2 loads this at inference and applies it after lr.predict_proba.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from scipy.optimize import minimize_scalar

from predictor_v2 import PredictorV2
from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, TEST_FIRST, TEST_LAST
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features,
    load_recent_form_from_db, add_recent_form_features,
)

EPS = 1e-6
OUT = Path("app/models/blend_v2")


# ─── calibrators ───────────────────────────────────────────────────────────

class TemperatureCalibrator:
    """p' = sigmoid(logit(p) / T). Fit T by minimizing NLL."""
    def fit(self, p, y):
        p = np.clip(p, EPS, 1 - EPS)
        logit = np.log(p / (1 - p))
        def nll(T):
            if T <= 0: return 1e9
            p_cal = 1 / (1 + np.exp(-logit / T))
            p_cal = np.clip(p_cal, EPS, 1 - EPS)
            return -(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal)).mean()
        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.T = float(res.x)
        return self
    def predict(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        logit = np.log(p / (1 - p))
        return 1 / (1 + np.exp(-logit / self.T))


class PlattCalibrator:
    """p' = sigmoid(A * logit(p) + B). Fit via LR on logit(p)."""
    def fit(self, p, y):
        p = np.clip(p, EPS, 1 - EPS)
        X = np.log(p / (1 - p)).reshape(-1, 1)
        self.lr = LogisticRegression(C=1e6, solver="lbfgs").fit(X, y)
        return self
    def predict(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        X = np.log(p / (1 - p)).reshape(-1, 1)
        return self.lr.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    """Monotonic non-parametric step function."""
    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return self
    def predict(self, p):
        return self.iso.predict(p)


class BetaCalibrator:
    """Kull, Silva Filho, Flach 2017. Three-param, designed for classifier outputs.
    p' = sigmoid(a * log(p) - b * log(1-p) + c).
    Fit by LR on features [log(p), -log(1-p)] with intercept.
    """
    def fit(self, p, y):
        p = np.clip(p, EPS, 1 - EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        self.lr = LogisticRegression(C=1e6, solver="lbfgs").fit(X, y)
        return self
    def predict(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        return self.lr.predict_proba(X)[:, 1]


class IdentityCalibrator:
    def fit(self, p, y): return self
    def predict(self, p): return p


METHODS = {
    "uncalibrated":       IdentityCalibrator,
    "temperature":        TemperatureCalibrator,
    "platt":              PlattCalibrator,
    "isotonic":           IsotonicCalibrator,
    "beta":               BetaCalibrator,
}


# ─── metrics ───────────────────────────────────────────────────────────────

def bucket_metrics(p, y, n_bins=10):
    """Expected Calibration Error + max bucket deviation + per-bucket table."""
    # Bin by PREDICTED confidence (max(p, 1-p)) in 5% bins from 0.5
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5) == (y == 1)).astype(int)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    buckets = []
    ece = 0.0
    n_total = len(y)
    max_dev_pp = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n_b = int(mask.sum())
        if n_b == 0:
            buckets.append((lo, hi, 0, None, None, None))
            continue
        pa = float(conf[mask].mean())
        ac = float(correct[mask].mean())
        dev = abs(pa - ac)
        ece += n_b / n_total * dev
        max_dev_pp = max(max_dev_pp, dev * 100)
        buckets.append((lo, hi, n_b, pa, ac, dev * 100))
    return dict(ece=float(ece), max_dev_pp=float(max_dev_pp), buckets=buckets)


def overall_metrics(p, y):
    p_c = np.clip(p, EPS, 1 - EPS)
    b = bucket_metrics(p, y)
    return dict(
        accuracy=float(accuracy_score(y, (p >= 0.5).astype(int))),
        log_loss=float(log_loss(y, p_c)),
        brier=float(brier_score_loss(y, p_c)),
        ece_pp=b["ece"] * 100,
        max_dev_pp=b["max_dev_pp"],
    )


# ─── 5-fold CV ─────────────────────────────────────────────────────────────

def kfold_calibrated_predictions(calibrator_cls, p, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.zeros_like(p, dtype=float)
    for train_idx, test_idx in kf.split(p):
        cal = calibrator_cls().fit(p[train_idx], y[train_idx])
        out[test_idx] = cal.predict(p[test_idx])
    return out


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    print("="*72)
    print("Calibration method comparison — v2 LR on 420-fight test set")
    print("="*72)

    v2 = PredictorV2(verbose=False)
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    wc_hist = load_wc_history_from_db()
    df = add_wc_features(df, wc_hist)
    # recent-form features tried & rejected — see retrain_lr_symmetric.py notes
    test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()

    X = v2.imputer.transform(test[v2.feat_cols].values)
    X = v2.scaler.transform(X)
    p_raw = v2.lr.predict_proba(X)[:, 1]
    y = test["win"].astype(int).values
    print(f"\nTest set: n={len(y)}  base-rate(y=1): {y.mean():.3f}")

    # 5-fold CV evaluation of each method
    print("\n5-fold CV held-out performance (calibrator fit on 4 folds, eval on 5th)")
    print(f"{'method':<14s} {'acc':>7s} {'logloss':>9s} {'brier':>7s} {'ECE':>7s} {'max_dev':>8s}")
    print("-" * 70)

    results = {}
    for name, cls in METHODS.items():
        p_cv = kfold_calibrated_predictions(cls, p_raw, y)
        m = overall_metrics(p_cv, y)
        results[name] = m
        print(f"  {name:<12s}  {m['accuracy']:>6.4f}  {m['log_loss']:>8.4f}  "
              f"{m['brier']:>6.4f}  {m['ece_pp']:>5.2f}pp  {m['max_dev_pp']:>6.2f}pp")

    # Winner selection — pick the safer choice, not just lowest ECE.
    # Rule: prefer order-preserving methods (temperature) when their ECE is
    # within 1pp of the best. Order-preservation matters for betting (it keeps
    # fighter rankings intact, preserves Strategy D's accuracy-dependent ROI).
    best_ece = min(m["ece_pp"] for n, m in results.items() if n != "uncalibrated")
    temp_ece = results["temperature"]["ece_pp"]
    if temp_ece - best_ece < 1.0 and results["temperature"]["accuracy"] >= results["uncalibrated"]["accuracy"] - 1e-9:
        winner_name = "temperature"
        reason = (f"tied within 1pp of best ECE ({temp_ece:.2f} vs best {best_ece:.2f}), "
                  f"preserves accuracy, 1 parameter (robust)")
    else:
        ranked = sorted([(name, m) for name, m in results.items() if name != "uncalibrated"],
                        key=lambda kv: (kv[1]["ece_pp"], kv[1]["log_loss"]))
        winner_name = ranked[0][0]
        reason = "best ECE overall"

    print(f"\n🏆 Picked: {winner_name}  ({reason})")
    print(f"   vs uncalibrated: "
          f"ECE {results['uncalibrated']['ece_pp']:.2f}pp → {results[winner_name]['ece_pp']:.2f}pp  "
          f"({results[winner_name]['ece_pp'] - results['uncalibrated']['ece_pp']:+.2f}pp)")
    print(f"   vs uncalibrated: "
          f"max_dev {results['uncalibrated']['max_dev_pp']:.2f}pp → {results[winner_name]['max_dev_pp']:.2f}pp")

    # Show per-bucket calibration BEFORE and AFTER (using the winner)
    print(f"\nPer-bucket comparison (uncalibrated vs {winner_name}):")
    winner_cv = kfold_calibrated_predictions(METHODS[winner_name], p_raw, y)
    b_raw = bucket_metrics(p_raw, y)["buckets"]
    b_cal = bucket_metrics(winner_cv, y)["buckets"]
    print(f"{'range':<10s} {'n':>4s}  {'raw_pred':>9s} {'raw_act':>9s} {'raw_dev':>8s}   "
          f"{'cal_pred':>9s} {'cal_act':>9s} {'cal_dev':>8s}")
    print("-" * 100)
    for (lo, hi, n, pr, ar, dr), (_, _, _, pc, ac, dc) in zip(b_raw, b_cal):
        if n == 0:
            print(f"  {int(lo*100)}-{int(hi*100)}%     {n:>3d}  empty")
            continue
        print(f"  {int(lo*100)}-{int(hi*100)}%     {n:>3d}   "
              f"{pr*100:>7.1f}%  {ar*100:>7.1f}%  {dr:>6.1f}pp   "
              f"{pc*100:>7.1f}%  {ac*100:>7.1f}%  {dc:>6.1f}pp")

    # Fit winner on ALL 420 fights for production use
    # (CV was only for fair comparison; deployment calibrator sees the full test set)
    winner_cls = METHODS[winner_name]
    final = winner_cls().fit(p_raw, y)

    # Save as a portable params dict — no class/module dependency at load time.
    # PredictorV2 reconstructs the calibrator from these params directly.
    if winner_name == "temperature":
        params = {"T": float(final.T)}
    elif winner_name == "platt":
        params = {"A": float(final.lr.coef_[0, 0]), "B": float(final.lr.intercept_[0])}
    elif winner_name == "isotonic":
        # Serialize isotonic breakpoints + values (sklearn allows pickling; we
        # round-trip via sklearn but strip the outer class so loader doesn't need us).
        params = {"X_": final.iso.X_thresholds_.tolist(),
                  "y_": final.iso.y_thresholds_.tolist()}
    elif winner_name == "beta":
        # Coefficients: [log(p), -log(1-p)], intercept
        params = {"a": float(final.lr.coef_[0, 0]),
                  "b": float(final.lr.coef_[0, 1]),
                  "c": float(final.lr.intercept_[0])}
    elif winner_name == "uncalibrated":
        params = {}
    else:
        params = {}

    out_path = OUT / "calibrator.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"method": winner_name, "params": params, "n_train": len(y)}, f)
    print(f"\n✓ Saved {out_path} ({winner_name}, trained on all n={len(y)})")
    print(f"   params: {params}")

    # Also save the comparison table as JSON for reference
    comp_path = OUT / "calibration_comparison.json"
    comp_path.write_text(json.dumps({
        "results": results, "winner": winner_name,
        "n_test": int(len(y)), "base_rate": float(y.mean()),
    }, indent=2))
    print(f"✓ Saved {comp_path}")


if __name__ == "__main__":
    main()
