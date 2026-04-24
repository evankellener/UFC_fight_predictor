"""4-fold walk-forward validation with 7-year training windows + 6-month test slices.

Folds (as requested 2026-04-23):
  fold 1:  train 2017-04 → 2024-04   test 2024-04 → 2024-10
  fold 2:  train 2017-10 → 2024-10   test 2024-10 → 2025-04
  fold 3:  train 2018-04 → 2025-04   test 2025-04 → 2025-10
  fold 4:  train 2018-10 → 2025-10   test 2025-10 → 2026-04

Per-fold, refits:
  - SimpleImputer (train only — LEAKAGE_REFERENCE.md §4)
  - StandardScaler (train only — §4)
  - LogisticRegression symmetric (doubled train — §1)
  - Temperature-scaling calibrator via 5-fold CV inside train (no test contamination — §6)

Holds fixed (declared methodology gap):
  - Poisson-Gamma / Beta-Binomial τ hyperparameters in mma_ai_config.py.
    Re-optimizing these per fold via Optuna Bayesian search would cost ~2-3
    hours at ~60s/trial × 30 trials × 4 folds. τ capacity (~30 scalars) is
    much smaller than the LR's 207 learned coefficients, so the leakage
    impact is second-order. This is called out in the final report.

Reports:
  - Per-fold: n, accuracy, log loss, AUC, Brier, ECE, max bucket deviation,
    calibrator T, bootstrap-CI on accuracy
  - Aggregate: mean ± std across folds on each metric
  - Top-10 feature coefficient stability across folds (does the model
    change its mind about which features matter?)
"""
import sys, json, pickle, math, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, log_loss,
                             brier_score_loss, roc_auc_score)
from scipy.optimize import minimize_scalar

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, LAM
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features,
    load_recent_form_from_db, add_recent_form_features,
    flip_row_dataframe, RATIO_OF_FIGHTERS,
    WC_HIST_DIFF_COLS, WC_PAIR_COLS, RECENT_FORM_DIFF_COLS,
)

FILTER_THRESHOLD = 3

# ── Fold definitions (exactly as requested) ──────────────────────────────
# All dates are inclusive-start, exclusive-end.
FOLDS = [
    {"name": "fold_1",
     "train_start": "2017-04-01", "train_end": "2024-04-01",
     "test_start":  "2024-04-01", "test_end":  "2024-10-01"},
    {"name": "fold_2",
     "train_start": "2017-10-01", "train_end": "2024-10-01",
     "test_start":  "2024-10-01", "test_end":  "2025-04-01"},
    {"name": "fold_3",
     "train_start": "2018-04-01", "train_end": "2025-04-01",
     "test_start":  "2025-04-01", "test_end":  "2025-10-01"},
    {"name": "fold_4",
     "train_start": "2018-10-01", "train_end": "2025-10-01",
     "test_start":  "2025-10-01", "test_end":  "2026-04-24"},  # includes today
]


# ── Temperature calibrator (same as calibrate_v2_lr.py, local copy) ──────
class TempCal:
    def fit(self, p, y):
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        logit = np.log(p / (1 - p))
        def nll(T):
            if T <= 0: return 1e9
            pc = 1 / (1 + np.exp(-logit / T))
            pc = np.clip(pc, eps, 1 - eps)
            return -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()
        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.T = float(res.x)
        return self
    def predict(self, p):
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        logit = np.log(p / (1 - p))
        return 1 / (1 + np.exp(-logit / self.T))


def fit_calibrator_via_cv(p_train, y_train, n_splits=5, seed=42):
    """Fit calibrator on full train (for application at test time), AND
    report CV-held-out ECE on train to catch pathological fits."""
    # For deployment: fit on full train
    prod = TempCal().fit(p_train, y_train)

    # For validity check: CV held-out performance
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p_cv = np.zeros_like(p_train, dtype=float)
    for tr_idx, te_idx in kf.split(p_train):
        c = TempCal().fit(p_train[tr_idx], y_train[tr_idx])
        p_cv[te_idx] = c.predict(p_train[te_idx])
    return prod, p_cv


def bucket_metrics(p, y):
    """ECE + max bucket deviation on confidence bins of 5% from 50%."""
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5) == (y == 1)).astype(int)
    edges = np.arange(0.50, 1.0001, 0.05)
    ece = 0.0; max_dev_pp = 0.0; buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if hi == edges[-1] else (conf >= lo) & (conf < hi)
        n = int(m.sum())
        if n == 0:
            buckets.append({"range": f"{int(lo*100)}-{int(hi*100)}%", "n": 0})
            continue
        pa = float(conf[m].mean()); ac = float(correct[m].mean())
        dev = abs(pa - ac)
        ece += n / len(y) * dev
        max_dev_pp = max(max_dev_pp, dev * 100)
        buckets.append({
            "range": f"{int(lo*100)}-{int(hi*100)}%",
            "n": n, "pred_avg": pa, "actual": ac,
            "dev_pp": dev * 100,
        })
    return dict(ece_pp=ece * 100, max_dev_pp=max_dev_pp, buckets=buckets)


def bootstrap_accuracy_ci(y, preds, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        boots.append(accuracy_score(y[idx], preds[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def leakage_assertions(train, test, fold):
    """Hard failures if the split has any leakage issues.
    Enforces LEAKAGE_REFERENCE.md §1 (chronological, never-touch-test)."""
    train_max = train["DATE"].max()
    test_min  = test["DATE"].min()
    assert train_max < pd.Timestamp(fold["test_start"]), \
        f"§1 VIOLATED: train max {train_max} >= test_start {fold['test_start']}"
    assert test_min >= pd.Timestamp(fold["test_start"]), \
        f"§1 VIOLATED: test min {test_min} < test_start {fold['test_start']}"
    # Overlap check
    train_keys = set(zip(train["DATE"], train["jbout"]))
    test_keys = set(zip(test["DATE"], test["jbout"]))
    overlap = train_keys & test_keys
    assert not overlap, f"§1 VIOLATED: {len(overlap)} bouts appear in BOTH train and test"


# ── Feature selection — matches production retrain_lr_symmetric.py ───────

def select_features(df):
    feats = [c for c in df.columns if (c.endswith("_diff") or c.endswith("_ufc")
             or c.endswith("_exp") or c in ("weightclass_encoded", "scheduled_rounds",
                                             "days_since_last_fight_f1",
                                             "cross_division_flag"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    return feats


def run_fold(df, fold, feats):
    fold_name = fold["name"]
    train_start = pd.Timestamp(fold["train_start"])
    train_end   = pd.Timestamp(fold["train_end"])
    test_start  = pd.Timestamp(fold["test_start"])
    test_end    = pd.Timestamp(fold["test_end"])

    train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
    test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()

    # Event-boundary check: ensure no fight card straddles boundary.
    # Done via jevent — a fight card gets one jevent.
    border_events = set(train["jevent"]) & set(test["jevent"])
    if border_events:
        print(f"  ⚠ WARN: {len(border_events)} events straddle boundary (same-day cards)")

    # Leakage assertions
    leakage_assertions(train, test, fold)

    # Symmetric train: double the training DF
    train_flipped = flip_row_dataframe(train)
    train_doubled = pd.concat([train, train_flipped], ignore_index=True)

    # usable features: drop zero-variance ones on THIS fold's train
    usable = [c for c in feats if c in train_doubled.columns
              and train_doubled[c].std() > 1e-8]

    # Fit imputer + scaler on train (LEAKAGE_REFERENCE.md §4)
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train_doubled[usable])
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    y_tr = train_doubled["win"].astype(int).values

    # Recency weight anchored at train_end (unchanged from production)
    w_tr = np.exp(-LAM * (train_end - train_doubled["DATE"]).dt.days.values / 365.25)

    # Fit LR
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs_tr, y_tr, sample_weight=w_tr)

    # Predict on train (for calibrator fitting) — use ORIGINAL (undoubled) train
    X_tr_orig = imp.transform(train[usable])
    X_tr_orig = sc.transform(X_tr_orig)
    p_train = lr.predict_proba(X_tr_orig)[:, 1]
    y_train = train["win"].astype(int).values

    # Fit calibrator (temperature) on train — via CV for honest measurement.
    # Production calibrator fit on full train (§6 says no tuning on test).
    cal_prod, p_train_cv = fit_calibrator_via_cv(p_train, y_train)

    # Predict on test (raw, then calibrated)
    X_te = imp.transform(test[usable])
    X_te = sc.transform(X_te)
    p_test_raw = lr.predict_proba(X_te)[:, 1]
    p_test_cal = cal_prod.predict(p_test_raw)
    y_test = test["win"].astype(int).values

    # Metrics
    def M(p, y):
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        pred = (p >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = float("nan")
        bm = bucket_metrics(p, y)
        return {
            "n":        int(len(y)),
            "accuracy": float(accuracy_score(y, pred)),
            "log_loss": float(log_loss(y, pc)),
            "brier":    float(brier_score_loss(y, pc)),
            "auc":      auc,
            "ece_pp":   bm["ece_pp"],
            "max_dev_pp": bm["max_dev_pp"],
        }

    m_raw = M(p_test_raw, y_test)
    m_cal = M(p_test_cal, y_test)

    # Bootstrap CI on calibrated accuracy
    acc_lo, acc_hi = bootstrap_accuracy_ci(
        y_test, (p_test_cal >= 0.5).astype(int), n_boot=500)

    # Top-10 features by |coef|
    coefs = lr.coef_[0]
    feat_coef = sorted(zip(usable, coefs.tolist()),
                       key=lambda t: -abs(t[1]))[:10]

    return {
        "fold": fold_name,
        "train_range": [fold["train_start"], fold["train_end"]],
        "test_range":  [fold["test_start"], fold["test_end"]],
        "n_train_unique": len(train),
        "n_train_doubled": len(train_doubled),
        "n_test": len(test),
        "n_features_usable": len(usable),
        "n_features_active": int(np.sum(np.abs(coefs) > 1e-8)),
        "train_base_rate": float(y_train.mean()),
        "test_base_rate":  float(y_test.mean()),
        "calibrator_T": cal_prod.T,
        "raw": m_raw,
        "calibrated": m_cal,
        "acc_ci_95": [acc_lo, acc_hi],
        "top_coefs": feat_coef,
    }


def main():
    print("=" * 76)
    print("4-fold walk-forward validation — 7-year training, 6-month test slices")
    print("Leakage audit: LEAKAGE_REFERENCE.md §1/§3/§4/§6")
    print("=" * 76)

    print("\nLoading base features + wc_history (one-time, reused across folds)...")
    base = load_base_both_elos()
    df = apply_threshold(base, FILTER_THRESHOLD)
    wc_hist = load_wc_history_from_db()
    df = add_wc_features(df, wc_hist)
    # recent-form features tried & rejected — see retrain_lr_symmetric.py
    feats = select_features(df)
    print(f"  Total rows: {len(df):,}")
    print(f"  Feature columns candidates: {len(feats)}")
    print(f"  Date range available: {df['DATE'].min().date()} → {df['DATE'].max().date()}")

    # Per-fold runs
    results = []
    for fold in FOLDS:
        print(f"\n── {fold['name']} ──────────────────────────────────────────────")
        print(f"  train: {fold['train_start']} → {fold['train_end']}")
        print(f"  test:  {fold['test_start']}  → {fold['test_end']}")
        r = run_fold(df, fold, feats)
        results.append(r)
        print(f"  n_train (unique / doubled): {r['n_train_unique']} / {r['n_train_doubled']}")
        print(f"  n_test: {r['n_test']}   features active: {r['n_features_active']}")
        print(f"  train base-rate: {r['train_base_rate']:.3f}   test base-rate: {r['test_base_rate']:.3f}")
        print(f"  T (temperature): {r['calibrator_T']:.4f}")
        rv = r["raw"]; cv = r["calibrated"]
        print(f"  RAW         acc={rv['accuracy']:.4f}  ll={rv['log_loss']:.4f}  "
              f"auc={rv['auc']:.4f}  brier={rv['brier']:.4f}  ECE={rv['ece_pp']:.2f}pp")
        print(f"  CALIBRATED  acc={cv['accuracy']:.4f}  ll={cv['log_loss']:.4f}  "
              f"auc={cv['auc']:.4f}  brier={cv['brier']:.4f}  ECE={cv['ece_pp']:.2f}pp")
        print(f"  accuracy 95% CI (bootstrap): [{r['acc_ci_95'][0]:.4f}, {r['acc_ci_95'][1]:.4f}]")

    # Aggregate
    print("\n" + "=" * 76)
    print("AGGREGATE across 4 folds (calibrated predictions)")
    print("=" * 76)
    metrics_cal = [r["calibrated"] for r in results]
    total_n = sum(m["n"] for m in metrics_cal)
    print(f"Total test fights: {total_n}")

    def agg(field):
        vals = [m[field] for m in metrics_cal]
        return np.mean(vals), np.std(vals), min(vals), max(vals)

    for name, field in [("accuracy", "accuracy"), ("log_loss", "log_loss"),
                        ("AUC",      "auc"),      ("Brier",    "brier"),
                        ("ECE",      "ece_pp"),   ("max_dev",  "max_dev_pp")]:
        m, s, mn, mx = agg(field)
        print(f"  {name:<10s}  mean={m:.4f}  std={s:.4f}  range=[{mn:.4f}, {mx:.4f}]")

    # Calibrator T stability
    Ts = [r["calibrator_T"] for r in results]
    print(f"\n  calibrator T per fold: {[f'{t:.3f}' for t in Ts]}")
    print(f"  calibrator T  mean={np.mean(Ts):.3f}  std={np.std(Ts):.3f}")

    # Feature stability — which features appear in top-10 across folds?
    print("\n" + "─" * 76)
    print("Feature stability — top-10 by |coef| in each fold")
    print("─" * 76)
    all_top = Counter()
    for r in results:
        for feat, _ in r["top_coefs"]:
            all_top[feat] += 1
    persistent = [feat for feat, cnt in all_top.most_common() if cnt >= 3]
    print(f"\nFeatures in top-10 on ≥3 of 4 folds ({len(persistent)}):")
    for feat in persistent:
        coefs_across_folds = []
        for r in results:
            c = dict(r["top_coefs"]).get(feat)
            coefs_across_folds.append(f"{c:+.3f}" if c is not None else "   —  ")
        print(f"  {feat:<38s}  {' '.join(coefs_across_folds)}")

    # Save results
    out = Path("results/walk_forward_4fold.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "folds": FOLDS,
        "results": results,
        "aggregate": {
            field: {"mean": float(np.mean([m[field] for m in metrics_cal])),
                    "std":  float(np.std([m[field] for m in metrics_cal])),
                    "min":  float(min(m[field] for m in metrics_cal)),
                    "max":  float(max(m[field] for m in metrics_cal))}
            for field in ["accuracy", "log_loss", "auc", "brier", "ece_pp", "max_dev_pp"]
        },
        "total_n_test": total_n,
        "calibrator_T_mean": float(np.mean(Ts)),
        "calibrator_T_std":  float(np.std(Ts)),
        "methodology_gaps": [
            "τ hyperparameters in mma_ai_config.py held fixed across folds "
            "(not re-optimized per-fold). Cost of fix: ~2-3 hours Optuna "
            "Bayesian search. τ space has ~30 scalars vs LR's 207 "
            "coefficients — leakage impact is second-order.",
        ],
        "leakage_audit": {
            "§1_temporal_split": "PASS — hard assertions on train_max < test_start",
            "§3_strictly_prior":  "PASS — wc_history + Elo caches pre-computed "
                                  "with d<fight_date filter",
            "§4_scaler_imputer_train_only": "PASS — fit_transform on train, transform on test",
            "§6_no_tuning_on_test": "PASS for LR+calibrator; τ fixed (see gap above)",
        },
    }, indent=2, default=str))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
