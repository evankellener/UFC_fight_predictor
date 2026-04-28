"""Clean per-fold walk-forward for the notebook.

Single Elastic Net, ZERO leakage. Audit at:
  docs/audits/notebook_01_clean_walkforward.md

For each fold:
  1. Compute_wc_priors using only fights with DATE < train_end (§3 fix)
  2. Recompute AdjPerf z-scores using fold-frozen priors (apply to ALL fights)
  3. Apply threshold (≥3 priors) + add WC features
  4. Train Elastic Net (C=0.05, l1_ratio=0.5) on fold's training window
  5. Calibrate via temperature scaling on train predictions
  6. Score test fights
  7. Report acc / log-loss / brier / AUC

No XGBoost, no blend, no style clustering, no market features.
"""
import sys, json, time, warnings, sqlite3
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
from scipy.optimize import minimize_scalar

import mma_ai_pipeline as mma

EPS = 1e-6
TRAIN_YEARS = 4
THRESHOLD = 3
LAM = 1.20  # production recency weighting
CONFIG = {"C": 0.05, "l1_ratio": 0.5}


def build_folds():
    starts = pd.date_range("2024-04-01", "2026-01-01", freq="3MS")
    return [{"name": f"fold_{i}",
             "train_start": (s - pd.DateOffset(years=TRAIN_YEARS)),
             "train_end":   s,
             "test_start":  s,
             "test_end":   (s + pd.DateOffset(months=3))}
            for i, s in enumerate(starts, 1)]


def temp_cal(p, y):
    p = np.clip(p, EPS, 1-EPS); logit = np.log(p/(1-p))
    def nll(T):
        if T <= 0: return 1e9
        pc = 1/(1+np.exp(-logit/T)); pc = np.clip(pc, EPS, 1-EPS)
        return -(y*np.log(pc) + (1-y)*np.log(1-pc)).mean()
    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)


def apply_temp(p, T):
    p = np.clip(p, EPS, 1-EPS); lg = np.log(p/(1-p))
    return 1/(1+np.exp(-lg/T))


def build_through_step6():
    """Pipeline Steps 1-6 (per-fight clean).
    These features only depend on fights with DATE < the current fight's DATE.
    No fold-cutoff needed at this stage."""
    print("Building pipeline through Step 6 (per-fight clean)...")
    df = mma.load_base_data()
    df = mma.beta_binomial_smooth(df)
    df = mma.poisson_gamma_smooth(df)
    df = mma.compute_derived_features(df)

    stat_cols = [c for c in df.columns if
                 c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
                 c.endswith("_ratio") or c.endswith("_per_ctrl") or
                 c in ["ko_smooth", "win_smooth", "decision_smooth",
                       "sub_land_smooth", "sub_land_rate", "ctrl_pm",
                       "ko_per_sig_str_land", "td_per_sig_str_att",
                       "ground_per_ctrl", "dist_per_sig_str_land",
                       "head_per_sig_str_land", "rev_per_ctrlopp",
                       "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
                       "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"]]
    stat_cols = sorted(set(c for c in stat_cols if c in df.columns and
                           not c.startswith("opp_") and not c.endswith("_raw")))

    df = mma.compute_decayed_averages(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    df = mma.compute_opponent_history(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    return df, stat_cols


def per_fold_features(df_full, stat_cols, train_end):
    """Compute fold-clean features:
       - WC priors computed using only fights with DATE < train_end (the §3 fix)
       - AdjPerf z-scores applied to all fights using those frozen priors
       - assemble_features (decayed AdjPerf + diffs)
    """
    train_only = df_full[df_full["DATE"] < train_end].copy()
    if len(train_only) < 100:
        return None
    priors = mma.compute_wc_priors(train_only, stat_cols)
    df_with_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    result = mma.assemble_features(df_with_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


def fit_elastic_net_one_fold(train, test, feats, train_anchor):
    """Single Elastic Net fit + temperature calibration. Returns calibrated test
    probabilities and metrics."""
    from retrain_lr_symmetric import flip_row_dataframe

    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values

    w = np.exp(-LAM * (train_anchor - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=CONFIG["C"], penalty="elasticnet",
                            l1_ratio=CONFIG["l1_ratio"],
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    p_tr_undoubled = lr.predict_proba(sc.transform(imp.transform(train[usable])))[:, 1]
    T = temp_cal(p_tr_undoubled, train["win"].astype(int).values)

    Xte = sc.transform(imp.transform(test[usable]))
    p_te_raw = lr.predict_proba(Xte)[:, 1]
    p_te = apply_temp(p_te_raw, T)

    y_test = test["win"].astype(int).values
    pc = np.clip(p_te, EPS, 1-EPS)
    metrics = {
        "n_test":   int(len(test)),
        "n_train":  int(len(train)),
        "n_features": int(len(usable)),
        "n_active": int((np.abs(lr.coef_[0]) > 1e-8).sum()),
        "T":        T,
        "accuracy": float(accuracy_score(y_test, (p_te >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_test, pc)),
        "brier":    float(brier_score_loss(y_test, pc)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_test, p_te))
    except ValueError:
        metrics["auc"] = None
    return p_te, metrics


def run_clean_walkforward(verbose=True):
    """Public entry point. Returns (per_fold_metrics, aggregate_metrics)."""
    overall_t0 = time.time()
    folds = build_folds()

    df_full, stat_cols = build_through_step6()
    if verbose:
        print(f"\n✓ Step 1-6 build done ({time.time()-overall_t0:.0f}s)  "
              f"df: {len(df_full):,} rows, {len(stat_cols)} stat cols")

    all_results = []
    for fold in folds:
        if verbose:
            print(f"\n── {fold['name'].upper()} test {fold['test_start'].date()} → "
                  f"{fold['test_end'].date()}  (priors frozen at "
                  f"train_end={fold['train_end'].date()})")
        fold_t0 = time.time()
        result = per_fold_features(df_full, stat_cols, fold["train_end"])
        if result is None:
            if verbose: print(f"  ⚠ insufficient training data, skipping")
            continue

        # Feed result through the existing run_threshold + add_wc_features path
        # by pointing the global CSV at it temporarily.
        feats_csv = Path("data/tmp/mmaai_features.csv")
        backup = Path("data/tmp/mmaai_features.csv.notebook_backup")
        if feats_csv.exists() and not backup.exists():
            import shutil; shutil.copy2(feats_csv, backup)
        result.to_csv(feats_csv, index=False)

        try:
            for mod in list(sys.modules):
                if mod.startswith("run_threshold_sweep_both_elos") or mod == "retrain_lr_symmetric" or mod == "walk_forward_4fold":
                    del sys.modules[mod]
            from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
            from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features
            from walk_forward_4fold import select_features

            base = load_base_both_elos()
            df = apply_threshold(base, THRESHOLD)
            df = add_wc_features(df, load_wc_history_from_db())
            feats = select_features(df)

            train = df[(df["DATE"] >= fold["train_start"]) & (df["DATE"] < fold["train_end"])].copy()
            test  = df[(df["DATE"] >= fold["test_start"]) & (df["DATE"] < fold["test_end"])].copy()

            # Hard leakage assert
            assert train["DATE"].max() < fold["test_start"], \
                f"§1 VIOLATION: train_max {train['DATE'].max()} >= test_start {fold['test_start']}"
            train_keys = set(zip(train["DATE"], train["jbout"]))
            test_keys = set(zip(test["DATE"], test["jbout"]))
            assert not (train_keys & test_keys), \
                f"§1 VIOLATION: bout overlap between train and test"

            if len(test) == 0:
                continue

            p_te, m = fit_elastic_net_one_fold(train, test, feats, fold["train_end"])
            m["fold"] = fold["name"]
            m["test_start"] = str(fold["test_start"].date())
            m["test_end"]   = str(fold["test_end"].date())
            m["train_end"]  = str(fold["train_end"].date())
            m["fold_elapsed_s"] = round(time.time() - fold_t0, 1)
            all_results.append(m)
            if verbose:
                print(f"  train {m['n_train']}  test {m['n_test']}  "
                      f"acc={m['accuracy']:.4f}  ll={m['log_loss']:.4f}  "
                      f"auc={m.get('auc') or 0:.4f}  fold_elapsed {m['fold_elapsed_s']}s")
        finally:
            if backup.exists():
                import shutil; shutil.copy2(backup, feats_csv)

    if backup.exists():
        backup.unlink()

    # Aggregate
    aggs = {}
    if all_results:
        for k in ("accuracy", "log_loss", "brier", "auc"):
            vals = [r[k] for r in all_results if r.get(k) is not None]
            if vals:
                aggs[f"{k}_mean"] = float(np.mean(vals))
                aggs[f"{k}_std"]  = float(np.std(vals))
        aggs["total_test_fights"] = sum(r["n_test"] for r in all_results)

    out = {"per_fold": all_results, "aggregate": aggs,
           "config": {"model": "ElasticNet", **CONFIG, "lambda_recency": LAM,
                      "train_years": TRAIN_YEARS, "threshold": THRESHOLD},
           "audit": "docs/audits/notebook_01_clean_walkforward.md",
           "total_runtime_min": round((time.time() - overall_t0) / 60, 1)}
    Path("results").mkdir(exist_ok=True)
    Path("results/notebook_clean_walkforward.json").write_text(json.dumps(out, indent=2, default=str))
    if verbose:
        print(f"\n✓ saved results/notebook_clean_walkforward.json  "
              f"({out['total_runtime_min']} min)")
        print(f"\nAGGREGATE: acc {aggs.get('accuracy_mean', 0)*100:.2f}% ± "
              f"{aggs.get('accuracy_std', 0)*100:.2f}pp   "
              f"ll {aggs.get('log_loss_mean', 0):.4f}   "
              f"auc {aggs.get('auc_mean', 0):.4f}   "
              f"n_test_total {aggs.get('total_test_fights', 0)}")
    return all_results, aggs


if __name__ == "__main__":
    run_clean_walkforward(verbose=True)
