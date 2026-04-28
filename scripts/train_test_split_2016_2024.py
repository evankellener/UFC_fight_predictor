"""Single train/test split:
   train: 2016-01-01 → 2024-10-01  (~8.75yr)
   test:  2024-10-01 → 2026-04-01

Elastic Net only. ZERO leakage (verified clean per-fold workflow).
Audit: docs/audits/train_test_split_2016_2024.md
"""
import sys, json, time, warnings
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
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             roc_auc_score)
from scipy.optimize import minimize_scalar

import mma_ai_pipeline as mma

EPS = 1e-6
LAM = 1.20             # recency-weight (production)
TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_END   = pd.Timestamp("2024-10-01")  # also = test_start
TEST_END    = pd.Timestamp("2026-04-01")
THRESHOLD   = 3
EN_C        = 0.05
EN_L1       = 0.5


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
    """Pipeline Steps 1-6 (per-fight clean — verified zero leakage by
    test_leakage_per_fold.py)."""
    print("Building pipeline through Step 6 (per-fight clean)...")
    df = mma.load_base_data()
    df = mma.beta_binomial_smooth(df)
    df = mma.poisson_gamma_smooth(df)
    df = mma.compute_derived_features(df)
    stat_cols = sorted(set(c for c in df.columns if
                 (c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
                  c.endswith("_ratio") or c.endswith("_per_ctrl") or
                  c in ["ko_smooth", "win_smooth", "decision_smooth",
                        "sub_land_smooth", "sub_land_rate", "ctrl_pm",
                        "ko_per_sig_str_land", "td_per_sig_str_att",
                        "ground_per_ctrl", "dist_per_sig_str_land",
                        "head_per_sig_str_land", "rev_per_ctrlopp",
                        "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
                        "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"])
                 and c in df.columns and not c.startswith("opp_") and not c.endswith("_raw")))
    df = mma.compute_decayed_averages(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    df = mma.compute_opponent_history(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    return df, stat_cols


def split_features(df_full, stat_cols, train_end):
    """Compute features with WC priors frozen at train_end (no future leak)."""
    train_only = df_full[df_full["DATE"] < train_end].copy()
    print(f"  WC priors computed from {len(train_only):,} train-only rows "
          f"(DATE < {train_end.date()})")
    priors = mma.compute_wc_priors(train_only, stat_cols)
    df_with_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    result = mma.assemble_features(df_with_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


def main():
    overall_t0 = time.time()
    print("=" * 78)
    print(f"SINGLE TRAIN/TEST SPLIT — Elastic Net only")
    print(f"  Train: {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  Test:  {TRAIN_END.date()} → {TEST_END.date()}")
    print(f"  Threshold: ≥{THRESHOLD} prior UFC fights")
    print(f"  Recency λ: {LAM}    EN(C={EN_C}, l1={EN_L1})")
    print(f"  Audit: docs/audits/train_test_split_2016_2024.md")
    print("=" * 78)

    df_full, stat_cols = build_through_step6()
    print(f"\n✓ Step 1-6 build done ({time.time()-overall_t0:.0f}s)")

    print(f"\nComputing features with priors frozen at train_end={TRAIN_END.date()}...")
    feats_df = split_features(df_full, stat_cols, TRAIN_END)
    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup = Path("data/tmp/mmaai_features.csv.before_201624")
    if feats_csv.exists() and not backup.exists():
        import shutil; shutil.copy2(feats_csv, backup)
    feats_df.to_csv(feats_csv, index=False)

    try:
        for mod in list(sys.modules):
            if mod.startswith("run_threshold_sweep_both_elos") or mod == "retrain_lr_symmetric" or mod == "walk_forward_4fold":
                del sys.modules[mod]
        from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
        from retrain_lr_symmetric import (load_wc_history_from_db, add_wc_features,
                                          flip_row_dataframe)
        from walk_forward_4fold import select_features

        base = load_base_both_elos()
        df = apply_threshold(base, THRESHOLD)
        df = add_wc_features(df, load_wc_history_from_db())
        feats = select_features(df)

        train = df[(df["DATE"] >= TRAIN_START) & (df["DATE"] < TRAIN_END)].copy()
        test  = df[(df["DATE"] >= TRAIN_END) & (df["DATE"] < TEST_END)].copy()

        # Hard leakage assertions (§1)
        assert train["DATE"].max() < TRAIN_END, \
            f"§1 violated: train_max {train['DATE'].max()} >= test_start {TRAIN_END}"
        train_keys = set(zip(train["DATE"], train["jbout"]))
        test_keys = set(zip(test["DATE"], test["jbout"]))
        assert not (train_keys & test_keys), \
            "§1 violated: bout overlap between train and test"
        print(f"\n✓ Leakage assertions pass")
        print(f"  Train: {len(train):,} fights ({train['DATE'].min().date()} → {train['DATE'].max().date()})")
        print(f"  Test:  {len(test):,} fights ({test['DATE'].min().date()} → {test['DATE'].max().date()})")

        # Symmetric doubled training
        train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
        usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]
        print(f"  Usable features: {len(usable)}")

        # Fit imputer + scaler on TRAIN ONLY (§4)
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
        ytr = train_d["win"].astype(int).values
        w = np.exp(-LAM * (TRAIN_END - train_d["DATE"]).dt.days.values / 365.25)

        # Elastic Net
        lr = LogisticRegression(C=EN_C, penalty="elasticnet", l1_ratio=EN_L1,
                                solver="saga", max_iter=8000, random_state=42)
        lr.fit(Xtr, ytr, sample_weight=w)
        n_active = int((np.abs(lr.coef_[0]) > 1e-8).sum())
        print(f"\n  Elastic Net fit: {len(usable)} features, {n_active} active after L1")

        # Calibrate on undoubled-train predictions
        p_tr = lr.predict_proba(sc.transform(imp.transform(train[usable])))[:, 1]
        T = temp_cal(p_tr, train["win"].astype(int).values)
        print(f"  Temperature calibrator: T = {T:.4f}")

        # Predict on test
        Xte = sc.transform(imp.transform(test[usable]))
        p_te_raw = lr.predict_proba(Xte)[:, 1]
        p_te = apply_temp(p_te_raw, T)
        y_test = test["win"].astype(int).values

        # Metrics
        pc = np.clip(p_te, EPS, 1-EPS)
        metrics = {
            "n_train": int(len(train)),
            "n_test":  int(len(test)),
            "n_features": int(len(usable)),
            "n_active": n_active,
            "T_calibrator": T,
            "accuracy": float(accuracy_score(y_test, (p_te >= 0.5).astype(int))),
            "log_loss": float(log_loss(y_test, pc)),
            "brier":    float(brier_score_loss(y_test, pc)),
        }
        try:
            metrics["auc"] = float(roc_auc_score(y_test, p_te))
        except ValueError:
            metrics["auc"] = None

        print()
        print("=" * 60)
        print("TEST METRICS  (zero-leakage Elastic Net)")
        print("=" * 60)
        print(f"  n_test       {metrics['n_test']:>6d}")
        print(f"  Accuracy     {metrics['accuracy']*100:>6.2f}%")
        print(f"  Log loss     {metrics['log_loss']:>7.4f}")
        print(f"  Brier        {metrics['brier']:>7.4f}")
        print(f"  AUC          {metrics['auc'] or 0:>7.4f}")
        print(f"  Active feat  {metrics['n_active']}/{metrics['n_features']}")

        # Save predictions for downstream analysis
        Path("results").mkdir(exist_ok=True)
        out_pred = test[["DATE", "jevent", "jbout", "jfighter", "opp_jfighter", "win"]].copy()
        out_pred["p_pred"] = p_te
        out_pred.to_parquet("results/train_test_2016_2024_predictions.parquet", index=False)

        out = {"config": {"train_start": str(TRAIN_START.date()),
                          "train_end":   str(TRAIN_END.date()),
                          "test_end":    str(TEST_END.date()),
                          "threshold": THRESHOLD,
                          "recency_lambda": LAM,
                          "model": "ElasticNet",
                          "C": EN_C, "l1_ratio": EN_L1},
               "metrics": metrics,
               "audit": "docs/audits/train_test_split_2016_2024.md",
               "total_runtime_min": round((time.time() - overall_t0)/60, 1)}
        Path("results/train_test_2016_2024.json").write_text(json.dumps(out, indent=2))
        print(f"\n✓ Saved results/train_test_2016_2024.json + predictions parquet")
        print(f"  Total runtime: {(time.time() - overall_t0)/60:.1f} minutes")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
