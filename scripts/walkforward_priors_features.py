"""Test whether priors-derived features close the rookie weakness.

Pre-registered hypothesis (see docs/audits/walkforward_priors_features.md):
  Adding {priors_diff, min_priors_x, max_priors_x, priors_log_ratio,
  rookie_diff, both_rookie} should narrow the rookie 3-4 slice gap
  (currently model 73.45% vs Vegas 80.53%, n=113) without harming
  pooled metrics.

Setup:
  - Same train (2016-01 → 2024-10) / test (2024-10 → 2026-04) as baseline
  - Same per-fold clean pipeline (mma_ai_pipeline + WC priors frozen at
    train_end + apply_threshold(3))
  - EN(C=0.05, l1=0.5), recency λ=1.20  — identical to baseline
  - Two prediction columns saved:
      p_uncal     — raw EN
      p_beta_cvoof — beta calibrator fit on 5-fold CV-OOF train predictions

Outputs:
  - results/walkforward_priors_features.json
  - results/walkforward_priors_features_predictions.parquet
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
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             roc_auc_score)

import mma_ai_pipeline as mma

EPS         = 1e-6
LAM         = 1.20
TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_END   = pd.Timestamp("2024-10-01")
TEST_END    = pd.Timestamp("2026-04-01")
THRESHOLD   = 3
EN_C        = 0.05
EN_L1       = 0.5
N_CV_FOLDS  = 5

NEW_FEATURES = ["priors_diff", "min_priors_x", "max_priors_x",
                "priors_log_ratio", "rookie_diff", "both_rookie"]


# ─── Beta calibrator (same as in train_calib_compare_v2.py) ─────────────

def _logit(p): p = np.clip(p, EPS, 1-EPS); return np.log(p / (1 - p))
def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


class BetaCalib:
    def fit(self, p, y):
        p = np.clip(p, EPS, 1-EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(X, y)
        self.a = float(lr.coef_[0, 0]); self.b = float(lr.coef_[0, 1])
        self.c = float(lr.intercept_[0]); return self
    def predict(self, p):
        p = np.clip(p, EPS, 1-EPS)
        z = self.a * np.log(p) + self.b * (-np.log(1 - p)) + self.c
        return _sigmoid(z)


# ─── Pipeline (same per-fold clean as baseline) ─────────────────────────

def build_through_step6():
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
    train_only = df_full[df_full["DATE"] < train_end].copy()
    print(f"  WC priors from {len(train_only):,} train-only rows  "
          f"(DATE < {train_end.date()})")
    priors = mma.compute_wc_priors(train_only, stat_cols)
    df_with_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    result = mma.assemble_features(df_with_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


def add_priors_features(df: pd.DataFrame) -> pd.DataFrame:
    """All formulas from f1_priors / f2_priors, no fitting required."""
    f1 = df["f1_priors"].astype(float).values
    f2 = df["f2_priors"].astype(float).values
    df = df.copy()
    df["priors_diff"]      = f1 - f2
    df["min_priors_x"]     = np.minimum(f1, f2)
    df["max_priors_x"]     = np.maximum(f1, f2)
    df["priors_log_ratio"] = np.log(f1 + 1.0) - np.log(f2 + 1.0)
    df["rookie_diff"]      = ((f1 <= 5).astype(int) - (f2 <= 5).astype(int)).astype(float)
    df["both_rookie"]      = ((f1 <= 5) & (f2 <= 5)).astype(float)
    return df


def fit_base_en(train_df, usable, lam):
    from retrain_lr_symmetric import flip_row_dataframe
    train_d = pd.concat([train_df, flip_row_dataframe(train_df)], ignore_index=True)
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    w   = np.exp(-lam * (TRAIN_END - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=EN_C, penalty="elasticnet", l1_ratio=EN_L1,
                            solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)
    return lr, imp, sc


def predict_base(lr, imp, sc, df, usable):
    return lr.predict_proba(sc.transform(imp.transform(df[usable])))[:, 1]


# ─── Slice helpers (same as error_analysis) ─────────────────────────────

def slice_metrics(d: pd.DataFrame, p_col: str, y_col: str = "win") -> dict:
    if len(d) == 0: return {"n": 0}
    p = d[p_col].values
    y = d[y_col].astype(int).values
    pc = np.clip(p, EPS, 1-EPS)
    pick = (p >= 0.5).astype(int)
    won_pick = ((pick == 1) == (y == 1)).astype(int)
    return {
        "n":             int(len(d)),
        "model_acc_pct": float(won_pick.mean() * 100),
        "log_loss":      float(-(y*np.log(pc) + (1-y)*np.log(1-pc)).mean()),
        "brier":         float(np.mean((p - y) ** 2)),
    }


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    overall_t0 = time.time()
    print("=" * 78)
    print("WALK-FORWARD WITH PRIORS FEATURES — clean Elastic Net")
    print(f"  Train: {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  Test:  {TRAIN_END.date()} → {TEST_END.date()}")
    print(f"  EN(C={EN_C}, l1={EN_L1})  threshold={THRESHOLD}  λ={LAM}")
    print(f"  NEW features: {NEW_FEATURES}")
    print(f"  Audit: docs/audits/walkforward_priors_features.md")
    print("=" * 78)

    df_full, stat_cols = build_through_step6()
    print(f"\n✓ Step 1-6 build done ({time.time()-overall_t0:.0f}s)")

    print(f"\nFreezing WC priors at train_end={TRAIN_END.date()}...")
    feats_df = split_features(df_full, stat_cols, TRAIN_END)
    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup = Path("data/tmp/mmaai_features.csv.before_priorsfx")
    if feats_csv.exists() and not backup.exists():
        import shutil; shutil.copy2(feats_csv, backup)
    feats_df.to_csv(feats_csv, index=False)

    try:
        for mod in list(sys.modules):
            if (mod.startswith("run_threshold_sweep_both_elos")
                or mod == "retrain_lr_symmetric"
                or mod == "walk_forward_4fold"):
                del sys.modules[mod]
        from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
        from retrain_lr_symmetric import (load_wc_history_from_db, add_wc_features)
        from walk_forward_4fold import select_features

        base = load_base_both_elos()
        df = apply_threshold(base, THRESHOLD)
        df = add_wc_features(df, load_wc_history_from_db())
        df = add_priors_features(df)            # ← NEW

        feats_baseline = select_features(df)
        feats_with_priors = feats_baseline + NEW_FEATURES
        # ensure new features are present and non-degenerate
        for f in NEW_FEATURES:
            assert f in df.columns, f"missing {f}"
        usable_new = [c for c in feats_with_priors
                      if c in df.columns and df[c].std() > 1e-8]
        usable_baseline = [c for c in feats_baseline
                           if c in df.columns and df[c].std() > 1e-8]
        added = [c for c in usable_new if c not in usable_baseline]

        train = df[(df["DATE"] >= TRAIN_START) & (df["DATE"] < TRAIN_END)].copy()
        test  = df[(df["DATE"] >= TRAIN_END)   & (df["DATE"] < TEST_END)].copy()
        train = train.sort_values("DATE").reset_index(drop=True)

        # §1 hard-asserts
        assert train["DATE"].max() < TRAIN_END, "§1 violation"
        assert test["DATE"].min()  >= TRAIN_END, "§1 violation"
        assert not (set(zip(train["DATE"], train["jbout"])) &
                    set(zip(test["DATE"],  test["jbout"]))), "§1 bout overlap"
        print(f"\n✓ Leakage assertions pass")
        print(f"  Train: {len(train):>5,d} fights  Test: {len(test):>5,d} fights")
        print(f"  Baseline features:   {len(usable_baseline)}")
        print(f"  + priors features:   {len(usable_new)}  (added: {added})")

        # ─── Run BOTH baseline and priors-augmented in one job ──────────
        results = {}
        per_test_pred = {}
        for tag, usable in [("baseline", usable_baseline),
                            ("priors",   usable_new)]:
            print(f"\n[{tag}]  fit EN on {len(usable)} features...")
            t0 = time.time()
            lr, imp, sc = fit_base_en(train, usable, LAM)
            n_active = int((np.abs(lr.coef_[0]) > 1e-8).sum())
            print(f"  done in {time.time()-t0:.0f}s  ({n_active}/{len(usable)} active)")
            p_test_raw = predict_base(lr, imp, sc, test, usable)

            # 5-fold CV-OOF for beta calibrator
            kf = KFold(n_splits=N_CV_FOLDS, shuffle=False)
            p_oof = np.zeros(len(train), dtype=np.float64)
            y_oof = train["win"].astype(int).values
            for k, (idx_tr, idx_te) in enumerate(kf.split(train)):
                tr_k, te_k = train.iloc[idx_tr], train.iloc[idx_te]
                lr_k, imp_k, sc_k = fit_base_en(tr_k, usable, LAM)
                p_oof[idx_te] = predict_base(lr_k, imp_k, sc_k, te_k, usable)
            beta = BetaCalib().fit(p_oof, y_oof)
            p_test_beta = beta.predict(p_test_raw)
            print(f"  beta params: a={beta.a:.3f}  b={beta.b:.3f}  c={beta.c:.3f}")

            results[tag] = {"n_active": n_active, "n_features": len(usable),
                            "added_features": added if tag == "priors" else []}
            per_test_pred[tag] = {"uncal": p_test_raw, "beta": p_test_beta}

        # ─── Attach Vegas to test ───────────────────────────────────────
        from build_walkforward_vegas_multi_threshold import attach_vegas_rich
        keys = test[["DATE", "jbout", "jfighter"]].drop_duplicates()
        tv = attach_vegas_rich(keys)
        test_v = test.merge(
            tv[["DATE", "jbout", "jfighter", "p_vegas_f1", "dec_odds_f1", "dec_odds_f2"]],
            on=["DATE", "jbout", "jfighter"], how="left",
        )
        # pre-compute Vegas-pick correctness for slice analysis
        y_v = test_v["win"].astype(int).values
        test_v["vegas_pick_a"] = (test_v["p_vegas_f1"] >= 0.5).astype(int)
        test_v["vegas_won"]    = np.where(test_v["vegas_pick_a"]==1, y_v == 1, y_v == 0).astype(int)

        # min_priors for slice analysis
        test_v["min_priors_test"] = test_v[["f1_priors", "f2_priors"]].min(axis=1)

        # ─── Pooled metrics ─────────────────────────────────────────────
        print()
        print("=" * 78)
        print("POOLED METRICS  (all 391 test fights)")
        print("=" * 78)
        print(f"  {'cell':<22s}  {'n':>4s}  {'acc':>6s}  {'logloss':>7s}  "
              f"{'brier':>6s}")
        print("  " + "-" * 60)
        y_test = test["win"].astype(int).values
        for tag in ("baseline", "priors"):
            for cal in ("uncal", "beta"):
                p = per_test_pred[tag][cal]
                m = slice_metrics(test.assign(_p=p), "_p")
                results[tag][f"pooled_{cal}"] = m
                print(f"  {tag:<10s} {cal:<10s}  {m['n']:>4d}  "
                      f"{m['model_acc_pct']:>5.2f}%  {m['log_loss']:>7.4f}  "
                      f"{m['brier']:>6.4f}")

        # ─── Rookie-slice metrics (pre-registered target) ──────────────
        print()
        print("=" * 78)
        print("ROOKIE 3-4 SLICE  (min(f1_priors, f2_priors) ∈ [3, 4])")
        print("(matched to error_analysis_test_set definition)")
        print("=" * 78)
        rookie_mask = (test_v["min_priors_test"] >= 3) & (test_v["min_priors_test"] < 5)
        print(f"  rookie slice n = {int(rookie_mask.sum())}")
        # Vegas-pick accuracy on this slice (constant — picks unchanged)
        vegas_acc = float(test_v.loc[rookie_mask, "vegas_won"].mean() * 100)
        print(f"  Vegas-pick accuracy on slice: {vegas_acc:.2f}%")
        print(f"  {'cell':<22s}  {'n':>4s}  {'acc':>6s}  {'logloss':>7s}  "
              f"{'brier':>6s}  {'Δ vs Vegas':>10s}")
        print("  " + "-" * 75)
        # Need to align test ↔ test_v ordering
        test_aligned = test.merge(test_v[["DATE","jbout","jfighter","min_priors_test"]],
                                  on=["DATE","jbout","jfighter"], how="left")
        rookie_idx = (test_aligned["min_priors_test"] >= 3) & (test_aligned["min_priors_test"] < 5)
        for tag in ("baseline", "priors"):
            for cal in ("uncal", "beta"):
                p = per_test_pred[tag][cal][rookie_idx.values]
                d = test_aligned[rookie_idx].assign(_p=p)
                m = slice_metrics(d, "_p")
                results[tag][f"rookie_{cal}"] = m
                delta = m["model_acc_pct"] - vegas_acc
                print(f"  {tag:<10s} {cal:<10s}  {m['n']:>4d}  "
                      f"{m['model_acc_pct']:>5.2f}%  {m['log_loss']:>7.4f}  "
                      f"{m['brier']:>6.4f}  {delta:>+9.2f}pp")
        results["rookie_vegas_acc_pct"] = vegas_acc

        # ─── Other slices (lighter — just for sanity, not for selection)
        print()
        print("=" * 78)
        print("SANITY SLICES  (developing 5-9 priors; established 10+)")
        print("=" * 78)
        for label, mask_cond in [
            ("developing 5-9", (test_aligned["min_priors_test"] >= 5) & (test_aligned["min_priors_test"] < 10)),
            ("established 10+", test_aligned["min_priors_test"] >= 10),
        ]:
            n = int(mask_cond.sum())
            v_acc = float(test_v[mask_cond.values]["vegas_won"].mean() * 100) if n else 0
            print(f"\n  {label} (n={n}, Vegas={v_acc:.2f}%)")
            for tag in ("baseline", "priors"):
                for cal in ("uncal", "beta"):
                    p = per_test_pred[tag][cal][mask_cond.values]
                    d = test_aligned[mask_cond].assign(_p=p)
                    m = slice_metrics(d, "_p")
                    delta = m["model_acc_pct"] - v_acc if n else 0
                    print(f"    {tag:<10s} {cal:<10s}  acc={m['model_acc_pct']:>5.2f}%  "
                          f"Δ vs Vegas={delta:>+5.2f}pp")

        # ─── Save predictions parquet ──────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        out_pred = test[["DATE","jevent","jbout","jfighter","opp_jfighter","win",
                         "f1_priors","f2_priors"]].copy()
        for tag in ("baseline", "priors"):
            for cal in ("uncal", "beta"):
                out_pred[f"p_{tag}_{cal}"] = per_test_pred[tag][cal]
        out_pred.to_parquet("results/walkforward_priors_features_predictions.parquet",
                            index=False)

        out = {
            "config": {"train_start": str(TRAIN_START.date()),
                       "train_end": str(TRAIN_END.date()),
                       "test_end": str(TEST_END.date()),
                       "threshold": THRESHOLD, "recency_lambda": LAM,
                       "model": "ElasticNet", "C": EN_C, "l1_ratio": EN_L1,
                       "n_train": int(len(train)), "n_test": int(len(test)),
                       "new_features": NEW_FEATURES},
            "results": results,
            "audit":   "docs/audits/walkforward_priors_features.md",
            "total_runtime_min": round((time.time() - overall_t0)/60, 1),
        }
        Path("results/walkforward_priors_features.json").write_text(
            json.dumps(out, indent=2, default=str))
        print(f"\n✓ Saved results/walkforward_priors_features.json")
        print(f"✓ Saved predictions parquet")
        print(f"\nTotal runtime: {(time.time() - overall_t0)/60:.1f} minutes")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
