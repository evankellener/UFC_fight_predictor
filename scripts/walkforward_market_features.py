"""Re-test market/contextual features on the clean pipeline.

Hypothesis: home_advantage, travel, card_position, timezone, and stance
features showed no lift on the leaky MMA-AI pipeline (pre-2026-04-27)
because 58 *_adjperf_dec_avg_diff columns absorbed the same variance.
On the clean pipeline (~37 active features), they may now contribute
independent signal.

Pre-registered decision rule:
  PASS  → ≥1 feature survives EN-L1 (|coef| > 1e-8) AND
           pooled log-loss / Brier / AUC improve vs baseline
  FAIL  → all features zeroed out, OR pooled metrics degrade

Same train (2016-01 → 2024-10) / test (2024-10 → 2026-04) split as the
honest baseline. Identical EN(C=0.05, l1=0.5), recency λ=1.20, threshold=3.

IMPORTANT ordering invariant (see audit §11):
  `feats_baseline = select_features(df)` is called BEFORE novel market
  features are merged into `df`. This prevents home_advantage_diff and
  card_position_norm_career_diff (both end in `_diff`) from silently
  leaking into the baseline feature list.

Outputs:
  results/walkforward_market_features.json
  results/walkforward_market_features_predictions.parquet

Audit: docs/audits/walkforward_market_features.md
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

# Novel features loaded from market_features_clean.csv.
# Excludes coming_off_loss_diff / win_streak_entering_diff / fights_last_12m_diff
# because those are ALREADY present and active in the baseline.
NOVEL_MARKET_COLS = [
    "home_advantage_diff",
    "travel_distance_diff_km",
    "tz_diff_diff_hr",
    "is_main_event",
    "card_position_norm_career_diff",
]
# Stance features are already in the df from load_base_both_elos but are
# explicitly excluded from select_features() — test them here.
STANCE_COLS = [
    "stance_mismatch",
    "southpaw_advantage_diff",
]
NEW_FEATURES = NOVEL_MARKET_COLS + STANCE_COLS


# ─── Beta calibrator ────────────────────────────────────────────────────────

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class BetaCalib:
    def fit(self, p, y):
        p = np.clip(p, EPS, 1 - EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(X, y)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.coef_[0, 1])
        self.c = float(lr.intercept_[0])
        return self

    def predict(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        z = self.a * np.log(p) + self.b * (-np.log(1 - p)) + self.c
        return _sigmoid(z)


# ─── Pipeline helpers ────────────────────────────────────────────────────────

def build_through_step6():
    print("Building pipeline through Step 6 (per-fight clean)...")
    df = mma.load_base_data()
    df = mma.beta_binomial_smooth(df)
    df = mma.poisson_gamma_smooth(df)
    df = mma.compute_derived_features(df)
    stat_cols = sorted(set(c for c in df.columns if (
        c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
        c.endswith("_ratio") or c.endswith("_per_ctrl") or
        c in ["ko_smooth", "win_smooth", "decision_smooth",
              "sub_land_smooth", "sub_land_rate", "ctrl_pm",
              "ko_per_sig_str_land", "td_per_sig_str_att",
              "ground_per_ctrl", "dist_per_sig_str_land",
              "head_per_sig_str_land", "rev_per_ctrlopp",
              "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
              "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"]
    ) and c in df.columns and not c.startswith("opp_") and not c.endswith("_raw")))
    df = mma.compute_decayed_averages(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    df = mma.compute_opponent_history(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    return df, stat_cols


def split_features(df_full, stat_cols, train_end):
    train_only = df_full[df_full["DATE"] < train_end].copy()
    print(f"  WC priors from {len(train_only):,} train-only rows "
          f"(DATE < {train_end.date()})")
    priors = mma.compute_wc_priors(train_only, stat_cols)
    df_with_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    result = mma.assemble_features(df_with_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


def fit_base_en(train_df, usable, lam):
    from retrain_lr_symmetric import flip_row_dataframe
    train_d = pd.concat([train_df, flip_row_dataframe(train_df)],
                        ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    w   = np.exp(-lam * (TRAIN_END - train_d["DATE"]).dt.days.values / 365.25)
    lr  = LogisticRegression(C=EN_C, penalty="elasticnet", l1_ratio=EN_L1,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)
    return lr, imp, sc


def predict_en(lr, imp, sc, df, usable):
    return lr.predict_proba(sc.transform(imp.transform(df[usable])))[:, 1]


def slice_metrics(d, p_col, y_col="win"):
    if len(d) == 0:
        return {"n": 0}
    p  = d[p_col].values
    y  = d[y_col].astype(int).values
    pc = np.clip(p, EPS, 1 - EPS)
    pick     = (p >= 0.5).astype(int)
    won_pick = ((pick == 1) == (y == 1)).astype(int)
    return {
        "n":             int(len(d)),
        "model_acc_pct": float(won_pick.mean() * 100),
        "log_loss":      float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
        "brier":         float(np.mean((p - y) ** 2)),
        "auc":           float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0_total = time.time()
    print("=" * 78)
    print("WALK-FORWARD — MARKET / CONTEXTUAL FEATURES  (clean Elastic Net)")
    print(f"  Train: {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  Test:  {TRAIN_END.date()} → {TEST_END.date()}")
    print(f"  EN(C={EN_C}, l1={EN_L1})  threshold={THRESHOLD}  λ={LAM}")
    print(f"  NEW features: {NEW_FEATURES}")
    print(f"  Audit: docs/audits/walkforward_market_features.md")
    print("=" * 78)

    # ── Step 1-6: build per-fight features ──────────────────────────────────
    df_full, stat_cols = build_through_step6()
    print(f"\n✓ Step 1-6 done ({time.time() - t0_total:.0f}s)")

    # ── Freeze WC priors at train_end ────────────────────────────────────────
    print(f"\nFreezing WC priors at {TRAIN_END.date()}...")
    feats_df = split_features(df_full, stat_cols, TRAIN_END)
    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_marketfx")
    if feats_csv.exists() and not backup.exists():
        import shutil
        shutil.copy2(feats_csv, backup)
    feats_df.to_csv(feats_csv, index=False)

    try:
        # ── Reload modules after CSV rewrite ──────────────────────────────
        for mod in list(sys.modules):
            if mod.startswith("run_threshold_sweep_both_elos") \
               or mod in ("retrain_lr_symmetric", "walk_forward_4fold"):
                del sys.modules[mod]

        from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
        from retrain_lr_symmetric import (load_wc_history_from_db,
                                          add_wc_features,
                                          flip_row_dataframe)
        from walk_forward_4fold import select_features

        base = load_base_both_elos()
        df   = apply_threshold(base, THRESHOLD)
        df   = add_wc_features(df, load_wc_history_from_db())

        # ── CRITICAL: compute baseline features BEFORE merging novel market
        #    columns, so that home_advantage_diff / card_position_norm_career_diff
        #    (both end in _diff) cannot silently enter the baseline.
        feats_baseline_raw = select_features(df)
        print(f"\nBaseline features (pre-market merge): {len(feats_baseline_raw)}")

        # ── Merge novel market features ──────────────────────────────────
        mf_path = Path("data/tmp/market_features_clean.csv")
        if not mf_path.exists():
            print(f"❌ Missing {mf_path}. Run scripts/build_market_features.py first.")
            return
        mf = pd.read_csv(mf_path, parse_dates=["DATE"])
        mf_cols = ["DATE", "jbout", "jfighter"] + NOVEL_MARKET_COLS
        mf = mf[[c for c in mf_cols if c in mf.columns]].drop_duplicates(
            subset=["DATE", "jbout", "jfighter"])

        # Check which novel cols are actually present
        missing_novel = [c for c in NOVEL_MARKET_COLS if c not in mf.columns]
        if missing_novel:
            print(f"⚠ Novel cols missing from CSV: {missing_novel}")
        available_novel = [c for c in NOVEL_MARKET_COLS if c in mf.columns]

        df = df.merge(mf[["DATE", "jbout", "jfighter"] + available_novel],
                      on=["DATE", "jbout", "jfighter"], how="left")
        print(f"Merged {len(available_novel)} novel market cols: {available_novel}")

        # Stance features are already in df from load_base_both_elos; confirm.
        stance_present = [c for c in STANCE_COLS if c in df.columns]
        stance_missing = [c for c in STANCE_COLS if c not in df.columns]
        if stance_missing:
            print(f"⚠ Stance cols missing from df: {stance_missing}")

        # ── Build usable feature lists ────────────────────────────────────
        usable_baseline = [c for c in feats_baseline_raw
                           if c in df.columns and df[c].std() > 1e-8]
        # Confirm no market features snuck into baseline
        market_in_baseline = [c for c in NEW_FEATURES if c in usable_baseline]
        if market_in_baseline:
            raise RuntimeError(
                f"ORDERING BUG: market features in baseline: {market_in_baseline}"
            )

        new_feats_usable = [
            c for c in (available_novel + stance_present)
            if c not in usable_baseline and df[c].std() > 1e-8
        ]
        usable_market = usable_baseline + new_feats_usable

        # ── Train / test split ────────────────────────────────────────────
        train = df[(df["DATE"] >= TRAIN_START) & (df["DATE"] < TRAIN_END)].copy()
        test  = df[(df["DATE"] >= TRAIN_END)   & (df["DATE"] < TEST_END)].copy()
        train = train.sort_values("DATE").reset_index(drop=True)

        assert train["DATE"].max() < TRAIN_END,          "§1 train/test violation"
        assert test["DATE"].min()  >= TRAIN_END,         "§1 train/test violation"
        assert not (set(zip(train["DATE"], train["jbout"])) &
                    set(zip(test["DATE"],  test["jbout"]))), "§1 bout overlap"

        print(f"\n✓ Leakage assertions pass")
        print(f"  Train: {len(train):>5,d} rows  Test: {len(test):>5,d} rows")
        print(f"  Baseline features : {len(usable_baseline)}")
        print(f"  New market + stance: {len(new_feats_usable)}")
        print(f"  Market+stance total: {len(usable_market)}")
        print(f"  Added: {new_feats_usable}")

        # ── Fit, predict, calibrate ───────────────────────────────────────
        results     = {}
        per_test_pred = {}

        for tag, usable in [("baseline", usable_baseline),
                             ("market",   usable_market)]:
            print(f"\n[{tag}]  fitting EN on {len(usable)} features...")
            t0 = time.time()
            lr, imp, sc = fit_base_en(train, usable, LAM)
            coefs   = lr.coef_[0]
            n_active = int((np.abs(coefs) > 1e-8).sum())
            print(f"  done in {time.time() - t0:.0f}s  ({n_active}/{len(usable)} active)")

            # Print market/stance feature coefficients for the market run
            if tag == "market":
                print(f"\n  Market + stance feature coefficients:")
                for feat in new_feats_usable:
                    if feat in usable:
                        idx = usable.index(feat)
                        print(f"    {feat:<40s}  coef = {coefs[idx]:+.6f}"
                              f"{'  ← ACTIVE' if abs(coefs[idx]) > 1e-8 else ''}")

            p_test_raw = predict_en(lr, imp, sc, test, usable)

            # 5-fold CV-OOF for beta calibrator (KFold shuffle=False → temporal order)
            kf    = KFold(n_splits=N_CV_FOLDS, shuffle=False)
            p_oof = np.zeros(len(train), dtype=np.float64)
            y_oof = train["win"].astype(int).values
            for k, (idx_tr, idx_te) in enumerate(kf.split(train)):
                tr_k, te_k     = train.iloc[idx_tr], train.iloc[idx_te]
                lr_k, imp_k, sc_k = fit_base_en(tr_k, usable, LAM)
                p_oof[idx_te]  = predict_en(lr_k, imp_k, sc_k, te_k, usable)

            beta        = BetaCalib().fit(p_oof, y_oof)
            p_test_beta = beta.predict(p_test_raw)
            print(f"  beta: a={beta.a:.3f}  b={beta.b:.3f}  c={beta.c:.3f}")

            results[tag]    = {
                "n_features": len(usable), "n_active": n_active,
                "new_features_tested": new_feats_usable if tag == "market" else [],
                "active_new": [f for f in new_feats_usable
                               if f in usable
                               and abs(coefs[usable.index(f)]) > 1e-8],
            }
            per_test_pred[tag] = {"uncal": p_test_raw, "beta": p_test_beta}

        # ── Pooled metrics ────────────────────────────────────────────────
        print()
        print("=" * 78)
        print("POOLED METRICS")
        print("=" * 78)
        print(f"  {'cell':<20s}  {'n':>4s}  {'acc':>6s}  {'logloss':>7s}  "
              f"{'brier':>6s}  {'auc':>6s}")
        print("  " + "-" * 65)
        y_test = test["win"].astype(int).values
        for tag in ("baseline", "market"):
            for cal in ("uncal", "beta"):
                p = per_test_pred[tag][cal]
                m = slice_metrics(test.assign(_p=p), "_p")
                results[tag][f"pooled_{cal}"] = m
                auc_str = f"{m['auc']:.4f}" if m["auc"] else "  n/a"
                print(f"  {tag:<10s} {cal:<8s}  {m['n']:>4d}  "
                      f"{m['model_acc_pct']:>5.2f}%  {m['log_loss']:>7.4f}  "
                      f"{m['brier']:>6.4f}  {auc_str}")

        # Δ row
        print()
        print("  Δ (market − baseline):")
        for cal in ("uncal", "beta"):
            bm = results["baseline"][f"pooled_{cal}"]
            mm = results["market"][f"pooled_{cal}"]
            d_acc = mm["model_acc_pct"] - bm["model_acc_pct"]
            d_ll  = mm["log_loss"]      - bm["log_loss"]
            d_br  = mm["brier"]         - bm["brier"]
            d_auc = (mm["auc"] - bm["auc"]) if mm["auc"] and bm["auc"] else None
            auc_str = f"{d_auc:+.4f}" if d_auc else "   n/a"
            print(f"  {'':10s} {cal:<8s}         "
                  f"  {d_acc:>+5.2f}pp  {d_ll:>+7.4f}  "
                  f"{d_br:>+6.4f}  {auc_str}")

        # ── Attach Vegas for slice reporting ──────────────────────────────
        from build_walkforward_vegas_multi_threshold import attach_vegas_rich
        keys = test[["DATE", "jbout", "jfighter"]].drop_duplicates()
        tv   = attach_vegas_rich(keys)
        test_v = test.merge(
            tv[["DATE", "jbout", "jfighter", "p_vegas_f1"]],
            on=["DATE", "jbout", "jfighter"], how="left",
        )
        y_v = test_v["win"].astype(int).values
        test_v["vegas_pick_a"] = (test_v["p_vegas_f1"] >= 0.5).astype(int)
        test_v["vegas_won"]    = np.where(
            test_v["vegas_pick_a"] == 1, y_v == 1, y_v == 0).astype(int)
        test_v["min_priors"] = test_v[["f1_priors", "f2_priors"]].min(axis=1)

        # Re-align test ↔ test_v (merge result may differ in order/length)
        test_aligned = test.merge(
            test_v[["DATE", "jbout", "jfighter", "min_priors", "vegas_won"]],
            on=["DATE", "jbout", "jfighter"], how="left",
        )

        # ── Rookie slice (3-4 priors) ─────────────────────────────────────
        print()
        print("=" * 78)
        print("ROOKIE SLICE  (min(f1_priors, f2_priors) ∈ [3, 4])")
        print("=" * 78)
        rookie_mask = ((test_aligned["min_priors"] >= 3) &
                       (test_aligned["min_priors"] < 5))
        n_rookie    = int(rookie_mask.sum())
        v_acc_rook  = float(test_v[rookie_mask.values]["vegas_won"].mean() * 100) \
                      if n_rookie else 0.0
        print(f"  n={n_rookie}  Vegas accuracy={v_acc_rook:.2f}%")
        print(f"  {'cell':<20s}  {'acc':>6s}  {'Δ vs Vegas':>10s}")
        for tag in ("baseline", "market"):
            for cal in ("uncal", "beta"):
                p = per_test_pred[tag][cal][rookie_mask.values]
                d = test_aligned[rookie_mask].assign(_p=p)
                m = slice_metrics(d, "_p")
                results[tag][f"rookie_{cal}"] = m
                if n_rookie:
                    delta = m["model_acc_pct"] - v_acc_rook
                    print(f"  {tag:<10s} {cal:<8s}  {m['model_acc_pct']:>5.2f}%  "
                          f"{delta:>+9.2f}pp")
        results["rookie_vegas_acc_pct"] = v_acc_rook

        # ── Pre-registered decision ────────────────────────────────────────
        active_new = results["market"].get("active_new", [])
        b_ll  = results["baseline"]["pooled_beta"]["log_loss"]
        m_ll  = results["market"]["pooled_beta"]["log_loss"]
        b_brier = results["baseline"]["pooled_beta"]["brier"]
        m_brier = results["market"]["pooled_beta"]["brier"]
        b_auc = results["baseline"]["pooled_beta"]["auc"] or 0
        m_auc = results["market"]["pooled_beta"]["auc"] or 0

        passed = bool(active_new) and (m_ll < b_ll or m_brier < b_brier or m_auc > b_auc)
        print()
        print("=" * 78)
        print("PRE-REGISTERED DECISION")
        print("=" * 78)
        print(f"  Active new features (|coef|>1e-8): {active_new if active_new else 'NONE'}")
        print(f"  ΔLL    = {m_ll - b_ll:+.4f}  ({'improvement' if m_ll < b_ll else 'degradation'})")
        print(f"  ΔBrier = {m_brier - b_brier:+.4f}  ({'improvement' if m_brier < b_brier else 'degradation'})")
        print(f"  ΔAUC   = {m_auc - b_auc:+.4f}  ({'improvement' if m_auc > b_auc else 'degradation'})")
        print()
        if passed:
            print("  ✅ HYPOTHESIS SUPPORTED — at least one feature is active "
                  "AND ≥1 probability metric improved.")
        else:
            print("  ❌ HYPOTHESIS REJECTED — either no features survived EN-L1 "
                  "or all probability metrics degraded.")

        # ── Save outputs ──────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)

        out_pred = test[["DATE", "jevent", "jbout", "jfighter",
                          "opp_jfighter", "win",
                          "f1_priors", "f2_priors"]].copy()
        for tag in ("baseline", "market"):
            for cal in ("uncal", "beta"):
                out_pred[f"p_{tag}_{cal}"] = per_test_pred[tag][cal]
        out_pred.to_parquet(
            "results/walkforward_market_features_predictions.parquet",
            index=False,
        )

        out = {
            "config": {
                "train_start":     str(TRAIN_START.date()),
                "train_end":       str(TRAIN_END.date()),
                "test_end":        str(TEST_END.date()),
                "threshold":       THRESHOLD,
                "recency_lambda":  LAM,
                "model":           "ElasticNet",
                "C":               EN_C,
                "l1_ratio":        EN_L1,
                "n_train":         int(len(train)),
                "n_test":          int(len(test)),
                "new_features":    NEW_FEATURES,
                "new_features_usable": new_feats_usable,
            },
            "results":  results,
            "decision": {
                "passed":      passed,
                "active_new":  active_new,
                "delta_ll":    float(m_ll - b_ll),
                "delta_brier": float(m_brier - b_brier),
                "delta_auc":   float(m_auc - b_auc),
            },
            "audit":    "docs/audits/walkforward_market_features.md",
            "total_runtime_min": round((time.time() - t0_total) / 60, 1),
        }
        Path("results/walkforward_market_features.json").write_text(
            json.dumps(out, indent=2, default=str)
        )
        print(f"\n✓ Saved results/walkforward_market_features.json")
        print(f"✓ Saved results/walkforward_market_features_predictions.parquet")
        print(f"\nTotal runtime: {(time.time() - t0_total) / 60:.1f} min")

    finally:
        # Restore mmaai_features.csv unconditionally
        if backup.exists():
            import shutil
            shutil.copy2(backup, feats_csv)
            backup.unlink()


if __name__ == "__main__":
    main()
