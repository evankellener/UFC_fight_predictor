"""Sweep training recency-weight λ on the clean market-features pipeline.

Evaluates λ ∈ {0.13, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8} with all else fixed:
  - Same train (2016-01 → 2024-10) / test (2024-10 → 2026-04)
  - Same feature set as walkforward_market_features.py
  - EN(C=0.05, l1=0.5) — only λ changes
  - Uncalibrated predictions (for clean λ comparison; calibration adds noise)

Per-quarter fold breakdown included to show Q1-2026 drift behaviour.

⚠ MULTIPLE-COMPARISON WARNING (see audit §6):
  λ is swept on the same test set used by all prior experiments.
  Selecting the best λ from this sweep = test-set selection.
  Treat results as exploratory. Pre-register the chosen λ and
  validate on future data before deploying.

Audit: docs/audits/lambda_sweep_clean.md
Outputs: results/lambda_sweep_clean.json
"""
import sys, json, time, warnings
from pathlib import Path
from itertools import combinations
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

import mma_ai_pipeline as mma

LAMBDA_SWEEP = [0.13, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8]
TRAIN_START  = pd.Timestamp("2016-01-01")
TRAIN_END    = pd.Timestamp("2024-10-01")
TEST_END     = pd.Timestamp("2026-04-01")
THRESHOLD    = 3
EN_C         = 0.05
EN_L1        = 0.5
EPS          = 1e-6

QUARTERS = [
    ("Q4-2024", "2024-10-01", "2025-01-01"),
    ("Q1-2025", "2025-01-01", "2025-04-01"),
    ("Q2-2025", "2025-04-01", "2025-07-01"),
    ("Q3-2025", "2025-07-01", "2025-10-01"),
    ("Q4-2025", "2025-10-01", "2026-01-01"),
    ("Q1-2026", "2026-01-01", "2026-04-01"),
]

NOVEL_MARKET_COLS = [
    "home_advantage_diff",
    "travel_distance_diff_km",
    "tz_diff_diff_hr",
    "is_main_event",
    "card_position_norm_career_diff",
]


# ── Pipeline build (once) ─────────────────────────────────────────────────

def build_through_step6():
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
    priors = mma.compute_wc_priors(train_only, stat_cols)
    df_with_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    result = mma.assemble_features(df_with_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


def load_dataset():
    """Build once, return (train_df, test_df, usable_features)."""
    df_full, stat_cols = build_through_step6()
    feats_df = split_features(df_full, stat_cols, TRAIN_END)
    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_lamsweep")
    if feats_csv.exists() and not backup.exists():
        import shutil; shutil.copy2(feats_csv, backup)
    feats_df.to_csv(feats_csv, index=False)

    for mod in list(sys.modules):
        if mod.startswith("run_threshold_sweep_both_elos") \
           or mod in ("retrain_lr_symmetric", "walk_forward_4fold"):
            del sys.modules[mod]

    from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
    from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features
    from walk_forward_4fold import select_features

    base = load_base_both_elos()
    df   = apply_threshold(base, THRESHOLD)
    df   = add_wc_features(df, load_wc_history_from_db())

    # Compute baseline features BEFORE merging market cols
    feats_baseline = select_features(df)

    mf_path = Path("data/tmp/market_features_clean.csv")
    if mf_path.exists():
        mf = pd.read_csv(mf_path, parse_dates=["DATE"])
        avail = [c for c in NOVEL_MARKET_COLS if c in mf.columns]
        df = df.merge(mf[["DATE","jbout","jfighter"]+avail],
                      on=["DATE","jbout","jfighter"], how="left")
        new_feats = [c for c in avail
                     if c not in feats_baseline and df[c].std() > 1e-8]
    else:
        new_feats = []

    usable = [c for c in feats_baseline + new_feats
              if c in df.columns and df[c].std() > 1e-8]

    train = df[(df["DATE"] >= TRAIN_START) & (df["DATE"] < TRAIN_END)].copy()
    test  = df[(df["DATE"] >= TRAIN_END)   & (df["DATE"] < TEST_END)].copy()
    train = train.sort_values("DATE").reset_index(drop=True)

    assert train["DATE"].max() < TRAIN_END,          "§1 violation"
    assert test["DATE"].min()  >= TRAIN_END,         "§1 violation"
    assert not (set(zip(train["DATE"], train["jbout"])) &
                set(zip(test["DATE"],  test["jbout"]))), "§1 bout overlap"

    return train, test, usable, backup, feats_csv


# ── Fit / predict ─────────────────────────────────────────────────────────

def fit_predict(train, test, usable, lam):
    from retrain_lr_symmetric import flip_row_dataframe
    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    w   = np.exp(-lam * (TRAIN_END - train_d["DATE"]).dt.days.values / 365.25)
    lr  = LogisticRegression(C=EN_C, penalty="elasticnet", l1_ratio=EN_L1,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)
    Xte = sc.transform(imp.transform(test[usable]))
    return lr.predict_proba(Xte)[:, 1]


# ── Metrics helpers ───────────────────────────────────────────────────────

def pooled_metrics(y, p):
    pc   = np.clip(p, EPS, 1 - EPS)
    pred = (p >= 0.5).astype(int)
    return dict(
        acc   = float(accuracy_score(y, pred) * 100),
        ll    = float(log_loss(y, pc)),
        brier = float(brier_score_loss(y, pc)),
        auc   = float(roc_auc_score(y, p)),
    )


def roi_straight(d, mask=None):
    d2 = d[mask] if mask is not None else d
    if len(d2) == 0:
        return None, 0
    pnl = np.where(d2["won_pick"] == 1, d2["dec_odds_pick"] - 1.0, -1.0)
    return float(pnl.mean() * 100), int(len(d2))


def attach_pick_cols(test_df, p_arr):
    d = test_df.copy()
    d["p_pred"]        = p_arr
    d["pick_a"]        = (d["p_pred"] >= 0.5).astype(int)
    d["dec_odds_pick"] = np.where(d["pick_a"]==1, d["dec_odds_f1"], d["dec_odds_f2"])
    d["p_pick"]        = np.where(d["pick_a"]==1, d["p_pred"],      1 - d["p_pred"])
    d["p_vegas_pick"]  = np.where(d["pick_a"]==1, d["p_vegas_f1"],  1 - d["p_vegas_f1"])
    d["edge"]          = d["p_pick"] - d["p_vegas_pick"]
    d["ev"]            = d["p_pick"] * d["dec_odds_pick"] - 1.0
    y = d["win"].astype(int).values
    d["won_pick"] = np.where(d["pick_a"]==1, y==1, y==0).astype(int)
    return d


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 78)
    print("LAMBDA SWEEP — clean market-features pipeline")
    print(f"  λ values: {LAMBDA_SWEEP}")
    print(f"  ⚠  exploratory — do not select λ from these test-set results")
    print(f"  Audit: docs/audits/lambda_sweep_clean.md")
    print("=" * 78)

    print("\nBuilding pipeline (once)...")
    train, test, usable, backup, feats_csv = load_dataset()
    print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(usable)}")

    # Attach Vegas to test
    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    keys   = test[["DATE","jbout","jfighter"]].drop_duplicates()
    tv     = attach_vegas_rich(keys)
    test_v = test.merge(
        tv[["DATE","jbout","jfighter","p_vegas_f1","dec_odds_f1","dec_odds_f2"]],
        on=["DATE","jbout","jfighter"], how="left",
    )
    test_v = test_v[test_v["p_vegas_f1"].notna()].copy()
    # Dedupe to one row per bout for ROI analysis
    test_v_dedup = test_v.sort_values("p_vegas_f1", ascending=False)\
                         .drop_duplicates(subset=["DATE","jbout"])\
                         .reset_index(drop=True)
    y_full = test["win"].astype(int).values
    print(f"  Vegas-matched (all rows): {len(test_v)}")
    print(f"  Vegas-matched (deduped):  {len(test_v_dedup)}")

    try:
        results = {}

        # Header
        qnames = [q[0] for q in QUARTERS]
        col_w  = 8

        print(f"\n{'':>6s}  {'acc':>6s}  {'ll':>6s}  {'auc':>6s}  "
              f"{'ROI_all':>8s}  {'ROI_+EV':>8s}  {'ROI_p65':>8s}  "
              + "  ".join(f"{q:>{col_w}s}" for q in qnames)
              + "  notes")
        print("-" * (6+6+6+6+8+8+8+4 + (col_w+2)*6 + 10))

        for lam in LAMBDA_SWEEP:
            t1 = time.time()
            p  = fit_predict(train, test, usable, lam)

            # Pooled metrics (all test rows, both perspectives)
            m  = pooled_metrics(y_full, p)

            # Align predictions onto deduped Vegas rows
            # test rows are in the same order as test_v (before dedup);
            # we need to get p for each row in test_v_dedup
            p_series = pd.Series(p, index=test.index)
            test_v2  = test_v_dedup.copy()
            test_v2["p_raw"] = test_v2.index.map(
                lambda i: p_series.get(i, np.nan)
            )
            # Some indices may not align if test_v has different index;
            # safer: merge on jbout
            p_map = (test.assign(p_raw=p)
                        .drop_duplicates(subset=["DATE","jbout"])
                        [["DATE","jbout","p_raw"]])
            test_v2 = test_v_dedup.merge(p_map, on=["DATE","jbout"], how="left")
            test_v2 = test_v2.dropna(subset=["p_raw"])

            d = attach_pick_cols(test_v2, test_v2["p_raw"].values)

            roi_all, n_all = roi_straight(d)
            roi_ev,  n_ev  = roi_straight(d, d["ev"] > 0)
            roi_p65, n_p65 = roi_straight(d, (d["ev"]>0) & (d["p_pick"]>=0.65))

            # Per-quarter ROI
            q_rois = []
            for _, qs, qe in QUARTERS:
                dq = d[(d["DATE"]>=qs)&(d["DATE"]<qe)]
                roi_q, nq = roi_straight(dq)
                q_rois.append((roi_q, nq))

            results[lam] = {
                "acc": m["acc"], "ll": m["ll"], "auc": m["auc"], "brier": m["brier"],
                "roi_all": roi_all, "roi_ev": roi_ev, "roi_p65": roi_p65,
                "n_all": n_all, "n_ev": n_ev, "n_p65": n_p65,
                "quarters": {q[0]: {"roi": r, "n": n}
                             for q, (r, n) in zip(QUARTERS, q_rois)},
            }

            # Count positive quarters
            pos_q = sum(1 for r, n in q_rois if r is not None and r > 0)

            q_str = "  ".join(
                f"{r:>+{col_w-1}.1f}%" if r is not None else f"{'—':>{col_w}s}"
                for r, n in q_rois
            )
            note = "← current" if lam == 1.2 else ""
            print(f"λ={lam:<4g}  {m['acc']:>5.2f}%  {m['ll']:>6.4f}  {m['auc']:>6.4f}  "
                  f"{roi_all:>+7.1f}%  {roi_ev:>+7.1f}%  {roi_p65:>+7.1f}%  "
                  f"{q_str}  {pos_q}/6  {note}")

        # ── Best λ summary ───────────────────────────────────────────────
        print()
        print("=" * 78)
        print("BEST λ PER METRIC (exploratory — do not treat as validated)")
        print("=" * 78)
        for metric, key, better in [
            ("Accuracy ↑",   "acc",     max),
            ("Log loss ↓",   "ll",      min),
            ("AUC ↑",        "auc",     max),
            ("ROI all ↑",    "roi_all", max),
            ("ROI +EV ↑",    "roi_ev",  max),
            ("ROI p≥0.65 ↑", "roi_p65", max),
        ]:
            vals   = {lam: results[lam][key] for lam in LAMBDA_SWEEP
                      if results[lam][key] is not None}
            best   = better(vals, key=lambda k: vals[k])
            print(f"  {metric:<16s}  best λ={best:<5g}  value={vals[best]:.4f}")

        # ── Q1-2026 specifically ─────────────────────────────────────────
        print()
        print("Q1-2026 ROI by λ (the drift quarter):")
        for lam in LAMBDA_SWEEP:
            q = results[lam]["quarters"].get("Q1-2026", {})
            roi = q.get("roi"); n = q.get("n", 0)
            if roi is not None:
                print(f"  λ={lam:<4g}  ROI={roi:>+6.1f}%  n={n}")

        # ── Save ─────────────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        Path("results/lambda_sweep_clean.json").write_text(
            json.dumps({
                "lambda_sweep": LAMBDA_SWEEP,
                "train_end": str(TRAIN_END.date()),
                "test_end":  str(TEST_END.date()),
                "warning":   "exploratory — λ swept on test set; not validated",
                "results":   results,
                "audit":     "docs/audits/lambda_sweep_clean.md",
                "runtime_min": round((time.time()-t0)/60, 1),
            }, indent=2, default=str)
        )
        print(f"\n✓ Saved results/lambda_sweep_clean.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil
            shutil.copy2(backup, feats_csv)
            backup.unlink()


if __name__ == "__main__":
    main()
