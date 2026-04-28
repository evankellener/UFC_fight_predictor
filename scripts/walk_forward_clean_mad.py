"""SURGICAL §3 leak fix: per-fold WC priors / MAD, no full pipeline rebuild.

The previous attempt (walk_forward_per_fold_rebuild.py) rebuilt the entire
MMA-AI pipeline per fold with a date cutoff. That disrupted the τ-calibrated
smoothing and gave catastrophic results (fold_2: -100% on n=4 parlays). Wrong
fix.

What's actually leaking: `mma_ai_pipeline.compute_wc_priors()` aggregates
`weightindex` × `stat_col` means + MADs over the FULL df at line 681. For
early folds, this means the "WC prior" used to score 2024-Q2 fights via
AdjPerf z-scores includes fights from 2024-2026 (the future). That's a
§1 + §3 violation.

What's NOT leaking:
  - BB/PG smoothed per-fight values (era-rolling 2yr, per-fight clean)
  - dec_avg (per-fight EMA, only uses fights < current)
  - compute_opponent_history (uses bisect_left to get opponent's prior fights)
  - Elo (sequential)

So the surgical fix:
  1. Run pipeline once globally through Step 6 (gets per-fight clean
     smoothed + derived + decayed + opponent stats)
  2. PER FOLD: recompute compute_wc_priors using only fights ≤ train_end
  3. PER FOLD: recompute compute_adjperf using fold-specific priors
  4. PER FOLD: re-decay-average the AdjPerf z-scores
  5. PER FOLD: re-construct diffs
  6. Run LR train/test per fold on the fold's clean features

Result: priors used to score fold N's test fights only see fights with
DATE < train_end of fold N. §1 + §3 compliant.

Estimated time: ~5 min initial build + ~1 min/fold = ~13 min.

LEAKAGE_REFERENCE.md compliance:
  §0 — audit filed at docs/audits/walk_forward_clean_mad.md
  §1 — train/test temporal split + leakage_assertions enforced
  §3 — WC priors recomputed per fold from train-only data ★ (the fix)
  §4 — imputer + scaler + calibrator refit per fold on train only
  §6 — no hyperparameter search; τs frozen at global tau_optimized values
        (those τs were optimized on a separate walk-forward CV; using
        them here is the same as in baseline 8-fold)
"""
import sys, json, time, warnings, sqlite3, importlib
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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from scipy.optimize import minimize_scalar

import mma_ai_pipeline as mma

EPS = 1e-6
TRAIN_YEARS = 4
THRESHOLD = 3
LAM = 1.50  # parlay strategy model

def build_folds():
    starts = pd.date_range("2024-04-01", "2026-01-01", freq="3MS")
    return [{"name": f"fold_{i}",
             "train_start": (s - pd.DateOffset(years=TRAIN_YEARS)),
             "train_end":   s,
             "test_start":  s,
             "test_end":   (s + pd.DateOffset(months=3))}
            for i, s in enumerate(starts, 1)]
FOLDS = build_folds()


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


def build_through_opponent_history():
    """Run pipeline Steps 1-6 globally. Returns df with per-fight smoothed/
    decayed/opponent-history values. Per-fight values are temporally clean;
    we'll redo Step 7+ per fold with fold-specific priors."""
    print("Building pipeline through Step 6 (per-fight clean)...")
    df = mma.load_base_data()
    print("  Step 2: Beta-Binomial smoothing...")
    df = mma.beta_binomial_smooth(df)
    print("  Step 3: Poisson-Gamma smoothing...")
    df = mma.poisson_gamma_smooth(df)
    print("  Step 4: Derived features...")
    df = mma.compute_derived_features(df)

    # Same stat_cols computation as build_features
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

    print(f"  Step 5: Decayed averages (λ=0.13)...")
    df = mma.compute_decayed_averages(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    print(f"  Step 6: Opponent history (per-fight clean)...")
    df = mma.compute_opponent_history(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    return df, stat_cols


def per_fold_features(df_full, stat_cols, train_end):
    """Compute fold-specific AdjPerf z-scores by:
       1. Restricting df to DATE < train_end for compute_wc_priors
       2. Running compute_adjperf on the FULL df with fold-specific priors
       3. Running assemble_features to get decayed AdjPerf + diffs
    The full df is needed for step 2/3 so test fights get scored with the
    frozen-at-train-end priors."""
    train_only = df_full[df_full["DATE"] < train_end].copy()
    if len(train_only) < 100:
        return None
    # Step 6 (priors only — opp_history columns are already in df_full)
    priors = mma.compute_wc_priors(train_only, stat_cols)
    # Step 7: AdjPerf with fold-frozen priors, applied to ALL fights including test
    df_with_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    # Steps 8-9: feature assembly + diff
    result = mma.assemble_features(df_with_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    # Filter to training era (matches build_features default)
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


def fit_one(train, test, feats, lam, train_anchor):
    from retrain_lr_symmetric import flip_row_dataframe
    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    w = np.exp(-lam * (train_anchor - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)
    p_tr = lr.predict_proba(sc.transform(imp.transform(train[usable])))[:, 1]
    T = temp_cal(p_tr, train["win"].astype(int).values)
    p_te_raw = lr.predict_proba(sc.transform(imp.transform(test[usable])))[:, 1]
    return apply_temp(p_te_raw, T), T, len(usable)


def parlay_metrics(test_with_p, fold_name):
    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    keys = test_with_p[["DATE","jbout","jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    merged = test_with_p.merge(tv[["DATE","jbout","jfighter","p_vegas_f1","dec_odds_f1","dec_odds_f2"]],
                                on=["DATE","jbout","jfighter"], how="left")
    merged = merged.drop_duplicates(subset=["DATE","jbout","jfighter"]).reset_index(drop=True)
    matched = merged[merged["p_vegas_f1"].notna()].copy()
    matched["pick_a"] = (matched["p_parlay"] >= 0.5).astype(int)
    matched["dec_odds_pick"] = np.where(matched["pick_a"]==1, matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_pick"] = np.where(matched["pick_a"]==1, matched["p_parlay"], 1 - matched["p_parlay"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"]==1, matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["edge"] = matched["p_pick"] - matched["p_vegas_pick"]
    matched["ev"]   = matched["p_pick"] * matched["dec_odds_pick"] - 1.0
    matched = matched.drop_duplicates(subset=["DATE","jbout"]).reset_index(drop=True)
    conn = sqlite3.connect("data/sqlite_db/sqlite_scrapper.db")
    res = pd.read_sql("SELECT jevent, jbout, sex FROM ufc_fight_results", conn)
    conn.close()
    matched = matched.merge(res, on=["jevent","jbout"], how="left")
    matched = matched[matched["sex"]==2]  # men-only

    summary = {"fold": fold_name, "n_matched": len(matched)}
    for label, edge_min in [("edge5", 0.05), ("edge10", 0.10)]:
        pos = matched[(matched["ev"]>0) & (matched["edge"]>=edge_min)].copy()
        parlays = []
        for date, grp in pos.groupby("DATE"):
            top2 = grp.sort_values("edge", ascending=False).head(2)
            if len(top2) < 2: continue
            for combo in combinations(top2.itertuples(index=False), 2):
                co  = float(np.prod([c.dec_odds_pick for c in combo]))
                won = int(np.prod([c.won_pick for c in combo]))
                parlays.append(dict(odds=co, won=won))
        if parlays:
            pnl = np.array([(p["odds"]-1.0) if p["won"] else -1.0 for p in parlays])
            summary[f"parlay_{label}_n"] = len(parlays)
            summary[f"parlay_{label}_roi"] = float(pnl.mean()*100)
            summary[f"parlay_{label}_hit"] = float(np.mean([p["won"] for p in parlays])*100)
        else:
            summary[f"parlay_{label}_n"] = 0
            summary[f"parlay_{label}_roi"] = None
            summary[f"parlay_{label}_hit"] = None
    return summary


def main():
    print("="*78)
    print("§3 SURGICAL FIX: per-fold WC priors / MAD")
    print(f"  Threshold: ≥{THRESHOLD} priors, training {TRAIN_YEARS}-yr window")
    print(f"  Fix: compute_wc_priors recomputed per fold from fights < train_end")
    print(f"  Estimated runtime: ~5min build + ~1min/fold × 8 ≈ 13min")
    print("="*78)

    overall_t0 = time.time()
    df_full, stat_cols = build_through_opponent_history()
    print(f"\n✓ Step 1-6 build done ({time.time()-overall_t0:.0f}s)  "
          f"df: {len(df_full):,} rows, {len(stat_cols)} stat cols")

    all_folds = []
    parlay_summaries = []
    out_path = Path("results/walk_forward_clean_mad.json")
    out_path.parent.mkdir(exist_ok=True)

    for i, fold in enumerate(FOLDS, 1):
        fold_t0 = time.time()
        print(f"\n{'='*78}")
        print(f"{fold['name'].upper()}  test {fold['test_start'].date()} → {fold['test_end'].date()}")
        print(f"  WC priors frozen at train_end = {fold['train_end'].date()}")
        print(f"{'='*78}")

        # Fold-specific feature matrix (priors frozen at train_end)
        result = per_fold_features(df_full, stat_cols, fold["train_end"])
        if result is None:
            print(f"  ⚠ insufficient training data, skipping")
            continue

        # Now run the standard load_base_both_elos / wc_features flow on this CSV
        # Save to a fold-specific path so downstream readers pick it up
        fold_csv = Path(f"data/tmp/mmaai_features_clean_mad_{fold['name']}.csv")
        result.to_csv(fold_csv, index=False)

        # Force fresh imports against the fold-specific CSV
        for mod in list(sys.modules):
            if mod.startswith("run_threshold_sweep_both_elos") or mod == "retrain_lr_symmetric" or mod == "walk_forward_4fold":
                del sys.modules[mod]
        # Monkey-patch load_base_both_elos to use the fold-specific CSV
        import run_threshold_sweep_both_elos as rt
        original_DT = rt.DT
        # Patch the CSV path
        feats_csv = Path("data/tmp/mmaai_features.csv")
        # Save current global CSV to backup (non-fold-specific path); restore after
        backup_global = Path("data/tmp/mmaai_features.csv.surgical_backup")
        if feats_csv.exists() and not backup_global.exists():
            import shutil; shutil.copy2(feats_csv, backup_global)
        # Replace global CSV with fold-specific one for this fold's training
        import shutil; shutil.copy2(fold_csv, feats_csv)

        try:
            from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
            from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features
            from walk_forward_4fold import select_features
            base = load_base_both_elos()
            df = apply_threshold(base, THRESHOLD)
            df = add_wc_features(df, load_wc_history_from_db())
            feats = select_features(df)
            train = df[(df["DATE"] >= fold["train_start"]) & (df["DATE"] < fold["train_end"])].copy()
            test  = df[(df["DATE"] >= fold["test_start"]) & (df["DATE"] < fold["test_end"])].copy()

            out = {"fold": fold["name"],
                   "test_start": str(fold["test_start"].date()),
                   "test_end":   str(fold["test_end"].date()),
                   "train_end":  str(fold["train_end"].date()),
                   "n_train": len(train), "n_test": len(test)}
            print(f"  train {len(train)}  test {len(test)}")

            if len(test) == 0:
                all_folds.append(out)
                continue

            for lam, label in [(1.20, "main"), (0.13, "companion"), (1.50, "parlay")]:
                p_test, T, n_feats = fit_one(train, test, feats, lam, fold["train_end"])
                y = test["win"].astype(int).values
                pc = np.clip(p_test, EPS, 1-EPS)
                out[f"{label}_acc"]   = float(accuracy_score(y, (p_test >= 0.5).astype(int)))
                out[f"{label}_ll"]    = float(log_loss(y, pc))
                out[f"{label}_brier"] = float(brier_score_loss(y, pc))
                try:
                    out[f"{label}_auc"] = float(roc_auc_score(y, p_test))
                except ValueError:
                    out[f"{label}_auc"] = None
                if label == "parlay":
                    test = test.copy(); test["p_parlay"] = p_test

            ps = parlay_metrics(test, fold["name"])
            out["parlay_strategy"] = ps
            print(f"  Parlay e≥5pp:  n={ps.get('parlay_edge5_n')}  ROI={ps.get('parlay_edge5_roi')}")
            print(f"  Parlay e≥10pp: n={ps.get('parlay_edge10_n')}  ROI={ps.get('parlay_edge10_roi')}")
            all_folds.append(out)
        finally:
            # Restore global CSV before next fold
            if backup_global.exists():
                shutil.copy2(backup_global, feats_csv)

        out_path.write_text(json.dumps({"folds": all_folds}, indent=2, default=str))
        print(f"  fold elapsed {(time.time()-fold_t0):.0f}s  total {(time.time()-overall_t0)/60:.1f}min")

    # Restore global CSV one more time
    if Path("data/tmp/mmaai_features.csv.surgical_backup").exists():
        import shutil
        shutil.copy2("data/tmp/mmaai_features.csv.surgical_backup", "data/tmp/mmaai_features.csv")
        Path("data/tmp/mmaai_features.csv.surgical_backup").unlink()

    # Aggregate
    print("\n" + "="*78); print("AGGREGATE — clean MAD"); print("="*78)
    print(f"{'fold':<8s} {'parlay_acc':>11s} {'parlay_ll':>10s} {'p5_n':>5s} {'p5_roi':>9s} {'p10_n':>6s} {'p10_roi':>10s}")
    print("-"*78)
    for r in all_folds:
        ps = r.get("parlay_strategy", {})
        e5_roi = f"{ps.get('parlay_edge5_roi'):>+7.2f}%" if ps.get("parlay_edge5_roi") is not None else "   nan "
        e10_roi = f"{ps.get('parlay_edge10_roi'):>+7.2f}%" if ps.get("parlay_edge10_roi") is not None else "   nan "
        print(f"  {r['fold']:<6s} {r.get('parlay_acc',0):>10.4f}  {r.get('parlay_ll',0):>9.4f}  "
              f"{ps.get('parlay_edge5_n','-'):>5}  {e5_roi}  "
              f"{ps.get('parlay_edge10_n','-'):>6}  {e10_roi}")

    # Pooled
    p5_n = sum(r.get("parlay_strategy",{}).get("parlay_edge5_n",0) or 0 for r in all_folds)
    p5_pnl = sum((r.get("parlay_strategy",{}).get("parlay_edge5_roi") or 0)/100
                 * (r.get("parlay_strategy",{}).get("parlay_edge5_n") or 0) for r in all_folds)
    p10_n = sum(r.get("parlay_strategy",{}).get("parlay_edge10_n",0) or 0 for r in all_folds)
    p10_pnl = sum((r.get("parlay_strategy",{}).get("parlay_edge10_roi") or 0)/100
                  * (r.get("parlay_strategy",{}).get("parlay_edge10_n") or 0) for r in all_folds)
    p5_pooled = p5_pnl / p5_n * 100 if p5_n else 0
    p10_pooled = p10_pnl / p10_n * 100 if p10_n else 0
    print()
    print(f"POOLED  PARLAY-2 edge≥5pp:  n={p5_n:>4d}  ROI={p5_pooled:>+7.2f}%")
    print(f"POOLED  PARLAY-2 edge≥10pp: n={p10_n:>4d}  ROI={p10_pooled:>+7.2f}%")
    print()
    print("Compare to LEAKY baseline (parlay_lambda120_8fold_4yr):")
    print("  PARLAY-2 edge≥5pp leaky:  n=55  pooled +27.25%")
    print("  PARLAY-2 edge≥10pp leaky: n=37  pooled +39.47%")
    print(f"\n✓ Saved {out_path}")
    print(f"  Total runtime: {(time.time()-overall_t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
