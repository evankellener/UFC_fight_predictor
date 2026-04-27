"""Per-fold τ re-optimization across 8 folds × 3-month walk-forward.

For each of 8 folds:
  1. Define inner-validation = last 6mo of fold's training window
     (split into 2× 3-month inner test slices for stability).
  2. Optuna 30 trials searching τ subspace, minimizing inner-val log loss.
  3. Rebuild features with best τ (single rebuild per fold).
  4. Train MAIN (λ=1.20), COMPANION (λ=0.13), PARLAY (λ=1.50) on full
     fold-train window using the rebuilt features.
  5. Evaluate all 3 on the fold's test window.
  6. Compute parlay-strategy ROI (men-only, edge≥5pp + edge≥10pp variants,
     top-2 by edge).

Then aggregates results across 8 folds and compares to:
  - Baseline 8-fold from results/parlay_lambda120_8fold_4yr_results.json
    and results/parlay_lambda_sweep_4yr.json (τ FIXED).

LEAKAGE_REFERENCE.md compliance:
  §1: Test fold (2024-04 → 2026-04 sliced 8-ways) NEVER touched during
      Optuna search. Optuna only sees inner-validation slices that are
      INSIDE each fold's training window (not the fold's test window).
  §3: Strict pre-fight threshold (≥3 priors) applied at apply_threshold(3).
  §4: Imputer + scaler refit per inner fold AND per fold on training only.
  §6: τ tuned on inner validation, evaluated on held-out test fold.
  τ search range matches scripts/reoptimize_tau_fold4.py (curated subset
  of PG_TAU + BB_TAU known to dominate signal).

Backup safety: backs up data/tmp/mmaai_features.csv before run, restores
at the end (or on Ctrl-C). The optimized τ JSON saved at
results/tau_per_fold.json. Per-fold prediction artifacts cached for
later analysis.

Estimated time: 30 trials × ~75s rebuild = ~37min/fold × 8 = ~5 hours.
"""
import sys, json, time, warnings, shutil, signal, atexit
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import sqlite3
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, roc_auc_score
from scipy.optimize import minimize_scalar


# ── Backup + restore the shared features CSV ────────────────────────────
FEATS_CSV = Path("data/tmp/mmaai_features.csv")
BACKUP    = Path("data/tmp/mmaai_features.csv.pre_8fold_tau_reopt")

if FEATS_CSV.exists() and not BACKUP.exists():
    shutil.copy2(FEATS_CSV, BACKUP)
    print(f"✓ Backed up {FEATS_CSV} → {BACKUP}")

def restore_features():
    if BACKUP.exists():
        shutil.copy2(BACKUP, FEATS_CSV)
        print(f"\n✓ Restored {FEATS_CSV} from backup")

atexit.register(restore_features)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))


# ── Fold + search-space config ──────────────────────────────────────────
TRAIN_YEARS = 4
N_TRIALS    = 30
THRESHOLD   = 3

def build_folds():
    starts = pd.date_range("2024-04-01", "2026-01-01", freq="3MS")
    return [{"name": f"fold_{i}",
             "train_start": (s - pd.DateOffset(years=TRAIN_YEARS)),
             "train_end":   s,
             "test_start":  s,
             "test_end":   (s + pd.DateOffset(months=3))}
            for i, s in enumerate(starts, 1)]
FOLDS = build_folds()

# Curated τ search subspaces (same as fold4 reopt; well-known to dominate signal)
PG_SEARCH = {
    "sig_str_land": (0.1, 5.0),
    "head_land":    (0.1, 5.0),
    "body_land":    (0.5, 8.0),
    "td_land":      (3.0, 30.0),
    "sub_att":      (5.0, 50.0),
    "kd":           (2.0, 40.0),
}
BB_RANGES = {
    "ko":       (2, 50),
    "win":      (5, 100),
    "decision": (2, 30),
    "sub_land": (2, 20),
    "ctrl":     (1, 10),
}


# ── Calibration helper (temperature) ────────────────────────────────────
EPS = 1e-6
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


# ── Feature rebuild driver ──────────────────────────────────────────────
def rebuild_and_load(pg_overrides, bb_overrides):
    """Rebuild features with overridden τ; return loaded df + feats list."""
    from mma_ai_config import PG_TAU_GLOBAL, BB_TAU_GLOBAL
    PG_TAU_GLOBAL.update(pg_overrides)
    BB_TAU_GLOBAL.update(bb_overrides)
    import importlib
    import mma_ai_pipeline
    importlib.reload(mma_ai_pipeline)
    from mma_ai_pipeline import build_features
    df_feats = build_features(config_name="v7")
    df_feats.to_csv(FEATS_CSV, index=False)

    for mod in list(sys.modules):
        if (mod.startswith("run_threshold_sweep_both_elos") or
            mod == "retrain_lr_symmetric" or mod == "walk_forward_4fold"):
            del sys.modules[mod]
    from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
    from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features
    from walk_forward_4fold import select_features
    base = load_base_both_elos()
    df = apply_threshold(base, THRESHOLD)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)
    return df, feats


def fit_one(train, test, feats, lam, train_anchor):
    """Symmetric-doubled LR fit + temp calibrate. Returns calibrated test
    probabilities + the model artifacts."""
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


# ── Inner-validation objective (the Optuna target) ──────────────────────
def make_inner_objective(fold, lam=1.50):
    """Build an objective that, given τ overrides, rebuilds features and
    measures inner-fold log loss within fold['train_start':'train_end']."""
    train_start = fold["train_start"]; train_end = fold["train_end"]
    inner_step = pd.DateOffset(months=3)
    inner_starts = [train_end - inner_step*2, train_end - inner_step]
    state = {"trial": 0, "t0": time.time()}

    def objective(trial):
        state["trial"] += 1
        pg_tau = {k: trial.suggest_float(f"pg_{k}", *rng, log=True)
                  for k, rng in PG_SEARCH.items()}
        bb_tau = {k: trial.suggest_float(f"bb_{k}", *rng, log=True)
                  for k, rng in BB_RANGES.items()}
        try:
            df, feats = rebuild_and_load(pg_tau, bb_tau)
        except Exception as e:
            print(f"    trial {trial.number:>2d}: REBUILD FAILED — {type(e).__name__}: {str(e)[:60]}")
            return 1.0
        train_full = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
        inner_lls = []
        for inner_start in inner_starts:
            inner_end = inner_start + inner_step
            inner_train = train_full[train_full["DATE"] < inner_start].copy()
            inner_val   = train_full[(train_full["DATE"] >= inner_start) &
                                     (train_full["DATE"] < inner_end)].copy()
            if len(inner_val) == 0: continue
            try:
                p_val, _, _ = fit_one(inner_train, inner_val, feats, lam, inner_start)
                yv = inner_val["win"].astype(int).values
                inner_lls.append(float(log_loss(yv, np.clip(p_val, EPS, 1-EPS))))
            except Exception as e:
                print(f"    trial {trial.number} inner fold failed: {e}")
        if not inner_lls: return 1.0
        ll = float(np.mean(inner_lls))
        elapsed = time.time() - state["t0"]
        print(f"    trial {trial.number:>2d}/{N_TRIALS}: inner_ll={ll:.4f}  t={elapsed:>5.0f}s")
        return ll
    return objective


# ── Per-fold full pipeline ──────────────────────────────────────────────
def run_fold(fold):
    print(f"\n{'='*78}")
    print(f"{fold['name'].upper()}  test window: {fold['test_start'].date()} → {fold['test_end'].date()}")
    print(f"{'='*78}")
    t0 = time.time()
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_inner_objective(fold, lam=1.50), n_trials=N_TRIALS,
                   show_progress_bar=False)
    best = study.best_params
    pg_best = {k.removeprefix("pg_"): v for k, v in best.items() if k.startswith("pg_")}
    bb_best = {k.removeprefix("bb_"): v for k, v in best.items() if k.startswith("bb_")}
    print(f"\n  Best inner-val log loss: {study.best_value:.4f}  "
          f"(t={time.time()-t0:.0f}s)")

    # Rebuild once with best τ, then train all three models on the full
    # fold training window, evaluate on the fold test window.
    df, feats = rebuild_and_load(pg_best, bb_best)
    train = df[(df["DATE"] >= fold["train_start"]) & (df["DATE"] < fold["train_end"])].copy()
    test  = df[(df["DATE"] >= fold["test_start"]) & (df["DATE"] < fold["test_end"])].copy()

    out = {"fold": fold["name"],
           "test_start": str(fold["test_start"].date()),
           "test_end":   str(fold["test_end"].date()),
           "best_pg_tau": pg_best, "best_bb_tau": bb_best,
           "best_inner_ll": float(study.best_value),
           "n_train": len(train), "n_test": len(test),
           "elapsed_s": round(time.time() - t0, 1)}

    if len(test) == 0:
        return df, feats, test, out

    for lam, label in [(1.20, "main"), (0.13, "companion"), (1.50, "parlay")]:
        try:
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
            out[f"{label}_T"] = T
            if label == "parlay":
                test = test.copy()
                test["p_parlay"] = p_test
        except Exception as e:
            print(f"    [{label}] fit failed: {e}")
    return df, feats, test, out


# ── Vegas + parlay strategy on the parlay-model preds ───────────────────
def parlay_metrics(test_with_p, fold_name):
    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    keys = test_with_p[["DATE", "jbout", "jfighter"]].drop_duplicates()
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
    print(f"PER-FOLD τ RE-OPTIMIZATION — 8 folds × 3-mo, λ=1.50, 4-yr training")
    print(f"  {N_TRIALS} Optuna trials per fold (search 11 τ scalars)")
    print(f"  Inner validation: last 6mo of each fold's training window (2× 3-mo)")
    print(f"  ESTIMATED time: ~30-45 min/fold × 8 = 4-6 hours")
    print("="*78)

    all_folds = []
    parlay_summaries = []
    out_path = Path("results/tau_per_fold_8.json")
    out_path.parent.mkdir(exist_ok=True)

    overall_t0 = time.time()
    for i, fold in enumerate(FOLDS, 1):
        try:
            df, feats, test_with_p, fold_result = run_fold(fold)
            all_folds.append(fold_result)
            if "p_parlay" in test_with_p.columns:
                ps = parlay_metrics(test_with_p, fold["name"])
                parlay_summaries.append(ps)
                fold_result["parlay_strategy"] = ps
                print(f"\n  PARLAY metrics: edge≥5pp n={ps.get('parlay_edge5_n')} "
                      f"ROI={ps.get('parlay_edge5_roi')}  "
                      f"edge≥10pp n={ps.get('parlay_edge10_n')} ROI={ps.get('parlay_edge10_roi')}")
            # Save incrementally so we don't lose progress on a crash
            out_path.write_text(json.dumps({"folds": all_folds, "parlay": parlay_summaries},
                                            indent=2, default=str))
            elapsed_total = (time.time() - overall_t0)/60
            print(f"  [{i}/{len(FOLDS)}] saved checkpoint  total elapsed: {elapsed_total:.1f}min")
        except Exception as e:
            print(f"\n  {fold['name']} CRASHED: {e}")
            import traceback; traceback.print_exc()

    # ── Aggregate ─────────────────────────────────────────────────────
    print("\n" + "="*78); print("AGGREGATE SUMMARY"); print("="*78)
    print(f"{'fold':<8s} {'main_acc':>9s} {'parlay_acc':>11s} {'main_ll':>9s} {'parlay_ll':>10s} "
          f"{'p_e5_n':>7s} {'p_e5_roi':>9s} {'p_e10_n':>8s} {'p_e10_roi':>10s}")
    print("-"*88)
    for r in all_folds:
        ps = r.get("parlay_strategy", {})
        e5_roi = f"{ps.get('parlay_edge5_roi'):>+7.2f}%" if ps.get("parlay_edge5_roi") is not None else "   nan "
        e10_roi = f"{ps.get('parlay_edge10_roi'):>+7.2f}%" if ps.get("parlay_edge10_roi") is not None else "   nan "
        print(f"  {r['fold']:<6s} {r.get('main_acc',0):>8.4f}  {r.get('parlay_acc',0):>10.4f}  "
              f"{r.get('main_ll',0):>8.4f}  {r.get('parlay_ll',0):>9.4f}  "
              f"{ps.get('parlay_edge5_n','-'):>7}  {e5_roi}  "
              f"{ps.get('parlay_edge10_n','-'):>8}  {e10_roi}")

    # Pooled across folds (parlay ROI weighted by n_bets)
    p5_total_n = sum(r.get("parlay_strategy", {}).get("parlay_edge5_n",0) or 0 for r in all_folds)
    p5_total_pnl = sum((r.get("parlay_strategy",{}).get("parlay_edge5_roi") or 0)/100
                       * (r.get("parlay_strategy",{}).get("parlay_edge5_n") or 0)
                       for r in all_folds)
    p10_total_n = sum(r.get("parlay_strategy", {}).get("parlay_edge10_n",0) or 0 for r in all_folds)
    p10_total_pnl = sum((r.get("parlay_strategy",{}).get("parlay_edge10_roi") or 0)/100
                       * (r.get("parlay_strategy",{}).get("parlay_edge10_n") or 0)
                       for r in all_folds)
    p5_pooled = (p5_total_pnl / p5_total_n * 100) if p5_total_n else 0
    p10_pooled = (p10_total_pnl / p10_total_n * 100) if p10_total_n else 0
    print()
    print(f"POOLED  PARLAY-2 edge≥5pp:  n={p5_total_n:>4d}  ROI={p5_pooled:>+7.2f}%")
    print(f"POOLED  PARLAY-2 edge≥10pp: n={p10_total_n:>4d}  ROI={p10_pooled:>+7.2f}%")
    print()
    print("Compare to baseline (τ FIXED), parlay_lambda120_8fold_4yr_results.json:")
    print("  PARLAY-2 edge≥5pp baseline:  n=55  pooled +27.25%")
    print("  PARLAY-2 edge≥10pp baseline: n=37  pooled +39.47%")

    print(f"\n✓ Saved {out_path}")
    print(f"  Total runtime: {(time.time()-overall_t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
