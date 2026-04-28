"""Walk-forward 8-fold × 3-mo with PER-FOLD pipeline rebuild.

Closes the leakage loophole the Apr 23 entry of LEAKAGE_REFERENCE.md left
open: in the previous 8-fold walk-forward, the `mma_ai_features.csv` was
built ONCE on the full dataset before any folds ran. Per-fight features
(Elo, dec_avg) are technically temporally clean because they only use
fights < that fight's date, but the AdjPerf z-score's MAD population
statistic and `n_eff` use ALL fights in the population — including future
ones from the test fold's perspective.

Magnitude of the leak: small (MAD is robust + uniform across fights), but
non-zero and worth closing.

This script rebuilds the entire MMA-AI pipeline per fold with a hard date
cutoff at fold's train_end. All downstream computations (BB/PG smoothing,
WC priors, MAD, n_eff, AdjPerf z-scores) use only fights ≤ train_end.

For each of 8 folds:
  1. Patch load_base_data to filter at train_end
  2. Run full build_features pipeline
  3. Train MAIN (λ=1.20), COMPANION (λ=0.13), PARLAY (λ=1.50) on full
     fold-train window
  4. Evaluate on fold's test window
  5. Apply men-only PARLAY-2 strategy at edge≥5pp + edge≥10pp variants

LEAKAGE_REFERENCE.md compliance:
  §1: Test fold strictly post-cutoff; never seen during pipeline build,
      MAD computation, or LR training
  §3: Strict prior threshold (≥3 priors) applied at apply_threshold(3)
  §4: Imputer + scaler refit per fold on training data only
  §6: No hyperparameter tuning on test fold (τ kept at global tau_optimized values)

Backup safety: backs up data/tmp/mmaai_features.csv at start, restores
on exit (atexit + SIGINT handler).

Estimated time: ~7-9 min/fold × 8 = ~1 hour.
"""
import sys, json, time, warnings, shutil, signal, atexit, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from scipy.optimize import minimize_scalar


# ── Backup features CSV ─────────────────────────────────────────────────
FEATS_CSV = Path("data/tmp/mmaai_features.csv")
BACKUP    = Path("data/tmp/mmaai_features.csv.pre_per_fold_rebuild")

if FEATS_CSV.exists() and not BACKUP.exists():
    shutil.copy2(FEATS_CSV, BACKUP)
    print(f"✓ Backed up {FEATS_CSV} → {BACKUP}")

def restore_features():
    if BACKUP.exists():
        shutil.copy2(BACKUP, FEATS_CSV)
        print(f"\n✓ Restored {FEATS_CSV} from backup")
atexit.register(restore_features)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))


# ── Config ──────────────────────────────────────────────────────────────
TRAIN_YEARS = 4
THRESHOLD   = 3
EPS = 1e-6

def build_folds():
    starts = pd.date_range("2024-04-01", "2026-01-01", freq="3MS")
    return [{"name": f"fold_{i}",
             "train_start": (s - pd.DateOffset(years=TRAIN_YEARS)),
             "train_end":   s,
             "test_start":  s,
             "test_end":   (s + pd.DateOffset(months=3))}
            for i, s in enumerate(starts, 1)]
FOLDS = build_folds()


# ── Calibration helper ──────────────────────────────────────────────────
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


# ── Build features capped at max_date ───────────────────────────────────
def build_features_capped(max_date):
    """Run the MMA-AI pipeline restricted to fights with DATE <= max_date.
    Monkey-patches load_base_data to inject the filter at the source —
    all downstream computations use only the restricted population."""
    # Force re-import to clear any cached module state
    for mod in list(sys.modules):
        if mod.startswith("mma_ai_pipeline") or mod.startswith("run_threshold"):
            del sys.modules[mod]
    import mma_ai_pipeline as mma
    original_load = mma.load_base_data

    def patched_load():
        df = original_load()
        df["DATE"] = pd.to_datetime(df["DATE"])
        before = len(df)
        df = df[df["DATE"] <= max_date].copy().reset_index(drop=True)
        print(f"  ↓ load_base_data: capped at {max_date.date()} → {len(df):,} / {before:,} rows")
        return df

    mma.load_base_data = patched_load
    df_feats = mma.build_features(config_name="v7")
    mma.load_base_data = original_load  # restore
    df_feats.to_csv(FEATS_CSV, index=False)
    return df_feats


# ── LR fit for one model variant ────────────────────────────────────────
def fit_one(train, test, feats, lam, train_anchor):
    """Symmetric-doubled LR fit + temp calibrate. Returns calibrated test probs."""
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


# ── Per-fold runner ─────────────────────────────────────────────────────
def run_fold(fold):
    print(f"\n{'='*78}")
    print(f"{fold['name'].upper()}  test {fold['test_start'].date()} → {fold['test_end'].date()}")
    print(f"  REBUILD pipeline capped at train_end={fold['train_end'].date()}")
    print(f"{'='*78}")
    t0 = time.time()
    build_features_capped(fold["train_end"])

    # Re-import downstream so they pick up new CSV
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

    train = df[(df["DATE"] >= fold["train_start"]) & (df["DATE"] < fold["train_end"])].copy()
    test  = df[(df["DATE"] >= fold["test_start"]) & (df["DATE"] < fold["test_end"])].copy()
    print(f"  train {len(train)}  test {len(test)}  (build elapsed {time.time()-t0:.0f}s)")

    out = {"fold": fold["name"],
           "test_start": str(fold["test_start"].date()),
           "test_end":   str(fold["test_end"].date()),
           "train_end":  str(fold["train_end"].date()),
           "n_train": len(train), "n_test": len(test)}

    if len(test) == 0:
        return df, feats, test, out

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
        out[f"{label}_T"] = T
        if label == "parlay":
            test = test.copy(); test["p_parlay"] = p_test

    out["fold_elapsed_s"] = round(time.time() - t0, 1)
    return df, feats, test, out


# ── Vegas + parlay strategy ─────────────────────────────────────────────
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
    print("WALK-FORWARD 8-FOLD × 3-mo with PER-FOLD PIPELINE REBUILD")
    print(f"  Threshold: ≥{THRESHOLD} prior UFC fights")
    print(f"  Training window: {TRAIN_YEARS} years per fold")
    print(f"  Each fold: rebuild features capped at train_end")
    print(f"  Estimated runtime: ~7-9 min/fold × 8 = ~1 hour")
    print("="*78)

    all_folds = []
    parlay_summaries = []
    out_path = Path("results/walk_forward_per_fold_rebuild.json")
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
                p5_n = ps.get("parlay_edge5_n", 0)
                p5_roi = ps.get("parlay_edge5_roi")
                p10_n = ps.get("parlay_edge10_n", 0)
                p10_roi = ps.get("parlay_edge10_roi")
                roi5 = f"{p5_roi:>+6.2f}%" if p5_roi is not None else "  nan "
                roi10 = f"{p10_roi:>+6.2f}%" if p10_roi is not None else "  nan "
                print(f"  Parlay e≥5pp:  n={p5_n} ROI={roi5}")
                print(f"  Parlay e≥10pp: n={p10_n} ROI={roi10}")
            out_path.write_text(json.dumps({"folds": all_folds, "parlay": parlay_summaries},
                                            indent=2, default=str))
            elapsed_total = (time.time() - overall_t0)/60
            print(f"  [{i}/{len(FOLDS)}] saved checkpoint  total elapsed: {elapsed_total:.1f}min")
        except Exception as e:
            print(f"\n  {fold['name']} CRASHED: {e}")
            import traceback; traceback.print_exc()

    # Aggregate
    print("\n" + "="*78); print("AGGREGATE SUMMARY"); print("="*78)
    print(f"{'fold':<8s} {'main_acc':>9s} {'parlay_acc':>11s} {'main_ll':>9s} {'parlay_ll':>10s} "
          f"{'p5_n':>5s} {'p5_roi':>9s} {'p10_n':>6s} {'p10_roi':>10s}")
    print("-"*88)
    for r in all_folds:
        ps = r.get("parlay_strategy", {})
        e5_roi = f"{ps.get('parlay_edge5_roi'):>+7.2f}%" if ps.get("parlay_edge5_roi") is not None else "   nan "
        e10_roi = f"{ps.get('parlay_edge10_roi'):>+7.2f}%" if ps.get("parlay_edge10_roi") is not None else "   nan "
        print(f"  {r['fold']:<6s} {r.get('main_acc',0):>8.4f}  {r.get('parlay_acc',0):>10.4f}  "
              f"{r.get('main_ll',0):>8.4f}  {r.get('parlay_ll',0):>9.4f}  "
              f"{ps.get('parlay_edge5_n','-'):>5}  {e5_roi}  "
              f"{ps.get('parlay_edge10_n','-'):>6}  {e10_roi}")

    # Pooled across folds
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
    print("Compare to baseline (pipeline built ONCE on full data):")
    print("  PARLAY-2 edge≥5pp baseline:  n=55  pooled +27.25%")
    print("  PARLAY-2 edge≥10pp baseline: n=37  pooled +39.47%")
    print(f"\n✓ Saved {out_path}")
    print(f"  Total runtime: {(time.time()-overall_t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
