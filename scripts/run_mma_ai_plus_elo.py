"""MMA-AI replication + Elo ablation.

Adds 6 Elo-derived features on top of the 192-feature MMA-AI set and measures
the lift for both model architectures:

  Config           No Elo (baseline)             + 6 Elo features (treatment)
  ───────────────  ──────────────────────────    ──────────────────────────
  LR+CatBoost      69.19 / 0.5974 / 0.7348       NEW
  AutoGluon        69.43 / 0.6058 / 0.7337       NEW

Elo features added (from src/elo_feature.py::compute_elo):
  precomp_elo_diff, elo_win_prob, elo_momentum_diff,
  peak_elo_diff, avg_opp_elo_diff, elo_consist_diff

Elo params — same as deployed blend model (`scripts/train_and_save_blend.py:37-39`):
  K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
  decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
  logistic_scale=449.205,
  opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2

Leakage guardrails (per LEAKAGE_REFERENCE.md):
  §1  Temporal split: train < 2024-05-04, test ∈ [2024-05-04, 2025-11-08]. No shuffle.
  §2  compute_elo output is precomp (rating BEFORE the fight) — confirmed-safe
      in LEAKAGE_REFERENCE.md §2 list. No shift required.
  §3  Filter (threshold=3 priors, strict methods) applied uniformly.
  §4  Imputer + scaler fit on train-only; AG fits internals on train-only.
  §5  Elo params are frozen (same as deployed), NOT tuned on test window.
  §6  Model hyperparams frozen across Elo and no-Elo runs — only feature set differs.
  §9  NO best_quality AG preset.
  §10 Single run per config; report once.

Usage:  python3 scripts/run_mma_ai_plus_elo.py [--skip-ag]
"""
import argparse
import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, log_loss,
                             brier_score_loss, roc_auc_score)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from elo_feature import compute_elo  # noqa: E402

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"
TEST_START = pd.Timestamp("2024-05-04")
TEST_END   = pd.Timestamp("2025-11-08")
TRAIN_START = pd.Timestamp("2016-01-01")
FILTER_THRESHOLD = 3

ELO_COLS = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
            "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]

# Elo params — identical to deployed blend model
ELO_PARAMS = dict(
    K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
    decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
    logistic_scale=449.205,
    opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2,
)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(
        acc=float(accuracy_score(y, pred)),
        ll=float(log_loss(y, p)),
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
    )


def fmt(m):
    return (f"acc={m['acc']*100:5.2f}%  ll={m['ll']:.4f}  "
            f"auc={m['auc']:.4f}  brier={m['brier']:.4f}")


def apply_filter(df):
    conn = sqlite3.connect(DB)
    hist = pd.read_sql("""
        SELECT w.jfighter, e.DATE FROM ufc_winlossko w
        JOIN ufc_event_details e ON e.jevent = w.jevent
    """, conn)
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fighter_dates = {f: grp["DATE"].values
                     for f, grp in hist.sort_values(["jfighter", "DATE"]).groupby("jfighter")}

    def prior_count(j, d):
        dates = fighter_dates.get(j, np.array([], dtype="datetime64[ns]"))
        return int((dates < np.datetime64(d)).sum()) if len(dates) else 0

    df = df.copy()
    df["f1_priors"] = df.apply(lambda r: prior_count(r["jfighter"], r["DATE"]), axis=1)
    df["f2_priors"] = df.apply(lambda r: prior_count(r["opp_jfighter"], r["DATE"]), axis=1)
    df = df[(df["f1_priors"] >= FILTER_THRESHOLD) & (df["f2_priors"] >= FILTER_THRESHOLD)]

    results = pd.read_sql("SELECT jevent, jbout, METHOD FROM ufc_fight_results", conn)
    results["METHOD_norm"] = results["METHOD"].str.lower().fillna("")
    conn.close()
    df = df.merge(results[["jevent", "jbout", "METHOD_norm"]], on=["jevent", "jbout"], how="left")
    unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
    mask = df["METHOD_norm"].apply(
        lambda m: any(u in str(m) for u in unwanted) if pd.notna(m) else False
    )
    df = df[~mask]
    df = df[df["DATE"] >= TRAIN_START]
    return df.reset_index(drop=True)


def build_elo_features():
    """Compute per-bout Elo features and return a DataFrame keyed on (jbout, DATE, f1, f2).
    §2 confirmed-safe: compute_elo produces precomp_elo (BEFORE the fight).
    """
    print("  Computing Elo features (params: K=48, ko=1.80, sub=1.20, sigmoid decay)...")
    bouts = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    elo_df, *_ = compute_elo(bouts, **ELO_PARAMS)
    keep = ["jbout", "DATE", "f1", "f2"] + ELO_COLS
    return elo_df[keep].copy()


def merge_elo(df, elo_df):
    """Merge Elo features onto training rows with sign-flipping for fighter2."""
    df = df.copy()
    elo_df = elo_df.copy()
    elo_df["DATE"] = pd.to_datetime(elo_df["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(elo_df, on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] == df["f2"]
    for c in ELO_COLS:
        if c == "elo_win_prob":
            df.loc[flip, c] = 1 - df.loc[flip, c]
        else:
            df.loc[flip, c] = -df.loc[flip, c]
    df.drop(columns=["f1", "f2"], inplace=True, errors="ignore")
    for c in ELO_COLS:
        df[c] = df[c].fillna(0.5 if c == "elo_win_prob" else 0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────
# Model trainers (§4, §6: fit on train only, hyperparams frozen)
# ─────────────────────────────────────────────────────────────────────────

def train_lr_cb_blend(train, test, feature_cols, label="win"):
    usable = [c for c in feature_cols if c in train.columns and train[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median")
    X_tr_raw = imp.fit_transform(train[usable])
    X_te_raw = imp.transform(test[usable])
    sc = StandardScaler()
    X_tr_lr = sc.fit_transform(X_tr_raw)
    X_te_lr = sc.transform(X_te_raw)

    y_tr = train[label].astype(int).values
    y_te = test[label].astype(int).values
    lam = 0.13
    w_tr = np.exp(-lam * (TEST_START - train["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr_lr, y_tr, sample_weight=w_tr)
    p_lr = lr.predict_proba(X_te_lr)[:, 1]

    cb = CatBoostClassifier(
        iterations=800, depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.8,
        random_seed=42, verbose=False, bootstrap_type="Bernoulli",
    )
    cb.fit(X_tr_raw, y_tr, sample_weight=w_tr)
    p_cb = cb.predict_proba(X_te_raw)[:, 1]

    p_bl = 0.5 * p_lr + 0.5 * p_cb
    return dict(lr=metrics(y_te, p_lr),
                cb=metrics(y_te, p_cb),
                blend=metrics(y_te, p_bl)), len(usable)


def train_autogluon(train, test, feature_cols, label="win", time_limit=600):
    from autogluon.tabular import TabularPredictor
    usable = [c for c in feature_cols if c in train.columns and train[c].std() > 1e-8]
    lam = 0.13

    train_sorted = train.sort_values("DATE").reset_index(drop=True)
    cal_start = int(len(train_sorted) * 0.85)
    fit_df = train_sorted.iloc[:cal_start].copy()
    val_df = train_sorted.iloc[cal_start:].copy()

    fit_df["sample_weight"] = np.exp(
        -lam * (TEST_START - fit_df["DATE"]).dt.days.values / 365.25
    )

    ag_train = fit_df[usable + [label, "sample_weight"]].copy()
    ag_val = val_df[usable + [label]].copy()
    ag_test = test[usable + [label]].copy()

    predictor = TabularPredictor(
        label=label, eval_metric="log_loss",
        problem_type="binary", sample_weight="sample_weight",
        verbosity=1,
    )
    predictor.fit(
        train_data=ag_train,
        tuning_data=ag_val,
        use_bag_holdout=True,
        hyperparameters={
            "CAT": {},
            "GBM": [{"extra_trees": True}, {}],
            "NN_TORCH": {},
        },
        num_stack_levels=2,
        num_bag_folds=4,
        num_bag_sets=2,
        time_limit=time_limit,
        calibrate=True,
        ag_args_fit={"num_cpus": 4},
    )

    p = predictor.predict_proba(ag_test.drop(columns=[label]))
    if hasattr(p, "iloc"):
        p = p[1].values if 1 in p.columns else p.iloc[:, 1].values
    y_te = ag_test[label].astype(int).values
    return metrics(y_te, p), len(usable)


# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ag", action="store_true", help="Skip AutoGluon runs")
    args = ap.parse_args()

    # Load + filter
    print("="*70)
    print("STEP 1: Load + filter + build Elo features")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    elo_df = build_elo_features()
    df_with_elo = merge_elo(df, elo_df)

    # Sanity
    print(f"  After filter: {len(df)} fights")
    print(f"  Elo feature coverage: "
          f"{(~df_with_elo['precomp_elo_diff'].isna()).sum()}/{len(df_with_elo)} rows non-null")

    # Split
    train_no = df[df["DATE"] < TEST_START].copy()
    test_no  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    train_el = df_with_elo[df_with_elo["DATE"] < TEST_START].copy()
    test_el  = df_with_elo[(df_with_elo["DATE"] >= TEST_START) & (df_with_elo["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train_no):,}   Test: {len(test_no):,}")

    base_feats = [c for c in df.columns if c.endswith("_diff") or c in
                  ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1")]
    elo_feats = base_feats + ELO_COLS
    print(f"  Base features: {len(base_feats)}   With Elo: {len(elo_feats)}")

    # ── Leakage sanity check on Elo features ──
    corrs = df_with_elo[ELO_COLS + ["win"]].corr()["win"].drop("win")
    print(f"\n  Elo feature |corr(win)| (should all be < 0.5):")
    for c in ELO_COLS:
        flag = " ⚠ HIGH" if abs(corrs[c]) > 0.5 else ""
        print(f"    {c:22s}  {corrs[c]:+.4f}{flag}")

    results = {}

    # ── LR + CatBoost blend: no Elo ─────────────────────────────
    print("\n" + "="*70)
    print("STEP 2A: LR+CatBoost blend (baseline, no Elo)")
    print("="*70)
    r_lr_no, nfeat = train_lr_cb_blend(train_no, test_no, base_feats)
    print(f"  ({nfeat} features)")
    print(f"  LR    : {fmt(r_lr_no['lr'])}")
    print(f"  CB    : {fmt(r_lr_no['cb'])}")
    print(f"  Blend : {fmt(r_lr_no['blend'])}")
    results["blend_no_elo"] = r_lr_no["blend"]

    # ── LR + CatBoost blend: with Elo ────────────────────────────
    print("\n" + "="*70)
    print("STEP 2B: LR+CatBoost blend (+ 6 Elo features)")
    print("="*70)
    r_lr_el, nfeat = train_lr_cb_blend(train_el, test_el, elo_feats)
    print(f"  ({nfeat} features)")
    print(f"  LR    : {fmt(r_lr_el['lr'])}")
    print(f"  CB    : {fmt(r_lr_el['cb'])}")
    print(f"  Blend : {fmt(r_lr_el['blend'])}")
    results["blend_with_elo"] = r_lr_el["blend"]

    # ── AutoGluon: no Elo ───────────────────────────────────────
    if not args.skip_ag:
        print("\n" + "="*70)
        print("STEP 3A: AutoGluon (baseline, no Elo)")
        print("="*70)
        r_ag_no, nfeat = train_autogluon(train_no, test_no, base_feats)
        print(f"  ({nfeat} features)  {fmt(r_ag_no)}")
        results["ag_no_elo"] = r_ag_no

        # ── AutoGluon: with Elo ─────────────────────────────────
        print("\n" + "="*70)
        print("STEP 3B: AutoGluon (+ 6 Elo features)")
        print("="*70)
        r_ag_el, nfeat = train_autogluon(train_el, test_el, elo_feats)
        print(f"  ({nfeat} features)  {fmt(r_ag_el)}")
        results["ag_with_elo"] = r_ag_el

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "="*75)
    print("FINAL SUMMARY — Elo lift on top of MMA-AI replication (422 test fights)")
    print("="*75)
    print(f"{'Config':<32s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    for label, r in [
        ("LR+CatBoost  (no Elo)", results["blend_no_elo"]),
        ("LR+CatBoost  (+ Elo)",  results["blend_with_elo"]),
    ]:
        print(f"{label:<32s}  {r['acc']*100:>6.2f}%  {r['ll']:>8.4f}  {r['auc']:>7.4f}  {r['brier']:>7.4f}")
    if "ag_no_elo" in results:
        for label, r in [
            ("AutoGluon    (no Elo)", results["ag_no_elo"]),
            ("AutoGluon    (+ Elo)",  results["ag_with_elo"]),
        ]:
            print(f"{label:<32s}  {r['acc']*100:>6.2f}%  {r['ll']:>8.4f}  {r['auc']:>7.4f}  {r['brier']:>7.4f}")

    print("\nElo lift:")
    delta_blend = {
        "acc_pp": (results["blend_with_elo"]["acc"] - results["blend_no_elo"]["acc"]) * 100,
        "ll":     results["blend_no_elo"]["ll"]  - results["blend_with_elo"]["ll"],
        "auc":    results["blend_with_elo"]["auc"] - results["blend_no_elo"]["auc"],
        "brier":  results["blend_no_elo"]["brier"] - results["blend_with_elo"]["brier"],
    }
    print(f"  LR+CatBoost:  Δacc={delta_blend['acc_pp']:+.2f}pp  "
          f"Δll={-delta_blend['ll']:+.4f}  Δauc={delta_blend['auc']:+.4f}  "
          f"Δbrier={-delta_blend['brier']:+.4f}  "
          f"(+/-: positive means Elo helped)")
    if "ag_no_elo" in results:
        delta_ag = {
            "acc_pp": (results["ag_with_elo"]["acc"] - results["ag_no_elo"]["acc"]) * 100,
            "ll":     results["ag_no_elo"]["ll"]  - results["ag_with_elo"]["ll"],
            "auc":    results["ag_with_elo"]["auc"] - results["ag_no_elo"]["auc"],
            "brier":  results["ag_no_elo"]["brier"] - results["ag_with_elo"]["brier"],
        }
        print(f"  AutoGluon :   Δacc={delta_ag['acc_pp']:+.2f}pp  "
              f"Δll={-delta_ag['ll']:+.4f}  Δauc={delta_ag['auc']:+.4f}  "
              f"Δbrier={-delta_ag['brier']:+.4f}  "
              f"(+/-: positive means Elo helped)")

    (DT / "mmaai_plus_elo_results.json").write_text(json.dumps(
        {"results": results, "elo_cols": ELO_COLS, "elo_params": ELO_PARAMS}, indent=2, default=str
    ))
    print(f"\nSaved to {DT/'mmaai_plus_elo_results.json'}")


if __name__ == "__main__":
    main()
