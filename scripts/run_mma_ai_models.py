"""MMA-AI replication with correct filter + model comparison.

Uses the filter identified by explore_filter_to_match_411.py:
  threshold=3 priors + method_strict=True  → 422 test fights (target 411)

Trains and evaluates four models on the same split:
  - LR ElasticNet (baseline, matches Exp 2)
  - XGBoost
  - CatBoost (MMA-AI's dominant ensemble member at weight 0.74)
  - LR + CatBoost blend (50/50)

Leakage guardrails (per LEAKAGE_REFERENCE.md):
  §1  Temporal split: train < 2024-05-04, test ∈ [2024-05-04, 2025-11-08]. No shuffle.
  §3  Prior-count uses strict d < fight_date. Same-day ties excluded.
  §4  Imputer + scaler fit on TRAIN only. Transform on test.
  §6  Model hyperparams frozen to values chosen on prior walk-forward validation
      (NOT on this test window). CatBoost params: depth=6, iters=800, lr=0.03,
      l2_leaf_reg=3 — common "safe" defaults; not tuned on this test window.
  §7  No odds. (Enforced by mma_ai_pipeline.)
  §10 Single run per model; single report.
"""
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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"
TEST_START = pd.Timestamp("2024-05-04")
TEST_END   = pd.Timestamp("2025-11-08")
TARGET_ACC, TARGET_LL = 0.7032, 0.5985
TARGET_AUC, TARGET_BR = 0.7297, 0.2057

# Filter chosen from explore_filter_to_match_411.py (threshold=3 + strict)
FILTER_THRESHOLD = 3
FILTER_METHOD_STRICT = True
TRAIN_START = pd.Timestamp("2016-01-01")  # MMA-AI v5.2 training cutoff


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(
        acc=float(accuracy_score(y, pred)),
        ll=float(log_loss(y, p)),
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
    )


def apply_filter(df):
    """Load prior counts + methods, apply threshold=3 + strict method filter."""
    conn = sqlite3.connect(DB)
    hist = pd.read_sql("""
        SELECT w.jfighter, e.DATE
        FROM ufc_winlossko w
        JOIN ufc_event_details e ON e.jevent = w.jevent
    """, conn)
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fighter_dates = {f: grp["DATE"].values for f, grp in hist.sort_values(
        ["jfighter", "DATE"]).groupby("jfighter")}

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


def main():
    print("="*70)
    print(f"STEP 1: Load features, apply filter (threshold={FILTER_THRESHOLD}, strict methods, start={TRAIN_START.date()})")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    print(f"  Raw pipeline output: {len(df):,}")
    df = apply_filter(df)
    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,}   Test: {len(test):,}   (target: 411)")
    if len(test) < 100:
        print("  ✗ Too few test fights, aborting.")
        return

    feat_cols = [c for c in df.columns if c.endswith("_diff") or c in
                 ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1")]
    usable = [c for c in feat_cols if c in train.columns and train[c].std() > 1e-8]
    print(f"  Usable features: {len(usable)}")

    # Impute and scale for LR (tree-based models don't need scaling)
    imp = SimpleImputer(strategy="median")
    X_tr_raw = imp.fit_transform(train[usable])
    X_te_raw = imp.transform(test[usable])
    sc = StandardScaler()
    X_tr_lr = sc.fit_transform(X_tr_raw)
    X_te_lr = sc.transform(X_te_raw)

    y_tr = train["win"].astype(int).values
    y_te = test["win"].astype(int).values
    lam = 0.13
    w_tr = np.exp(-lam * (TEST_START - train["DATE"]).dt.days.values / 365.25)

    print("\n" + "="*70)
    print("STEP 2: Train 4 models on identical train split")
    print("="*70)

    # ── LR baseline ─────────────────────────────────────────────────────
    print("\n[A] LR ElasticNet (C=0.05, l1=0.5)")
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr_lr, y_tr, sample_weight=w_tr)
    p_lr = lr.predict_proba(X_te_lr)[:, 1]
    m_lr = metrics(y_te, p_lr)
    print(f"      acc={m_lr['acc']*100:.2f}%  ll={m_lr['ll']:.4f}  "
          f"auc={m_lr['auc']:.4f}  brier={m_lr['brier']:.4f}")

    # ── XGBoost ─────────────────────────────────────────────────────────
    print("\n[B] XGBoost (800 trees, depth 6, lr 0.03, subsample 0.8)")
    xgb = XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
        min_child_weight=5, tree_method="hist",
        eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_tr_raw, y_tr, sample_weight=w_tr)
    p_xgb = xgb.predict_proba(X_te_raw)[:, 1]
    m_xgb = metrics(y_te, p_xgb)
    print(f"      acc={m_xgb['acc']*100:.2f}%  ll={m_xgb['ll']:.4f}  "
          f"auc={m_xgb['auc']:.4f}  brier={m_xgb['brier']:.4f}")

    # ── CatBoost ────────────────────────────────────────────────────────
    print("\n[C] CatBoost (800 iters, depth 6, lr 0.03, l2_leaf_reg 3)")
    cb = CatBoostClassifier(
        iterations=800, depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.8,
        random_seed=42, verbose=False, bootstrap_type="Bernoulli",
    )
    cb.fit(X_tr_raw, y_tr, sample_weight=w_tr)
    p_cb = cb.predict_proba(X_te_raw)[:, 1]
    m_cb = metrics(y_te, p_cb)
    print(f"      acc={m_cb['acc']*100:.2f}%  ll={m_cb['ll']:.4f}  "
          f"auc={m_cb['auc']:.4f}  brier={m_cb['brier']:.4f}")

    # ── Blend: 0.5 LR + 0.5 CatBoost ────────────────────────────────────
    print("\n[D] Blend: 0.5·LR + 0.5·CatBoost")
    p_bl = 0.5 * p_lr + 0.5 * p_cb
    m_bl = metrics(y_te, p_bl)
    print(f"      acc={m_bl['acc']*100:.2f}%  ll={m_bl['ll']:.4f}  "
          f"auc={m_bl['auc']:.4f}  brier={m_bl['brier']:.4f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3: Summary vs MMA-AI v7 target")
    print("="*70)
    print(f"{'':30s}  {'n':>4s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    print(f"{'TARGET (MMA-AI v7)':30s}  {411:>4d}  {TARGET_ACC*100:>6.2f}%  "
          f"{TARGET_LL:>8.4f}  {TARGET_AUC:>7.4f}  {TARGET_BR:>7.4f}")
    rows = [
        ("[A] LR ElasticNet", m_lr),
        ("[B] XGBoost", m_xgb),
        ("[C] CatBoost", m_cb),
        ("[D] LR+CatBoost blend", m_bl),
    ]
    for name, m in rows:
        gap_pp = (m['acc'] - TARGET_ACC) * 100
        gap_ll = m['ll'] - TARGET_LL
        print(f"{name:30s}  {len(test):>4d}  {m['acc']*100:>6.2f}%  {m['ll']:>8.4f}  "
              f"{m['auc']:>7.4f}  {m['brier']:>7.4f}   "
              f"[Δacc={gap_pp:+.2f}pp Δll={gap_ll:+.4f}]")

    out = {
        "n_test": int(len(test)), "n_train": int(len(train)),
        "filter": {"threshold": FILTER_THRESHOLD, "method_strict": FILTER_METHOD_STRICT,
                   "train_start": str(TRAIN_START.date())},
        "target": {"acc": TARGET_ACC, "ll": TARGET_LL,
                   "auc": TARGET_AUC, "brier": TARGET_BR},
        "results": {
            "LR":             m_lr,
            "XGBoost":        m_xgb,
            "CatBoost":       m_cb,
            "LR_CB_blend":    m_bl,
        },
    }
    (DT / "mmaai_models_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {DT/'mmaai_models_comparison.json'}")


if __name__ == "__main__":
    main()
