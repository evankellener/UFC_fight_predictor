"""MMA-AI replication, Exp 2 — apply his exact test-set filter to match 411 fights.

Builds on scripts/run_mma_ai_replication.py. Loads the freshly regenerated
mmaai_features.csv and filters to MMA-AI's exact test criteria (PDF p.45):

  filter_fights(threshold=2):
    - Binary results (win/loss, no draws/NCs)
    - Both fighters must have ≥2 prior UFC fights
    - Drop unwanted methods: DQ, other, overturned, decision-split, decision-majority
    - Date >= 2015-01-01 (implicit, since our features start 2014-04)

The filter is applied to BOTH train and test for consistency (matches MMA-AI).

Leakage guardrails (per LEAKAGE_REFERENCE.md):
  §1  Temporal split preserved: train < 2024-05-04, test ∈ [2024-05-04, 2025-11-08].
      Filter doesn't shuffle or leak across splits.
  §3  Prior-count filter uses strictly-earlier fights (event_date < current_date).
      Same-day ties excluded.
  §4  Scaler/imputer fit on TRAIN-only post-filter.
  §6  LR hyperparams frozen (C=0.05, l1=0.5) — identical to Exp 1.
  §10 Single run, single report.
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

warnings.filterwarnings("ignore")

DT = Path("data/tmp")
FEATURES = DT / "mmaai_features.csv"
DB = "data/sqlite_db/sqlite_scrapper.db"

TEST_START = pd.Timestamp("2024-05-04")
TEST_END   = pd.Timestamp("2025-11-08")
TARGET_ACC, TARGET_LL = 0.7032, 0.5985
TARGET_AUC, TARGET_BR = 0.7297, 0.2057


def main():
    print("="*70)
    print("STEP 1: Load regenerated mmaai_features.csv (post-WC-fix)")
    print("="*70)
    df = pd.read_csv(FEATURES, parse_dates=["DATE"])
    print(f"  Raw pipeline output: {len(df):,} fights")

    # ── STEP 2: Compute prior UFC fight counts per fighter at each fight date ──
    # §3 compliance: strict inequality d < fight_date, same-day ties excluded.
    print("\n" + "="*70)
    print("STEP 2: Count prior UFC fights per fighter (MMA-AI threshold=2)")
    print("="*70)

    # Use ufc_winlossko as the source of truth for "prior UFC fight count"
    conn = sqlite3.connect(DB)
    history = pd.read_sql("""
        SELECT w.jfighter, e.DATE, w.jbout
        FROM ufc_winlossko w
        JOIN ufc_event_details e ON e.jevent = w.jevent
    """, conn)
    history["DATE"] = pd.to_datetime(history["DATE"])
    history = history.sort_values(["jfighter", "DATE"]).reset_index(drop=True)

    # For each fighter, record their fight dates in order
    fighter_dates = {f: grp["DATE"].values for f, grp in history.groupby("jfighter")}

    def prior_count(jfighter, fight_date):
        dates = fighter_dates.get(jfighter, np.array([], dtype="datetime64[ns]"))
        if len(dates) == 0:
            return 0
        # Strict inequality: count only fights STRICTLY BEFORE fight_date
        return int((dates < np.datetime64(fight_date)).sum())

    df["f1_priors"] = df.apply(lambda r: prior_count(r["jfighter"], r["DATE"]), axis=1)
    df["f2_priors"] = df.apply(lambda r: prior_count(r["opp_jfighter"], r["DATE"]), axis=1)

    before = len(df)
    df = df[(df["f1_priors"] >= 2) & (df["f2_priors"] >= 2)].copy()
    print(f"  After ≥2-priors filter: {len(df):,} / {before:,} fights "
          f"({100*len(df)/before:.1f}% retained)")

    # ── STEP 3: Drop non-binary methods ──────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3: Drop non-binary outcome methods (DQ, overturned, split/majority dec)")
    print("="*70)

    conn = sqlite3.connect(DB)
    results = pd.read_sql("""
        SELECT r.jevent, r.jbout, r.METHOD
        FROM ufc_fight_results r
    """, conn)
    results["METHOD_norm"] = results["METHOD"].str.lower().fillna("")
    conn.close()

    # Extract jevent, jbout from df's composite key. The features CSV has jevent+jbout
    # embedded — check columns:
    if "jevent" not in df.columns or "jbout" not in df.columns:
        # Reconstruct from index if needed
        print("  (jevent/jbout not in df columns — skipping method filter)")
    else:
        df = df.merge(results[["jevent", "jbout", "METHOD_norm"]],
                      on=["jevent", "jbout"], how="left")
        unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
        mask_unwanted = df["METHOD_norm"].apply(
            lambda m: any(u in str(m) for u in unwanted) if pd.notna(m) else False
        )
        before = len(df)
        df = df[~mask_unwanted].copy()
        print(f"  After method filter: {len(df):,} / {before:,} fights "
              f"({100*len(df)/before:.1f}% retained)")

    # ── STEP 4: Temporal split ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4: Train/test split")
    print("="*70)
    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,}   Test: {len(test):,}   (MMA-AI reported: 411 test)")

    if len(test) < 50:
        print("  ✗ Too few test fights, aborting.")
        return

    # ── STEP 5: Train LR ElasticNet, same hyperparams as Exp 1 ───────────────
    print("\n" + "="*70)
    print("STEP 5: Train LR (C=0.05, l1=0.5) on filtered train set")
    print("="*70)
    feat_cols = [c for c in df.columns if c.endswith("_diff") or c in
                 ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1")]
    usable = [c for c in feat_cols if c in train.columns and train[c].std() > 1e-8]
    print(f"  Usable feature cols: {len(usable)}/{len(feat_cols)}")

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train[usable])
    X_te = imp.transform(test[usable])

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    y_tr = train["win"].astype(int).values
    y_te = test["win"].astype(int).values

    decay_lambda = 0.13
    w_tr = np.exp(-decay_lambda * (TEST_START - train["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr, y_tr, sample_weight=w_tr)
    p = lr.predict_proba(X_te)[:, 1]

    # ── STEP 6: Metrics + comparison ─────────────────────────────────────────
    pred = (p >= 0.5).astype(int)
    m = dict(
        acc=accuracy_score(y_te, pred),
        ll=log_loss(y_te, p),
        auc=roc_auc_score(y_te, p),
        brier=brier_score_loss(y_te, p),
    )

    print("\n" + "="*70)
    print("STEP 6: Filtered vs MMA-AI v7 target")
    print("="*70)
    print(f"{'':28s}  {'n':>4s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    print(f"{'TARGET (MMA-AI v7)':28s}  {411:>4d}  {TARGET_ACC*100:>6.2f}%  "
          f"{TARGET_LL:>8.4f}  {TARGET_AUC:>7.4f}  {TARGET_BR:>7.4f}")
    print(f"{'Exp 1 unfiltered (LR)':28s}  {807:>4d}  {65.30:>6.2f}%  "
          f"{0.6352:>8.4f}  {0.6902:>7.4f}  {0.2216:>7.4f}")
    print(f"{'Exp 2 filtered (LR)':28s}  {len(test):>4d}  {m['acc']*100:>6.2f}%  "
          f"{m['ll']:>8.4f}  {m['auc']:>7.4f}  {m['brier']:>7.4f}")
    print(f"{'Gap (Exp 2 vs target)':28s}        {(m['acc']-TARGET_ACC)*100:>+6.2f}pp  "
          f"{m['ll']-TARGET_LL:>+8.4f}  {m['auc']-TARGET_AUC:>+7.4f}  "
          f"{m['brier']-TARGET_BR:>+7.4f}")

    out = {
        "n_test": int(len(test)), "n_train": int(len(train)),
        "test_window": [str(TEST_START.date()), str(TEST_END.date())],
        "filter": "both_fighters_ge_2_priors + drop_dq/overturned/split_decision/majority_decision",
        "metrics": {k: float(v) for k, v in m.items()},
        "target": {"acc": TARGET_ACC, "ll": TARGET_LL, "auc": TARGET_AUC, "brier": TARGET_BR},
        "gap": {
            "acc_pp": (m['acc']-TARGET_ACC)*100,
            "ll": m['ll']-TARGET_LL,
            "auc": m['auc']-TARGET_AUC,
            "brier": m['brier']-TARGET_BR,
        },
        "model": "LogReg ElasticNet C=0.05 l1=0.5, recency-weighted λ=0.13",
    }
    (DT / "mmaai_replication_filtered_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {DT/'mmaai_replication_filtered_results.json'}")


if __name__ == "__main__":
    main()
