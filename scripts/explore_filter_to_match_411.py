"""Filter exploration — find the combination that lands at 411 test fights.

Sweeps prior-fight thresholds, start-date cutoffs, and method filters. Reports
test-window size (2024-05-04 → 2025-11-08) under each combination, so we can
pick the one that matches MMA-AI's 411 exactly.

Does not train anything. Pure diagnostic.

Leakage guardrails: prior-count uses strict d < fight_date (§3). Filter is
applied uniformly across data; no cross-split contamination.
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"
TEST_START = pd.Timestamp("2024-05-04")
TEST_END   = pd.Timestamp("2025-11-08")
TARGET_N_TEST = 411

df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
print(f"Loaded {len(df):,} fights from mmaai_features.csv  "
      f"({df['DATE'].min().date()} → {df['DATE'].max().date()})\n")

# ── Prior fight counts per fighter (UFC only, strict d < fight_date) ────
conn = sqlite3.connect(DB)
hist = pd.read_sql("""
    SELECT w.jfighter, e.DATE
    FROM ufc_winlossko w
    JOIN ufc_event_details e ON e.jevent = w.jevent
""", conn)
hist["DATE"] = pd.to_datetime(hist["DATE"])
hist = hist.sort_values(["jfighter", "DATE"])
fighter_dates = {f: grp["DATE"].values for f, grp in hist.groupby("jfighter")}

def prior_count(j, d):
    dates = fighter_dates.get(j, np.array([], dtype="datetime64[ns]"))
    return int((dates < np.datetime64(d)).sum()) if len(dates) else 0

df["f1_priors"] = df.apply(lambda r: prior_count(r["jfighter"], r["DATE"]), axis=1)
df["f2_priors"] = df.apply(lambda r: prior_count(r["opp_jfighter"], r["DATE"]), axis=1)

# ── Method labels (for method filter variants) ──────────────────────────
results = pd.read_sql("""
    SELECT jevent, jbout, METHOD FROM ufc_fight_results
""", conn)
results["METHOD_norm"] = results["METHOD"].str.lower().fillna("")
conn.close()
df = df.merge(results[["jevent", "jbout", "METHOD_norm"]], on=["jevent", "jbout"], how="left")

# ── Sweep ───────────────────────────────────────────────────────────────
print(f"Target n_test = {TARGET_N_TEST} fights (MMA-AI reported)\n")
print(f"{'threshold':>10}  {'start_date':>12}  {'method_strict':>14}  "
      f"{'n_test':>7}  {'n_train':>8}  {'Δ vs 411':>9}")
print("-" * 75)

results_rows = []
unwanted_strict = ["dq", "other", "overturned", "decision - split", "decision - majority"]
unwanted_lenient = ["dq", "other", "overturned"]  # keep split/majority

for threshold in [2, 3, 4, 5]:
    for start_date in ["2014-04-01", "2015-01-01", "2016-01-01"]:
        for method_strict in [True, False]:
            d = df[
                (df["f1_priors"] >= threshold)
                & (df["f2_priors"] >= threshold)
                & (df["DATE"] >= pd.Timestamp(start_date))
            ].copy()
            unw = unwanted_strict if method_strict else unwanted_lenient
            mask = d["METHOD_norm"].apply(
                lambda m: any(u in str(m) for u in unw) if pd.notna(m) else False
            )
            d = d[~mask]
            n_test = ((d["DATE"] >= TEST_START) & (d["DATE"] <= TEST_END)).sum()
            n_train = (d["DATE"] < TEST_START).sum()
            delta = n_test - TARGET_N_TEST
            flag = " ← match" if abs(delta) <= 5 else ""
            print(f"{threshold:>10}  {start_date:>12}  {'yes' if method_strict else 'no':>14}  "
                  f"{n_test:>7}  {n_train:>8}  {delta:>+9d}{flag}")
            results_rows.append(dict(
                threshold=threshold, start_date=start_date,
                method_strict=method_strict,
                n_test=int(n_test), n_train=int(n_train),
                delta=int(delta),
            ))

# Best match
best = min(results_rows, key=lambda r: abs(r["delta"]))
print("\nClosest match to 411:")
print(f"  threshold={best['threshold']}  start={best['start_date']}  "
      f"method_strict={best['method_strict']}  → n_test={best['n_test']} (Δ={best['delta']:+d})")

import json
(DT / "filter_exploration_results.json").write_text(
    json.dumps({"target": TARGET_N_TEST, "best": best, "all": results_rows}, indent=2)
)
print(f"\nSaved to {DT/'filter_exploration_results.json'}")
