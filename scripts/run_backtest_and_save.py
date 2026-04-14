"""Run the 8-fold walk-forward backtest and persist per-fold per-bout
predictions to app/models/blend/backtest_predictions.json.

This file is the single source of truth for:
  GET /api/model/blend_sweep
  GET /api/model/folds
  GET /api/model/calibration
  GET /api/model/tier_bouts

Output schema:
{
  "folds": [
    {"fold_num": 1, "train_start": "...", "train_end": "...",
     "test_start": "...", "test_end": "...", "n_bouts": int}, ...
  ],
  "predictions": [
    {"fold_num": int, "bout_date": "YYYY-MM-DD",
     "fighter_a": jfighter, "fighter_b": jfighter,
     "display_a": "First Last", "display_b": "First Last",
     "p_lr": float (= P(fighter_a wins) from LR alone),
     "p_xgb": float (= P(fighter_a wins) from XGB alone),
     "actual_winner": jfighter}, ...
  ],
  "config": {
     "lr_c": 0.05, "lr_l1": 0.5, "xgb_params": {...},
     "n_folds": 8, "fold_length_days": ~45,
     "test_first": "2025-04-05", "test_last": "2026-04-05",
     "train_years": 8, "generated_at": isoformat
  }
}

fighter_a/b are alphabetized (f1 < f2) so the sign convention is fixed across
the dataset; p_lr / p_xgb are always P(fighter_a wins).
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, "src")
from elo_feature import compute_elo

DT = Path("data/tmp")
OUT = Path("app/models/blend/backtest_predictions.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

TEST_FIRST = pd.Timestamp("2025-04-05")
TEST_LAST  = pd.Timestamp("2026-04-05")
N_FOLDS    = 8
TRAIN_YEARS = 8
DATA_START = pd.Timestamp("2016-01-01")
TRAIN_ERA  = pd.Timestamp("2018-01-01")  # minimum train start

tau = json.load(open(DT / "tau_optimized.json"))
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
XGB_PARAMS = dict(n_estimators=1200, max_depth=4, learning_rate=0.015,
                  subsample=0.7, colsample_bytree=0.6, reg_lambda=4.0,
                  min_child_weight=20, eval_metric="logloss",
                  tree_method="hist", random_state=42)

# ── Load full feature matrix (same assembly as train_and_save_blend.py) ────
# Fast path: the production train_and_save_blend script already writes a
# ready-to-use training_df.csv with all 228 feature columns pre-computed.
# We just need to load it and re-run WF folds on it.
TRAIN_DF = Path("app/models/blend/training_df.csv")
if not TRAIN_DF.exists():
    print(f"ERROR: {TRAIN_DF} not found. Run scripts/train_and_save_blend.py first.")
    sys.exit(1)

df = pd.read_csv(TRAIN_DF, parse_dates=["DATE"])
df["win"] = df["win"].astype(int)

with open("data/tmp/model_feat_cols.json") as f:
    lr_cols = [c for c in json.load(f) if c in df.columns]
with open("app/models/blend/feat_lists.json") as f:
    feat_lists = json.load(f)
xgb_cols = [c for c in feat_lists["xgb_cols"] if c in df.columns]
print(f"Loaded training_df: {len(df)} rows, "
      f"LR feats={len(lr_cols)}, XGB feats={len(xgb_cols)}")

# ── Pretty display names (camelCase -> "Camel Case") ───────────────────────
_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')
def display_name(jf):
    # insert spaces before capital letters: PaddyPimblett -> Paddy Pimblett
    return _CAMEL_RE.sub(' ', str(jf)).strip()

# ── Build folds ────────────────────────────────────────────────────────────
span_days = (TEST_LAST - TEST_FIRST).days
folds = []
for i in range(N_FOLDS):
    fs = TEST_FIRST + pd.Timedelta(days=int(round(i * span_days / N_FOLDS)))
    fe = TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span_days / N_FOLDS))) if i < N_FOLDS-1 else TEST_LAST
    folds.append((fs, fe))

# ── Run WF, collect per-bout predictions ───────────────────────────────────
all_preds = []
fold_meta = []

for idx, (fs, fe) in enumerate(folds, start=1):
    train_start = max(TRAIN_ERA, fs - pd.DateOffset(years=TRAIN_YEARS))
    tr = df[(df["DATE"] >= train_start) & (df["DATE"] < fs)].copy()
    te = df[(df["DATE"] >= fs) & (df["DATE"] < fe)].copy()
    if len(te) == 0:
        print(f"Fold {idx}: no test bouts, skipping")
        continue
    ytr = tr["win"].values
    w_tr = np.exp(-LAM * (fs - tr["DATE"]).dt.days.values / 365.0)

    # LR
    sc = StandardScaler()
    X_lr_tr = sc.fit_transform(tr[lr_cols])
    X_lr_te = sc.transform(te[lr_cols])
    lr = LogisticRegression(C=LR_C, penalty="elasticnet", l1_ratio=LR_L1,
                            solver="saga", max_iter=4000)
    lr.fit(X_lr_tr, ytr, sample_weight=w_tr)
    p_lr = lr.predict_proba(X_lr_te)[:, 1]

    # XGB
    xb = XGBClassifier(**XGB_PARAMS)
    xb.fit(tr[xgb_cols], ytr, sample_weight=w_tr)
    p_xgb = xb.predict_proba(te[xgb_cols])[:, 1]

    # In the training convention: each row's "jfighter" is a specific fighter
    # (not alphabetically ordered). "win" = whether jfighter won. Treat
    # fighter_a = jfighter, fighter_b = opp_jfighter, p = P(jfighter wins).
    for te_i, (_, row) in enumerate(te.iterrows()):
        fa, fb = row["jfighter"], row["opp_jfighter"]
        actual_winner = fa if int(row["win"]) == 1 else fb
        all_preds.append({
            "fold_num":    idx,
            "bout_date":   row["DATE"].strftime("%Y-%m-%d"),
            "fighter_a":   fa,
            "fighter_b":   fb,
            "display_a":   display_name(fa),
            "display_b":   display_name(fb),
            "p_lr":        float(p_lr[te_i]),
            "p_xgb":       float(p_xgb[te_i]),
            "actual_winner": actual_winner,
        })

    fold_meta.append({
        "fold_num":    idx,
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end":   fs.strftime("%Y-%m-%d"),
        "test_start":  fs.strftime("%Y-%m-%d"),
        "test_end":    fe.strftime("%Y-%m-%d"),
        "n_bouts":     int(len(te)),
    })
    print(f"Fold {idx}: train {train_start.date()} → {fs.date()} "
          f"({len(tr):>4} rows), test → {fe.date()} ({len(te):>3} bouts)")

# ── Save ───────────────────────────────────────────────────────────────────
payload = {
    "folds": fold_meta,
    "predictions": all_preds,
    "config": {
        "lr_c": LR_C, "lr_l1": LR_L1,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if k != "eval_metric"},
        "n_folds": N_FOLDS,
        "test_first": TEST_FIRST.strftime("%Y-%m-%d"),
        "test_last":  TEST_LAST.strftime("%Y-%m-%d"),
        "train_years": TRAIN_YEARS,
        "blend_weight_xgb": 0.5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    },
}
OUT.write_text(json.dumps(payload, indent=1))
print(f"\nWrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
print(f"  folds: {len(fold_meta)}  predictions: {len(all_preds)}")

# Quick sanity: compute aggregate metrics at w=0.5 and print
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
preds_df = pd.DataFrame(all_preds)
# Build y (did fighter_a win?) per row
preds_df["y"] = (preds_df["actual_winner"] == preds_df["fighter_a"]).astype(int)
preds_df["p_blend"] = 0.5 * preds_df["p_lr"] + 0.5 * preds_df["p_xgb"]
y = preds_df["y"].values
p = preds_df["p_blend"].values
pred = (p >= 0.5).astype(int)
print(f"\nSanity check (blend w=0.5 pooled across folds, n={len(y)}):")
print(f"  acc={accuracy_score(y, pred):.4f}  "
      f"ll={log_loss(y, p):.4f}  "
      f"brier={brier_score_loss(y, p):.4f}  "
      f"auc={roc_auc_score(y, p):.4f}")
