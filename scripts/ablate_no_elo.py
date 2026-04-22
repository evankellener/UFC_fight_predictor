"""C0 ablation — strip ALL Elo-touching features from the deployed blend.

Answers the question: how much does Elo actually contribute to the final model?
This is the missing ablation from ablate_elo_configs.py (which only compared
Elo *formulations*, not Elo presence/absence).

Features stripped:
  LR : elo_win_prob
  XGB: elo_win_prob, elo_trajectory_diff, ix_age_x_elo, ix_elo_x_streak,
       ix_elo_x_layoff, ix_elo_x_age_ratio, ix_elo_x_card, ix_sos_x_elo

Leakage guardrails (per LEAKAGE_REFERENCE.md):
  §1  Same 8-fold walk-forward, test window 2025-04-05 → 2026-04-05, no shuffle.
  §4  StandardScaler fit on train only, per fold.
  §6  LR (C, l1_ratio, recency_lambda) and XGB params FROZEN — identical to C1.
  §10 Single run per config.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

TEST_FIRST = pd.Timestamp("2025-04-05")
TEST_LAST  = pd.Timestamp("2026-04-05")
N_FOLDS, TRAIN_YEARS = 8, 8
TRAIN_ERA = pd.Timestamp("2018-01-01")

DT = Path("data/tmp")
APP = Path("app/models/blend")
tau = json.load(open(DT / "tau_optimized.json"))
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
XGB_PARAMS = dict(n_estimators=1200, max_depth=4, learning_rate=0.015,
                  subsample=0.7, colsample_bytree=0.6, reg_lambda=4.0,
                  min_child_weight=20, eval_metric="logloss",
                  tree_method="hist", random_state=42)

df = pd.read_csv(APP / "training_df.csv", parse_dates=["DATE"])
df["win"] = df["win"].astype(int)
fl = json.load(open(APP / "feat_lists.json"))

ELO_STRIP = {"elo_win_prob", "elo_trajectory_diff", "precomp_elo_diff",
             "elo_momentum_diff", "peak_elo_diff", "avg_opp_elo_diff",
             "elo_consist_diff",
             "ix_age_x_elo", "ix_elo_x_streak", "ix_elo_x_layoff",
             "ix_elo_x_age_ratio", "ix_elo_x_card", "ix_sos_x_elo"}

# C1: deployed
lr_deploy  = [c for c in fl["lr_cols"]  if c in df.columns]
xgb_deploy = [c for c in fl["xgb_cols"] if c in df.columns]
# C0: Elo stripped
lr_noelo   = [c for c in lr_deploy  if c not in ELO_STRIP]
xgb_noelo  = [c for c in xgb_deploy if c not in ELO_STRIP]

stripped_lr  = sorted(set(lr_deploy)  - set(lr_noelo))
stripped_xgb = sorted(set(xgb_deploy) - set(xgb_noelo))
print(f"Stripped from LR  ({len(stripped_lr)}): {stripped_lr}")
print(f"Stripped from XGB ({len(stripped_xgb)}): {stripped_xgb}")
print(f"LR  feats: {len(lr_deploy)} → {len(lr_noelo)}")
print(f"XGB feats: {len(xgb_deploy)} → {len(xgb_noelo)}")

span = (TEST_LAST - TEST_FIRST).days
folds = []
for i in range(N_FOLDS):
    fs = TEST_FIRST + pd.Timedelta(days=int(round(i * span / N_FOLDS)))
    fe = TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span / N_FOLDS))) \
         if i < N_FOLDS-1 else TEST_LAST
    folds.append((fs, fe))

def wf(lr_cols, xgb_cols):
    rows = []
    for fs, fe in folds:
        train_start = max(TRAIN_ERA, fs - pd.DateOffset(years=TRAIN_YEARS))
        tr = df[(df["DATE"] >= train_start) & (df["DATE"] < fs)].copy()
        te = df[(df["DATE"] >= fs) & (df["DATE"] < fe)].copy()
        if len(te) == 0: continue
        ytr = tr["win"].values; yte = te["win"].values
        w = np.exp(-LAM * (fs - tr["DATE"]).dt.days.values / 365.0)
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[lr_cols])
        Xte = sc.transform(te[lr_cols])
        lr = LogisticRegression(C=LR_C, penalty="elasticnet", l1_ratio=LR_L1,
                                solver="saga", max_iter=4000)
        lr.fit(Xtr, ytr, sample_weight=w)
        p_lr = lr.predict_proba(Xte)[:, 1]
        xb = XGBClassifier(**XGB_PARAMS)
        xb.fit(tr[xgb_cols], ytr, sample_weight=w)
        p_xgb = xb.predict_proba(te[xgb_cols])[:, 1]
        for i in range(len(te)):
            rows.append((yte[i], p_lr[i], p_xgb[i]))
    a = np.array(rows, dtype=float)
    y, p_lr, p_xgb = a[:, 0].astype(int), a[:, 1], a[:, 2]
    p_bl = 0.5*p_lr + 0.5*p_xgb
    def m(p):
        pr = (p >= 0.5).astype(int)
        return dict(acc=accuracy_score(y,pr), ll=log_loss(y,p),
                    auc=roc_auc_score(y,p), brier=brier_score_loss(y,p))
    return dict(n=len(y), lr=m(p_lr), xgb=m(p_xgb), blend=m(p_bl))

def fmt(m):
    return f"acc={m['acc']*100:5.2f}%  ll={m['ll']:.4f}  auc={m['auc']:.4f}  brier={m['brier']:.4f}"

print("\n[C1] baseline (deployed, with Elo)")
r1 = wf(lr_deploy, xgb_deploy)
print(f"  LR   : {fmt(r1['lr'])}")
print(f"  XGB  : {fmt(r1['xgb'])}")
print(f"  BLEND: {fmt(r1['blend'])}   n={r1['n']}")

print("\n[C0] Elo stripped (no elo_win_prob, no ix_*_elo*, no elo_trajectory)")
r0 = wf(lr_noelo, xgb_noelo)
print(f"  LR   : {fmt(r0['lr'])}")
print(f"  XGB  : {fmt(r0['xgb'])}")
print(f"  BLEND: {fmt(r0['blend'])}   n={r0['n']}")

print("\n" + "="*90)
print("Elo contribution (C1 WITH Elo  vs  C0 Elo-stripped):")
print("="*90)
for name, k in [("LR", "lr"), ("XGB", "xgb"), ("BLEND", "blend")]:
    b1, b0 = r1[k], r0[k]
    print(f"  {name:6s}  Δacc={100*(b1['acc']-b0['acc']):+5.2f}pp  "
          f"Δll={b0['ll']-b1['ll']:+.4f}  Δauc={b1['auc']-b0['auc']:+.4f}  "
          f"Δbrier={b0['brier']-b1['brier']:+.4f}")
print("  (positive = Elo helps)")

out = {"C1_with_elo": r1, "C0_no_elo": r0, "stripped_lr": stripped_lr, "stripped_xgb": stripped_xgb}
(DT / "elo_contribution.json").write_text(json.dumps(out, indent=2, default=str))
print(f"\nSaved to {DT / 'elo_contribution.json'}")
