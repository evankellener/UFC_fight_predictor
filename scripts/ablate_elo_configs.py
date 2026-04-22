"""Elo ablation — 4 configs through the same 8-fold walk-forward.

Configs:
  C1 baseline        : deployed Elo params + deployed feature list (1 Elo feat in LR,
                       elo_win_prob + elo_trajectory_diff + 6 ix_* in XGB)
  C2 restored feats  : deployed Elo params + add back peak_elo_diff, precomp_elo_diff,
                       elo_momentum_diff, avg_opp_elo_diff, elo_consist_diff to LR & XGB
  C3 best params     : "best" Elo params from memory (r1_finish_mult=1.25,
                       streak_bonus=0.40/cap=5, sigmoid decay max=0.80/mid=365/steep=40)
                       + deployed feature list
  C4 both            : best params + restored features

Leakage guardrails (per LEAKAGE_REFERENCE.md):
  §1  Identical walk-forward folds across all 4 configs; train=DATE<fs, test=fs<=DATE<fe.
      Test window 2025-04-05 → 2026-04-05 frozen. No shuffle.
  §2  Elo features come from compute_elo; precomp_elo_* is by construction
      rating-before-the-fight. No additional rolling/EMA added in this script.
  §4  StandardScaler fit on train only, per fold.
  §5  Elo time-decay params come from prior Bayesian / auto-research runs,
      NOT retuned on the test set in this script.
  §6  LR (C, l1_ratio, recency_lambda) and XGB params FROZEN across all 4
      configs — identical to the deployed backtest. We are varying ONLY the
      Elo feature set / Elo params, not re-searching hyperparams on test.
  §10 Each config run exactly once, pooled across folds, reported once.

Usage:  python3 scripts/ablate_elo_configs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, "src")
from elo_feature import compute_elo  # noqa: E402


# ── Fold + hyperparam config (frozen; copied from run_backtest_and_save.py) ──
TEST_FIRST = pd.Timestamp("2025-04-05")
TEST_LAST  = pd.Timestamp("2026-04-05")
N_FOLDS    = 8
TRAIN_YEARS = 8
TRAIN_ERA  = pd.Timestamp("2018-01-01")

DT = Path("data/tmp")
APP = Path("app/models/blend")

tau = json.load(open(DT / "tau_optimized.json"))
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
XGB_PARAMS = dict(
    n_estimators=1200, max_depth=4, learning_rate=0.015,
    subsample=0.7, colsample_bytree=0.6, reg_lambda=4.0,
    min_child_weight=20, eval_metric="logloss",
    tree_method="hist", random_state=42,
)

# ── Elo param sets ──────────────────────────────────────────────────────────
# Current deployed (from scripts/train_and_save_blend.py:37-39)
ELO_DEPLOYED = dict(
    K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
    decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
    logistic_scale=449.205,
    opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2,
)

# "Best" from memory (session_summary_mar2026 + feature_sigmoid_decay)
# Adds r1_finish_mult=1.25, streak_bonus=0.40/cap=5, tightens sigmoid decay.
ELO_BEST = dict(
    K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
    r1_finish_mult=1.25,
    streak_bonus=0.40, streak_cap=5,
    decay_max=0.80, decay_midpoint=365.0, decay_steepness=40.0,
    logistic_scale=449.205,
    opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2,
)

ELO_6 = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
         "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
RESTORED_EXTRA = ["precomp_elo_diff", "elo_momentum_diff",
                  "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]


def build_elo_cols(params: dict) -> pd.DataFrame:
    """Run compute_elo and return per-fight Elo feature columns keyed by (jbout, DATE)."""
    bouts = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    elo_df, *_ = compute_elo(bouts, **params)
    keep = ["jbout", "DATE", "f1", "f2"] + ELO_6
    return elo_df[keep].copy()


def merge_elo_into_df(base_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Elo features into training rows, flipping sign where jfighter == f2.
    Mirrors the logic in scripts/train_and_save_blend.py so new Elo values sit in
    the same columns the feature lists reference.
    """
    df = base_df.drop(columns=ELO_6, errors="ignore").copy()
    elo_df = elo_df.copy()
    elo_df["DATE"] = pd.to_datetime(elo_df["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(elo_df, on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] == df["f2"]
    for c in ELO_6:
        if c == "elo_win_prob":
            df.loc[flip, c] = 1 - df.loc[flip, c]
        else:
            df.loc[flip, c] = -df.loc[flip, c]
    df.drop(columns=["f1", "f2"], inplace=True, errors="ignore")
    for c in ELO_6:
        df[c] = df[c].fillna(0.5 if c == "elo_win_prob" else 0.0)
    return df


def run_walk_forward(df: pd.DataFrame, lr_cols: list, xgb_cols: list) -> dict:
    """Run 8-fold WF with frozen hyperparams. Returns pooled metrics."""
    span = (TEST_LAST - TEST_FIRST).days
    folds = []
    for i in range(N_FOLDS):
        fs = TEST_FIRST + pd.Timedelta(days=int(round(i * span / N_FOLDS)))
        fe = TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span / N_FOLDS))) \
             if i < N_FOLDS-1 else TEST_LAST
        folds.append((fs, fe))

    rows = []
    for fs, fe in folds:
        train_start = max(TRAIN_ERA, fs - pd.DateOffset(years=TRAIN_YEARS))
        tr = df[(df["DATE"] >= train_start) & (df["DATE"] < fs)].copy()
        te = df[(df["DATE"] >= fs) & (df["DATE"] < fe)].copy()
        if len(te) == 0:
            continue
        ytr = tr["win"].astype(int).values
        yte = te["win"].astype(int).values
        w_tr = np.exp(-LAM * (fs - tr["DATE"]).dt.days.values / 365.0)

        # §4 guardrail: scaler fit on train only
        sc = StandardScaler()
        X_lr_tr = sc.fit_transform(tr[lr_cols])
        X_lr_te = sc.transform(te[lr_cols])
        lr = LogisticRegression(C=LR_C, penalty="elasticnet", l1_ratio=LR_L1,
                                solver="saga", max_iter=4000)
        lr.fit(X_lr_tr, ytr, sample_weight=w_tr)
        p_lr = lr.predict_proba(X_lr_te)[:, 1]

        xb = XGBClassifier(**XGB_PARAMS)
        xb.fit(tr[xgb_cols], ytr, sample_weight=w_tr)
        p_xgb = xb.predict_proba(te[xgb_cols])[:, 1]

        for i in range(len(te)):
            rows.append((yte[i], p_lr[i], p_xgb[i]))

    arr = np.array(rows, dtype=float)
    y, p_lr, p_xgb = arr[:, 0].astype(int), arr[:, 1], arr[:, 2]
    p_blend = 0.5 * p_lr + 0.5 * p_xgb

    def metrics(p):
        pred = (p >= 0.5).astype(int)
        return dict(
            acc=accuracy_score(y, pred),
            ll=log_loss(y, p),
            auc=roc_auc_score(y, p),
            brier=brier_score_loss(y, p),
        )
    return {"n": len(y), "lr": metrics(p_lr), "xgb": metrics(p_xgb),
            "blend": metrics(p_blend)}


def fmt(m):
    return f"acc={m['acc']*100:5.2f}%  ll={m['ll']:.4f}  auc={m['auc']:.4f}  brier={m['brier']:.4f}"


# ── Assemble configs ────────────────────────────────────────────────────────
print("Loading training_df + feature lists...")
base_df = pd.read_csv(APP / "training_df.csv", parse_dates=["DATE"])
base_df["win"] = base_df["win"].astype(int)

feat_lists = json.load(open(APP / "feat_lists.json"))
lr_cols_deployed = [c for c in feat_lists["lr_cols"] if c in base_df.columns]
xgb_cols_deployed = [c for c in feat_lists["xgb_cols"] if c in base_df.columns]
print(f"  base_df rows={len(base_df)}, LR={len(lr_cols_deployed)}, XGB={len(xgb_cols_deployed)}")

lr_cols_restored = lr_cols_deployed + [c for c in RESTORED_EXTRA if c not in lr_cols_deployed]
xgb_cols_restored = xgb_cols_deployed + [c for c in RESTORED_EXTRA if c not in xgb_cols_deployed]

print("\nBuilding Elo features under DEPLOYED params (should match training_df values)...")
elo_dep = build_elo_cols(ELO_DEPLOYED)
df_dep = merge_elo_into_df(base_df, elo_dep)

print("Building Elo features under BEST params (r1=1.25, streak=0.40/5, sigmoid 0.80/365/40)...")
elo_best = build_elo_cols(ELO_BEST)
df_best = merge_elo_into_df(base_df, elo_best)

# Sanity: log how much the Elo features actually moved
delta = (df_best[ELO_6].values - df_dep[ELO_6].values)
nonzero_rows = np.abs(delta).sum(axis=1) > 1e-9
print(f"  Rows with any Elo feature change: {nonzero_rows.sum()}/{len(delta)}")
for i, c in enumerate(ELO_6):
    d = df_best[c].values - df_dep[c].values
    print(f"    {c:22s}  mean|Δ|={np.mean(np.abs(d)):.4f}  max|Δ|={np.max(np.abs(d)):.4f}")


# ── Leakage pre-flight (§1, §11) ────────────────────────────────────────────
assert df_dep["DATE"].equals(df_best["DATE"]), "DATE ordering diverged between configs"
assert (df_dep["DATE"].values == base_df["DATE"].values).all(), "DATE reorder during merge"
# Confirm test window bouts exist in all configs
for name, d in [("dep", df_dep), ("best", df_best)]:
    n_test = ((d["DATE"] >= TEST_FIRST) & (d["DATE"] < TEST_LAST)).sum()
    assert n_test > 400, f"{name} has only {n_test} test-window bouts, expected >400"


# ── Run configs ─────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("Running 4 configs through identical 8-fold walk-forward...")
print("="*90)

print("\n[C1] baseline (deployed params + deployed features)")
r1 = run_walk_forward(df_dep,  lr_cols_deployed, xgb_cols_deployed)
print(f"  LR   : {fmt(r1['lr'])}")
print(f"  XGB  : {fmt(r1['xgb'])}")
print(f"  BLEND: {fmt(r1['blend'])}   n={r1['n']}")

print("\n[C2] restore dropped Elo features (deployed params)")
r2 = run_walk_forward(df_dep,  lr_cols_restored, xgb_cols_restored)
print(f"  LR   : {fmt(r2['lr'])}")
print(f"  XGB  : {fmt(r2['xgb'])}")
print(f"  BLEND: {fmt(r2['blend'])}   n={r2['n']}")

print("\n[C3] best Elo params (deployed features)")
r3 = run_walk_forward(df_best, lr_cols_deployed, xgb_cols_deployed)
print(f"  LR   : {fmt(r3['lr'])}")
print(f"  XGB  : {fmt(r3['xgb'])}")
print(f"  BLEND: {fmt(r3['blend'])}   n={r3['n']}")

print("\n[C4] best params + restored features")
r4 = run_walk_forward(df_best, lr_cols_restored, xgb_cols_restored)
print(f"  LR   : {fmt(r4['lr'])}")
print(f"  XGB  : {fmt(r4['xgb'])}")
print(f"  BLEND: {fmt(r4['blend'])}   n={r4['n']}")


# ── Delta table ─────────────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"SUMMARY — BLEND (0.5 LR + 0.5 XGB), 8-fold WF, test window {TEST_FIRST.date()} → {TEST_LAST.date()}")
print("="*90)
print(f"{'Config':<40s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
for name, r in [("C1 baseline (deployed)", r1),
                ("C2 + restored Elo features",  r2),
                ("C3 best Elo params",           r3),
                ("C4 best params + restored",    r4)]:
    b = r["blend"]
    print(f"{name:<40s}  {b['acc']*100:>6.2f}%  {b['ll']:>8.4f}  {b['auc']:>7.4f}  {b['brier']:>7.4f}")

print("\nDeltas vs C1 baseline (positive acc/auc = better, negative ll/brier = better):")
for name, r in [("C2",  r2), ("C3", r3), ("C4", r4)]:
    b, b1 = r["blend"], r1["blend"]
    print(f"  {name}: Δacc={100*(b['acc']-b1['acc']):+5.2f}pp  "
          f"Δll={b['ll']-b1['ll']:+.4f}  Δauc={b['auc']-b1['auc']:+.4f}  "
          f"Δbrier={b['brier']-b1['brier']:+.4f}")

# Save results
out = {
    "test_window": [str(TEST_FIRST.date()), str(TEST_LAST.date())],
    "n_folds": N_FOLDS, "train_years": TRAIN_YEARS,
    "hyperparams_frozen": {"lr_C": LR_C, "lr_l1": LR_L1, "recency_lambda": LAM,
                           "xgb": {k: v for k, v in XGB_PARAMS.items() if k != "eval_metric"}},
    "configs": {
        "C1_baseline":   {"elo": ELO_DEPLOYED, "features": "deployed", **r1},
        "C2_restored":   {"elo": ELO_DEPLOYED, "features": "deployed+restored", **r2},
        "C3_best_params":{"elo": ELO_BEST,     "features": "deployed", **r3},
        "C4_combined":   {"elo": ELO_BEST,     "features": "deployed+restored", **r4},
    },
}
out_path = DT / "elo_ablation_results.json"
out_path.write_text(json.dumps(out, indent=2, default=str))
print(f"\nResults saved to {out_path}")
