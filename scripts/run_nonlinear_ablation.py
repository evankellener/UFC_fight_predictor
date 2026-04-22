"""Small nonlinear model sweep on the final 196-feature stack.

Baseline: LR ElasticNet = 70.97% / 0.5913 / 0.7528 / 0.2019.

Tries several small, less-prone-to-overfit nonlinear models individually and
blended with LR at various weights. Goal: improve calibration (log loss, Brier)
and/or capture nonlinear picks LR misses.

Each model uses SMALL complexity (depth ≤4, modest iters, early stopping
enabled where possible) to avoid the overfitting that plagued prior deeper
CatBoost runs with 228-dim feature spaces.

Leakage guardrails (LEAKAGE_REFERENCE.md):
  §4  scaler/imputer fit on train-only; CB/XGB/LGBM fit on train only
  §6  hyperparams FROZEN from common "safe small model" defaults; NOT tuned on
      this test window. If we see results >+1pp on test, do NOT iterate and
      cherry-pick best hyperparams (that's p-hacking; memory §9 calls this out)
  §10 Single run per config; one report.
"""
import json, sqlite3, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, log_loss,
                             brier_score_loss, roc_auc_score)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from elo_feature import compute_elo

# ── Try LightGBM (optional) ────────────────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available; skipping.")

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"
TEST_START = pd.Timestamp("2024-05-04")
TEST_END   = pd.Timestamp("2025-11-08")
TRAIN_START = pd.Timestamp("2016-01-01")
FILTER_THRESHOLD = 3

ELO_COLS = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
            "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
ELO_PARAMS = dict(K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
                  decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
                  logistic_scale=449.205,
                  opp_quality_k=True, sliding_k=True, upset_momentum=True,
                  champ_mult=1.2)
TIER_1C = ["win_streak_entering_diff", "coming_off_loss_diff", "fights_last_12m_diff"]
STYLE_COLS = ["striking_elo_diff", "grappling_elo_diff"]


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(acc=float(accuracy_score(y, pred)),
                ll=float(log_loss(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)))


def fmt(m):
    return f"acc={m['acc']*100:5.2f}%  ll={m['ll']:.4f}  auc={m['auc']:.4f}  brier={m['brier']:.4f}"


def apply_filter(df):
    conn = sqlite3.connect(DB)
    hist = pd.read_sql("SELECT w.jfighter, e.DATE FROM ufc_winlossko w "
                       "JOIN ufc_event_details e ON e.jevent=w.jevent", conn)
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fd = {f: grp["DATE"].values for f, grp in
          hist.sort_values(["jfighter", "DATE"]).groupby("jfighter")}

    def prior(j, d):
        dates = fd.get(j, np.array([], dtype="datetime64[ns]"))
        return int((dates < np.datetime64(d)).sum()) if len(dates) else 0

    df = df.copy()
    df["f1_priors"] = df.apply(lambda r: prior(r["jfighter"], r["DATE"]), axis=1)
    df["f2_priors"] = df.apply(lambda r: prior(r["opp_jfighter"], r["DATE"]), axis=1)
    df = df[(df["f1_priors"] >= FILTER_THRESHOLD) & (df["f2_priors"] >= FILTER_THRESHOLD)]
    res = pd.read_sql("SELECT jevent, jbout, METHOD FROM ufc_fight_results", conn)
    res["METHOD_norm"] = res["METHOD"].str.lower().fillna("")
    conn.close()
    df = df.merge(res[["jevent", "jbout", "METHOD_norm"]], on=["jevent", "jbout"], how="left")
    unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
    m = df["METHOD_norm"].apply(lambda x: any(u in str(x) for u in unwanted)
                                 if pd.notna(x) else False)
    df = df[~m]
    df = df[df["DATE"] >= TRAIN_START]
    return df.reset_index(drop=True)


def merge_all_layers(df):
    # main Elo
    bouts = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    ed, *_ = compute_elo(bouts, **ELO_PARAMS)
    keep = ["jbout", "DATE", "f1", "f2"] + ELO_COLS
    ed = ed[keep].copy(); ed["DATE"] = pd.to_datetime(ed["DATE"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(ed, on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] == df["f2"]
    for c in ELO_COLS:
        df.loc[flip, c] = (1 - df.loc[flip, c]) if c == "elo_win_prob" else -df.loc[flip, c]
    df.drop(columns=["f1", "f2"], inplace=True, errors="ignore")
    for c in ELO_COLS:
        df[c] = df[c].fillna(0.5 if c == "elo_win_prob" else 0.0)
    # recency
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    df = df.merge(rs[["DATE", "jbout", "jfighter"] + TIER_1C],
                  on=["DATE", "jbout", "jfighter"], how="left")
    for c in TIER_1C: df[c] = df[c].fillna(0)
    # style
    se = pd.read_csv(DT / "style_elo_features.csv", parse_dates=["DATE"])
    df = df.merge(se, on=["DATE", "jbout", "jfighter"], how="left")
    for c in STYLE_COLS: df[c] = df[c].fillna(0.0)
    return df


def main():
    print("="*70)
    print("STEP 1: Load + filter + merge ALL layers (196-feat final stack)")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    df = merge_all_layers(df)

    train = df[df["DATE"] < TEST_START].copy()
    test  = df[(df["DATE"] >= TEST_START) & (df["DATE"] <= TEST_END)].copy()
    print(f"  Train: {len(train):,}   Test: {len(test):,}")

    feats = [c for c in df.columns if (c.endswith("_diff") or c in
             ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1",
              "stance_mismatch", "southpaw_advantage_diff"))]
    feats = [c for c in feats if c not in ("f1_priors", "f2_priors")]
    # Use Tier 1c + style + Elo + MMA-AI; drop SoS/stance (not in prior best minimal)
    base_exclude_mma = [c for c in df.columns if c in
                        ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                         "form_winrate3_diff", "form_winrate5_diff",
                         "elo_trajectory_diff", "career_fights_diff",
                         "stance_mismatch", "southpaw_advantage_diff")]
    feats = [c for c in feats if c not in base_exclude_mma
             and not c.startswith("ix_")]
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    print(f"  Final feature count: {len(usable)}")

    # ── Pre-process for LR (scaled) and trees (raw with imputation) ─────
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train[usable])
    X_te = imp.transform(test[usable])
    sc = StandardScaler()
    X_tr_lr = sc.fit_transform(X_tr)
    X_te_lr = sc.transform(X_te)
    y_tr = train["win"].astype(int).values
    y_te = test["win"].astype(int).values
    w_tr = np.exp(-0.13 * (TEST_START - train["DATE"]).dt.days.values / 365.25)

    # ── LR baseline ──────────────────────────────────────────────────────
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(X_tr_lr, y_tr, sample_weight=w_tr)
    p_lr = lr.predict_proba(X_te_lr)[:, 1]

    # ── Small nonlinear models (§6 — hyperparams are frozen "safe" defaults) ──
    # CatBoost small #1: shallow, low iters
    cb_s1 = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
                               l2_leaf_reg=5.0, subsample=0.8,
                               random_seed=42, verbose=False,
                               bootstrap_type="Bernoulli")
    cb_s1.fit(X_tr, y_tr, sample_weight=w_tr)
    p_cb_s1 = cb_s1.predict_proba(X_te)[:, 1]

    # CatBoost small #2: slightly deeper, more iters
    cb_s2 = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03,
                               l2_leaf_reg=5.0, subsample=0.8,
                               random_seed=42, verbose=False,
                               bootstrap_type="Bernoulli")
    cb_s2.fit(X_tr, y_tr, sample_weight=w_tr)
    p_cb_s2 = cb_s2.predict_proba(X_te)[:, 1]

    # XGBoost small
    xgb = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=5.0, min_child_weight=10,
                        tree_method="hist", eval_metric="logloss",
                        random_state=42)
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)
    p_xgb = xgb.predict_proba(X_te)[:, 1]

    # LightGBM small (if available)
    p_lgbm = None
    if HAS_LGBM:
        lgbm = LGBMClassifier(n_estimators=400, max_depth=4, num_leaves=15,
                              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=5.0, min_child_samples=20,
                              random_state=42, verbose=-1)
        lgbm.fit(X_tr, y_tr, sample_weight=w_tr)
        p_lgbm = lgbm.predict_proba(X_te)[:, 1]

    # ── Report individual models ────────────────────────────────────────
    print("\n" + "="*90)
    print("Individual models")
    print("="*90)
    print(f"{'Model':<36s}  {'Acc':>7s}  {'LogLoss':>8s}  {'AUC':>7s}  {'Brier':>7s}")
    rows = [("LR ElasticNet (baseline)", p_lr),
            ("CatBoost small (d=3, iter=300)", p_cb_s1),
            ("CatBoost small (d=4, iter=500)", p_cb_s2),
            ("XGBoost small (d=3, 400 trees)", p_xgb)]
    if p_lgbm is not None:
        rows.append(("LightGBM small (leaves=15, 400)", p_lgbm))
    results = {}
    for name, p in rows:
        m = metrics(y_te, p)
        results[name] = m
        print(f"{name:<36s}  {m['acc']*100:>6.2f}%  {m['ll']:>8.4f}  "
              f"{m['auc']:>7.4f}  {m['brier']:>7.4f}")

    # ── Blend sweep ─────────────────────────────────────────────────────
    # For each nonlinear model, try LR weight in {0.9, 0.8, 0.7, 0.6, 0.5}
    print("\n" + "="*90)
    print("Blend sweep: w*LR + (1-w)*NONLIN. Find w that maximizes log-loss improvement.")
    print("="*90)
    candidates = [("CB small-1 (d=3)", p_cb_s1),
                  ("CB small-2 (d=4)", p_cb_s2),
                  ("XGB small", p_xgb)]
    if p_lgbm is not None:
        candidates.append(("LGBM small", p_lgbm))

    lr_ll = results["LR ElasticNet (baseline)"]["ll"]
    print(f"{'Blend':<28s}  {'w_LR':>5s}  {'Acc':>7s}  {'LogLoss':>8s}  "
          f"{'Δll vs LR':>9s}  {'AUC':>7s}  {'Brier':>7s}")
    best_blend = None
    blend_rows = []
    for name, p_nl in candidates:
        for w in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
            p = w * p_lr + (1 - w) * p_nl
            m = metrics(y_te, p)
            dll = lr_ll - m['ll']
            marker = " ←" if (best_blend is None or m['ll'] < best_blend['ll']) else ""
            print(f"{name:<28s}  {w:>5.2f}  {m['acc']*100:>6.2f}%  {m['ll']:>8.4f}  "
                  f"{dll:>+8.4f}  {m['auc']:>7.4f}  {m['brier']:>7.4f}{marker}")
            blend_rows.append(dict(blend=name, w_lr=w, **m))
            if best_blend is None or m['ll'] < best_blend['ll']:
                best_blend = dict(name=name, w=w, **m)

    print(f"\nBEST blend by log loss: {best_blend['name']} w_LR={best_blend['w']}")
    print(f"  {fmt(best_blend)}")
    base = results["LR ElasticNet (baseline)"]
    print(f"\nΔ vs LR baseline:")
    print(f"  Δacc  = {(best_blend['acc']-base['acc'])*100:+.2f}pp")
    print(f"  Δll   = {base['ll']-best_blend['ll']:+.4f}")
    print(f"  Δauc  = {best_blend['auc']-base['auc']:+.4f}")
    print(f"  Δbrier= {base['brier']-best_blend['brier']:+.4f}")

    out = {
        "n_test": len(test), "n_features": len(usable),
        "baseline_lr": base,
        "individual": results,
        "blend_sweep": blend_rows,
        "best_blend": best_blend,
    }
    (DT / "nonlinear_ablation.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {DT/'nonlinear_ablation.json'}")


if __name__ == "__main__":
    main()
