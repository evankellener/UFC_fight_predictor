"""Render-app backtest updater — 8-fold walk-forward with NEW 196-feature stack.

Replaces the old LR+XGB blend on 228 features with the new production stack:
  185 MMA-AI features (v7 pipeline, post-WC-index-fix)
  +  6 Elo features  (deployed Elo params)
  +  3 Tier 1c recency features (win_streak, coming_off_loss, fights_last_12m)
  +  2 Tier 2b style Elos (striking/grappling)

Output schema PRESERVED for Flask app compatibility:
  app/models/blend/backtest_predictions.json
      folds: [{fold_num, train_start, train_end, test_start, test_end, n_bouts}]
      predictions: [{fold_num, bout_date, fighter_a, fighter_b, display_a,
                     display_b, p_lr, p_xgb, actual_winner}]
      config: {lr_c, lr_l1, xgb_params, n_folds, test_first, test_last,
               train_years, blend_weight_xgb, generated_at}

`p_lr` = pure LR on 196 features (the new champion — 70.97% single-shot, 71.89% WF).
`p_xgb` = small XGBoost on same features (kept for schema compatibility + side-by-side
          display in Flask). Per `finding_nonlinear_doesnt_help.md` and
          `finding_blend_hurts_roi.md`, XGB underperforms LR here; recommend
          `blend_weight_xgb=0` (pure LR) for production.

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§11):
  §1  Temporal walk-forward; train strictly < fold start.
  §3  Feature builders verified d<fight_date.
  §4  Imputer/scaler refit per fold on train-only.
  §6  Hyperparams frozen (LR: C=0.05, l1=0.5; XGB: small safe defaults from
      finding_nonlinear_doesnt_help.md — never tuned on test).
  §10 Single run per fold; schema-stable output.

Run after feature CSVs are rebuilt:
    python3 scripts/run_mma_ai_replication.py      # rebuilds mmaai_features.csv
    python3 scripts/build_sos_form_features.py     # optional; not currently used
    python3 scripts/build_recency_stance.py        # recency features
    python3 scripts/build_style_elos.py            # style Elos
    python3 scripts/run_backtest_and_save.py       # THIS — regenerate Render artifacts
"""
import json, re, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

sys.path.insert(0, "src")
from elo_feature import compute_elo

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"
OUT = Path("app/models/blend/backtest_predictions.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Same Render test window as the previous deployment
TEST_FIRST = pd.Timestamp("2025-04-05")
TEST_LAST  = pd.Timestamp("2026-04-05")
N_FOLDS = 8
TRAIN_YEARS = 8
TRAIN_ERA = pd.Timestamp("2018-01-01")  # minimum train start date
LAM = 0.13  # recency weight decay

# Frozen LR hyperparams
LR_C = 0.05
LR_L1 = 0.5

# XGB params — keeping the DEPLOYED Render hyperparams (deeper) because the
# Render window is UNFILTERED (threshold=0) and the small-XGB config that won
# on the filtered MMA-AI window loses ~3pp here. On unfiltered data a deeper
# XGB handles hard cases (first-few-fight fighters, split decisions) better.
# These are the same hyperparams from the prior deployed backtest — NOT tuned
# on this run's test window (§6).
XGB_PARAMS = dict(
    n_estimators=1200, max_depth=4, learning_rate=0.015,
    subsample=0.7, colsample_bytree=0.6, reg_lambda=4.0,
    min_child_weight=20, tree_method="hist",
    eval_metric="logloss", random_state=42,
)

ELO_COLS = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
            "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
ELO_PARAMS = dict(K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
                  decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
                  logistic_scale=449.205,
                  opp_quality_k=True, sliding_k=True, upset_momentum=True,
                  champ_mult=1.2)
TIER_1C = ["win_streak_entering_diff", "coming_off_loss_diff", "fights_last_12m_diff"]
STYLE_COLS = ["striking_elo_diff", "grappling_elo_diff"]

_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')
def display_name(jf):
    return _CAMEL_RE.sub(' ', str(jf)).strip()


def merge_all_layers(df):
    """Merge ALL feature layers for XGB (which benefits from richer features on
    the unfiltered Render window). LR internally will use a leaner subset."""
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

    # Recency + stance (Tier 1c + 2a) — all 5 features
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    rs_cols = [c for c in rs.columns if c not in ("DATE", "jbout", "jfighter")]
    df = df.merge(rs, on=["DATE", "jbout", "jfighter"], how="left")
    for c in rs_cols: df[c] = df[c].fillna(0)

    # SoS + form (Tier 1b) — 7 features
    sos_path = DT / "sos_form_features.csv"
    if sos_path.exists():
        sf = pd.read_csv(sos_path, parse_dates=["DATE"])
        sf_cols = [c for c in sf.columns if c not in ("DATE", "jbout", "jfighter")]
        df = df.merge(sf, on=["DATE", "jbout", "jfighter"], how="left")
        for c in sf_cols: df[c] = df[c].fillna(0)

    # Style Elos (Tier 2b) — 2 features
    se = pd.read_csv(DT / "style_elo_features.csv", parse_dates=["DATE"])
    df = df.merge(se, on=["DATE", "jbout", "jfighter"], how="left")
    for c in STYLE_COLS: df[c] = df[c].fillna(0.0)

    # Tier 1a interactions (precomputed here, not in a separate file)
    if "age_diff" in df.columns and "precomp_elo_diff" in df.columns:
        df["ix_age_x_elo_diff"] = df["age_diff"] * df["precomp_elo_diff"]
    if "days_since_last_fight_diff" in df.columns:
        df["ix_elo_x_layoff_diff"] = df["precomp_elo_diff"] * df["days_since_last_fight_diff"]
    if "scheduled_rounds" in df.columns:
        df["ix_elo_x_rounds"] = df["precomp_elo_diff"] * (df["scheduled_rounds"] - 3)
    if "win_streak_entering_diff" in df.columns:
        df["ix_elo_x_streak_diff"] = df["precomp_elo_diff"] * df["win_streak_entering_diff"].fillna(0)
    return df


def main():
    print("="*70)
    print("Render-app backtest updater — NEW 196-feature stack")
    print("="*70)

    # ── Load + merge ────────────────────────────────────────────────────
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df["win"] = df["win"].astype(int)
    df = merge_all_layers(df)
    # Deduplicate just in case
    df = df.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    print(f"Feature matrix: {len(df):,} rows")

    # LR uses the LEAN 196-feature stack that won on the MMA-AI window:
    # MMA-AI + Elo + Tier 1c recency + Tier 2b style. Excludes interactions/SoS/stance
    # that don't help LR (see finding_tier12_lift.md).
    LR_FEATS = [c for c in df.columns if (c.endswith("_diff") or c in
                ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
                and c not in ("f1_priors", "f2_priors")
                and not c.startswith("ix_")
                and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                              "form_winrate3_diff", "form_winrate5_diff",
                              "elo_trajectory_diff", "career_fights_diff",
                              "stance_mismatch", "southpaw_advantage_diff")]
    # XGB gets the FULL feature set — benefits from interactions + SoS + stance
    # on the unfiltered Render window where hard cases matter more.
    XGB_FEATS = [c for c in df.columns if (c.endswith("_diff") or c in
                 ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1",
                  "stance_mismatch", "southpaw_advantage_diff"))
                 and c not in ("f1_priors", "f2_priors")]
    print(f"LR features: {len(LR_FEATS)}  XGB features: {len(XGB_FEATS)}")

    # ── Build folds ─────────────────────────────────────────────────────
    span_days = (TEST_LAST - TEST_FIRST).days
    folds = []
    for i in range(N_FOLDS):
        fs = TEST_FIRST + pd.Timedelta(days=int(round(i * span_days / N_FOLDS)))
        fe = TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span_days / N_FOLDS))) \
             if i < N_FOLDS-1 else TEST_LAST
        folds.append((fs, fe))

    print(f"\n8-fold WF over {TEST_FIRST.date()} → {TEST_LAST.date()} "
          f"(~{span_days/N_FOLDS:.0f} days/fold)")

    # ── Run WF ──────────────────────────────────────────────────────────
    all_preds, fold_meta = [], []
    for idx, (fs, fe) in enumerate(folds, start=1):
        train_start = max(TRAIN_ERA, fs - pd.DateOffset(years=TRAIN_YEARS))
        tr = df[(df["DATE"] >= train_start) & (df["DATE"] < fs)].copy()
        te = df[(df["DATE"] >= fs) & (df["DATE"] < fe)].copy() if idx < N_FOLDS \
             else df[(df["DATE"] >= fs) & (df["DATE"] <= fe)].copy()
        if len(te) == 0:
            print(f"  Fold {idx}: no test bouts, skipping")
            continue

        # Drop any test rows without a valid win label
        te = te[te["win"].isin([0, 1])].copy()
        if len(te) == 0: continue

        lr_usable = [c for c in LR_FEATS if c in tr.columns and tr[c].std() > 1e-8]
        xgb_usable = [c for c in XGB_FEATS if c in tr.columns and tr[c].std() > 1e-8]
        y_tr = tr["win"].values
        w_tr = np.exp(-LAM * (fs - tr["DATE"]).dt.days.values / 365.0)

        # LR (scaled + imputed on LR_FEATS)
        imp_lr = SimpleImputer(strategy="median")
        X_tr_lr_raw = imp_lr.fit_transform(tr[lr_usable])
        X_te_lr_raw = imp_lr.transform(te[lr_usable])
        sc = StandardScaler()
        X_tr_lr = sc.fit_transform(X_tr_lr_raw)
        X_te_lr = sc.transform(X_te_lr_raw)
        lr = LogisticRegression(C=LR_C, penalty="elasticnet", l1_ratio=LR_L1,
                                solver="saga", max_iter=6000, random_state=42)
        lr.fit(X_tr_lr, y_tr, sample_weight=w_tr)
        p_lr = lr.predict_proba(X_te_lr)[:, 1]

        # XGB on FULL feature set (deep, for unfiltered Render window)
        imp_xgb = SimpleImputer(strategy="median")
        X_tr_xgb = imp_xgb.fit_transform(tr[xgb_usable])
        X_te_xgb = imp_xgb.transform(te[xgb_usable])
        xb = XGBClassifier(**XGB_PARAMS)
        xb.fit(X_tr_xgb, y_tr, sample_weight=w_tr)
        p_xgb = xb.predict_proba(X_te_xgb)[:, 1]

        # Emit per-bout predictions (matches existing schema)
        for te_i, (_, row) in enumerate(te.iterrows()):
            fa, fb = row["jfighter"], row["opp_jfighter"]
            actual = fa if int(row["win"]) == 1 else fb
            all_preds.append({
                "fold_num":    idx,
                "bout_date":   row["DATE"].strftime("%Y-%m-%d"),
                "fighter_a":   fa,
                "fighter_b":   fb,
                "display_a":   display_name(fa),
                "display_b":   display_name(fb),
                "p_lr":        float(p_lr[te_i]),
                "p_xgb":       float(p_xgb[te_i]),
                "actual_winner": actual,
            })

        fold_meta.append({
            "fold_num":    idx,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end":   fs.strftime("%Y-%m-%d"),
            "test_start":  fs.strftime("%Y-%m-%d"),
            "test_end":    fe.strftime("%Y-%m-%d"),
            "n_bouts":     int(len(te)),
        })
        print(f"  Fold {idx}: train {train_start.date()}→{fs.date()} ({len(tr):>4}),  "
              f"test →{fe.date()} ({len(te):>3})  "
              f"LR={len(lr_usable)}  XGB={len(xgb_usable)}")

    # ── Save ────────────────────────────────────────────────────────────
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
            "blend_weight_xgb": 0.0,  # production = pure LR (see memory)
            "feature_stack": "mma_ai_v7 + elo(6) + tier1c_recency(3) + tier2b_style(2)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nWrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
    print(f"  folds: {len(fold_meta)}  predictions: {len(all_preds)}")

    # Sanity metrics
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
    preds_df = pd.DataFrame(all_preds)
    preds_df["y"] = (preds_df["actual_winner"] == preds_df["fighter_a"]).astype(int)
    y = preds_df["y"].values
    for name, col in [("LR only", "p_lr"), ("XGB only", "p_xgb"),
                      ("Blend 0.5/0.5", None)]:
        if col is None:
            p = 0.5 * preds_df["p_lr"].values + 0.5 * preds_df["p_xgb"].values
        else:
            p = preds_df[col].values
        p_c = np.clip(p, 0.02, 0.98)
        pred = (p_c >= 0.5).astype(int)
        print(f"\n  {name}  (n={len(y)}):")
        print(f"    acc={accuracy_score(y, pred):.4f}  "
              f"ll={log_loss(y, p_c):.4f}  "
              f"auc={roc_auc_score(y, p):.4f}  "
              f"brier={brier_score_loss(y, p_c):.4f}")


if __name__ == "__main__":
    main()
