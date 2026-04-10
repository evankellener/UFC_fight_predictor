"""
UFC Fight Predictor — MMA-AI Pipeline Predictions
Uses the MMA-AI pipeline (Poisson-Gamma + Beta-Binomial smoothing, AdjPerf z-scores)
with clean walk-forward-optimized taus and LR+CB ensemble.

Replaces predict_event.py for the web app.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

from elo_feature import load_bouts, compute_elo, get_fighter_elo, BASE_ELO

warnings.filterwarnings("ignore")

_DATA = Path(__file__).parent.parent / "data" / "tmp"
TAU_PATH = _DATA / "tau_optimized.json"
FEATURES_CSV = _DATA / "mmaai_features.csv"
FIGHTER_STATS_CSV = _DATA / "mmaai_fighter_stats.csv"
FEATURE_COLS_JSON = _DATA / "mmaai_feature_cols.json"
ELO_BOUTS_CSV = _DATA / "elo_bouts.csv"

# ELO params (from optimize_pipeline.py, Bayesian-optimized)
ELO_K               = 48.0
ELO_KO_MULT         = 1.80
ELO_SUB_MULT        = 1.20
ELO_DECAY           = 0.923
ELO_WC_PENALTY      = None
ELO_STREAK_BONUS    = 0.0
ELO_STREAK_CAP      = 5
ELO_R1_MULT         = 1.25
ELO_LOGISTIC_SCALE  = 449.205
ELO_DECAY_MAX       = 0.25
ELO_DECAY_MIDPOINT  = 730.0
ELO_DECAY_STEEPNESS = 80.0

# Features kept from forward selection (all others removed from custom feats)
SELECTED_ELO_FEATURES = ["peak_elo_diff", "elo_win_prob"]

# Style matchup features
STYLE_STATS = [
    'head_land_pm_dec_avg', 'body_land_pm_dec_avg', 'leg_land_pm_dec_avg',
    'td_att_pm_dec_avg', 'td_def_dec_avg', 'sub_att_pm_dec_avg',
    'ctrl_pm_dec_avg', 'kd_pm_dec_avg', 'head_acc_dec_avg', 'head_def_dec_avg',
    'ground_acc_dec_avg', 'clinch_acc_dec_avg', 'ko_smooth_dec_avg',
    'win_smooth_dec_avg', 'distance_acc_dec_avg',
]
STYLE_FEAT_NAMES = ['style_distance', 'striking_matchup', 'wrestling_matchup',
                    'power_matchup', 'grappling_matchup', 'sub_matchup']
STYLE_CONFIG_PATH = _DATA / "style_config.json"

# All ELO feature names (for building live predictions)
ALL_ELO_FEATURES = [
    "precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
    "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff", "elo_predictability",
]


def load_taus():
    """Load clean walk-forward-optimized taus and LR params."""
    with open(TAU_PATH) as f:
        opt = json.load(f)
    return opt


def build_training_data():
    """Load pre-computed MMA-AI features from CSV, add Elo, return everything needed.

    Features are pre-built by running the MMA-AI pipeline locally and saving to CSV.
    This avoids needing the full SQLite DB on the deploy server.

    Returns dict with:
        df: full feature DataFrame (for training)
        fighter_stats: per-fighter latest individual stats
        feature_cols: individual feature column names
        feat_cols: final diff feature column names used by the model
    """
    opt = load_taus()

    # Load pre-computed features
    print("  Loading pre-computed MMA-AI features from CSV...")
    df = pd.read_csv(FEATURES_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    print(f"  Features: {df.shape[0]:,} fights × {df.shape[1]} columns")

    # Load per-fighter stats
    stats_df = pd.read_csv(FIGHTER_STATS_CSV)
    with open(FEATURE_COLS_JSON) as f:
        individual_feature_cols = json.load(f)

    fighter_stats = {}
    for _, row in stats_df.iterrows():
        stats = {"DATE": row.get("last_fight_date", "")}
        for col in individual_feature_cols:
            if col in row.index:
                stats[col] = float(row[col]) if pd.notna(row[col]) else 0.0
        if "REACH" in row.index:
            stats["REACH"] = float(row["REACH"]) if pd.notna(row["REACH"]) else 0.0
        if "weightindex" in row.index:
            stats["weightindex"] = int(row["weightindex"]) if pd.notna(row["weightindex"]) else 0
        if "days_since_last_fight" in row.index:
            stats["days_since_last_fight"] = float(row["days_since_last_fight"]) if pd.notna(row["days_since_last_fight"]) else 0.0
        fighter_stats[row["jfighter"]] = stats
    print(f"  Fighter stats: {len(fighter_stats)} fighters")

    # Add Elo features (skip if already in CSV from pre-computation)
    if "precomp_elo_diff" in df.columns and df["precomp_elo_diff"].notna().sum() > 0:
        print("  Elo features already in CSV, skipping merge.")
        coverage = (df["precomp_elo_diff"] != 0).sum()
        print(f"  Elo coverage: {coverage:,} / {len(df):,}")
    else:
        print("  Computing Elo ratings...")
        bouts = pd.read_csv(ELO_BOUTS_CSV)
        bouts["DATE"] = pd.to_datetime(bouts["DATE"])
        from elo_feature import compute_elo as _compute_elo
        elo_df, _, _, _ = _compute_elo(
            bouts, K=ELO_K, ko_mult=ELO_KO_MULT, sub_mult=ELO_SUB_MULT,
            decay_lambda=ELO_DECAY,
            decay_max=ELO_DECAY_MAX, decay_midpoint=ELO_DECAY_MIDPOINT,
            decay_steepness=ELO_DECAY_STEEPNESS,
            logistic_scale=ELO_LOGISTIC_SCALE,
            opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2,
        )
        ELO_DIFF_COLS = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
                         "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
        elo_merge = elo_df[["jbout", "DATE", "f1", "f2"] + ELO_DIFF_COLS].copy()
        elo_merge["DATE"] = pd.to_datetime(elo_merge["DATE"])
        df = df.merge(elo_merge, on=["jbout", "DATE"], how="left")
        is_flipped = (df["jfighter"] == df["f2"])
        for col in ELO_DIFF_COLS:
            if col == "elo_win_prob":
                df.loc[is_flipped, col] = 1.0 - df.loc[is_flipped, col]
            else:
                df.loc[is_flipped, col] = -df.loc[is_flipped, col]
        df.drop(columns=["f1", "f2"], inplace=True, errors="ignore")
        for col in ELO_DIFF_COLS:
            if col == "elo_win_prob":
                df[col] = df[col].fillna(0.5)
            else:
                df[col] = df[col].fillna(0.0)
        coverage = (df["precomp_elo_diff"] != 0).sum()
        print(f"  Elo coverage: {coverage:,} / {len(df):,}")

    # Load model feature list (includes MMA-AI diffs + Elo + style matchup features)
    model_feat_path = _DATA / "model_feat_cols.json"
    if model_feat_path.exists():
        with open(model_feat_path) as f:
            feat_cols = json.load(f)
        # Ensure all columns exist in df
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
    else:
        # Fallback: build feature list manually
        feat_cols = [c for c in df.columns if c.endswith("_diff")]
        for c in ["weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"]:
            if c in df.columns:
                feat_cols.append(c)
        elo_all = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
                   "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff",
                   "elo_predictability"]
        feat_cols = [f for f in feat_cols if f not in elo_all]
        feat_cols.extend(SELECTED_ELO_FEATURES)

    # Clean features
    for c in feat_cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100)

    return {
        "df": df,
        "fighter_stats": fighter_stats,
        "feature_cols": individual_feature_cols,
        "feat_cols": feat_cols,
        "taus": opt,
    }


def train_ensemble(data: dict):
    """Train LR + CatBoost ensemble on MMA-AI features.

    Returns dict with trained models and metadata.
    """
    df = data["df"]
    feat_cols = data["feat_cols"]
    opt = data["taus"]

    lr_C = opt.get("lr_C", 0.1)
    lr_l1 = opt.get("lr_l1", 0.4)
    recency_lambda = opt.get("recency_lambda", 0.10)
    train_era = opt.get("train_era", "2018-01-01")

    # Filter to training era (drop old noisy fights)
    train = df[df["DATE"] >= train_era].copy()
    print(f"  Training era: >= {train_era} ({len(train):,} fights)")

    max_date = train["DATE"].max()
    days = (max_date - train["DATE"]).dt.days.clip(lower=0)
    weights = np.exp(-recency_lambda * days / 365.25)
    weights = weights / weights.mean()

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(train[feat_cols]), columns=feat_cols)
    y = train["win"].values

    # LR
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feat_cols)
    lr = LogisticRegression(C=lr_C, solver="saga", max_iter=3000,
                            penalty="elasticnet", l1_ratio=lr_l1, random_state=42)
    lr.fit(X_scaled, y, sample_weight=weights.values)

    # CatBoost (use last year as eval set for early stopping)
    cutoff = max_date - pd.Timedelta(days=365)
    train_mask = train["DATE"] < cutoff
    val_mask = train["DATE"] >= cutoff

    if val_mask.sum() > 20:
        Xtr_cb = pd.DataFrame(imputer.transform(train.loc[train_mask, feat_cols]), columns=feat_cols)
        Xval_cb = pd.DataFrame(imputer.transform(train.loc[val_mask, feat_cols]), columns=feat_cols)
        ytr_cb = train.loc[train_mask, "win"].values
        yval_cb = train.loc[val_mask, "win"].values
        wtr_cb = weights[train_mask].values

        cb = CatBoostClassifier(depth=3, iterations=500, learning_rate=0.03,
                                l2_leaf_reg=10, random_seed=42, verbose=0,
                                eval_metric="Logloss", early_stopping_rounds=50)
        cb.fit(Xtr_cb, ytr_cb, sample_weight=wtr_cb,
               eval_set=(Xval_cb, yval_cb), verbose=0)
    else:
        cb = CatBoostClassifier(depth=3, iterations=300, learning_rate=0.03,
                                l2_leaf_reg=10, random_seed=42, verbose=0)
        cb.fit(X, y, sample_weight=weights.values, verbose=0)

    n_active = int((lr.coef_[0] != 0).sum())
    print(f"  LR: {n_active}/{len(feat_cols)} active features  |  "
          f"C={lr_C:.4f}  l1={lr_l1:.3f}  decay={recency_lambda:.3f}")
    print(f"  CatBoost: {cb.tree_count_} trees")
    print(f"  Trained on {len(train):,} fights")

    return {
        "lr": lr,
        "cb": cb,
        "scaler": scaler,
        "imputer": imputer,
        "feat_cols": feat_cols,
        "lr_weight": 0.85,  # LR*0.85 + CB*0.15
    }


def predict_fight(name_a: str, name_b: str,
                  models: dict, fighter_stats: dict,
                  feature_cols: list,
                  elo_ratings: dict = None, elo_last_date: dict = None,
                  elo_extra: dict = None, event_date: str = None,
                  verbose: bool = True) -> dict:
    """Predict a single fight using MMA-AI pipeline features + LR/CB ensemble."""

    if name_a not in fighter_stats:
        raise ValueError(f"Fighter not found: {name_a}")
    if name_b not in fighter_stats:
        raise ValueError(f"Fighter not found: {name_b}")

    sa = fighter_stats[name_a]
    sb = fighter_stats[name_b]

    # Alphabetical order (matches training convention)
    if name_a < name_b:
        f1, f2, sf1, sf2 = name_a, name_b, sa, sb
    else:
        f1, f2, sf1, sf2 = name_b, name_a, sb, sa

    feat_cols = models["feat_cols"]

    # Build diff features
    row = {}
    for feat in feat_cols:
        if feat in SELECTED_ELO_FEATURES or feat in ALL_ELO_FEATURES:
            continue  # handled below
        elif feat in STYLE_FEAT_NAMES:
            continue  # handled below
        elif feat == "weightclass_encoded":
            row[feat] = sf1.get("weightindex", 0)
        elif feat == "scheduled_rounds":
            row[feat] = 3.0
        elif feat == "days_since_last_fight_f1":
            row[feat] = sf1.get("days_since_last_fight", 0)
        elif feat.endswith("_diff"):
            col = feat[:-5]
            v1 = sf1.get(col, 0.0)
            v2 = sf2.get(col, 0.0)
            row[feat] = v1 - v2
        else:
            row[feat] = 0.0

    # Style matchup features (non-transitive interactions)
    s1 = {s: sf1.get(s, 0.0) for s in STYLE_STATS}
    s2 = {s: sf2.get(s, 0.0) for s in STYLE_STATS}
    has_style = any(s1[s] != 0 for s in STYLE_STATS) and any(s2[s] != 0 for s in STYLE_STATS)
    if has_style:
        # Load style config for distance calculation
        if STYLE_CONFIG_PATH.exists():
            with open(STYLE_CONFIG_PATH) as _f:
                _cfg = json.load(_f)
            _mean = np.array(_cfg['scaler_mean'])
            _scale = np.array(_cfg['scaler_scale'])
            sv1 = (np.array([s1[s] for s in STYLE_STATS]) - _mean) / _scale
            sv2 = (np.array([s2[s] for s in STYLE_STATS]) - _mean) / _scale
            row['style_distance'] = float(np.linalg.norm(sv1 - sv2))
        else:
            row['style_distance'] = 0.0
        row['striking_matchup'] = (
            s1['head_land_pm_dec_avg'] * s2['head_def_dec_avg'] -
            s2['head_land_pm_dec_avg'] * s1['head_def_dec_avg'])
        row['wrestling_matchup'] = (
            s1['td_att_pm_dec_avg'] * (1 - s2['td_def_dec_avg']) -
            s2['td_att_pm_dec_avg'] * (1 - s1['td_def_dec_avg']))
        row['power_matchup'] = (
            s1['kd_pm_dec_avg'] * s2['head_def_dec_avg'] -
            s2['kd_pm_dec_avg'] * s1['head_def_dec_avg'])
        row['grappling_matchup'] = (
            s1['ctrl_pm_dec_avg'] * (1 - s2['td_def_dec_avg']) -
            s2['ctrl_pm_dec_avg'] * (1 - s1['td_def_dec_avg']))
        row['sub_matchup'] = (
            s1['sub_att_pm_dec_avg'] * (1 - s2['ground_acc_dec_avg']) -
            s2['sub_att_pm_dec_avg'] * (1 - s1['ground_acc_dec_avg']))
    else:
        for sf in STYLE_FEAT_NAMES:
            row[sf] = 0.0

    # Elo features
    wc = sf1.get("weightindex", 0)
    if elo_ratings is not None:
        elo_f1 = get_fighter_elo(f1, elo_ratings, elo_last_date or {},
                                 event_date or pd.Timestamp.now(), ELO_DECAY,
                                 ELO_DECAY_MAX, ELO_DECAY_MIDPOINT, ELO_DECAY_STEEPNESS)
        elo_f2 = get_fighter_elo(f2, elo_ratings, elo_last_date or {},
                                 event_date or pd.Timestamp.now(), ELO_DECAY,
                                 ELO_DECAY_MAX, ELO_DECAY_MIDPOINT, ELO_DECAY_STEEPNESS)
        row["peak_elo_diff"] = (
            (elo_extra or {}).get("peak_elo", {}).get(f1, BASE_ELO) -
            (elo_extra or {}).get("peak_elo", {}).get(f2, BASE_ELO)
        )
        row["elo_win_prob"] = 1.0 / (1.0 + 10 ** (-(elo_f1 - elo_f2) / ELO_LOGISTIC_SCALE))
    else:
        row["peak_elo_diff"] = 0.0
        row["elo_win_prob"] = 0.5

    # Build feature vector
    X = pd.DataFrame([row])[feat_cols].fillna(0.0).clip(-100, 100)

    # Impute + predict with ensemble
    X_imp = pd.DataFrame(models["imputer"].transform(X), columns=feat_cols)
    X_scaled = pd.DataFrame(models["scaler"].transform(X_imp), columns=feat_cols)

    lr_prob = float(models["lr"].predict_proba(X_scaled)[0][1])
    cb_prob = float(models["cb"].predict_proba(X_imp)[0][1])

    w = models["lr_weight"]
    prob_f1 = w * lr_prob + (1 - w) * cb_prob

    winner = f1 if prob_f1 >= 0.5 else f2
    confidence = prob_f1 if prob_f1 >= 0.5 else 1 - prob_f1

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {f1} vs {f2}")
        print(f"  LR: {lr_prob:.1%}  CB: {cb_prob:.1%}  Ensemble: {prob_f1:.1%}")
        print(f"  --> PICK: {winner}  ({confidence:.1%} confidence)")

    return {
        "f1": f1, "f2": f2, "prob_f1": prob_f1,
        "lr_prob": lr_prob, "cb_prob": cb_prob,
        "winner": winner, "confidence": confidence,
        "name_a": name_a, "name_b": name_b,
        "prob_a": prob_f1 if name_a == f1 else 1 - prob_f1,
    }
