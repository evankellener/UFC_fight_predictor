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
from mma_ai_pipeline import get_fighter_stats_lookup
from mma_ai_config import PG_TAU_GLOBAL, BB_TAU_GLOBAL
from combined_features import add_elo_features, ELO_DIFF_COLS, ELO_NODIFF_COLS

warnings.filterwarnings("ignore")

TAU_PATH = Path(__file__).parent.parent / "data" / "tmp" / "tau_optimized.json"

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
    """Build MMA-AI features with clean taus, add Elo, return everything needed.

    Returns dict with:
        df: full feature DataFrame (for training)
        fighter_stats: per-fighter latest individual stats
        feature_cols: individual feature column names
        feat_cols: final diff feature column names used by the model
    """
    opt = load_taus()

    # Set clean-optimized taus
    PG_TAU_GLOBAL.update(opt["pg_tau"])
    BB_TAU_GLOBAL.update(opt["bb_tau"])

    # Build pipeline and get per-fighter stats
    result = get_fighter_stats_lookup("jan26")
    df = result["df"]
    fighter_stats = result["fighter_stats"]
    individual_feature_cols = result["feature_cols"]

    # Add Elo features
    df = add_elo_features(df)

    # Build feature list: all MMA-AI diffs + selected Elo features
    feat_cols = [c for c in df.columns if c.endswith("_diff")]
    for c in ["weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"]:
        if c in df.columns:
            feat_cols.append(c)

    # Remove custom features, add back only selected ones
    custom_feats = ELO_DIFF_COLS + ELO_NODIFF_COLS + [
        "weight_cut_pct_diff", "missed_weight_diff", "missed_weight_history_diff",
        "home_advantage_diff", "card_position_norm_diff", "is_main_event",
    ]
    feat_cols = [f for f in feat_cols if f not in custom_feats]
    feat_cols.extend(SELECTED_ELO_FEATURES)

    # Clean features
    for c in feat_cols:
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
    recency_lambda = opt.get("recency_lambda", 0.347)

    # Train on all data (the app trains on everything, like predict_event.py)
    train = df.copy()

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

        cb = CatBoostClassifier(depth=5, iterations=1000, learning_rate=0.05,
                                l2_leaf_reg=3, random_seed=42, verbose=0,
                                eval_metric="Logloss", early_stopping_rounds=50)
        cb.fit(Xtr_cb, ytr_cb, sample_weight=wtr_cb,
               eval_set=(Xval_cb, yval_cb), verbose=0)
    else:
        cb = CatBoostClassifier(depth=5, iterations=300, learning_rate=0.05,
                                l2_leaf_reg=3, random_seed=42, verbose=0)
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
        "lr_weight": 0.7,  # LR*0.7 + CB*0.3 = best ensemble
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
        elif feat == "weightclass_encoded":
            row[feat] = sf1.get("weightindex", 0)
        elif feat == "scheduled_rounds":
            row[feat] = 3.0
        elif feat == "days_since_last_fight_f1":
            row[feat] = sf1.get("days_since_last_fight", 0)
        elif feat.endswith("_diff"):
            # Strip _diff to get individual stat name
            col = feat[:-5]
            v1 = sf1.get(col, 0.0)
            v2 = sf2.get(col, 0.0)
            row[feat] = v1 - v2
        else:
            row[feat] = 0.0

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
