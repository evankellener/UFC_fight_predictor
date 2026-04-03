"""
UFC Fight Predictor — Live Event Prediction
Trains LogReg ElasticNet (C=0.007, l1_ratio=0.1, RobustScaler) on ALL DB data,
then predicts a specified fight card.
"""

import sqlite3, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from elo_feature import load_bouts, compute_elo, get_fighter_elo, BASE_ELO

warnings.filterwarnings("ignore")

_app_db  = Path(__file__).parent.parent / "data/sqlite_db/app.db"
_full_db = Path(__file__).parent.parent / "data/sqlite_db/sqlite_scrapper.db"
DB_PATH  = _app_db if _app_db.exists() else _full_db
ERA_CUTOFF  = "2014-01-01"
RECENCY_LAM = 0.13

# Best ELO params (Bayesian-optimized via optimize_pipeline.py, fitness=-0.0778)
ELO_K            = 48.0       # was 59.507 — recalibrated for higher finish multipliers
ELO_KO_MULT      = 1.80      # was 0.937 — KO/TKO wins now heavily rewarded
ELO_SUB_MULT     = 1.20      # was 1.481 — slight sub reward (less than KO)
ELO_DECAY        = 0.923
ELO_WC_PENALTY   = None       # disabled — was causing spurious resets from weightindex noise
ELO_STREAK_BONUS = 0.0       # disabled — opp-quality K-factor handles this better
ELO_STREAK_CAP   = 5
ELO_R1_MULT      = 1.25      # was 0.800 — R1 finishes now rewarded (dominance signal)
ELO_LOGISTIC_SCALE = 449.205

# Sigmoid decay params (set to None to use legacy exponential ELO_DECAY)
ELO_DECAY_MAX       = 0.25    # max fraction of Elo deviation lost at long inactivity
ELO_DECAY_MIDPOINT  = 730.0   # days where decay is steepest (~2 years)
ELO_DECAY_STEEPNESS = 80.0    # transition sharpness (lower = sharper)

# ── Feature groups (identical to ml_pipeline_v2.py) ────────────────────────
DECAY_DIFF = [
    "sig_str_land_per_min_dec_avg_diff","total_str_land_per_min_dec_avg_diff",
    "td_land_per_min_dec_avg_diff","sub_att_per_min_dec_avg_diff",
    "ctrl_per_min_dec_avg_diff","sig_str_acc_dec_avg_diff",
    "total_str_acc_dec_avg_diff","head_acc_dec_avg_diff","body_acc_dec_avg_diff",
    "leg_acc_dec_avg_diff","distance_acc_dec_avg_diff","clinch_acc_dec_avg_diff",
    "ground_acc_dec_avg_diff","sig_str_def_dec_avg_diff","td_def_dec_avg_diff",
    "ko_dec_avg_diff","ko5_dec_avg_diff","win_dec_avg_diff","loss_dec_avg_diff",
    "finish_rate_dec_avg_diff","grapple_strike_mix_dec_avg_diff",
    "str_eff_diff_dec_avg_diff","age_dec_avg_diff","days_since_last_fight_dec_avg_diff",
    "age_ratio_diff","reach_ratio_dec_avg_diff","td_att_per_min_dec_avg_diff",
    "kd_per_min_dec_avg_diff","ko_eff_dec_avg_diff",
]
ADJPERF_DIFF = [
    "adjperf_sigstr_pm_dec_avg_diff","adjperf_totalstr_pm_dec_avg_diff",
    "adjperf_td_per15_dec_avg_diff","adjperf_sub_per15_dec_avg_diff",
    "adjperf_ctrl_per_min_dec_avg_diff","adjperf_td_att_pm_dec_avg_diff",
    "adjperf_kd_pm_dec_avg_diff","adjperf_ko_eff_dec_avg_diff",
]
ADJPERF_NEW_DIFF = [
    "adjperf_sigstr_ratio_dec_avg_diff","adjperf_distacc_dec_avg_diff",
    "adjperf_headratio_dec_avg_diff","adjperf_bodyacc_dec_avg_diff",
    "adjperf_grounddef_dec_avg_diff","adjperf_headacc_dec_avg_diff",
    "adjperf_distdef_dec_avg_diff",
]
NEW_DIFF = [
    "td_acc_dec_avg_diff","sig_str_land_ratio_dec_avg_diff","head_ratio_dec_avg_diff",
    "dist_ratio_dec_avg_diff","clinch_pm_dec_avg_diff","head_def_dec_avg_diff",
    "body_def_dec_avg_diff","opp_leg_pm_dec_avg_diff","opp_ctrl_pm_dec_avg_diff",
    "opp_kd_pm_dec_avg_diff","opp_td_att_pm_dec_avg_diff","opp_sub_att_pm_dec_avg_diff",
    "win_ratio_dec_avg_diff","ufc_fight_count_diff",
]
NEW_DIFF2 = [
    "head_pm_dec_avg_diff","body_pm_dec_avg_diff","leg_pm_dec_avg_diff",
    "dist_pm_dec_avg_diff","ground_pm_dec_avg_diff",
    "opp_sig_str_pm_dec_avg_diff","opp_total_str_pm_dec_avg_diff",
    "opp_head_pm_dec_avg_diff","opp_body_pm_dec_avg_diff","opp_dist_pm_dec_avg_diff",
    "opp_clinch_pm_dec_avg_diff","opp_ground_pm_dec_avg_diff","opp_td_pm_dec_avg_diff",
    "leg_def_dec_avg_diff","dist_def_dec_avg_diff","clinch_def_dec_avg_diff",
    "ground_def_dec_avg_diff","total_str_def_dec_avg_diff",
    "body_ratio_dec_avg_diff","leg_ratio_dec_avg_diff",
    "clinch_ratio_dec_avg_diff","ground_ratio_dec_avg_diff",
]
ADJPERF_NEW_DIFF2 = [
    "adjperf_head_pm_dec_avg_diff","adjperf_body_pm_dec_avg_diff",
    "adjperf_leg_pm_dec_avg_diff","adjperf_dist_pm_dec_avg_diff",
    "adjperf_clinch_pm_dec_avg_diff","adjperf_ground_pm_dec_avg_diff",
    "adjperf_legacc_dec_avg_diff","adjperf_clinchacc_dec_avg_diff",
    "adjperf_groundacc_dec_avg_diff","adjperf_sigstracc_dec_avg_diff",
    "adjperf_totalstracc_dec_avg_diff","adjperf_tdacc_dec_avg_diff",
    "adjperf_headdef_dec_avg_diff","adjperf_bodydef_dec_avg_diff",
    "adjperf_legdef_dec_avg_diff","adjperf_clinchdef_dec_avg_diff",
    "adjperf_sigstrdef_dec_avg_diff","adjperf_tddef_dec_avg_diff",
    "adjperf_totalstrdef_dec_avg_diff",
    "adjperf_bodyratio_dec_avg_diff","adjperf_legratio_dec_avg_diff",
    "adjperf_clinch_ratio_dec_avg_diff","adjperf_ground_ratio_dec_avg_diff",
    "adjperf_dist_ratio_dec_avg_diff",
]
NEW_DIFF3 = [
    "td_per_sigstr_att_dec_avg_diff","td_land_per_ctrl_dec_avg_diff",
    "ground_land_per_ctrl_dec_avg_diff","sub_att_ratio_dec_avg_diff",
    "opp_decision_rate_dec_avg_diff","rev_per_ctrlopp_dec_avg_diff",
]
ADJPERF_NEW_DIFF3 = [
    "adjperf_win_dec_avg_diff","adjperf_td_per_sigstr_att_dec_avg_diff",
    "adjperf_ctrl_ratio_dec_avg_diff","adjperf_ko_ratio_dec_avg_diff",
    "adjperf_sub_att_ratio_dec_avg_diff","adjperf_rev_per_ctrlopp_dec_avg_diff",
    "adjperf_sub_def_dec_avg_diff","adjperf_td_land_per_ctrl_dec_avg_diff",
    "adjperf_rev_ratio_dec_avg_diff",
]

# Phase-7 durability/absorption/efficiency from ufc_sql_new_features4.sql
NEW_DIFF4 = [
    "avg_fight_duration_dec_avg_diff","round_ended_dec_avg_diff",
    "ko_in_r1_rate_dec_avg_diff","decision_rate_dec_avg_diff",
    "kod_rate_dec_avg_diff","subwd_rate_dec_avg_diff",
    "sigstr_absorbed_pm_dec_avg_diff","ostr_absorbed_pm_dec_avg_diff",
    "damage_efficiency_dec_avg_diff","kd_absorbed_pm_dec_avg_diff",
    "counter_ratio_dec_avg_diff","output_per_damage_dec_avg_diff",
    "striking_diff_pm_dec_avg_diff","volume_pm_dec_avg_diff",
]

# Phase-8 grappling from ufc_sql_new_features5.sql
NEW_DIFF5 = [
    "sub_per_ctrl_min_dec_avg_diff","top_position_output_dec_avg_diff",
    "td_to_ctrl_conversion_dec_avg_diff","grappling_dominance_pm_dec_avg_diff",
]

# Streak features from ufc_sql_streak_features.sql
STREAK_DIFF = [
    "win_streak_diff","loss_streak_diff","finish_streak_diff",
    "recent_ko_rate_3_diff","recent_finish_rate_3_diff",
]

# Cardio features from ufc_sql_cardio_features.sql
CARDIO_DIFF = [
    "r3_vs_r1_sigstr_ratio_dec_avg_diff","r3_vs_r1_ctrl_ratio_dec_avg_diff",
    "r3_vs_r1_td_ratio_dec_avg_diff","r1_output_share_dec_avg_diff",
    "r1_kd_rate_dec_avg_diff","late_round_td_rate_dec_avg_diff",
]

# Opponent quality from ufc_sql_opp_quality.sql
OPP_QUALITY_DIFF = [
    "opp_avg_win_ratio_diff","opp_avg_finish_rate_diff","opp_avg_kod_rate_diff",
]

# Static features from ufc_sql_final_rebuild.sql
STATIC_DIFF = [
    "height_ratio_diff","reach_to_height_diff","weight_diff",
    "stance_southpaw_diff","age_squared_diff","scheduled_rounds",
]

# Round 1 AdjPerf features (computed in SQL: ufc_sql_adjperf_r1.sql)
ADJPERF_R1_DIFF = [
    "adjperf_r1_sigstr_dec_avg_diff","adjperf_r1_ctrl_dec_avg_diff",
    "adjperf_r1_kd_dec_avg_diff","adjperf_r1_td_dec_avg_diff",
]

# ELO features (computed in Python)
ELO_FEATURES = [
    "precomp_elo_diff","elo_win_prob","elo_momentum_diff",
    "peak_elo_diff","avg_opp_elo_diff","elo_consist_diff",
    "elo_predictability",
]

# Age peak feature (computed in Python from age_dec_avg)
AGE_PEAK_AGE  = 27.0   # center of peak performance window
AGE_PEAK_WIDTH = 3.0   # std dev of the Gaussian (wider = more forgiving)
AGE_FEATURES = ["age_prime_diff", "ufc_age_diff"]

ALL_FEATURES = (DECAY_DIFF + ADJPERF_DIFF + ADJPERF_NEW_DIFF + NEW_DIFF
                + NEW_DIFF2 + ADJPERF_NEW_DIFF2
                + NEW_DIFF3 + ADJPERF_NEW_DIFF3
                + NEW_DIFF4 + NEW_DIFF5 + STREAK_DIFF + CARDIO_DIFF
                + OPP_QUALITY_DIFF + STATIC_DIFF + ADJPERF_R1_DIFF
                + ["weightindex", "event_rolling_ema"]
                + ELO_FEATURES + AGE_FEATURES)


def compute_age_prime(age: float, peak: float = AGE_PEAK_AGE,
                      width: float = AGE_PEAK_WIDTH) -> float:
    """Gaussian prime factor: 1.0 at peak age, decaying for younger/older."""
    if pd.isna(age) or age <= 0:
        return 0.5  # neutral
    return float(np.exp(-((age - peak) / width) ** 2))

ALL_ADJPERF_DIFF = ADJPERF_DIFF + ADJPERF_NEW_DIFF + ADJPERF_NEW_DIFF2 + ADJPERF_NEW_DIFF3


# ── Event-level rolling EMA ────────────────────────────────────────────────
def add_event_rolling_ema(df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """
    Compute a rolling EMA of UFC-wide win rate on a per-EVENT basis (not per-fight).

    Each event gets one data point = average win rate across all bouts that night.
    EMA is computed across events (span=20 ≈ ~6 months of UFC events), then
    shifted by 1 event to prevent leakage — every fight row gets the EMA value
    from the PREVIOUS event, i.e. the meta-game state heading into that card.
    """
    df = df.copy()

    # One row per event: mean win rate on that date
    event_win_rate = (
        df.groupby("DATE")["win"]
        .mean()
        .sort_index()
        .reset_index()
        .rename(columns={"win": "event_win_rate"})
    )

    # EMA across events, shifted 1 event back (no leakage)
    event_win_rate["event_rolling_ema"] = (
        event_win_rate["event_win_rate"]
        .ewm(span=span, min_periods=5)
        .mean()
        .shift(1)
        .fillna(0.5)
    )

    df = df.merge(event_win_rate[["DATE", "event_rolling_ema"]], on="DATE", how="left")
    return df


def get_current_event_ema(df: pd.DataFrame, event_date: str, span: int = 20) -> float:
    """
    Get the event_rolling_ema to use when predicting a future event.
    Uses all events strictly BEFORE event_date (no leakage).
    """
    past = df[df["DATE"] < pd.to_datetime(event_date)].copy()

    event_win_rate = (
        past.groupby("DATE")["win"]
        .mean()
        .sort_index()
        .reset_index()
        .rename(columns={"win": "event_win_rate"})
    )

    if event_win_rate.empty:
        return 0.5

    val = float(
        event_win_rate["event_win_rate"]
        .ewm(span=span, min_periods=5)
        .mean()
        .iloc[-1]
    )
    print(f"  event_rolling_ema ({event_date}): {val:.4f}  "
          f"({len(event_win_rate)} events before cutoff, span={span})")
    return val


# ── ELO builder ───────────────────────────────────────────────────────────
def build_elo(anchor_date: str = None):
    """
    Compute ELO for all bouts up to anchor_date (or all time).
    Returns (elo_df keyed on (jbout, jfighter), ratings dict, last_date dict,
             extra dict with elo_3ago/peak_elo/opp_elos for inference).
    """
    bouts = load_bouts()
    if anchor_date:
        bouts = bouts[bouts["DATE"] < pd.to_datetime(anchor_date)]
    elo_df, ratings, last_date, extra = compute_elo(
        bouts, K=ELO_K, ko_mult=ELO_KO_MULT,
        sub_mult=ELO_SUB_MULT, decay_lambda=ELO_DECAY,
        wc_change_penalty=ELO_WC_PENALTY,
        streak_bonus=ELO_STREAK_BONUS, streak_cap=ELO_STREAK_CAP,
        r1_finish_mult=ELO_R1_MULT, logistic_scale=ELO_LOGISTIC_SCALE,
        decay_max=ELO_DECAY_MAX, decay_midpoint=ELO_DECAY_MIDPOINT,
        decay_steepness=ELO_DECAY_STEEPNESS,
    )
    elo_df = elo_df[["jbout", "f1", "precomp_elo_diff",
                      "elo_win_prob", "elo_momentum_diff",
                      "peak_elo_diff", "avg_opp_elo_diff",
                      "elo_consist_diff", "elo_predictability"]].rename(
        columns={"f1": "jfighter"})
    return elo_df, ratings, last_date, extra


# ── Load all data ──────────────────────────────────────────────────────────
def load_all_data(min_prior_fights: int = 1) -> pd.DataFrame:
    """Load training data from final_features_fast.

    min_prior_fights: both fighters must have this many prior UFC fights.
        0 = keep everything (noisiest, most coverage)
        1 = exclude true debut matchups (good balance)
        2 = original strict filter (cleanest, fewest rows)
    """
    conn = sqlite3.connect(DB_PATH)
    # All SQL-based feature columns (ELO computed in Python separately)
    all_feature_cols = (DECAY_DIFF + ADJPERF_DIFF + ADJPERF_NEW_DIFF
                        + NEW_DIFF + NEW_DIFF2 + ADJPERF_NEW_DIFF2
                        + NEW_DIFF3 + ADJPERF_NEW_DIFF3
                        + NEW_DIFF4 + NEW_DIFF5 + STREAK_DIFF + CARDIO_DIFF
                        + OPP_QUALITY_DIFF + STATIC_DIFF + ADJPERF_R1_DIFF)
    query = (
        "SELECT f.DATE, f.jbout, f.jfighter, f.opp_jfighter, f.weightindex, w.win, "
        "f.ufc_fight_count, f.age_dec_avg, "
        + ", ".join(f"f.{c}" for c in all_feature_cols)
        + """
        FROM final_features_fast f
        JOIN ufc_winlossko w
          ON w.jfighter=f.jfighter AND w.jevent=f.jevent AND w.jbout=f.jbout
        LEFT JOIN ufc_fight_results fr
          ON fr.jevent=f.jevent AND fr.jbout=f.jbout
        WHERE f.jfighter < f.opp_jfighter
          AND f.DATE >= ?
          AND (fr.sex IS NULL OR fr.sex != 1)
        """
    )
    df = pd.read_sql_query(query, conn, params=(ERA_CUTOFF,))
    conn.close()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["win"]  = df["win"].astype(int)

    # Compute age prime diff (Gaussian peak curve)
    age_f1 = df["age_dec_avg"]
    age_f2 = age_f1 - df["age_dec_avg_diff"]  # recover opponent age
    prime_f1 = np.exp(-((age_f1 - AGE_PEAK_AGE) / AGE_PEAK_WIDTH) ** 2)
    prime_f2 = np.exp(-((age_f2 - AGE_PEAK_AGE) / AGE_PEAK_WIDTH) ** 2)
    df["age_prime_diff"] = (prime_f1 - prime_f2).fillna(0.0)

    # Compute UFC age (years since UFC debut) for each fighter
    debut_dates = pd.read_sql_query(
        "SELECT jfighter, MIN(DATE) AS debut_date FROM ufc_winlossko GROUP BY jfighter",
        conn_ufc_age := sqlite3.connect(DB_PATH),
    )
    conn_ufc_age.close()
    debut_dates["debut_date"] = pd.to_datetime(debut_dates["debut_date"])
    # f1 UFC age
    df = df.merge(debut_dates.rename(columns={"debut_date": "_f1_debut"}),
                  on="jfighter", how="left")
    # f2 UFC age
    df = df.merge(debut_dates.rename(columns={"jfighter": "opp_jfighter", "debut_date": "_f2_debut"}),
                  on="opp_jfighter", how="left")
    df["ufc_age_diff"] = (
        (df["DATE"] - df["_f1_debut"]).dt.days / 365.25 -
        (df["DATE"] - df["_f2_debut"]).dt.days / 365.25
    ).fillna(0.0)
    df.drop(columns=["_f1_debut", "_f2_debut"], inplace=True)

    if min_prior_fights > 0:
        opp_fc = df["ufc_fight_count"] - df["ufc_fight_count_diff"]
        n_before = len(df)
        df = df[(df["ufc_fight_count"] >= min_prior_fights) & (opp_fc >= min_prior_fights)].copy()
        print(f"  After >={min_prior_fights}-prior-fights filter: {len(df):,} / {n_before:,}")
    print(f"Loaded {len(df):,} fights  "
          f"({df['DATE'].min().date()} → {df['DATE'].max().date()})")
    print(f"  Win rate: {df['win'].mean():.3f}")
    return df


# ── Train on all data ──────────────────────────────────────────────────────
def train_on_all(df: pd.DataFrame, anchor_date: str = None):
    # Add event-level rolling EMA
    df = add_event_rolling_ema(df, span=20)

    # Add ELO features (build_elo already returns jfighter, not f1)
    elo_df, ratings, last_date, extra = build_elo(anchor_date)
    df = df.merge(elo_df, on=["jbout", "jfighter"], how="left")
    for col in ELO_FEATURES:
        if col in ("elo_win_prob", "elo_predictability"):
            df[col] = df[col].fillna(0.5)
        else:
            df[col] = df[col].fillna(0.0)
    print(f"  ELO coverage: {(df['precomp_elo_diff'] != 0).sum():,} / {len(df):,} rows")

    # Clip and fill adjperf diffs
    df[ADJPERF_DIFF]      = df[ADJPERF_DIFF].clip(-14, 14).fillna(0.0)
    df[ADJPERF_NEW_DIFF]  = df[ADJPERF_NEW_DIFF].clip(-14, 14).fillna(0.0)
    df[ADJPERF_NEW_DIFF2] = df[ADJPERF_NEW_DIFF2].clip(-14, 14).fillna(0.0)
    df[ADJPERF_NEW_DIFF3] = df[ADJPERF_NEW_DIFF3].clip(-14, 14).fillna(0.0)
    df[NEW_DIFF]  = df[NEW_DIFF].fillna(0.0)
    df[NEW_DIFF2] = df[NEW_DIFF2].fillna(0.0)
    df[NEW_DIFF3] = df[NEW_DIFF3].fillna(0.0)
    df[NEW_DIFF4]        = df[NEW_DIFF4].fillna(0.0)
    df[NEW_DIFF5]        = df[NEW_DIFF5].fillna(0.0)
    df[STREAK_DIFF]      = df[STREAK_DIFF].fillna(0.0)
    df[CARDIO_DIFF]      = df[CARDIO_DIFF].fillna(0.0)
    df[OPP_QUALITY_DIFF] = df[OPP_QUALITY_DIFF].fillna(0.0)
    df[STATIC_DIFF]      = df[STATIC_DIFF].fillna(0.0)

    # Recency weights anchored to anchor_date (or DB max if not specified)
    max_date = pd.to_datetime(anchor_date) if anchor_date else df["DATE"].max()
    days_before = (max_date - df["DATE"]).dt.days.clip(lower=0)
    weights = np.exp(-RECENCY_LAM * days_before / 365.25)
    weights = weights / weights.mean()

    print(f"\nRecency weights: min={weights.min():.2f}  max={weights.max():.2f}  "
          f"ratio={weights.max()/weights.min():.1f}x  (anchor: {max_date.date()})")

    ema_range = df["event_rolling_ema"].agg(["min", "max", "mean"])
    print(f"  event_rolling_ema: min={ema_range['min']:.4f}  "
          f"max={ema_range['max']:.4f}  mean={ema_range['mean']:.4f}")

    # Impute remaining NULLs (age, days_since, reach, etc.)
    imputer = SimpleImputer(strategy="median")
    X_raw = df[ALL_FEATURES]
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=ALL_FEATURES)
    y = df["win"].values

    pipe = Pipeline([
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(C=0.007, solver="saga", max_iter=3000,
                                   penalty="elasticnet", l1_ratio=0.1,
                                   random_state=42)),
    ])
    pipe.fit(X, y, clf__sample_weight=weights.values)
    n_nonzero = int((pipe["clf"].coef_[0] != 0).sum())
    print(f"Trained on {len(df):,} fights  |  Active features: {n_nonzero}/{len(ALL_FEATURES)}")

    return pipe, imputer, df, ratings, last_date, extra


# ── Pull fighter stats ────────────────────────────────────────────────────
def _get_fighter_stats(conn, jfighter: str) -> dict:
    SKIP_SUFFIX = ("_diff",)
    SKIP_EXACT  = {"DATE","jevent","jbout","jfighter","opp_jfighter","win"}
    tables = [
        "final_features_fast",
        "new_features_dec_avg",
        "adjperf_new_fast",
        "new_features2_dec_avg",
        "adjperf_new2_fast",
        "new_features3_dec_avg",
        "adjperf_new3_fast",
        "new_features4_dec_avg",
        "new_features5_dec_avg",
        "streak_features",
        "cardio_features_dec_avg",
        "opp_quality_features",
    ]
    stats = {}
    for tbl in tables:
        try:
            row = pd.read_sql(
                f"SELECT * FROM {tbl} WHERE jfighter=? ORDER BY DATE DESC LIMIT 1",
                conn, params=(jfighter,)
            )
        except Exception:
            continue
        if row.empty:
            continue
        for col in row.columns:
            if col in SKIP_EXACT or col.endswith(SKIP_SUFFIX):
                continue
            stats[col] = row[col].iloc[0]
    return stats


# ── Predict a single fight ────────────────────────────────────────────────
def predict_fight(name_a: str, name_b: str, pipe, verbose: bool = True,
                  weightindex: int = None, event_ema: float = 0.5,
                  elo_ratings: dict = None, elo_last_date: dict = None,
                  elo_extra: dict = None, event_date: str = None) -> dict:
    conn = sqlite3.connect(DB_PATH)
    sa = _get_fighter_stats(conn, name_a)
    sb = _get_fighter_stats(conn, name_b)
    conn.close()

    if not sa:
        raise ValueError(f"Fighter not found in DB: {name_a}")
    if not sb:
        raise ValueError(f"Fighter not found in DB: {name_b}")

    # Alphabetical order matches WHERE jfighter < opp_jfighter in training
    if name_a < name_b:
        f1, f2, sf1, sf2 = name_a, name_b, sa, sb
    else:
        f1, f2, sf1, sf2 = name_b, name_a, sb, sa

    wc = weightindex if weightindex is not None else (sf1.get("weightindex") or 0)

    # ELO features for this matchup
    elo_vals = {}
    if elo_ratings is not None:
        elo_f1 = get_fighter_elo(f1, elo_ratings, elo_last_date or {},
                                 event_date or pd.Timestamp.now(), ELO_DECAY,
                                 ELO_DECAY_MAX, ELO_DECAY_MIDPOINT, ELO_DECAY_STEEPNESS)
        elo_f2 = get_fighter_elo(f2, elo_ratings, elo_last_date or {},
                                 event_date or pd.Timestamp.now(), ELO_DECAY,
                                 ELO_DECAY_MAX, ELO_DECAY_MIDPOINT, ELO_DECAY_STEEPNESS)
        elo_vals["precomp_elo_diff"] = elo_f1 - elo_f2
        elo_vals["elo_win_prob"] = 1.0 / (1.0 + 10 ** (-(elo_f1 - elo_f2) / ELO_LOGISTIC_SCALE))
        # Momentum: current ELO minus ELO 3 fights ago
        def _momentum(fighter, cur_elo):
            if elo_extra is None:
                return 0.0
            hist = list(elo_extra.get("elo_3ago", {}).get(fighter, []))
            return (cur_elo - hist[0]) if len(hist) >= 4 else 0.0
        elo_vals["elo_momentum_diff"] = _momentum(f1, elo_f1) - _momentum(f2, elo_f2)
        # Peak ELO
        pk1 = elo_extra.get("peak_elo", {}).get(f1, BASE_ELO) if elo_extra else BASE_ELO
        pk2 = elo_extra.get("peak_elo", {}).get(f2, BASE_ELO) if elo_extra else BASE_ELO
        elo_vals["peak_elo_diff"] = pk1 - pk2
        # Avg opponent ELO
        oq1 = elo_extra.get("opp_elos", {}).get(f1, []) if elo_extra else []
        oq2 = elo_extra.get("opp_elos", {}).get(f2, []) if elo_extra else []
        elo_vals["avg_opp_elo_diff"] = (float(np.mean(oq1)) if oq1 else BASE_ELO) - (float(np.mean(oq2)) if oq2 else BASE_ELO)
        # Consistency (std of recent deltas) — not available for inference, use 0
        elo_vals["elo_consist_diff"] = 0.0
        # Elo predictability: use latest per-weight-class EMA from elo_extra
        wc_pred_map = elo_extra.get("latest_elo_predictability_by_wc", {}) if elo_extra else {}
        elo_vals["elo_predictability"] = wc_pred_map.get(wc, 0.5)
    else:
        for col in ELO_FEATURES:
            elo_vals[col] = 0.5 if col in ("elo_win_prob", "elo_predictability") else 0.0

    row = {}
    for feat in ALL_FEATURES:
        if feat == "weightindex":
            row[feat] = wc
        elif feat == "event_rolling_ema":
            row[feat] = event_ema
        elif feat in elo_vals:
            row[feat] = elo_vals[feat]
        elif feat == "scheduled_rounds":
            row[feat] = 3.0  # default; could be overridden
        elif feat == "age_prime_diff":
            a1 = sf1.get("age_dec_avg") or 0
            a2 = sf2.get("age_dec_avg") or 0
            row[feat] = compute_age_prime(a1) - compute_age_prime(a2)
        elif feat == "age_ratio_diff":
            a1 = sf1.get("age_dec_avg") or 0
            a2 = sf2.get("age_dec_avg") or 0
            row[feat] = (a1 / a2) if a2 else 1.0
        elif feat == "reach_ratio_dec_avg_diff":
            r1 = sf1.get("REACH") or 0
            r2 = sf2.get("REACH") or 0
            row[feat] = (r1 / r2) if r2 else 1.0
        elif feat == "ufc_fight_count_diff":
            row[feat] = (sf1.get("ufc_fight_count") or 0) - (sf2.get("ufc_fight_count") or 0)
        else:
            col = feat[:-5]  # strip '_diff'
            v1 = sf1.get(col) or 0
            v2 = sf2.get(col) or 0
            row[feat] = v1 - v2

    X = pd.DataFrame([row])[ALL_FEATURES]
    adj_cols = [c for c in ALL_ADJPERF_DIFF if c in X.columns]
    X[adj_cols] = X[adj_cols].clip(-14, 14).fillna(0.0)
    X = X.fillna(0.0)

    prob_f1    = float(pipe.predict_proba(X)[0][1])
    winner     = f1 if prob_f1 >= 0.5 else f2
    confidence = prob_f1 if prob_f1 >= 0.5 else 1 - prob_f1

    if verbose:
        f1_date = sa.get("DATE","?") if name_a == f1 else sb.get("DATE","?")
        f2_date = sb.get("DATE","?") if name_b == f2 else sa.get("DATE","?")
        print(f"\n{'─'*60}")
        print(f"  {f1} vs {f2}")
        print(f"  Last DB fight: {f1}={f1_date}  |  {f2}={f2_date}")
        print(f"  P({f1} wins) = {prob_f1:.1%}    P({f2} wins) = {1-prob_f1:.1%}")
        print(f"  --> PICK: {winner}  ({confidence:.1%} confidence)")

        if hasattr(pipe, "named_steps"):
            coef = pipe.named_steps["clf"].coef_[0]
            top5 = sorted(zip(ALL_FEATURES, X.values[0], coef),
                          key=lambda x: abs(x[1]*x[2]), reverse=True)[:5]
            print(f"  Top drivers:")
            for fn, fv, fc in top5:
                direction = "favors " + (f1 if fv * fc > 0 else f2)
                print(f"    {fn:<48} val={fv:+.3f}  ({direction})")

    return {"f1": f1, "f2": f2, "prob_f1": prob_f1,
            "winner": winner, "confidence": confidence,
            "name_a": name_a, "name_b": name_b,
            "prob_a": prob_f1 if name_a == f1 else 1 - prob_f1}


# ── Predict a full card ───────────────────────────────────────────────────
def predict_card(fight_card: list, pipe, df_train: pd.DataFrame,
                 event_date: str, label: str = "EVENT",
                 elo_ratings: dict = None, elo_last_date: dict = None,
                 elo_extra: dict = None) -> None:
    print("\n" + "="*60)
    print(f"  {label} — PREDICTIONS")
    print("="*60)
    event_ema = get_current_event_ema(df_train, event_date)
    results = []
    for i, (a, b) in enumerate(fight_card, 1):
        try:
            r = predict_fight(a, b, pipe, event_ema=event_ema,
                              elo_ratings=elo_ratings,
                              elo_last_date=elo_last_date,
                              elo_extra=elo_extra,
                              event_date=event_date)
            r["bout_num"] = i
            results.append(r)
        except ValueError as e:
            print(f"\n  [{i}] ERROR: {e}")

    print("\n" + "="*60)
    print("  CARD SUMMARY")
    print("="*60)
    print(f"  {'#':<3}  {'PICK':<28}  {'CONF':>6}  {'vs':<3}  {'Opponent'}")
    print("  " + "-"*65)
    for r in results:
        loser = r["f2"] if r["winner"] == r["f1"] else r["f1"]
        bar = "█" * int(r["confidence"] * 20)
        print(f"  {r['bout_num']:<3}  {r['winner']:<28}  {r['confidence']:>5.1%}  {'vs':<3}  {loser}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("UFC Fight Predictor — Train-All + Live Event Prediction")
    print("=" * 60)

    EVENT_DATE = "2026-01-24"
    df   = load_all_data()
    pipe, imputer, df_trained, elo_ratings, elo_last_date, elo_extra = train_on_all(df, anchor_date=EVENT_DATE)

    # ── UFC 311 / Event 01-24-26 ─────────────────────────────────────────
    fight_card = [
        ("AlexPerez",         "CharlesJohnson"),
        ("NikitaKrylov",      "ModestasBukauskas"),
        ("AtebaGautier",      "AndreyPulyaev"),
        ("UmarNurmagomedov",  "DeivesonFigueiredo"),
        ("JeanSilva",         "ArnoldAllen"),
        ("NataliaSilva",      "RoseNamajunas"),
        ("WaldoCortesAcosta", "DerrickLewis"),
        ("SeanO'Malley",      "SongYadong"),
        ("JustinGaethje",     "PaddyPimblett"),
    ]

    predict_card(fight_card, pipe, df_trained, event_date=EVENT_DATE,
                 label="UFC Event 01-24-26",
                 elo_ratings=elo_ratings, elo_last_date=elo_last_date,
                 elo_extra=elo_extra)
