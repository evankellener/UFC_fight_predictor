"""
MMA-AI Pipeline Replication — complete feature engineering from raw fight stats.

Implements the MMA-AI.net v7 pipeline:
1. Base data loading (from SQL)
2. Beta-Binomial smoothing (binary outcomes)
3. Poisson-Gamma smoothing (count stats)
4. Derived features (accuracy, defense, ratios, per-minute)
5. Time-decayed averages
6. Opponent history + weight-class priors (with true MAD)
7. Adjusted Performance (AdjPerf) z-scores
8. Three-layer feature assembly
9. Final diff construction (unbalanced, red corner = fighter1)

Usage:
    from mma_ai_pipeline import build_features
    df = build_features(config="v7")  # or "jan26"
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from mma_ai_config import *

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: BASE DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_base_data() -> pd.DataFrame:
    """Load per-fighter per-fight raw stats from DB.

    Returns one row per fighter per fight with raw counts + metadata.
    Fighter ordering preserves BOUT order (fighter1 = first name in BOUT = red corner).
    """
    conn = sqlite3.connect(DB_PATH)

    # Get full-fight aggregated stats (sum across rounds)
    fight_stats = pd.read_sql("""
        SELECT
            r.jevent, r.jbout, r.jfighter,
            SUM(r.sigstracc) AS sig_str_land,
            SUM(r.sigstratt) AS sig_str_att,
            SUM(r.headacc)   AS head_land,
            SUM(r.headatt)   AS head_att,
            SUM(r.bodyacc)   AS body_land,
            SUM(r.bodyatt)   AS body_att,
            SUM(r.legacc)    AS leg_land,
            SUM(r.legatt)    AS leg_att,
            SUM(r.distacc)   AS dist_land,
            SUM(r.distatt)   AS dist_att,
            SUM(r.clinchacc) AS clinch_land,
            SUM(r.clinchatt) AS clinch_att,
            SUM(r.groundacc) AS ground_land,
            SUM(r.groundatt) AS ground_att,
            SUM(r.tdacc)     AS td_land,
            SUM(r.tdatt)     AS td_att,
            SUM(r.subatt)    AS sub_att,
            SUM(r.kd)        AS kd,
            SUM(r.rev)       AS rev,
            SUM(r.ctrl)      AS ctrl_sec,
            SUM(r.round_minutes) AS time_minutes,
            r.weightindex
        FROM ufc_fighter_match_stats_round_smooth r
        GROUP BY r.jevent, r.jbout, r.jfighter
    """, conn)

    # Get Round 1 stats separately
    rd1_stats = pd.read_sql("""
        SELECT
            r.jevent, r.jbout, r.jfighter,
            r.sigstracc AS sig_str_land_rd1,
            r.sigstratt AS sig_str_att_rd1,
            r.headacc   AS head_land_rd1,
            r.headatt   AS head_att_rd1,
            r.bodyacc   AS body_land_rd1,
            r.bodyatt   AS body_att_rd1,
            r.legacc    AS leg_land_rd1,
            r.legatt    AS leg_att_rd1,
            r.tdacc     AS td_land_rd1,
            r.tdatt     AS td_att_rd1,
            r.subatt    AS sub_att_rd1,
            r.kd        AS kd_rd1,
            r.rev       AS rev_rd1,
            r.ctrl      AS ctrl_sec_rd1,
            MIN(r.round_minutes, 5.0) AS time_minutes_rd1
        FROM ufc_fighter_match_stats_round_smooth r
        WHERE r.round = 1
    """, conn)

    # Get fight outcomes
    outcomes = pd.read_sql("""
        SELECT w.jevent, w.jbout, w.jfighter, w.win, w.ko, w.subw,
               e.DATE, e.LOCATION,
               fr.METHOD, fr.ROUND AS finish_round, fr."TIME FORMAT" as time_format
        FROM ufc_winlossko w
        JOIN ufc_event_details e ON e.jevent = w.jevent
        JOIN ufc_fight_results fr ON fr.jevent = w.jevent AND fr.jbout = w.jbout
    """, conn)

    # Get fighter physical stats
    tott = pd.read_sql("""
        SELECT jfighter, HEIGHT, WEIGHT, REACH, STANCE, DOB, weightindex as tott_wc
        FROM ufc_fighter_tott
    """, conn)

    # Get BOUT ordering (red corner = first fighter in BOUT)
    bout_order = pd.read_sql("""
        SELECT jevent, jbout, BOUT FROM ufc_fight_results
    """, conn)

    conn.close()

    # Parse dates
    outcomes["DATE"] = pd.to_datetime(outcomes["DATE"])

    # Determine decision outcome
    outcomes["decision"] = ((outcomes["win"] == 1) & (outcomes["ko"] == 0) &
                            (outcomes["subw"] == 0)).astype(int)

    # Merge fight stats + outcomes
    df = fight_stats.merge(outcomes, on=["jevent", "jbout", "jfighter"], how="inner")
    df = df.merge(rd1_stats, on=["jevent", "jbout", "jfighter"], how="left")

    # Determine opponent
    fighter_pairs = df[["jevent", "jbout", "jfighter"]].copy()
    opp = fighter_pairs.merge(fighter_pairs, on=["jevent", "jbout"], suffixes=("", "_opp"))
    opp = opp[opp["jfighter"] != opp["jfighter_opp"]]
    opp = opp.rename(columns={"jfighter_opp": "opp_jfighter"})
    df = df.merge(opp[["jevent", "jbout", "jfighter", "opp_jfighter"]],
                  on=["jevent", "jbout", "jfighter"], how="inner")

    # Determine fighter1/fighter2 from BOUT order (red corner = fighter1)
    # Parse BOUT: "FighterA  vs. FighterB" → fighter1 = FighterA (red corner)
    bout_order["f1_name"] = bout_order["BOUT"].str.split(r"\s+vs\.\s+").str[0].str.strip()
    bout_order["f1_jfighter"] = bout_order["f1_name"].str.replace(" ", "", regex=False)
    df = df.merge(bout_order[["jevent", "jbout", "f1_jfighter"]].drop_duplicates(),
                  on=["jevent", "jbout"], how="left")
    df["is_fighter1"] = (df["jfighter"] == df["f1_jfighter"]).astype(int)

    # Merge physical stats
    df = df.merge(tott, on="jfighter", how="left")

    # Compute age at fight time
    tott_dob = pd.to_datetime(df["DOB"], errors="coerce")
    df["age"] = (df["DATE"] - tott_dob).dt.days / 365.25

    # Sort by date
    df = df.sort_values(["DATE", "jevent", "jbout", "jfighter"]).reset_index(drop=True)

    # Dedupe: SQL upstream JOINs occasionally produce duplicate rows for the
    # same (jevent, jbout, jfighter) tuple — most prominently old tournament
    # bouts where round-by-round stats were merged with header rows multiple
    # times. Up to 20 identical copies of single fights have been observed.
    # If left in place, the merge in compute_derived_features fans out via
    # cross-product on duplicate keys, distorting EMAs and per-fighter
    # histories. ALSO causes the empirical leakage test to report spurious
    # "leaks" because truncation changes the duplication pattern.
    n_before = len(df)
    df = df.drop_duplicates(subset=["jevent", "jbout", "jfighter"],
                            keep="first").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Deduplicated {n_dropped} duplicate (jevent, jbout, jfighter) rows")

    # Fill NaN counts with 0
    count_cols = [c for c in df.columns if c.endswith("_land") or c.endswith("_att") or
                  c in ["kd", "rev", "ctrl_sec", "sub_att"] or c.endswith("_rd1")]
    for c in count_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    print(f"Loaded {len(df):,} fighter-fight rows  "
          f"({df['DATE'].min().date()} → {df['DATE'].max().date()})")
    print(f"  Unique fighters: {df['jfighter'].nunique():,}")
    print(f"  Unique bouts: {df['jbout'].nunique():,}")
    print(f"  fighter1 win rate: {df[df['is_fighter1']==1]['win'].mean():.3f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: BETA-BINOMIAL SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════

def beta_binomial_smooth(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Beta-Binomial smoothing to binary outcomes.

    Must run BEFORE Poisson-Gamma so sub_att is still raw.
    """
    df = df.copy()

    # Binary rate stats (ko, win, decision) — use era-rolling mean as prior.
    # Previously this was df.groupby("weightindex")[stat].mean() — all-time,
    # which leaked era info across fights. LEAKAGE_REFERENCE.md §1/§3.
    for stat, tau_key in [("ko", "ko"), ("win", "win"), ("decision", "decision")]:
        rolling_rate = _era_rolling_mean(df, stat)
        # Strictly-prior fallback chain: 2yr rolling → all-prior expanding →
        # constant. Replaces the leaky df[stat].mean() that included future
        # fights. LEAKAGE_REFERENCE.md §1/§3.
        all_prior_rate = _all_prior_rolling_mean(df, stat)
        df[f"_rate_prior_{stat}"] = rolling_rate \
            .fillna(all_prior_rate) \
            .fillna(_CONST_BINARY_PRIOR)

        def _smooth(row, _stat=stat, _tau_key=tau_key):
            wc = row["weightindex"]
            tau = BB_TAU_GLOBAL[_tau_key]
            if wc in BB_TAU_WC_OVERRIDES and _tau_key in BB_TAU_WC_OVERRIDES[wc]:
                tau = BB_TAU_WC_OVERRIDES[wc][_tau_key]
            rate_prior = row[f"_rate_prior_{_stat}"]
            successes = row[_stat]
            attempts = 1
            return (rate_prior * tau + successes) / (tau + attempts)

        df[f"{stat}_smooth"] = df.apply(_smooth, axis=1)
        df.drop(columns=[f"_rate_prior_{stat}"], inplace=True)

    # sub_land = successful submissions (≈ subw). Rate = sub_land / sub_att.
    df["sub_land"] = df["subw"].fillna(0)
    rolling_sub_rate = _era_rolling_ratio(df, "sub_land", "sub_att", eps=1.0)
    # Strictly-prior fallback chain (no df-wide aggregation).
    all_prior_sub = _all_prior_rolling_ratio(df, "sub_land", "sub_att", eps=1.0)
    df["_rate_prior_sub"] = rolling_sub_rate \
        .fillna(all_prior_sub) \
        .fillna(_CONST_BINARY_PRIOR)

    def _smooth_sub(row):
        wc = row["weightindex"]
        tau = BB_TAU_GLOBAL["sub_land"]
        if wc in BB_TAU_WC_OVERRIDES and "sub_land" in BB_TAU_WC_OVERRIDES[wc]:
            tau = BB_TAU_WC_OVERRIDES[wc]["sub_land"]
        rate_prior = row["_rate_prior_sub"]
        successes = row["sub_land"]
        attempts = max(row["sub_att"], 1)
        return (rate_prior * tau + successes) / (tau + attempts)

    df["sub_land_smooth"] = df.apply(_smooth_sub, axis=1)
    df.drop(columns=["_rate_prior_sub"], inplace=True)

    # ctrl: time-share (seconds). Rate = ctrl_sec / (time_minutes * 60).
    # We compute the ratio on a derived "time_seconds" column so the rolling
    # ratio helper works cleanly.
    df["_time_seconds"] = (df["time_minutes"] * 60).clip(lower=1)
    rolling_ctrl_rate = _era_rolling_ratio(df, "ctrl_sec", "_time_seconds", eps=1.0)
    all_prior_ctrl = _all_prior_rolling_ratio(df, "ctrl_sec", "_time_seconds", eps=1.0)
    df["_rate_prior_ctrl"] = rolling_ctrl_rate \
        .fillna(all_prior_ctrl) \
        .fillna(_CONST_RATE_PRIOR)

    def _smooth_ctrl(row):
        wc = row["weightindex"]
        tau = BB_TAU_GLOBAL["ctrl"]
        if wc in BB_TAU_WC_OVERRIDES and "ctrl" in BB_TAU_WC_OVERRIDES[wc]:
            tau = BB_TAU_WC_OVERRIDES[wc]["ctrl"]
        rate_prior = row["_rate_prior_ctrl"]
        total_sec = max(row["time_minutes"] * 60, 1)
        ctrl_sec = row["ctrl_sec"]
        p_post = (rate_prior * tau * 60 + ctrl_sec) / (tau * 60 + total_sec)
        return p_post * total_sec

    df["ctrl_smooth"] = df.apply(_smooth_ctrl, axis=1)
    df.drop(columns=["_rate_prior_ctrl", "_time_seconds"], inplace=True)

    print(f"  Beta-Binomial smoothing (era-rolling 2yr priors): ko, win, decision, sub_land, ctrl")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# ERA-ROLLING WEIGHT-CLASS BASELINES
# ═══════════════════════════════════════════════════════════════════════════
#
# Historically this pipeline computed per-WC shrinkage priors as the mean of a
# stat across ALL fights in that weight class, cross-era. That's a leakage
# issue (§1 / §3: a 2017 fight's prior includes 2026 data) and a drift issue
# (a 2026 fighter gets smoothed toward a weight-class mean that was partly
# defined by 2017-era fighters throwing 27% fewer strikes per minute).
#
# Fix: per-row rolling baseline over [DATE - ERA_WINDOW_DAYS, DATE),
# restricted to same weightindex, using pandas groupby-rolling with
# closed='left' so the current row and any same-date fights are excluded.
#
# All rolling baselines are STRICTLY PRIOR to the current row — conforms to
# LEAKAGE_REFERENCE.md §1 (temporal splits) and §3 (career aggregates).

ERA_WINDOW_DAYS = 730  # 2-year rolling window


def _all_prior_rolling_mean(df, col):
    """Per-row expanding mean of `col` over ALL prior fights in same weight
    class (closed='left' excludes current row). Used as a strictly-time-
    respecting fallback when the 2yr rolling window is empty (no fights in
    the same WC in the last 2yr — happens for early-UFC fights and obscure
    weight classes).
    Returns Series aligned to df.index. NaN only when there are NO prior
    same-WC fights at all."""
    # ~100-year window = effectively "all prior fights in this WC"
    return _era_rolling_mean(df, col, window_days=36500)


def _all_prior_rolling_ratio(df, numer_col, denom_col, eps=1.0):
    """Per-row expanding ratio sum(numer)/sum(denom) over ALL prior fights
    in same WC. Used as strictly-time-respecting fallback for rate-style
    priors when the 2yr rolling window is empty."""
    return _era_rolling_ratio(df, numer_col, denom_col,
                              window_days=36500, eps=eps)


# Constant fallbacks when even all-prior is empty (very first fight in a
# weight class has no prior data anywhere). These are domain priors,
# NOT computed from data, so they do not leak. Loose values that won't
# bias predictions when applied to ~1-3 fights total in the dataset.
_CONST_BINARY_PRIOR = 0.20   # baseline rate for binary outcomes (ko/sub/etc)
_CONST_RATE_PRIOR   = 0.05   # baseline rate for time-share / per-min stats


def _era_rolling_mean(df, col, window_days=ERA_WINDOW_DAYS):
    """Per-row 2yr rolling mean of `col` within the same weightindex.
    Excludes the current row (closed='left'). Returns a Series aligned
    to df's index. NaN where no prior fights exist in the window OR
    where weightindex/DATE is missing on the row.
    """
    tmp = df[["DATE", "weightindex", col]].copy()
    tmp["_orig_idx"] = df.index
    # Rolling needs valid weightindex and DATE; NaNs in those get dropped
    # by groupby silently — handle explicitly so lengths stay consistent.
    valid = tmp.dropna(subset=["weightindex", "DATE"]).copy()
    valid = valid.sort_values(["weightindex", "DATE"]).reset_index(drop=True)
    if len(valid) == 0:
        return pd.Series(float("nan"), index=df.index)
    rolled = (valid.groupby("weightindex")
                   .rolling(f"{window_days}D", on="DATE", closed="left", min_periods=1)[col]
                   .mean()
                   .reset_index(level=0, drop=True))
    valid["_baseline"] = rolled.values
    # Map back using the original index we stashed
    lookup = valid.set_index("_orig_idx")["_baseline"]
    return df.index.to_series().map(lookup)


def _era_rolling_ratio(df, numer_col, denom_col, window_days=ERA_WINDOW_DAYS,
                       eps=0.1):
    """Per-row 2yr rolling ratio sum(numer) / sum(denom) within the same wc.
    Used for rate-style priors (e.g. total KOs / total fight minutes).
    Excludes the current row; returns Series aligned to df.index.
    """
    tmp = df[["DATE", "weightindex", numer_col, denom_col]].copy()
    tmp["_orig_idx"] = df.index
    valid = tmp.dropna(subset=["weightindex", "DATE"]).copy()
    valid = valid.sort_values(["weightindex", "DATE"]).reset_index(drop=True)
    if len(valid) == 0:
        return pd.Series(float("nan"), index=df.index)
    num = (valid.groupby("weightindex")
                .rolling(f"{window_days}D", on="DATE", closed="left", min_periods=1)[numer_col]
                .sum()
                .reset_index(level=0, drop=True))
    den = (valid.groupby("weightindex")
                .rolling(f"{window_days}D", on="DATE", closed="left", min_periods=1)[denom_col]
                .sum()
                .reset_index(level=0, drop=True))
    valid["_ratio"] = num.values / den.values.clip(min=eps)
    lookup = valid.set_index("_orig_idx")["_ratio"]
    return df.index.to_series().map(lookup)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: POISSON-GAMMA SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════

def poisson_gamma_smooth(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Poisson-Gamma smoothing to count statistics."""
    df = df.copy()

    # Full-fight stats: era-rolling ratio = rolling_sum(stat) / rolling_sum(time)
    # Replaces the all-time ratio computed from the whole df (§1/§3 leakage).
    for stat, tau_val in PG_TAU_GLOBAL.items():
        if stat not in df.columns:
            continue

        rolling_rate = _era_rolling_ratio(df, stat, "time_minutes", eps=0.1)
        all_prior_rate = _all_prior_rolling_ratio(df, stat, "time_minutes", eps=0.1)
        df[f"_rate_prior_{stat}"] = rolling_rate \
            .fillna(all_prior_rate) \
            .fillna(_CONST_RATE_PRIOR)

        def _smooth_pg(row, _stat=stat, _tau=tau_val):
            wc = row["weightindex"]
            tau = _tau
            if wc in PG_TAU_WC_OVERRIDES and _stat in PG_TAU_WC_OVERRIDES[wc]:
                tau = PG_TAU_WC_OVERRIDES[wc][_stat]
            rate_prior = row[f"_rate_prior_{_stat}"]
            t = max(row["time_minutes"], 0.01)
            observed = row[_stat]
            lambda_post = (rate_prior * tau + observed) / (tau + t)
            return t * lambda_post

        df[f"{stat}_smooth"] = df.apply(_smooth_pg, axis=1)
        df.drop(columns=[f"_rate_prior_{stat}"], inplace=True)

    # Round 1 stats — same treatment
    for stat, tau_val in PG_TAU_RD1.items():
        if stat not in df.columns:
            continue

        rolling_rate = _era_rolling_ratio(df, stat, "time_minutes_rd1", eps=0.1)
        all_prior_rate = _all_prior_rolling_ratio(df, stat, "time_minutes_rd1", eps=0.1)
        df[f"_rate_prior_{stat}"] = rolling_rate \
            .fillna(all_prior_rate) \
            .fillna(_CONST_RATE_PRIOR)

        def _smooth_rd1(row, _stat=stat, _tau=tau_val):
            wc = row["weightindex"]
            tau = _tau
            if wc in PG_TAU_WC_OVERRIDES and _stat in PG_TAU_WC_OVERRIDES[wc]:
                tau = PG_TAU_WC_OVERRIDES[wc][_stat]
            rate_prior = row[f"_rate_prior_{_stat}"]
            t = max(row.get("time_minutes_rd1", 0) or 0, 0.01)
            observed = row.get(_stat, 0) or 0
            lambda_post = (rate_prior * tau + observed) / (tau + t)
            return t * lambda_post

        df[f"{stat}_smooth"] = df.apply(_smooth_rd1, axis=1)
        df.drop(columns=[f"_rate_prior_{stat}"], inplace=True)

    print(f"  Poisson-Gamma smoothing (era-rolling 2yr priors): "
          f"{len(PG_TAU_GLOBAL)} full-fight + {len(PG_TAU_RD1)} R1 stats")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: DERIVED FEATURES (on smoothed values)
# ═══════════════════════════════════════════════════════════════════════════

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy, defense, per-minute, ratio features from smoothed stats."""
    df = df.copy()
    eps = 1e-8  # prevent division by zero
    t = df["time_minutes"].clip(lower=0.1)

    # ── Accuracy rates ────────────────────────────────────────────
    for prefix in ["sig_str", "head", "body", "leg", "dist", "clinch", "ground", "td"]:
        land_col = f"{prefix}_land_smooth"
        att_col = f"{prefix}_att_smooth"
        if land_col in df.columns and att_col in df.columns:
            df[f"{prefix}_acc"] = df[land_col] / df[att_col].clip(lower=eps)

    # ── Per-minute rates ──────────────────────────────────────────
    for prefix in ["sig_str_land", "head_land", "body_land", "leg_land",
                    "dist_land", "clinch_land", "ground_land",
                    "td_land", "td_att", "sub_att", "kd", "rev"]:
        smooth_col = f"{prefix}_smooth"
        if smooth_col in df.columns:
            df[f"{prefix}_pm"] = df[smooth_col] / t

    # ctrl per minute
    df["ctrl_pm"] = df["ctrl_smooth"] / (t * 60)  # ctrl_smooth is in seconds

    # ── Ratios ────────────────────────────────────────────────────
    sig = df["sig_str_land_smooth"].clip(lower=eps)
    df["head_land_ratio"] = df["head_land_smooth"] / sig
    df["body_land_ratio"] = df["body_land_smooth"] / sig
    df["leg_land_ratio"] = df["leg_land_smooth"] / sig
    df["dist_land_ratio"] = df["dist_land_smooth"] / sig
    df["clinch_land_ratio"] = df["clinch_land_smooth"] / sig
    df["ground_land_ratio"] = df["ground_land_smooth"] / sig

    # Style features (from Sep 2025 update)
    df["dist_per_sig_str_land"] = df["dist_land_smooth"] / sig
    df["ground_per_ctrl"] = df["ground_land_smooth"] / df["ctrl_smooth"].clip(lower=eps)
    df["ko_per_sig_str_land"] = df["ko_smooth"] / sig  # ko rate per sig strike
    df["td_per_sig_str_att"] = df["td_land_smooth"] / df["sig_str_att_smooth"].clip(lower=eps)

    # Submission-related
    df["sub_land_rate"] = df["sub_land_smooth"]  # already a probability from BB smoothing

    # ── Defense rates (1 - opp_land/opp_att) ──────────────────────
    # Need opponent's stats for this fight — merge from opponent rows
    # For now, compute from the fighter's own "allowed" perspective:
    # defense = 1 - (what opponent landed on me / what opponent threw at me)
    # We need opp stats per fight — get from the paired row
    opp_lookup = df.set_index(["jevent", "jbout", "jfighter"])[
        ["sig_str_land_smooth", "sig_str_att_smooth", "head_land_smooth", "head_att_smooth",
         "body_land_smooth", "body_att_smooth", "leg_land_smooth", "leg_att_smooth",
         "dist_land_smooth", "dist_att_smooth", "clinch_land_smooth", "clinch_att_smooth",
         "ground_land_smooth", "ground_att_smooth", "td_land_smooth", "td_att_smooth",
         "sub_att_smooth", "sub_land_smooth"]
    ].rename(columns=lambda c: f"opp_{c}")

    # Join opponent stats: my defense = 1 - what my opponent landed / what they threw
    df = df.merge(
        opp_lookup.reset_index().rename(columns={"jfighter": "opp_jfighter"}),
        on=["jevent", "jbout", "opp_jfighter"], how="left"
    )

    for prefix in ["sig_str", "head", "body", "leg", "dist", "clinch", "ground", "td"]:
        opp_land = f"opp_{prefix}_land_smooth"
        opp_att = f"opp_{prefix}_att_smooth"
        if opp_land in df.columns and opp_att in df.columns:
            df[f"{prefix}_def"] = 1.0 - df[opp_land] / df[opp_att].clip(lower=eps)

    # Sub defense (1 - opponent's sub success rate against me)
    opp_sub_land = df.get("opp_sub_land_smooth", pd.Series(0, index=df.index))
    opp_sub_att = df.get("opp_sub_att_smooth", pd.Series(eps, index=df.index)).clip(lower=eps)
    df["sub_def"] = 1.0 - opp_sub_land / opp_sub_att

    # ── Sig str landing ratio (same as accuracy, but explicit name match) ──
    df["sig_str_land_ratio"] = df["sig_str_acc"]  # alias for his naming convention

    # ── Ko ratio (ko / fights as rate) ────────────────────────────
    df["ko_ratio"] = df["ko_smooth"]  # from BB smoothing, already a rate

    # ── Sub att ratio ─────────────────────────────────────────────
    df["sub_att_ratio"] = df["sub_att_smooth"] / t.clip(lower=eps)

    # ── Reversal ratio ────────────────────────────────────────────
    opp_ctrl = df["ctrl_sec"].clip(lower=eps)
    df["rev_per_ctrlopp"] = df["rev_smooth"] / (opp_ctrl / 60)
    df["rev_ratio"] = df["rev_smooth"] / t.clip(lower=eps)

    # ── Control ratio ─────────────────────────────────────────────
    df["ctrl_ratio"] = df["ctrl_smooth"] / (t * 60).clip(lower=eps)

    # ── Ground land per ctrl ──────────────────────────────────────
    df["ground_land_per_ctrl"] = df["ground_land_smooth"] / df["ctrl_smooth"].clip(lower=eps)
    df["td_land_per_ctrl"] = df["td_land_smooth"] / df["ctrl_smooth"].clip(lower=eps)

    # ── Style features (from Sep 2025 update) ─────────────────────
    df["dist_per_sig_str_land"] = df["dist_land_smooth"] / sig
    df["ground_per_ctrl"] = df["ground_land_smooth"] / df["ctrl_smooth"].clip(lower=eps)
    df["ko_per_sig_str_land"] = df["ko_smooth"] / sig
    df["td_per_sig_str_att"] = df["td_land_smooth"] / df["sig_str_att_smooth"].clip(lower=eps)
    df["head_per_sig_str_land"] = df["head_land_smooth"] / sig

    # ── Round 1 per-minute ────────────────────────────────────────
    t_rd1 = df["time_minutes_rd1"].clip(lower=0.1)
    for stat in ["sig_str_land_rd1", "kd_rd1", "td_land_rd1", "rev_rd1"]:
        smooth_col = f"{stat}_smooth"
        if smooth_col in df.columns:
            df[f"{stat}_pm"] = df[smooth_col] / t_rd1

    df["ctrl_rd1_pm"] = df.get("ctrl_sec_rd1_smooth", df.get("ctrl_sec_rd1", 0)) / (t_rd1 * 60)

    # ── Round 1 ratios ────────────────────────────────────────────
    sig_rd1 = df.get("sig_str_land_rd1_smooth", pd.Series(0, index=df.index)).clip(lower=eps)
    if "rev_rd1_smooth" in df.columns:
        # rev_rd1 ratio relative to opponent ctrl in R1
        ctrl_rd1_sec = df.get("ctrl_sec_rd1", pd.Series(1, index=df.index)).clip(lower=eps)
        df["rev_rd1_ratio"] = df["rev_rd1_smooth"] / (ctrl_rd1_sec / 60)

    # ── Age ratio ─────────────────────────────────────────────────
    # Will be diffed as f1_age/f2_age in the assembly step — store raw for now
    df["age_ratio_raw"] = df["age"]  # will compute ratio in diff step

    # Drop temp opp columns
    opp_cols_to_drop = [c for c in df.columns if c.startswith("opp_") and c.endswith("_smooth")]
    df.drop(columns=opp_cols_to_drop, inplace=True, errors="ignore")

    print(f"  Derived features: accuracy, defense, per-minute, ratios, style metrics")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: TIME-DECAYED AVERAGES
# ═══════════════════════════════════════════════════════════════════════════

def compute_decayed_averages(df: pd.DataFrame, feature_cols: list,
                              decay_lambda: float = 0.13) -> pd.DataFrame:
    """Compute time-decayed career averages for each fighter, lagged by 1 fight.

    Vectorized: processes all features at once per fighter using matrix operations.
    """
    df = df.sort_values(["DATE", "jevent", "jbout"]).copy()

    # Initialize output columns
    for col in feature_cols:
        df[f"{col}_dec_avg"] = np.nan

    # Process per fighter (vectorized across features)
    feat_matrix = df[feature_cols].values  # N × F matrix
    dates_raw = df["DATE"].values
    fighters = df["jfighter"].values

    # Group indices by fighter
    fighter_groups = {}
    for i, f in enumerate(fighters):
        if f not in fighter_groups:
            fighter_groups[f] = []
        fighter_groups[f].append(i)

    result_matrix = np.full_like(feat_matrix, np.nan)

    for jfighter, indices in fighter_groups.items():
        if len(indices) < 2:
            continue

        idx = np.array(indices)
        dates = dates_raw[idx]
        vals = feat_matrix[idx]  # n_fights × n_features

        for i in range(1, len(idx)):
            # Days between current fight and all prior fights
            days_diff = (dates[i] - dates[:i]).astype("timedelta64[D]").astype(float)
            weights = np.exp(-decay_lambda * days_diff / 365.25)

            prior = vals[:i]  # i × F
            # Weighted mean across prior fights, handling NaN per column
            w_col = np.where(np.isnan(prior), 0, weights[:, None])
            v_col = np.where(np.isnan(prior), 0, prior)
            w_sum = w_col.sum(axis=0)
            mask = w_sum > 0
            avg = np.where(mask, (w_col * v_col).sum(axis=0) / w_sum, np.nan)
            result_matrix[idx[i]] = avg

    for j, col in enumerate(feature_cols):
        df[f"{col}_dec_avg"] = result_matrix[:, j]

    print(f"  Decayed averages (λ={decay_lambda}): {len(feature_cols)} features")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: OPPONENT HISTORY + WC PRIORS (with true MAD)
# ═══════════════════════════════════════════════════════════════════════════

def compute_opponent_history(df: pd.DataFrame, stat_cols: list,
                              decay_lambda: float = 0.13) -> pd.DataFrame:
    """For each fight, compute what the opponent typically allows.

    Per-column n_eff (Kish), true MAD, weighted mean.
    Vectorized: builds opponent history as arrays, processes per-fight.
    """
    df = df.sort_values(["DATE", "jevent", "jbout"]).copy()

    # Initialize output columns
    for col in stat_cols:
        df[f"{col}_opp_mean"] = np.nan
        df[f"{col}_opp_mad"] = np.nan
        df[f"{col}_opp_n_eff"] = np.nan

    # Build opponent history: what did fighters do AGAINST each opponent?
    # Key: opp_jfighter. Value: list of (date, feature_array)
    stat_idx = {col: i for i, col in enumerate(stat_cols)}
    n_stats = len(stat_cols)

    opp_dates = {}   # opp -> [date1, date2, ...]
    opp_vals = {}    # opp -> [[val1_stat0, val1_stat1, ...], [val2_stat0, ...], ...]

    feat_arr = df[stat_cols].values
    dates_arr = df["DATE"].values
    opp_arr = df["opp_jfighter"].values

    for i in range(len(df)):
        opp = opp_arr[i]
        if opp not in opp_dates:
            opp_dates[opp] = []
            opp_vals[opp] = []
        opp_dates[opp].append(dates_arr[i])
        opp_vals[opp].append(feat_arr[i])

    # Sort each opponent's history
    for opp in opp_dates:
        pairs = sorted(zip(opp_dates[opp], opp_vals[opp]), key=lambda x: x[0])
        opp_dates[opp] = [p[0] for p in pairs]
        opp_vals[opp] = np.array([p[1] for p in pairs])

    # For each fight, compute opponent history from PRIOR fights
    opp_mean_out = np.full((len(df), n_stats), np.nan)
    opp_mad_out = np.full((len(df), n_stats), np.nan)
    opp_neff_out = np.full((len(df), n_stats), np.nan)

    opp_col = df["opp_jfighter"].values
    date_col = df["DATE"].values

    for i in range(len(df)):
        opp = opp_col[i]
        fight_date = date_col[i]

        if opp not in opp_dates:
            continue

        all_dates = opp_dates[opp]
        all_vals = opp_vals[opp]

        # Binary search for prior fights
        import bisect
        cutoff = bisect.bisect_left(all_dates, fight_date)
        if cutoff < 1:
            continue

        prior_dates = all_dates[:cutoff]
        prior_vals = all_vals[:cutoff]  # cutoff × n_stats

        # Decay weights
        days_ago = np.array([(fight_date - d).astype("timedelta64[D]").astype(float)
                              for d in prior_dates])
        weights = np.exp(-decay_lambda * days_ago / 365.25)

        # Per-column: weighted mean, MAD, Kish n_eff
        for j in range(n_stats):
            col_vals = prior_vals[:, j]
            valid = ~np.isnan(col_vals)
            if valid.sum() < 1:
                continue

            v = col_vals[valid]
            w = weights[valid]

            # Weighted mean
            opp_mean_out[i, j] = np.sum(w * v) / np.sum(w)

            # True MAD
            med = np.median(v)
            opp_mad_out[i, j] = np.median(np.abs(v - med))

            # Per-column Kish n_eff
            opp_neff_out[i, j] = np.sum(w) ** 2 / np.sum(w ** 2)

    for j, col in enumerate(stat_cols):
        df[f"{col}_opp_mean"] = opp_mean_out[:, j]
        df[f"{col}_opp_mad"] = opp_mad_out[:, j]
        df[f"{col}_opp_n_eff"] = opp_neff_out[:, j]

    print(f"  Opponent history (per-column MAD + n_eff): {len(stat_cols)} stats")
    return df


def compute_wc_priors(df: pd.DataFrame, stat_cols: list) -> dict:
    """Compute per-weight-class means, MADs, and MAD floors for each stat."""
    priors = {}

    for col in stat_cols:
        priors[col] = {}
        for wc, grp in df.groupby("weightindex"):
            vals = grp[col].dropna().values
            if len(vals) < 3:
                continue
            wc_mean = np.mean(vals)
            median_val = np.median(vals)
            wc_mad = np.median(np.abs(vals - median_val))
            priors[col][wc] = {"mean": wc_mean, "mad": max(wc_mad, 1e-6)}

        # Global fallback
        all_vals = df[col].dropna().values
        if len(all_vals) > 0:
            global_mean = np.mean(all_vals)
            global_median = np.median(all_vals)
            global_mad = np.median(np.abs(all_vals - global_median))
            priors[col]["global"] = {"mean": global_mean, "mad": max(global_mad, 1e-6)}

        # MAD floor = 5th percentile of per-opponent MADs
        opp_mad_col = f"{col}_opp_mad"
        if opp_mad_col in df.columns:
            opp_mads = df[opp_mad_col].dropna().values
            if len(opp_mads) > 0:
                priors[col]["mad_floor"] = max(np.percentile(opp_mads, MAD_FLOOR_PERCENTILE), 1e-6)
            else:
                priors[col]["mad_floor"] = 1e-6
        else:
            priors[col]["mad_floor"] = 1e-6

    print(f"  WC priors computed: {len(stat_cols)} stats × {df['weightindex'].nunique()} weight classes")
    return priors


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: ADJUSTED PERFORMANCE (AdjPerf)
# ═══════════════════════════════════════════════════════════════════════════

def compute_adjperf(df: pd.DataFrame, stat_cols: list, priors: dict) -> pd.DataFrame:
    """Compute opponent-adjusted, weight-class-adjusted z-scores. Vectorized."""
    df = df.copy()

    # Pre-extract per-row data once (avoid pandas .get per row)
    wc_arr = df["weightindex"].values

    for col in stat_cols:
        family = "default"
        for prefix, fam in STAT_FAMILY.items():
            if col.startswith(prefix):
                family = fam
                break
        K_mean = ADJPERF_K[family]["K_mean"]
        K_mad = ADJPERF_K[family]["K_mad"]

        opp_mean_col = f"{col}_opp_mean"
        opp_mad_col = f"{col}_opp_mad"
        n_eff_col = f"{col}_opp_n_eff"

        col_priors = priors.get(col, {})
        global_prior = col_priors.get("global", {})
        global_mean = global_prior.get("mean", 0.0)
        global_mad  = global_prior.get("mad", 0.1)
        mad_floor   = col_priors.get("mad_floor", 1e-6)

        # Build per-row wc_mean and wc_mad arrays via map (vectorized)
        n = len(df)
        wc_mean_arr = np.full(n, global_mean)
        wc_mad_arr  = np.full(n, global_mad)
        for wc, p in col_priors.items():
            if wc == "global" or wc == "mad_floor":
                continue
            mask = wc_arr == wc
            if mask.any():
                wc_mean_arr[mask] = p.get("mean", global_mean)
                wc_mad_arr[mask]  = p.get("mad", global_mad)

        # Per-row vectors for opp values + n_eff + observed
        observed = df[col].values if col in df.columns else np.full(n, np.nan)
        opp_mean = df[opp_mean_col].values if opp_mean_col in df.columns else np.full(n, np.nan)
        opp_mad  = df[opp_mad_col].values  if opp_mad_col  in df.columns else np.full(n, np.nan)
        n_eff    = df[n_eff_col].values    if n_eff_col    in df.columns else np.zeros(n)
        n_eff = np.where(np.isfinite(n_eff), n_eff, 0.0)

        # Shrinkage weights
        w_mean = np.where(n_eff > 0, n_eff / (n_eff + K_mean), 0.0)
        w_mad  = np.where(n_eff > 0, n_eff / (n_eff + K_mad), 0.0)

        # Blended baseline (mu / sigma)
        opp_mean_valid = np.isfinite(opp_mean) & (n_eff >= 1)
        opp_mad_valid  = np.isfinite(opp_mad)  & (n_eff >= 1)
        mu = np.where(opp_mean_valid,
                      w_mean * np.where(opp_mean_valid, opp_mean, 0.0)
                      + (1 - w_mean) * wc_mean_arr,
                      wc_mean_arr)
        sigma = np.where(opp_mad_valid,
                         w_mad * np.where(opp_mad_valid, opp_mad, 0.0)
                         + (1 - w_mad) * wc_mad_arr,
                         wc_mad_arr)
        sigma = np.maximum(sigma, mad_floor)

        # Z-score with clipping; observed missing → NaN result
        with np.errstate(divide="ignore", invalid="ignore"):
            z_raw = (observed - mu) / sigma
        z = np.where(np.isfinite(observed) & (sigma > 0),
                     np.clip(z_raw, -ADJPERF_CLIP, ADJPERF_CLIP),
                     np.nan)
        df[f"{col}_adjperf"] = z

    print(f"  AdjPerf z-scores: {len(stat_cols)} stats (clip=±{ADJPERF_CLIP})")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8-9: FEATURE ASSEMBLY + FINAL DIFF
# ═══════════════════════════════════════════════════════════════════════════

def assemble_features(df: pd.DataFrame, stat_cols: list,
                       decay_lambda: float = 0.13,
                       return_individuals: bool = False) -> pd.DataFrame:
    """Build three-layer features and construct final fighter1-fighter2 diffs.

    If return_individuals=True, returns (result, fighter_stats, feature_cols) where
    fighter_stats is a dict mapping jfighter -> dict of latest individual stats.
    """

    # ── Compute static per-fight features BEFORE dec_avg ────────
    # UFC Age (years since first fight)
    debut_dates = df.groupby("jfighter")["DATE"].min().to_dict()
    df["ufc_age"] = df.apply(
        lambda r: (r["DATE"] - debut_dates.get(r["jfighter"], r["DATE"])).days / 365.25,
        axis=1
    )

    # Reach ratio — use a STRICTLY-PRIOR median (per-row expanding median over
    # all fights with DATE < this fight's DATE). Previously used df["REACH"].median()
    # over the full dataset which leaks future REACH values into early-fight ratios.
    # LEAKAGE_REFERENCE.md §1/§3.
    _reach_sorted = df.sort_values("DATE")[["DATE", "REACH"]].copy()
    _reach_sorted["_orig_idx"] = _reach_sorted.index
    # Expanding median, shifted by 1 so current row is excluded
    _reach_sorted["_prior_median"] = (
        _reach_sorted["REACH"].expanding(min_periods=1).median().shift(1)
    )
    # Constant fallback for the very first fight (no prior data anywhere)
    _CONST_REACH_MEDIAN = 70.0  # inches, league-typical
    _reach_lookup = (_reach_sorted.set_index("_orig_idx")["_prior_median"]
                                  .fillna(_CONST_REACH_MEDIAN))
    df["reach_ratio"] = df["REACH"] / df.index.to_series().map(_reach_lookup)

    # Days since last fight
    df["days_since_last_fight"] = df.groupby("jfighter")["DATE"].diff().dt.days.fillna(365)

    # ── Compute dec_avg for static features (age, ufc_age, days_since_last_fight) ──
    # These are his #1 (#age_dec_avg) and #15/#27 (days_since_last_fight_dec_avg) features
    static_da_cols = ["age", "ufc_age", "days_since_last_fight"]
    print(f"  Computing decayed averages for {len(static_da_cols)} static features...")
    df = compute_decayed_averages(df, static_da_cols, decay_lambda)

    # Layer 1: dec_avg of adjperf scores
    adjperf_cols = [f"{col}_adjperf" for col in stat_cols if f"{col}_adjperf" in df.columns]
    print(f"  Computing decayed averages for {len(adjperf_cols)} adjperf features...")
    df = compute_decayed_averages(df, adjperf_cols, decay_lambda)

    # Layer 2: opp dec_avg (what opponent allows) — already partially done in opponent_history
    # We need dec_avg of the opponent's "allowed" stats
    opp_cols = [f"{col}_opp_mean" for col in stat_cols if f"{col}_opp_mean" in df.columns]
    # These are already per-fight values, need dec_avg per fighter
    print(f"  Computing decayed averages for {len(opp_cols)} opponent features...")
    df = compute_decayed_averages(df, opp_cols, decay_lambda)

    # Layer 3: fighter's own dec_avg (already computed in Step 5)
    # These are the *_dec_avg columns from Step 5

    # ── Build final diff dataset ──────────────────────────────────
    # One row per fight, fighter1 - fighter2 (fighter1 = red corner)

    # Get all feature columns to diff
    feature_cols_to_diff = []

    # Static dec_avg features (age, ufc_age, days_since_last_fight)
    for col in static_da_cols:
        da_col = f"{col}_dec_avg"
        if da_col in df.columns:
            feature_cols_to_diff.append(da_col)

    # Layer 1: adjperf dec_avg
    for col in adjperf_cols:
        da_col = f"{col}_dec_avg"
        if da_col in df.columns:
            feature_cols_to_diff.append(da_col)

    # Layer 2: opp dec_avg
    for col in opp_cols:
        da_col = f"{col}_dec_avg"
        if da_col in df.columns:
            feature_cols_to_diff.append(da_col)

    # Layer 3: fighter's own dec_avg
    for col in stat_cols:
        da_col = f"{col}_dec_avg"
        if da_col in df.columns and da_col not in feature_cols_to_diff:
            feature_cols_to_diff.append(da_col)

    # Raw static features to diff (in addition to their dec_avg versions)
    static_diff = ["age", "ufc_age", "reach_ratio", "WEIGHT", "days_since_last_fight"]
    for col in static_diff:
        if col in df.columns and col not in feature_cols_to_diff:
            feature_cols_to_diff.append(col)

    print(f"  Total features to diff: {len(feature_cols_to_diff)}")

    # Split into fighter1 and fighter2 rows (deduplicate to handle tournament bouts)
    f1 = df[df["is_fighter1"] == 1].drop_duplicates(subset=["jevent", "jbout"]).set_index(["jevent", "jbout"])
    f2 = df[df["is_fighter1"] == 0].drop_duplicates(subset=["jevent", "jbout"]).set_index(["jevent", "jbout"])

    # Only keep bouts where both fighters exist
    common_idx = f1.index.intersection(f2.index)
    f1 = f1.loc[common_idx]
    f2 = f2.loc[common_idx]

    # Build diff dataframe
    result = pd.DataFrame(index=common_idx)
    result["DATE"] = f1["DATE"].values
    result["jfighter"] = f1["jfighter"].values
    result["opp_jfighter"] = f2["jfighter"].values
    result["win"] = f1["win"].values  # fighter1 (red corner) win
    result["weightindex"] = f1["weightindex"].values

    # Diff all feature columns
    for col in feature_cols_to_diff:
        if col in f1.columns and col in f2.columns:
            col_name = f"{col}_diff" if not col.endswith("_diff") else col
            result[col_name] = f1[col].values - f2[col].values

    # Age ratio (f1_age / f2_age, not a diff)
    f1_age = f1["age"].clip(lower=18).values
    f2_age = f2["age"].clip(lower=18).values
    result["age_ratio_diff"] = (f1_age / f2_age) - 1.0  # centered at 0

    # Reach ratio diff — use the per-row prior median computed earlier in this
    # function for NaN fill, NOT a global median. Previously used .median()
    # which leaks future fighter REACH values. LEAKAGE_REFERENCE.md §1/§3.
    # The per-row median is in df via reach_ratio computation; here we use it
    # if present, otherwise fall back to the constant.
    _CONST_REACH_MEDIAN = 70.0
    f1_reach = f1["REACH"].fillna(_CONST_REACH_MEDIAN).values
    f2_reach = f2["REACH"].fillna(_CONST_REACH_MEDIAN).values
    result["reach_ratio_diff"] = (f1_reach / np.clip(f2_reach, 1, None)) - 1.0

    # Non-diffed features
    if "days_since_last_fight" in f1.columns:
        result["days_since_last_fight_f1"] = f1["days_since_last_fight"].values
    result["weightclass_encoded"] = f1["weightindex"].values
    result["scheduled_rounds"] = f1["time_format"].apply(
        lambda x: 5.0 if "5" in str(x) else 3.0
    ).values if "time_format" in f1.columns else 3.0

    result = result.reset_index()
    result["DATE"] = pd.to_datetime(result["DATE"])
    result = result.sort_values("DATE").reset_index(drop=True)

    print(f"\n  Final dataset (all dates): {len(result):,} fights × {result.shape[1]} columns")
    print(f"  fighter1 win rate: {result['win'].mean():.3f}")
    print(f"  Date range: {result['DATE'].min().date()} → {result['DATE'].max().date()}")

    if not return_individuals:
        return result

    # Extract per-fighter latest individual stats (from pre-diff data)
    print("  Extracting per-fighter latest stats...")
    # Combine f1 and f2 back into a single df with all fighters
    all_fighters = pd.concat([f1.reset_index(), f2.reset_index()], ignore_index=True)
    all_fighters["DATE"] = pd.to_datetime(all_fighters["DATE"])
    latest_idx = all_fighters.groupby("jfighter")["DATE"].idxmax()
    latest = all_fighters.loc[latest_idx]

    fighter_stats = {}
    for _, row in latest.iterrows():
        stats = {"DATE": row["DATE"]}
        for col in feature_cols_to_diff:
            if col in row.index:
                stats[col] = float(row[col]) if pd.notna(row[col]) else 0.0
        if "REACH" in row.index:
            stats["REACH"] = float(row["REACH"]) if pd.notna(row["REACH"]) else 0.0
        if "weightindex" in row.index:
            stats["weightindex"] = int(row["weightindex"]) if pd.notna(row["weightindex"]) else 0
        if "days_since_last_fight" in row.index:
            stats["days_since_last_fight"] = float(row["days_since_last_fight"]) if pd.notna(row["days_since_last_fight"]) else 0.0
        fighter_stats[row["jfighter"]] = stats

    print(f"  Fighter stats: {len(fighter_stats)} fighters, {len(feature_cols_to_diff)} features each")
    return result, fighter_stats, feature_cols_to_diff


def filter_to_era(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Filter final dataset to training era (after features computed on full history)."""
    before = len(df)
    df = df[df["DATE"] >= pd.to_datetime(start_date)].copy().reset_index(drop=True)
    print(f"  Filtered to >={start_date}: {len(df):,} / {before:,} fights")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def build_features(config_name: str = "v7") -> pd.DataFrame:
    """Build complete feature set from scratch.

    Args:
        config_name: "v7" or "jan26"

    Returns:
        DataFrame with one row per fight, all features as diffs.
    """
    config = V7_CONFIG if config_name == "v7" else JAN26_CONFIG
    decay_lambda = config["decay_lambda"]

    print(f"{'='*60}")
    print(f"  MMA-AI PIPELINE — config: {config_name}")
    print(f"  decay λ={decay_lambda}, start={config['start_date']}")
    print(f"{'='*60}")

    # Step 1: Load base data (ALL history — date filter only at final output)
    print("\nStep 1: Loading base data...")
    df = load_base_data()

    # Step 2: Beta-Binomial smoothing
    print("\nStep 2: Beta-Binomial smoothing...")
    df = beta_binomial_smooth(df)

    # Step 3: Poisson-Gamma smoothing
    print("\nStep 3: Poisson-Gamma smoothing...")
    df = poisson_gamma_smooth(df)

    # Step 4: Derived features
    print("\nStep 4: Derived features...")
    df = compute_derived_features(df)

    # Define stat columns for AdjPerf — all smoothed/derived features
    stat_cols = [c for c in df.columns if
                 c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
                 c.endswith("_ratio") or c.endswith("_per_ctrl") or
                 c in ["ko_smooth", "win_smooth", "decision_smooth",
                        "sub_land_smooth", "sub_land_rate", "ctrl_pm",
                        "ko_per_sig_str_land", "td_per_sig_str_att",
                        "ground_per_ctrl", "dist_per_sig_str_land",
                        "head_per_sig_str_land", "rev_per_ctrlopp",
                        "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
                        "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"]]
    # Remove any temp/duplicate columns
    stat_cols = sorted(set(c for c in stat_cols if c in df.columns and
                           not c.startswith("opp_") and not c.endswith("_raw")))

    # Step 5: Decayed averages of base features
    print("\nStep 5: Decayed averages...")
    df = compute_decayed_averages(df, stat_cols, decay_lambda)

    # Step 6: Opponent history
    print("\nStep 6: Opponent history + WC priors...")
    df = compute_opponent_history(df, stat_cols, decay_lambda)
    priors = compute_wc_priors(df, stat_cols)

    # Step 7: AdjPerf
    print("\nStep 7: AdjPerf z-scores...")
    df = compute_adjperf(df, stat_cols, priors)

    # Steps 8-9: Feature assembly + diff
    print("\nSteps 8-9: Feature assembly + diff construction...")
    result = assemble_features(df, stat_cols, decay_lambda)

    # Filter to training era (features computed on full history, now trim)
    print(f"\nStep 10: Date filter...")
    result = filter_to_era(result, config["start_date"])

    return result


def get_fighter_stats_lookup(config_name: str = "jan26") -> dict:
    """Build MMA-AI features and return per-fighter latest individual stats.

    Returns dict with keys:
        "df": the full diffed DataFrame (for training)
        "fighter_stats": dict mapping jfighter -> dict of individual stats
        "feature_cols": list of feature column names used for diffs
    """
    config = V7_CONFIG if config_name == "v7" else JAN26_CONFIG
    decay_lambda = config["decay_lambda"]

    print(f"{'='*60}")
    print(f"  MMA-AI PIPELINE — config: {config_name}")
    print(f"  decay λ={decay_lambda}, start={config['start_date']}")
    print(f"{'='*60}")

    df = load_base_data()
    df = beta_binomial_smooth(df)
    df = poisson_gamma_smooth(df)
    df = compute_derived_features(df)

    stat_cols = [c for c in df.columns if
                 c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
                 c.endswith("_ratio") or c.endswith("_per_ctrl") or
                 c in ["ko_smooth", "win_smooth", "decision_smooth",
                        "sub_land_smooth", "sub_land_rate", "ctrl_pm",
                        "ko_per_sig_str_land", "td_per_sig_str_att",
                        "ground_per_ctrl", "dist_per_sig_str_land",
                        "head_per_sig_str_land", "rev_per_ctrlopp",
                        "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
                        "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"]]
    stat_cols = sorted(set(c for c in stat_cols if c in df.columns and
                           not c.startswith("opp_") and not c.endswith("_raw")))

    df = compute_decayed_averages(df, stat_cols, decay_lambda)
    df = compute_opponent_history(df, stat_cols, decay_lambda)
    priors = compute_wc_priors(df, stat_cols)
    df = compute_adjperf(df, stat_cols, priors)

    # Use assemble_features with return_individuals=True
    result, fighter_stats, feature_cols = assemble_features(
        df, stat_cols, decay_lambda, return_individuals=True
    )
    result = filter_to_era(result, config["start_date"])

    return {
        "df": result,
        "fighter_stats": fighter_stats,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = build_features("v7")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    feat_cols = [c for c in df.columns if c.endswith("_diff")]
    print(f"\n{len(feat_cols)} diff features")

    # LEAKAGE CHECK: no feature should correlate > 0.5 with win
    print(f"\nTop 15 features by |correlation| with win:")
    corrs = df[feat_cols + ["win"]].corr()["win"].drop("win").abs().sort_values(ascending=False)
    for feat, corr in corrs.head(15).items():
        flag = " *** LEAKAGE?" if corr > 0.5 else ""
        print(f"  {feat:<60} {corr:.4f}{flag}")

    # Check against his feature list
    his_features = [
        "age_dec_avg_diff", "sig_str_land_ratio_dec_adjperf_dec_avg_diff",
        "reach_ratio_dec_avg_diff", "sub_att_dec_avg_diff", "td_acc_dec_avg_diff",
        "head_land_dec_avg_diff", "age_ratio_diff", "head_def_dec_avg_diff",
        "ufc_age_diff", "head_land_ratio_adjperf_dec_avg_diff",
    ]
    print(f"\nFeature parity check (his top 10):")
    for hf in his_features:
        # Try to find a matching feature
        matches = [f for f in feat_cols if hf.replace("_diff","") in f or f.replace("_diff","") in hf]
        status = "FOUND" if matches else "MISSING"
        print(f"  {hf:<55} {status}  {matches[:2] if matches else ''}")
