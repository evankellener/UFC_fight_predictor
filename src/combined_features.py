"""
Combined Features — layer Elo + market edge onto MMA-AI pipeline output.

Takes the diff-format output from build_features() and adds:
1. Elo features (precomp_elo_diff, elo_win_prob, elo_momentum_diff, etc.)
2. Market edge features (weight_cut, home_advantage, card_position, etc.)

Handles the ordering mismatch: Elo uses alphabetical f1/f2,
MMA-AI pipeline uses red corner = fighter1.
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "sqlite_db" / "sqlite_scrapper.db")

# Elo params — import from predict_event.py to stay in sync
from predict_event import (
    ELO_K, ELO_KO_MULT, ELO_SUB_MULT, ELO_DECAY,
    ELO_DECAY_MAX, ELO_DECAY_MIDPOINT, ELO_DECAY_STEEPNESS,
    ELO_WC_PENALTY, ELO_STREAK_BONUS, ELO_STREAK_CAP,
    ELO_R1_MULT, ELO_LOGISTIC_SCALE,
)
ELO_PARAMS = dict(
    K=ELO_K,
    ko_mult=ELO_KO_MULT,
    sub_mult=ELO_SUB_MULT,
    decay_lambda=ELO_DECAY,
    decay_max=ELO_DECAY_MAX,
    decay_midpoint=ELO_DECAY_MIDPOINT,
    decay_steepness=ELO_DECAY_STEEPNESS,
    wc_change_penalty=ELO_WC_PENALTY,
    streak_bonus=ELO_STREAK_BONUS,
    streak_cap=ELO_STREAK_CAP,
    r1_finish_mult=ELO_R1_MULT,
    logistic_scale=ELO_LOGISTIC_SCALE,
    opp_quality_k=True,
    sliding_k=True,
    upset_momentum=True,
    champ_mult=1.2,
)

# Elo diff features to merge (all are f1-f2 diffs)
ELO_DIFF_COLS = [
    "precomp_elo_diff",
    "elo_win_prob",
    "elo_momentum_diff",
    "peak_elo_diff",
    "avg_opp_elo_diff",
    "elo_consist_diff",
]

# Elo features that are NOT diffs (same value regardless of ordering)
ELO_NODIFF_COLS = [
    "elo_predictability",
]


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Elo features to MMA-AI pipeline diff output.

    The pipeline output has jfighter (red corner) and opp_jfighter (blue corner).
    Elo computes f1/f2 alphabetically. We merge on (jevent, jbout) and flip
    signs when jfighter == elo's f2 (i.e., red corner is alphabetically second).
    """
    from elo_feature import load_bouts, compute_elo

    print("\n  Computing Elo ratings...")
    bouts = load_bouts()
    elo_df, ratings, last_date, extra = compute_elo(bouts, **ELO_PARAMS)

    # Keep only the columns we need for merging
    elo_merge = elo_df[["jevent", "jbout", "f1", "f2"] + ELO_DIFF_COLS + ELO_NODIFF_COLS].copy()

    # Merge onto pipeline output via (jevent, jbout)
    df = df.merge(elo_merge, on=["jevent", "jbout"], how="left")

    # Determine if pipeline's jfighter matches elo's f1 or f2
    # If jfighter == f1: diffs are already correct (f1-f2 = red-blue)
    # If jfighter == f2: need to negate diffs (f1-f2 is opposite of red-blue)
    is_flipped = (df["jfighter"] == df["f2"])

    for col in ELO_DIFF_COLS:
        if col == "elo_win_prob":
            # elo_win_prob = P(f1 wins). If red=f2, we want P(f2 wins) = 1 - P(f1 wins)
            df.loc[is_flipped, col] = 1.0 - df.loc[is_flipped, col]
        else:
            df.loc[is_flipped, col] = -df.loc[is_flipped, col]

    # Drop temp merge columns
    df.drop(columns=["f1", "f2"], inplace=True, errors="ignore")

    # Fill NaN Elo features with neutral values
    for col in ELO_DIFF_COLS:
        if col == "elo_win_prob":
            df[col] = df[col].fillna(0.5)
        else:
            df[col] = df[col].fillna(0.0)
    for col in ELO_NODIFF_COLS:
        df[col] = df[col].fillna(0.5)

    n_matched = df["precomp_elo_diff"].notna().sum()
    print(f"  Elo features added: {len(ELO_DIFF_COLS) + len(ELO_NODIFF_COLS)} features, "
          f"{n_matched}/{len(df)} fights matched")

    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market edge features to MMA-AI pipeline diff output.

    The pipeline output is already in diff format (one row per fight).
    Market edge features need per-fighter data, so we compute them from scratch
    and produce diffs directly.
    """
    conn = sqlite3.connect(DB_PATH)

    # ── Weight Cut Features ──────────────────────────────────────
    try:
        wi = pd.read_sql("SELECT * FROM ufc_weigh_in_results", conn)
        wi["jfighter"] = wi["fighter"].str.replace(" ", "", regex=False)
        wi["DATE"] = pd.to_datetime(wi["date"])
        wi["weight_cut_pct"] = (
            (wi["weight_limit"] - wi["weighin_weight"]) / wi["weight_limit"]
        ).clip(-0.1, 0.15)
        wi["missed_weight"] = wi["missed_weight"].fillna(0).astype(int)

        # Current fight weigh-in
        wi_dedup = wi[["jfighter", "DATE", "weight_cut_pct", "missed_weight"]].drop_duplicates(
            subset=["jfighter", "DATE"], keep="last"
        )
        # Fighter 1 (red corner = jfighter)
        df = df.merge(
            wi_dedup.rename(columns={"weight_cut_pct": "_f1_wcut", "missed_weight": "_f1_missed"}),
            on=["jfighter", "DATE"], how="left"
        )
        # Fighter 2 (blue corner = opp_jfighter)
        df = df.merge(
            wi_dedup.rename(columns={
                "jfighter": "opp_jfighter",
                "weight_cut_pct": "_f2_wcut",
                "missed_weight": "_f2_missed",
            }),
            on=["opp_jfighter", "DATE"], how="left"
        )

        df["weight_cut_pct_diff"] = (df["_f1_wcut"] - df["_f2_wcut"]).fillna(0.0)
        df["missed_weight_diff"] = (
            df["_f1_missed"].fillna(0) - df["_f2_missed"].fillna(0)
        ).astype(float)

        # Historical missed weight count (career prior to this fight)
        wi_sorted = wi.sort_values("DATE")
        missed_history = {}
        for _, row in wi_sorted.iterrows():
            jf = row["jfighter"]
            if jf not in missed_history:
                missed_history[jf] = []
            missed_history[jf].append((row["DATE"], int(row["missed_weight"])))

        def _career_missed_before(jfighter, fight_date):
            history = missed_history.get(jfighter, [])
            return sum(m for d, m in history if d < fight_date)

        df["_f1_miss_hist"] = df.apply(lambda r: _career_missed_before(r["jfighter"], r["DATE"]), axis=1)
        df["_f2_miss_hist"] = df.apply(lambda r: _career_missed_before(r["opp_jfighter"], r["DATE"]), axis=1)
        df["missed_weight_history_diff"] = (df["_f1_miss_hist"] - df["_f2_miss_hist"]).astype(float)

        print(f"  Weight cut features added (weigh-in records: {len(wi)})")
    except Exception as e:
        print(f"  Weight cut features skipped: {e}")
        df["weight_cut_pct_diff"] = 0.0
        df["missed_weight_diff"] = 0.0
        df["missed_weight_history_diff"] = 0.0

    # ── Home Advantage ───────────────────────────────────────────
    try:
        events = pd.read_sql("SELECT jevent, DATE, LOCATION FROM ufc_event_details", conn)
        events["DATE"] = pd.to_datetime(events["DATE"])
        events["event_country"] = events["LOCATION"].str.split(",").str[-1].str.strip().str.upper()

        nat = pd.read_sql("SELECT jfighter, country FROM ufc_fighter_nationality", conn)
        nat["country"] = nat["country"].str.upper()

        # Join event country via jevent
        event_country = events[["jevent", "event_country"]].drop_duplicates(subset=["jevent"], keep="last")
        df = df.merge(event_country, on="jevent", how="left")

        # Fighter nationalities
        df = df.merge(nat.rename(columns={"country": "_f1_country"}), on="jfighter", how="left")
        df = df.merge(
            nat.rename(columns={"jfighter": "opp_jfighter", "country": "_f2_country"}),
            on="opp_jfighter", how="left"
        )

        f1_home = df["_f1_country"] == df["event_country"]
        f2_home = df["_f2_country"] == df["event_country"]
        df["home_advantage_diff"] = (f1_home.astype(float) - f2_home.astype(float)).fillna(0.0)

        print(f"  Home advantage added (nationalities: {len(nat)})")
    except Exception as e:
        print(f"  Home advantage skipped: {e}")
        df["home_advantage_diff"] = 0.0

    # ── Card Position ────────────────────────────────────────────
    try:
        cp = pd.read_sql("SELECT jevent, jbout, position, total_fights FROM ufc_card_position", conn)

        if len(cp) > 0:
            cp["_card_pos_norm"] = (cp["position"] - 1) / (cp["total_fights"] - 1).clip(lower=1)
            cp["_is_main"] = (cp["position"] == 1).astype(float)

            df = df.merge(
                cp[["jevent", "jbout", "_card_pos_norm", "_is_main"]].drop_duplicates(
                    subset=["jevent", "jbout"], keep="last"
                ),
                on=["jevent", "jbout"], how="left"
            )

            # Career average card position per fighter (no leakage)
            fighter_cp = pd.read_sql("""
                SELECT w.jfighter, e.DATE, cp.position, cp.total_fights
                FROM ufc_winlossko w
                JOIN ufc_card_position cp ON cp.jevent = w.jevent AND cp.jbout = w.jbout
                JOIN ufc_event_details e ON e.jevent = w.jevent
                ORDER BY e.DATE
            """, conn)

            if len(fighter_cp) > 0:
                fighter_cp["DATE"] = pd.to_datetime(fighter_cp["DATE"])
                fighter_cp["pos_norm"] = (
                    (fighter_cp["position"] - 1) / (fighter_cp["total_fights"] - 1).clip(lower=1)
                )

                all_fighter_pos = {}
                for _, row in fighter_cp.iterrows():
                    jf = row["jfighter"]
                    if jf not in all_fighter_pos:
                        all_fighter_pos[jf] = []
                    all_fighter_pos[jf].append((row["DATE"], row["pos_norm"]))

                def _avg_pos_before(jfighter, fight_date):
                    history = all_fighter_pos.get(jfighter, [])
                    prior = [p for d, p in history if d < fight_date]
                    return float(np.mean(prior)) if prior else 0.5

                df["_f1_avg_pos"] = df.apply(lambda r: _avg_pos_before(r["jfighter"], r["DATE"]), axis=1)
                df["_f2_avg_pos"] = df.apply(lambda r: _avg_pos_before(r["opp_jfighter"], r["DATE"]), axis=1)
                df["card_position_norm_diff"] = (df["_f1_avg_pos"] - df["_f2_avg_pos"]).astype(float)
            else:
                df["card_position_norm_diff"] = 0.0

            df["is_main_event"] = df.get("_is_main", pd.Series(0.0, index=df.index)).fillna(0.0)
            print(f"  Card position added ({len(cp)} records)")
        else:
            df["card_position_norm_diff"] = 0.0
            df["is_main_event"] = 0.0
    except Exception as e:
        print(f"  Card position skipped: {e}")
        df["card_position_norm_diff"] = 0.0
        df["is_main_event"] = 0.0

    conn.close()

    # ── Cleanup temp columns ─────────────────────────────────────
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=drop_cols + ["event_country"], inplace=True, errors="ignore")

    return df


def add_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all extra features (Elo + market edge) to MMA-AI pipeline output."""
    print("\n" + "=" * 55)
    print("  ADDING COMBINED FEATURES (Elo + Market Edge)")
    print("=" * 55)

    df = add_elo_features(df)
    df = add_market_features(df)

    # List all new feature columns
    elo_feats = ELO_DIFF_COLS + ELO_NODIFF_COLS
    market_feats = [
        "weight_cut_pct_diff", "missed_weight_diff", "missed_weight_history_diff",
        "home_advantage_diff", "card_position_norm_diff", "is_main_event",
    ]
    all_new = [c for c in elo_feats + market_feats if c in df.columns]
    print(f"\n  Total new features: {len(all_new)}")
    for feat in all_new:
        nz = (df[feat] != 0).sum() if feat in df.columns else 0
        print(f"    {feat:<35} nonzero={nz:>5}/{len(df)}")

    return df
