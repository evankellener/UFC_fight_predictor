"""
ELO feature builder for UFC fight predictor.

Phase 1: Base ELO (K=32) — measure AUC of elo_diff predicting win
Phase 2: Grid search over K + method multipliers (ko_mult, sub_mult)
Phase 3: Add time decay — grid search over decay_lambda

Usage:
    python src/elo_feature.py
"""

import sqlite3, warnings
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

_app_db  = Path(__file__).parent.parent / "data/sqlite_db/app.db"
_full_db = Path(__file__).parent.parent / "data/sqlite_db/sqlite_scrapper.db"
DB_PATH  = _app_db if _app_db.exists() else _full_db
ERA_CUTOFF = "2014-01-01"
BASE_ELO   = 1500


# ── Load EXPANDED bout-level data (UFC + non-UFC) ──────────────────────────
def load_bouts_expanded() -> pd.DataFrame:
    """
    Load bouts from BOTH ufc_winlossko (UFC fights with full metadata)
    AND ufc_complete_fight_history (non-UFC fights: Pride, Strikeforce, WEC, DWCS, etc.)

    Non-UFC fights are added for Elo warm-up only — they give debuting UFC fighters
    a real Elo prior instead of defaulting to 1500.

    Returns same format as load_bouts().
    """
    conn = sqlite3.connect(DB_PATH)

    # 1. Load UFC bouts (same as load_bouts)
    q_ufc = """
    SELECT w.DATE, w.jevent, w.jbout, w.jfighter, w.win, w.ko, w.subw,
           f.weightindex,
           fr.ROUND AS finish_round
    FROM ufc_winlossko w
    LEFT JOIN final_features_fast f
      ON f.jevent = w.jevent AND f.jbout = w.jbout AND f.jfighter = w.jfighter
    LEFT JOIN ufc_fight_results fr
      ON fr.jevent = w.jevent AND fr.jbout = w.jbout
    ORDER BY w.DATE, w.jevent, w.jbout, w.jfighter
    """
    ufc_df = pd.read_sql_query(q_ufc, conn)
    ufc_df["DATE"] = pd.to_datetime(ufc_df["DATE"])

    # Build UFC bouts (same logic as load_bouts)
    ufc_bouts = []
    ufc_jbouts = set()
    for (jevent, jbout), grp in ufc_df.groupby(["jevent", "jbout"]):
        if len(grp) != 2:
            continue
        r1, r2 = grp.iloc[0], grp.iloc[1]
        if r1["jfighter"] > r2["jfighter"]:
            r1, r2 = r2, r1
        if r1["win"] == 1:
            winner, loser = r1["jfighter"], r2["jfighter"]
            method = "ko" if r1["ko"] == 1 else ("sub" if r1["subw"] == 1 else "dec")
        else:
            winner, loser = r2["jfighter"], r1["jfighter"]
            method = "ko" if r2["ko"] == 1 else ("sub" if r2["subw"] == 1 else "dec")
        wc = int(r1["weightindex"]) if pd.notna(r1["weightindex"]) else 0
        rnd = int(r1["finish_round"]) if pd.notna(r1["finish_round"]) else 0
        ufc_bouts.append({
            "DATE": r1["DATE"], "jevent": jevent, "jbout": jbout,
            "f1": r1["jfighter"], "f2": r2["jfighter"],
            "winner": winner, "loser": loser, "method": method,
            "f1_win": int(r1["win"]), "weightindex": wc, "finish_round": rnd,
            "source": "ufc",
        })
        ufc_jbouts.add(jbout)

    # 2. Load non-UFC fights from complete_fight_history
    # Build name mapping: full name -> jfighter (CamelCase)
    name_map_df = pd.read_sql("SELECT FIGHTER, jfighter FROM ufc_fighter_tott", conn)
    name_to_j = dict(zip(name_map_df["FIGHTER"].str.strip(), name_map_df["jfighter"]))

    # Also build reverse: for fighters not in tott, create jfighter from name
    def make_jfighter(name):
        if not name or pd.isna(name):
            return None
        mapped = name_to_j.get(name.strip())
        if mapped:
            return mapped
        # Fallback: remove spaces and capitalize
        return "".join(w.capitalize() for w in name.strip().split())

    cfh = pd.read_sql("""
        SELECT FIGHTER, OPPONENT, EVENT, DATE, RESULT, METHOD
        FROM ufc_complete_fight_history
        WHERE EVENT NOT LIKE '%UFC%'
          AND EVENT NOT LIKE '%Ultimate Fighter%'
    """, conn)
    conn.close()

    cfh["DATE"] = pd.to_datetime(cfh["DATE"], errors="coerce")
    cfh = cfh.dropna(subset=["DATE", "FIGHTER", "OPPONENT"])

    # Map names
    cfh["j_fighter"] = cfh["FIGHTER"].apply(make_jfighter)
    cfh["j_opponent"] = cfh["OPPONENT"].apply(make_jfighter)
    cfh = cfh.dropna(subset=["j_fighter", "j_opponent"])

    # Parse result and method
    non_ufc_bouts = []
    seen = set()

    for _, r in cfh.iterrows():
        f1_j = r["j_fighter"]
        f2_j = r["j_opponent"]
        if f1_j == f2_j:
            continue

        # Deduplicate (each fight appears twice in complete_fight_history)
        bout_key = tuple(sorted([f1_j, f2_j])) + (r["DATE"],)
        if bout_key in seen:
            continue
        seen.add(bout_key)

        # Determine winner
        result_str = str(r["RESULT"]).strip().lower()
        method_str = str(r["METHOD"]).strip().lower() if pd.notna(r["METHOD"]) else ""

        if result_str == "win":
            winner, loser = f1_j, f2_j
        elif result_str == "loss":
            winner, loser = f2_j, f1_j
        else:
            continue  # skip draws, NC, etc.

        # Parse method
        if "ko" in method_str or "tko" in method_str:
            method = "ko"
        elif "sub" in method_str:
            method = "sub"
        else:
            method = "dec"

        # Alphabetical ordering for f1/f2
        if f1_j > f2_j:
            f1_j, f2_j = f2_j, f1_j
        f1_win = 1 if winner == f1_j else 0

        jbout = f"{f1_j}vs{f2_j}"

        non_ufc_bouts.append({
            "DATE": r["DATE"],
            "jevent": r["EVENT"].replace(" ", ""),
            "jbout": jbout,
            "f1": f1_j, "f2": f2_j,
            "winner": winner, "loser": loser, "method": method,
            "f1_win": f1_win, "weightindex": 0, "finish_round": 0,
            "source": "non_ufc",
        })

    # Combine and sort
    all_bouts = pd.DataFrame(ufc_bouts + non_ufc_bouts)
    all_bouts = all_bouts.sort_values("DATE").reset_index(drop=True)

    n_ufc = len(ufc_bouts)
    n_non = len(non_ufc_bouts)
    print(f"Loaded {len(all_bouts):,} bouts  "
          f"(UFC={n_ufc:,}, non-UFC={n_non:,}, "
          f"{all_bouts['DATE'].min().date()} → {all_bouts['DATE'].max().date()})")
    return all_bouts


# ── Load bout-level data from DB (UFC only) ────────────────────────────────
def load_bouts() -> pd.DataFrame:
    """
    Load all bouts from 1994 (full history for accurate ELO warm-up).
    Includes weightindex and finish_round so all modifiers can be applied.
    Returns one row per bout: DATE, jevent, jbout, f1, f2, winner, loser,
    method, f1_win, weightindex, finish_round.
    """
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT w.DATE, w.jevent, w.jbout, w.jfighter, w.win, w.ko, w.subw,
           f.weightindex,
           fr.ROUND AS finish_round
    FROM ufc_winlossko w
    LEFT JOIN final_features_fast f
      ON f.jevent = w.jevent AND f.jbout = w.jbout AND f.jfighter = w.jfighter
    LEFT JOIN ufc_fight_results fr
      ON fr.jevent = w.jevent AND fr.jbout = w.jbout
    ORDER BY w.DATE, w.jevent, w.jbout, w.jfighter
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    df["DATE"] = pd.to_datetime(df["DATE"])

    bouts = []
    for (jevent, jbout), grp in df.groupby(["jevent", "jbout"]):
        if len(grp) != 2:
            continue
        r1, r2 = grp.iloc[0], grp.iloc[1]
        if r1["jfighter"] > r2["jfighter"]:
            r1, r2 = r2, r1
        if r1["win"] == 1:
            winner, loser = r1["jfighter"], r2["jfighter"]
            method = "ko" if r1["ko"] == 1 else ("sub" if r1["subw"] == 1 else "dec")
        else:
            winner, loser = r2["jfighter"], r1["jfighter"]
            method = "ko" if r2["ko"] == 1 else ("sub" if r2["subw"] == 1 else "dec")
        wc  = int(r1["weightindex"])  if pd.notna(r1["weightindex"])  else 0
        rnd = int(r1["finish_round"]) if pd.notna(r1["finish_round"]) else 0
        bouts.append({
            "DATE": r1["DATE"], "jevent": jevent, "jbout": jbout,
            "f1": r1["jfighter"], "f2": r2["jfighter"],
            "winner": winner, "loser": loser, "method": method,
            "f1_win": int(r1["win"]), "weightindex": wc, "finish_round": rnd,
        })

    bouts_df = pd.DataFrame(bouts).sort_values("DATE").reset_index(drop=True)
    print(f"Loaded {len(bouts_df):,} bouts  "
          f"({bouts_df['DATE'].min().date()} → {bouts_df['DATE'].max().date()})")
    return bouts_df


# ── ELO computation engine ──────────────────────────────────────────────────
def compute_elo(bouts: pd.DataFrame,
                K: float = 32,
                ko_mult: float = 1.0,
                sub_mult: float = 1.0,
                decay_lambda: float = 0.0,
                wc_change_penalty: float = 1.0,
                streak_bonus: float = 0.0,
                streak_cap: int = 5,
                r1_finish_mult: float = 1.0,
                logistic_scale: float = 400.0,
                decay_max: float = None,
                decay_midpoint: float = None,
                decay_steepness: float = None) -> tuple:
    """
    Compute precomp ELO for each bout (rating BEFORE the fight).

    Parameters
    ----------
    K                 : base K-factor
    ko_mult           : K multiplier for KO/TKO wins
    sub_mult          : K multiplier for submission wins
    decay_lambda      : per-year inactivity decay toward BASE_ELO (legacy exponential)
    wc_change_penalty : when a fighter moves weight class, their new rating is
                        BASE + (avg_last3 - BASE) * penalty, where avg_last3 is
                        the average of their last 3 precomp ELO ratings.
    logistic_scale    : denominator in the logistic function (standard=400)
                        (1.0 = full carry of 3-fight avg, 0.0 = reset to BASE_ELO)
    streak_bonus      : extra K multiplier per consecutive win in current streak
    streak_cap        : max streak length counted for the bonus
    r1_finish_mult    : extra K multiplier for KO/sub finishes in round 1
    decay_max         : sigmoid decay — maximum fraction of Elo deviation lost (0-1)
    decay_midpoint    : sigmoid decay — days of inactivity where decay is steepest
    decay_steepness   : sigmoid decay — controls transition sharpness (higher = sharper)
    """
    from collections import deque

    # Determine which decay mode to use
    use_sigmoid_decay = (decay_max is not None and decay_midpoint is not None
                         and decay_steepness is not None)

    ratings    = {}; last_date  = {}; last_wc    = {}; streaks    = {}
    elo_hist   = {}  # last 3 precomp for wc-change blend
    elo_3ago   = {}  # deque(maxlen=4): position [0] = 3 fights ago
    peak_elo   = {}; opp_elos   = {}; elo_deltas = {}

    def get_rating(fighter, fight_date, wc):
        if fighter not in ratings:
            ratings[fighter]    = BASE_ELO; last_date[fighter]  = fight_date
            last_wc[fighter]    = wc;       streaks[fighter]    = 0
            elo_hist[fighter]   = [];       elo_3ago[fighter]   = deque(maxlen=4)
            peak_elo[fighter]   = BASE_ELO; opp_elos[fighter]   = []
            elo_deltas[fighter] = deque(maxlen=5)
            return BASE_ELO
        r = ratings[fighter]
        days_inactive = max((fight_date - last_date[fighter]).days, 0)
        if use_sigmoid_decay and days_inactive > 0:
            # Sigmoid decay: gentle early, steep in middle, flattens late
            # decay_factor ranges from ~1.0 (short break) to ~(1 - decay_max) (long break)
            sig = 1.0 / (1.0 + np.exp(-(days_inactive - decay_midpoint) / decay_steepness))
            decay_factor = 1.0 - decay_max * sig
            r = BASE_ELO + (r - BASE_ELO) * decay_factor
        elif decay_lambda > 0 and days_inactive > 0:
            # Legacy exponential decay
            years = days_inactive / 365.25
            r = BASE_ELO + (r - BASE_ELO) * np.exp(-decay_lambda * years)
        if wc != 0 and last_wc[fighter] != 0 and wc != last_wc[fighter]:
            avg3 = float(np.mean(elo_hist[fighter])) if elo_hist[fighter] else BASE_ELO
            r = BASE_ELO + (avg3 - BASE_ELO) * wc_change_penalty
        return r

    pre_f1, pre_f2 = [], []
    mom_f1, mom_f2, pk_f1, pk_f2, oq_f1, oq_f2, cons_f1, cons_f2 = ([] for _ in range(8))

    for _, bout in bouts.iterrows():
        f1, f2       = bout["f1"], bout["f2"]
        winner       = bout["winner"]
        method       = bout["method"]
        date         = bout["DATE"]
        wc           = int(bout.get("weightindex",  0) or 0)
        finish_round = int(bout.get("finish_round", 0) or 0)

        r1 = get_rating(f1, date, wc)
        r2 = get_rating(f2, date, wc)

        # Update histories before recording extended features
        elo_hist[f1] = (elo_hist.get(f1, []) + [r1])[-3:]
        elo_hist[f2] = (elo_hist.get(f2, []) + [r2])[-3:]
        elo_3ago[f1].append(r1); elo_3ago[f2].append(r2)
        opp_elos.setdefault(f1, []).append(r2)
        opp_elos.setdefault(f2, []).append(r1)

        # Extended precomp features
        h1, h2 = list(elo_3ago[f1]), list(elo_3ago[f2])
        mom_f1.append((r1 - h1[0]) if len(h1) >= 4 else np.nan)
        mom_f2.append((r2 - h2[0]) if len(h2) >= 4 else np.nan)
        pk_f1.append(peak_elo.get(f1, BASE_ELO)); pk_f2.append(peak_elo.get(f2, BASE_ELO))
        oq1 = opp_elos[f1][:-1]; oq2 = opp_elos[f2][:-1]
        oq_f1.append(float(np.mean(oq1)) if oq1 else np.nan)
        oq_f2.append(float(np.mean(oq2)) if oq2 else np.nan)
        d1 = list(elo_deltas.get(f1, deque())); d2 = list(elo_deltas.get(f2, deque()))
        cons_f1.append(float(np.std(d1)) if len(d1) >= 2 else np.nan)
        cons_f2.append(float(np.std(d2)) if len(d2) >= 2 else np.nan)

        # ELO update
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / logistic_scale))
        mult = ko_mult if method == "ko" else (sub_mult if method == "sub" else 1.0)
        if finish_round == 1 and method in ("ko", "sub"):
            mult *= r1_finish_mult
        s1_win = (f1 == winner)
        sk1, sk2 = streaks.get(f1, 0), streaks.get(f2, 0)
        k_win  = K * mult * (1 + streak_bonus * min(sk1 if s1_win else sk2, streak_cap))
        k_lose = K * mult * (1 + streak_bonus * min(sk2 if s1_win else sk1, streak_cap))
        s1 = 1.0 if s1_win else 0.0
        new_r1 = r1 + (k_win  if s1_win else k_lose) * (s1 - exp1)
        new_r2 = r2 + (k_lose if s1_win else k_win)  * ((1-s1) - (1-exp1))

        pre_f1.append(r1); pre_f2.append(r2)
        ratings[f1] = new_r1; ratings[f2] = new_r2
        last_date[f1] = date; last_date[f2] = date
        last_wc[f1]   = wc;   last_wc[f2]   = wc
        peak_elo[f1]  = max(peak_elo.get(f1, BASE_ELO), new_r1)
        peak_elo[f2]  = max(peak_elo.get(f2, BASE_ELO), new_r2)
        streaks[f1]   = (streaks.get(f1, 0) + 1) if s1_win  else 0
        streaks[f2]   = (streaks.get(f2, 0) + 1) if not s1_win else 0
        elo_deltas.setdefault(f1, deque(maxlen=5)).append(abs(new_r1 - r1))
        elo_deltas.setdefault(f2, deque(maxlen=5)).append(abs(new_r2 - r2))

    result = bouts.copy()
    pf1, pf2 = np.array(pre_f1), np.array(pre_f2)
    result["precomp_elo_f1"]    = pf1;  result["precomp_elo_f2"]    = pf2
    result["precomp_elo_diff"]  = pf1 - pf2
    result["elo_win_prob"]      = 1 / (1 + 10 ** (-(pf1 - pf2) / logistic_scale))
    result["elo_momentum_f1"]   = mom_f1; result["elo_momentum_f2"]  = mom_f2
    result["elo_momentum_diff"] = np.array(mom_f1, dtype=float) - np.array(mom_f2, dtype=float)
    result["peak_elo_f1"]       = pk_f1;  result["peak_elo_f2"]      = pk_f2
    result["peak_elo_diff"]     = np.array(pk_f1) - np.array(pk_f2)
    result["avg_opp_elo_f1"]    = oq_f1;  result["avg_opp_elo_f2"]   = oq_f2
    result["avg_opp_elo_diff"]  = np.array(oq_f1, dtype=float) - np.array(oq_f2, dtype=float)
    result["elo_consist_f1"]    = cons_f1; result["elo_consist_f2"]   = cons_f2
    result["elo_consist_diff"]  = np.array(cons_f1, dtype=float) - np.array(cons_f2, dtype=float)

    # ── Elo predictability: per-weight-class EMA of correct predictions ──
    # 1 if higher-Elo fighter won, 0 otherwise; separate EMA per weight class
    elo_diff = pf1 - pf2
    elo_correct = np.where(
        elo_diff == 0, 0.5,
        np.where(
            (elo_diff > 0) == result["f1_win"].values.astype(bool),
            1.0, 0.0
        )
    )
    result["_elo_correct"] = elo_correct

    # Per-weight-class, event-level EMA: aggregate hit rate per (wc, event date),
    # then EMA across events within each wc, shifted by 1 event (no leakage)
    result_sorted = result.sort_values("DATE")

    # One row per (weightindex, DATE): mean hit rate for that wc on that night
    wc_event_hit = (
        result_sorted.groupby(["weightindex", "DATE"])["_elo_correct"]
        .mean()
        .reset_index()
        .rename(columns={"_elo_correct": "_wc_event_hit_rate"})
        .sort_values("DATE")
    )

    # EMA within each weight class, shifted by 1 event
    wc_event_hit["elo_predictability"] = (
        wc_event_hit.groupby("weightindex")["_wc_event_hit_rate"]
        .transform(lambda s: s.ewm(span=15, min_periods=5).mean().shift(1))
    )

    # Merge back onto result by (weightindex, DATE)
    result = result.merge(
        wc_event_hit[["weightindex", "DATE", "elo_predictability"]],
        on=["weightindex", "DATE"], how="left"
    )
    result["elo_predictability"] = result["elo_predictability"].fillna(0.5)

    # Latest per-wc predictability for inference (unshifted = value for next event)
    latest_by_wc = {}
    for wc, grp in wc_event_hit.sort_values("DATE").groupby("weightindex"):
        ema = grp["_wc_event_hit_rate"].ewm(span=15, min_periods=5).mean()
        latest_by_wc[int(wc)] = float(ema.iloc[-1]) if len(ema) >= 5 else 0.5
    result.drop(columns=["_elo_correct"], inplace=True)

    extra = {"elo_3ago": {f: list(d) for f, d in elo_3ago.items()},
             "peak_elo": peak_elo, "opp_elos": opp_elos,
             "latest_elo_predictability_by_wc": latest_by_wc}
    return result, ratings, last_date, extra


# ── Per-weight-class ELO ────────────────────────────────────────────────────
def load_bouts_with_wc() -> pd.DataFrame:
    """Load bouts with weightindex from final_features_fast (covers ~99% of bouts)."""
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT w.DATE, w.jevent, w.jbout, w.jfighter, w.win, w.ko, w.subw,
           f.weightindex
    FROM ufc_winlossko w
    LEFT JOIN final_features_fast f
      ON f.jevent = w.jevent AND f.jbout = w.jbout AND f.jfighter = w.jfighter
    ORDER BY w.DATE, w.jevent, w.jbout, w.jfighter
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    df["DATE"] = pd.to_datetime(df["DATE"])

    bouts = []
    for (jevent, jbout), grp in df.groupby(["jevent", "jbout"]):
        if len(grp) != 2:
            continue
        r1, r2 = grp.iloc[0], grp.iloc[1]
        if r1["jfighter"] > r2["jfighter"]:
            r1, r2 = r2, r1
        if r1["win"] == 1:
            winner, loser = r1["jfighter"], r2["jfighter"]
            method = "ko" if r1["ko"] == 1 else ("sub" if r1["subw"] == 1 else "dec")
        else:
            winner, loser = r2["jfighter"], r1["jfighter"]
            method = "ko" if r2["ko"] == 1 else ("sub" if r2["subw"] == 1 else "dec")
        wc = r1["weightindex"] if pd.notna(r1["weightindex"]) else 0
        bouts.append({
            "DATE": r1["DATE"], "jevent": jevent, "jbout": jbout,
            "f1": r1["jfighter"], "f2": r2["jfighter"],
            "winner": winner, "loser": loser, "method": method,
            "f1_win": int(r1["win"]), "weightindex": int(wc),
        })

    bouts_df = pd.DataFrame(bouts).sort_values("DATE").reset_index(drop=True)
    print(f"Loaded {len(bouts_df):,} bouts with weightindex  "
          f"({bouts_df['DATE'].min().date()} → {bouts_df['DATE'].max().date()})")
    return bouts_df


def compute_elo_per_wc(bouts: pd.DataFrame,
                       K: float = 64,
                       ko_mult: float = 1.5,
                       sub_mult: float = 1.3,
                       decay_lambda: float = 0.2) -> tuple:
    """
    Per-weight-class ELO: each fighter maintains separate ratings per division.
    Key: (jfighter, weightindex) → rating.
    Returns (result_df, ratings_wc dict, last_date_wc dict).
    """
    ratings   = {}   # (jfighter, wc) -> ELO
    last_date = {}   # (jfighter, wc) -> last fight date

    def get_rating(fighter, wc, fight_date):
        key = (fighter, wc)
        if key not in ratings:
            ratings[key]   = BASE_ELO
            last_date[key] = fight_date
            return BASE_ELO
        r = ratings[key]
        if decay_lambda > 0:
            years = (fight_date - last_date[key]).days / 365.25
            r = BASE_ELO + (r - BASE_ELO) * np.exp(-decay_lambda * max(years, 0))
        return r

    pre_f1, pre_f2 = [], []
    for _, bout in bouts.iterrows():
        f1, f2   = bout["f1"], bout["f2"]
        winner   = bout["winner"]
        method   = bout["method"]
        date     = bout["DATE"]
        wc       = bout["weightindex"]

        r1 = get_rating(f1, wc, date)
        r2 = get_rating(f2, wc, date)

        exp1  = 1 / (1 + 10 ** ((r2 - r1) / 400))
        mult  = ko_mult if method == "ko" else (sub_mult if method == "sub" else 1.0)
        k_eff = K * mult
        s1    = 1.0 if f1 == winner else 0.0

        ratings[(f1, wc)]   = r1 + k_eff * (s1 - exp1)
        ratings[(f2, wc)]   = r2 + k_eff * ((1 - s1) - (1 - exp1))
        last_date[(f1, wc)] = date
        last_date[(f2, wc)] = date

        pre_f1.append(r1)
        pre_f2.append(r2)

    result = bouts.copy()
    result["precomp_elo_wc_f1"]   = pre_f1
    result["precomp_elo_wc_f2"]   = pre_f2
    result["precomp_elo_wc_diff"] = [a - b for a, b in zip(pre_f1, pre_f2)]
    return result, ratings, last_date


def get_fighter_elo_wc(jfighter: str, wc: int, ratings_wc: dict,
                       last_date_wc: dict, event_date,
                       decay_lambda: float = 0.0) -> float:
    """Get a fighter's weight-class-specific ELO as of event_date."""
    key = (jfighter, wc)
    if key not in ratings_wc:
        return BASE_ELO
    r = ratings_wc[key]
    if decay_lambda > 0 and key in last_date_wc:
        years = (pd.to_datetime(event_date) - last_date_wc[key]).days / 365.25
        r = BASE_ELO + (r - BASE_ELO) * np.exp(-decay_lambda * max(years, 0))
    return r


def get_fighter_elo(jfighter: str, ratings: dict, last_date: dict,
                    event_date, decay_lambda: float = 0.0,
                    decay_max: float = None, decay_midpoint: float = None,
                    decay_steepness: float = None) -> float:
    """
    Return a fighter's ELO rating as of event_date, applying inactivity decay.
    Falls back to BASE_ELO for unknown fighters.
    Uses sigmoid decay if all three sigmoid params are provided, else exponential.
    """
    if jfighter not in ratings:
        return BASE_ELO
    r = ratings[jfighter]
    if jfighter in last_date:
        days_inactive = max((pd.to_datetime(event_date) - last_date[jfighter]).days, 0)
        if (decay_max is not None and decay_midpoint is not None
                and decay_steepness is not None and days_inactive > 0):
            sig = 1.0 / (1.0 + np.exp(-(days_inactive - decay_midpoint) / decay_steepness))
            decay_factor = 1.0 - decay_max * sig
            r = BASE_ELO + (r - BASE_ELO) * decay_factor
        elif decay_lambda > 0 and days_inactive > 0:
            years = days_inactive / 365.25
            r = BASE_ELO + (r - BASE_ELO) * np.exp(-decay_lambda * years)
    return r


# ── Evaluate: AUC of elo_diff predicting f1_win ────────────────────────────
def evaluate(bouts: pd.DataFrame, test_start: str = "2024-01-01") -> float:
    test = bouts[bouts["DATE"] >= test_start].copy()
    if len(test) < 10:
        return 0.0
    return roc_auc_score(test["f1_win"], test["precomp_elo_diff"])


# ── Phase 1: Baseline ───────────────────────────────────────────────────────
def phase1(bouts: pd.DataFrame):
    print("\n" + "="*60)
    print("PHASE 1 — Base ELO (K=32, no multipliers, no decay)")
    print("="*60)
    result, _, _, _ = compute_elo(bouts, K=32)
    auc = evaluate(result)
    # Also check correlation
    test = result[result["DATE"] >= "2024-01-01"]
    corr = test["precomp_elo_diff"].corr(test["f1_win"].astype(float))
    print(f"  Test AUC  : {auc:.4f}")
    print(f"  Test Corr : {corr:.4f}")
    print(f"  Test rows : {len(test):,}")
    return result


# ── Phase 2: Grid search K + method multipliers ─────────────────────────────
def phase2(bouts: pd.DataFrame):
    print("\n" + "="*60)
    print("PHASE 2 — Grid search: K + method multipliers")
    print("="*60)

    K_values   = [16, 24, 32, 40, 48, 64]
    ko_values  = [1.0, 1.1, 1.2, 1.3, 1.5]
    sub_values = [1.0, 1.1, 1.2, 1.3]

    best_auc  = 0.0
    best_params = {}
    results = []

    total = len(K_values) * len(ko_values) * len(sub_values)
    print(f"  Searching {total} combinations...")

    for K, ko_m, sub_m in product(K_values, ko_values, sub_values):
        r, _, _, _ = compute_elo(bouts, K=K, ko_mult=ko_m, sub_mult=sub_m)
        auc = evaluate(r)
        results.append({"K": K, "ko_mult": ko_m, "sub_mult": sub_m, "auc": auc})
        if auc > best_auc:
            best_auc    = auc
            best_params = {"K": K, "ko_mult": ko_m, "sub_mult": sub_m}

    df_results = pd.DataFrame(results).sort_values("auc", ascending=False)
    print("\n  Top 10 combinations:")
    print(df_results.head(10).to_string(index=False))
    print(f"\n  Best: K={best_params['K']}  ko_mult={best_params['ko_mult']}  "
          f"sub_mult={best_params['sub_mult']}  AUC={best_auc:.4f}")
    return best_params, best_auc


# ── Phase 3: Add time decay ─────────────────────────────────────────────────
def phase3(bouts: pd.DataFrame, best_params: dict, best_auc_p2: float):
    print("\n" + "="*60)
    print("PHASE 3 — Grid search: time decay (on top of best Phase 2 params)")
    print("="*60)

    decay_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    results = []
    best_auc    = best_auc_p2
    best_decay  = 0.0

    for lam in decay_values:
        r, _, _, _ = compute_elo(bouts,
                              K=best_params["K"],
                              ko_mult=best_params["ko_mult"],
                              sub_mult=best_params["sub_mult"],
                              decay_lambda=lam)
        auc = evaluate(r)
        results.append({"decay_lambda": lam, "auc": auc})
        if auc > best_auc:
            best_auc   = auc
            best_decay = lam

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print(f"\n  Best decay_lambda={best_decay}  AUC={best_auc:.4f}")
    return best_decay, best_auc


# ── Phase 4: Weight-class change penalty ────────────────────────────────────
def phase4(bouts: pd.DataFrame, best_params: dict, best_auc_p3: float):
    print("\n" + "="*60)
    print("PHASE 4 — Grid search: weight-class change penalty")
    print("="*60)

    wc_values = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0]
    results, best_auc, best_wc = [], best_auc_p3, 1.0

    for wc_p in wc_values:
        r, _, _, _ = compute_elo(bouts,
                              K=best_params["K"],
                              ko_mult=best_params["ko_mult"],
                              sub_mult=best_params["sub_mult"],
                              decay_lambda=best_params["decay_lambda"],
                              wc_change_penalty=wc_p)
        auc = evaluate(r)
        results.append({"wc_change_penalty": wc_p, "auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_wc  = wc_p

    print(pd.DataFrame(results).to_string(index=False))
    print(f"\n  Best wc_change_penalty={best_wc}  AUC={best_auc:.4f}")
    return best_wc, best_auc


# ── Phase 6: Round-1 finish multiplier ──────────────────────────────────────
def phase6(bouts: pd.DataFrame, best_params: dict, best_auc_p5: float):
    print("\n" + "="*60)
    print("PHASE 6 — Grid search: round-1 finish multiplier")
    print("="*60)

    r1_values = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
    results, best_auc, best_r1 = [], best_auc_p5, 1.0

    for r1m in r1_values:
        r, _, _, _ = compute_elo(bouts,
                              K=best_params["K"],
                              ko_mult=best_params["ko_mult"],
                              sub_mult=best_params["sub_mult"],
                              decay_lambda=best_params["decay_lambda"],
                              wc_change_penalty=best_params["wc_change_penalty"],
                              streak_bonus=best_params["streak_bonus"],
                              streak_cap=best_params["streak_cap"],
                              r1_finish_mult=r1m)
        auc = evaluate(r)
        results.append({"r1_finish_mult": r1m, "auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_r1  = r1m

    print(pd.DataFrame(results).to_string(index=False))
    print(f"\n  Best r1_finish_mult={best_r1}  AUC={best_auc:.4f}")
    return best_r1, best_auc


# ── Phase 5: Win streak bonus ────────────────────────────────────────────────
def phase5(bouts: pd.DataFrame, best_params: dict, best_auc_p4: float):
    print("\n" + "="*60)
    print("PHASE 5 — Grid search: win streak bonus")
    print("="*60)

    streak_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    cap_values    = [3, 5, 7]
    results, best_auc = [], best_auc_p4
    best_streak, best_cap = 0.0, 5

    total = len(streak_values) * len(cap_values)
    print(f"  Searching {total} combinations...")

    for sb, sc in product(streak_values, cap_values):
        r, _, _, _ = compute_elo(bouts,
                              K=best_params["K"],
                              ko_mult=best_params["ko_mult"],
                              sub_mult=best_params["sub_mult"],
                              decay_lambda=best_params["decay_lambda"],
                              wc_change_penalty=best_params["wc_change_penalty"],
                              streak_bonus=sb,
                              streak_cap=sc)
        auc = evaluate(r)
        results.append({"streak_bonus": sb, "streak_cap": sc, "auc": auc})
        if auc > best_auc:
            best_auc    = auc
            best_streak = sb
            best_cap    = sc

    df_r = pd.DataFrame(results).sort_values("auc", ascending=False)
    print(df_r.to_string(index=False))
    print(f"\n  Best streak_bonus={best_streak}  streak_cap={best_cap}  AUC={best_auc:.4f}")
    return best_streak, best_cap, best_auc


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bouts = load_bouts()

    # Phase 1
    phase1(bouts)

    # Phase 2
    best_params, best_auc_p2 = phase2(bouts)

    # Phase 3
    best_decay, best_auc_p3 = phase3(bouts, best_params, best_auc_p2)
    best_params["decay_lambda"] = best_decay

    # Phase 4
    best_wc, best_auc_p4 = phase4(bouts, best_params, best_auc_p3)
    best_params["wc_change_penalty"] = best_wc

    # Phase 5
    best_streak, best_cap, best_auc_p5 = phase5(bouts, best_params, best_auc_p4)
    best_params["streak_bonus"] = best_streak
    best_params["streak_cap"]   = best_cap

    # Phase 6
    best_r1, final_auc = phase6(bouts, best_params, best_auc_p5)
    best_params["r1_finish_mult"] = best_r1

    print("\n" + "="*60)
    print("FINAL BEST PARAMS")
    print("="*60)
    for k, v in best_params.items():
        print(f"  {k:<22} = {v}")
    print(f"  {'Test AUC (ELO alone)':<22} = {final_auc:.4f}")
