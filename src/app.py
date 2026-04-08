"""
UFC Fight Predictor — Web App
Flask app wrapping the MMA-AI pipeline with LR+CB ensemble.
Model trains once at startup, then serves predictions via API.
"""

import sqlite3
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Ensure src/ is on path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import math

from predict_mmaai import (
    build_training_data, train_ensemble,
    predict_fight as predict_fight_mmaai,
    ELO_K, ELO_KO_MULT, ELO_SUB_MULT, ELO_DECAY,
    ELO_DECAY_MAX, ELO_DECAY_MIDPOINT, ELO_DECAY_STEEPNESS,
    ELO_LOGISTIC_SCALE, ALL_ELO_FEATURES as ELO_FEATURES,
    SELECTED_ELO_FEATURES,
)
from elo_feature import get_fighter_elo, load_bouts, compute_elo, BASE_ELO

# ── Z-Score stat configuration ─────────────────────────────────────────────
# Maps category → list of stats for the fighter profile z-score display.
# Each stat: col (DB column), name (display), table (source), inverted (lower=better).
ZSCORE_STAT_CONFIG = {
    "Striking": [
        {"col": "sig_str_land_per_min_dec_avg", "name": "Sig. Strikes/Min", "table": "final_features_fast", "inverted": False},
        {"col": "sig_str_acc_dec_avg", "name": "Sig. Strike Accuracy", "table": "final_features_fast", "inverted": False},
        {"col": "sig_str_def_dec_avg", "name": "Sig. Strike Defense", "table": "final_features_fast", "inverted": False},
        {"col": "head_acc_dec_avg", "name": "Head Accuracy", "table": "final_features_fast", "inverted": False},
        {"col": "body_acc_dec_avg", "name": "Body Accuracy", "table": "final_features_fast", "inverted": False},
        {"col": "leg_acc_dec_avg", "name": "Leg Accuracy", "table": "final_features_fast", "inverted": False},
        {"col": "distance_acc_dec_avg", "name": "Distance Accuracy", "table": "final_features_fast", "inverted": False},
        {"col": "kd_per_min_dec_avg", "name": "Knockdowns/Min", "table": "final_features_fast", "inverted": False},
    ],
    "Power & Finishing": [
        {"col": "ko_eff_dec_avg", "name": "KO Efficiency", "table": "final_features_fast", "inverted": False},
        {"col": "ko_in_r1_rate_dec_avg", "name": "R1 KO Rate", "table": "new_features4_dec_avg", "inverted": False},
        {"col": "finish_rate_dec_avg", "name": "Finish Rate", "table": "final_features_fast", "inverted": False},
        {"col": "damage_efficiency_dec_avg", "name": "Damage Efficiency", "table": "new_features4_dec_avg", "inverted": False},
        {"col": "output_per_damage_dec_avg", "name": "Output per Damage", "table": "new_features4_dec_avg", "inverted": False},
    ],
    "Grappling": [
        {"col": "td_land_per_min_dec_avg", "name": "Takedowns/Min", "table": "final_features_fast", "inverted": False},
        {"col": "td_def_dec_avg", "name": "TD Defense", "table": "final_features_fast", "inverted": False},
        {"col": "sub_att_per_min_dec_avg", "name": "Sub Attempts/Min", "table": "final_features_fast", "inverted": False},
        {"col": "ctrl_per_min_dec_avg", "name": "Control Time/Min", "table": "final_features_fast", "inverted": False},
        {"col": "grappling_dominance_pm_dec_avg", "name": "Grappling Dominance", "table": "new_features5_dec_avg", "inverted": False},
        {"col": "top_position_output_dec_avg", "name": "Top Position Output", "table": "new_features5_dec_avg", "inverted": False},
        {"col": "td_to_ctrl_conversion_dec_avg", "name": "TD\u2192Control Conv.", "table": "new_features5_dec_avg", "inverted": False},
    ],
    "Cardio & Durability": [
        {"col": "r3_vs_r1_sigstr_ratio_dec_avg", "name": "Late Round Output", "table": "cardio_features_dec_avg", "inverted": False},
        {"col": "sigstr_absorbed_pm_dec_avg", "name": "Strikes Absorbed/Min", "table": "new_features4_dec_avg", "inverted": True},
        {"col": "kod_rate_dec_avg", "name": "KO'd Rate", "table": "new_features4_dec_avg", "inverted": True},
        {"col": "avg_fight_duration_dec_avg", "name": "Avg Fight Duration", "table": "new_features4_dec_avg", "inverted": False},
        {"col": "r1_output_share_dec_avg", "name": "R1 Output Share", "table": "cardio_features_dec_avg", "inverted": False},
    ],
    "Activity & Record": [
        {"col": "win_streak", "name": "Win Streak", "table": "streak_features", "inverted": False},
        {"col": "opp_avg_win_ratio", "name": "Opposition Quality", "table": "opp_quality_features", "inverted": False},
        {"col": "recent_ko_rate_3", "name": "Recent KO Rate", "table": "streak_features", "inverted": False},
        {"col": "ufc_fight_count", "name": "UFC Experience", "table": "new_features_dec_avg", "inverted": False},
        {"col": "days_since_last_fight_dec_avg", "name": "Days Since Last Fight", "table": "final_features_fast", "inverted": True},
    ],
}

_app_db = Path(__file__).parent.parent / "data/sqlite_db/app.db"
_full_db = Path(__file__).parent.parent / "data/sqlite_db/sqlite_scrapper.db"
DB_PATH = _app_db if _app_db.exists() else _full_db

app = Flask(__name__)

# ── Global model state (loaded once at startup) ─────────────────────────────
model_state = {}


def init_model():
    """Train the MMA-AI pipeline model on all available data. Called once at startup."""
    print("Building MMA-AI features (this takes ~60s)...")
    data = build_training_data()
    print("Training LR+CB ensemble...")
    models = train_ensemble(data)

    model_state["models"] = models
    model_state["fighter_stats"] = data["fighter_stats"]
    model_state["feature_cols"] = data["feature_cols"]
    model_state["feat_cols"] = data["feat_cols"]
    model_state["df_trained"] = data["df"]

    # Build Elo ratings for live predictions (from pre-saved CSV, no DB needed)
    print("Building Elo history...")
    from predict_mmaai import ELO_BOUTS_CSV
    bouts = pd.read_csv(ELO_BOUTS_CSV)
    bouts["DATE"] = pd.to_datetime(bouts["DATE"])
    elo_full_df, ratings, last_date, extra = compute_elo(
        bouts, K=ELO_K, ko_mult=ELO_KO_MULT, sub_mult=ELO_SUB_MULT,
        decay_lambda=ELO_DECAY,
        decay_max=ELO_DECAY_MAX, decay_midpoint=ELO_DECAY_MIDPOINT,
        decay_steepness=ELO_DECAY_STEEPNESS,
        logistic_scale=ELO_LOGISTIC_SCALE,
        opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2,
    )
    model_state["elo_ratings"] = ratings
    model_state["elo_last_date"] = last_date
    model_state["elo_extra"] = extra
    model_state["elo_full_df"] = elo_full_df

    # Fighter list from MMA-AI pipeline stats
    model_state["fighter_list"] = [
        {"jfighter": name, "last_fight": str(stats.get("DATE", ""))}
        for name, stats in data["fighter_stats"].items()
    ]

    print("Computing z-score baselines...")
    try:
        _compute_wc_baselines()
    except Exception as e:
        print(f"  Z-score baselines skipped (DB tables missing): {e}")
    print("Model ready.")


def _load_fighter_list():
    """Load all fighter names with their most recent fight date."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT jfighter, MAX(DATE) as last_fight
           FROM final_features_fast
           GROUP BY jfighter
           ORDER BY jfighter""",
        conn,
    )
    conn.close()
    return df.to_dict("records")


def _get_fighter_details(jfighter):
    """Get fighter physical stats + record from DB. Gracefully degrades if tables missing."""
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception:
        return {}

    try:
        tott = pd.read_sql_query(
            "SELECT * FROM ufc_fighter_tott WHERE jfighter = ?",
            conn, params=(jfighter,),
        )
    except Exception:
        tott = pd.DataFrame()

    try:
        record = pd.read_sql_query(
            """SELECT
                 COUNT(*) as total_fights,
                 SUM(win) as wins,
                 SUM(CASE WHEN win=0 THEN 1 ELSE 0 END) as losses,
                 SUM(ko) as ko_wins,
                 SUM(subw) as sub_wins
               FROM ufc_winlossko WHERE jfighter = ?""",
            conn, params=(jfighter,),
        )
    except Exception:
        record = pd.DataFrame()

    try:
        nat = pd.read_sql_query(
            "SELECT country FROM ufc_fighter_nationality WHERE jfighter = ?",
            conn, params=(jfighter,),
        )
    except Exception:
        nat = pd.DataFrame()
    # Elo
    elo = None
    if model_state.get("elo_ratings") and jfighter in model_state["elo_ratings"]:
        elo = model_state["elo_ratings"][jfighter]
        peak = (model_state.get("elo_extra") or {}).get("peak_elo", {}).get(jfighter)
    else:
        peak = None

    conn.close()

    details = {}
    if not tott.empty:
        row = tott.iloc[0]
        details["height"] = row.get("HEIGHT")
        details["weight"] = row.get("WEIGHT")
        details["reach"] = row.get("REACH")
        details["stance"] = row.get("STANCE")
        details["dob"] = row.get("DOB")
    if not record.empty:
        row = record.iloc[0]
        details["total_fights"] = int(row["total_fights"])
        details["wins"] = int(row["wins"])
        details["losses"] = int(row["losses"])
        details["ko_wins"] = int(row["ko_wins"])
        details["sub_wins"] = int(row["sub_wins"])
    if not nat.empty:
        details["country"] = nat.iloc[0]["country"]
    if elo is not None:
        details["elo"] = round(elo, 1)
    if peak is not None:
        details["peak_elo"] = round(peak, 1)

    return details


def _get_odds(fighter_a, fighter_b):
    """Look up most recent odds for this matchup."""
    conn = sqlite3.connect(DB_PATH)
    # Try both orderings
    for f1, f2 in [(fighter_a, fighter_b), (fighter_b, fighter_a)]:
        odds = pd.read_sql_query(
            """SELECT avg_odds_f1, avg_odds_f2, DATE
               FROM ufc_fight_odds
               WHERE jfighter = ? AND opp_jfighter = ?
               ORDER BY DATE DESC LIMIT 1""",
            conn, params=(f1, f2),
        )
        if not odds.empty:
            conn.close()
            row = odds.iloc[0]
            if f1 == fighter_a:
                return {"odds_a": row["avg_odds_f1"], "odds_b": row["avg_odds_f2"],
                        "odds_date": row["DATE"]}
            else:
                return {"odds_a": row["avg_odds_f2"], "odds_b": row["avg_odds_f1"],
                        "odds_date": row["DATE"]}
    conn.close()
    return None


def _american_to_implied(odds):
    """Convert American odds to implied probability."""
    if odds is None or pd.isna(odds):
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _format_fighter_name(jfighter):
    """Convert CamelCase jfighter to display name (e.g. AlexPereira -> Alex Pereira)."""
    import re
    # Insert space before uppercase letters that follow lowercase
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', jfighter)
    # Handle cases like O'Malley
    name = re.sub(r"(\w)'(\w)", r"\1'\2", name)
    return name


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": "pipe" in model_state,
        "fighters": len(model_state.get("fighter_list", [])),
    })


@app.route("/api/fighters")
def fighters():
    """Return fighter list for autocomplete. Optional ?q= filter."""
    q = (request.args.get("q") or "").lower()
    fighters = model_state.get("fighter_list", [])
    if q:
        fighters = [f for f in fighters if q in f["jfighter"].lower()]
    # Limit to 50 for performance
    return jsonify(fighters[:50])


@app.route("/api/fighter/<name>/elo_history")
def fighter_elo_history(name):
    """Return full Elo history: pre-fight and post-fight Elo for each bout."""
    elo_df = model_state.get("elo_full_df")
    if elo_df is None:
        return jsonify({"error": "Elo data not loaded"}), 503

    # Collect all bouts involving this fighter
    as_f1 = elo_df[elo_df["f1"] == name].copy()
    as_f2 = elo_df[elo_df["f2"] == name].copy()

    rows = []
    for _, r in as_f1.iterrows():
        rows.append({
            "date": str(r["DATE"])[:10],
            "pre_elo": float(r["precomp_elo_f1"]),
            "opp_pre_elo": float(r["precomp_elo_f2"]),
            "opponent": r["f2"],
            "won": bool(r.get("f1_win", r.get("winner", "") == name)),
            "method": r.get("method", ""),
        })
    for _, r in as_f2.iterrows():
        rows.append({
            "date": str(r["DATE"])[:10],
            "pre_elo": float(r["precomp_elo_f2"]),
            "opp_pre_elo": float(r["precomp_elo_f1"]),
            "opponent": r["f1"],
            "won": bool(r.get("f1_win", 0) == 0),
            "method": r.get("method", ""),
        })

    rows.sort(key=lambda x: x["date"])

    # Compute post-fight Elo using the same formula as elo_feature.py
    K = ELO_K
    points = []
    for row in rows:
        pre = row["pre_elo"]
        opp = row["opp_pre_elo"]
        exp = 1 / (1 + 10 ** ((opp - pre) / ELO_LOGISTIC_SCALE))
        method = row["method"]
        mult = ELO_KO_MULT if method == "ko" else (ELO_SUB_MULT if method == "sub" else 1.0)
        s = 1.0 if row["won"] else 0.0
        post = pre + K * mult * (s - exp)

        points.append({
            "date": row["date"],
            "pre_elo": round(pre, 1),
            "post_elo": round(post, 1),
            "opponent": row["opponent"],
            "opponent_display": _format_fighter_name(row["opponent"]),
            "result": "W" if row["won"] else "L",
            "method": method,
            "opp_elo": round(opp, 1),
        })

    return jsonify({"jfighter": name, "history": points})


@app.route("/api/fighter/<name>")
def fighter_detail(name):
    """Get fighter details: physical stats, record, Elo."""
    details = _get_fighter_details(name)
    if not details:
        return jsonify({"error": f"Fighter not found: {name}"}), 404
    details["jfighter"] = name
    details["display_name"] = _format_fighter_name(name)
    return jsonify(details)


@app.route("/fighter/<name>")
def fighter_profile_page(name):
    """Serve the fighter profile page."""
    return render_template("fighter.html", jfighter=name)


@app.route("/api/fighter/<name>/profile")
def fighter_full_profile(name):
    """Full fighter profile: details + recent fights + radar stats + Elo history."""
    details = _get_fighter_details(name)
    if not details:
        return jsonify({"error": f"Fighter not found: {name}"}), 404
    details["jfighter"] = name
    details["display_name"] = _format_fighter_name(name)

    conn = sqlite3.connect(DB_PATH)

    # Recent fights (last 8)
    recent = pd.read_sql_query(
        """SELECT w.DATE, w.jevent, w.jbout, w.win, w.ko, w.subw,
                  w.jfighter,
                  fr.METHOD, fr.ROUND, fr.TIME
           FROM ufc_winlossko w
           LEFT JOIN ufc_fight_results fr ON fr.jevent = w.jevent AND fr.jbout = w.jbout
           WHERE w.jfighter = ?
           ORDER BY w.DATE DESC LIMIT 20""",
        conn, params=(name,),
    )
    fights = []
    for _, r in recent.iterrows():
        # Extract opponent from jbout (format: "FighterAvs.FighterB")
        parts = r["jbout"].replace("vs.", "|").split("|")
        opp = parts[1] if parts[0] == name else parts[0]
        fights.append({
            "date": r["DATE"],
            "opponent": opp,
            "opponent_display": _format_fighter_name(opp),
            "result": "W" if r["win"] else "L",
            "method": r.get("METHOD") or "",
            "round": int(r["ROUND"]) if pd.notna(r.get("ROUND")) else None,
            "ko": bool(r["ko"]),
            "sub": bool(r["subw"]),
        })

    # Radar stats from AdjPerf z-scores (most recent fight in final_features_fast)
    radar = {}
    try:
        adjperf = pd.read_sql_query(
            """SELECT adjperf_sigstr_pm_dec_avg, adjperf_kd_pm_dec_avg,
                      adjperf_td_per15_dec_avg, adjperf_ctrl_per_min_dec_avg,
                      adjperf_ko_eff_dec_avg, adjperf_td_att_pm_dec_avg
               FROM final_features_fast
               WHERE jfighter = ? ORDER BY DATE DESC LIMIT 1""",
            conn, params=(name,),
        )
        if not adjperf.empty:
            row = adjperf.iloc[0]
            radar_map = {
                "Striking": "adjperf_sigstr_pm_dec_avg",
                "Power": "adjperf_kd_pm_dec_avg",
                "KO Eff.": "adjperf_ko_eff_dec_avg",
                "Wrestling": "adjperf_td_per15_dec_avg",
                "Grappling": "adjperf_ctrl_per_min_dec_avg",
                "TD Press.": "adjperf_td_att_pm_dec_avg",
            }
            for label, col in radar_map.items():
                val = row.get(col)
                if val is not None and pd.notna(val):
                    clipped = max(-3, min(3, float(val)))
                    radar[label] = round((clipped + 3) / 6 * 100, 1)
                else:
                    radar[label] = 50.0
    except Exception:
        pass

    # Elo history (all fights)
    elo_history = []
    if model_state.get("elo_extra"):
        elo_3ago = model_state["elo_extra"].get("elo_history", {}).get(name, [])
        # elo_history might not be stored; build from elo_df instead
    # Build from elo_df in model training data
    try:
        elo_df_full = model_state.get("df_trained")
        if elo_df_full is not None:
            fighter_rows = elo_df_full[
                (elo_df_full["jfighter"] == name)
            ][["DATE", "precomp_elo_diff"]].sort_values("DATE")
            # precomp_elo_diff is f1-f2; we need absolute Elo
            # Use the ratings dict for current, and approximate history
            if name in model_state.get("elo_ratings", {}):
                current_elo = model_state["elo_ratings"][name]
                elo_history.append({
                    "date": "current",
                    "elo": round(current_elo, 1),
                })
    except Exception:
        pass

    conn.close()

    return jsonify({
        **details,
        "recent_fights": fights,
        "radar": radar,
        "elo_history": elo_history,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict a single fight.
    POST JSON: {"fighter_a": "AlexPereira", "fighter_b": "MagommedAnkalaev",
                "event_date": "2026-04-05", "scheduled_rounds": 5}
    """
    data = request.get_json()
    if not data or "fighter_a" not in data or "fighter_b" not in data:
        return jsonify({"error": "Provide fighter_a and fighter_b"}), 400

    fighter_a = data["fighter_a"]
    fighter_b = data["fighter_b"]
    event_date = data.get("event_date", pd.Timestamp.now().strftime("%Y-%m-%d"))

    models = model_state.get("models")
    if models is None:
        return jsonify({"error": "Model not loaded yet"}), 503

    try:
        result = predict_fight_mmaai(
            fighter_a, fighter_b,
            models=models,
            fighter_stats=model_state["fighter_stats"],
            feature_cols=model_state["feature_cols"],
            elo_ratings=model_state["elo_ratings"],
            elo_last_date=model_state["elo_last_date"],
            elo_extra=model_state["elo_extra"],
            event_date=event_date,
            verbose=False,
        )

        # Get top feature drivers
        drivers = _get_top_drivers(result, models)

        # Get fighter details
        details_a = _get_fighter_details(fighter_a)
        details_b = _get_fighter_details(fighter_b)

        # Get Vegas odds
        odds = _get_odds(fighter_a, fighter_b)
        vegas = {}
        if odds:
            vegas["odds_a"] = odds["odds_a"]
            vegas["odds_b"] = odds["odds_b"]
            imp_a = _american_to_implied(odds["odds_a"])
            imp_b = _american_to_implied(odds["odds_b"])
            if imp_a is not None and imp_b is not None:
                # Normalize to remove vig
                total = imp_a + imp_b
                vegas["implied_prob_a"] = round(imp_a / total, 4)
                vegas["implied_prob_b"] = round(imp_b / total, 4)

        response = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "display_a": _format_fighter_name(fighter_a),
            "display_b": _format_fighter_name(fighter_b),
            "prob_a": round(result["prob_a"], 4),
            "prob_b": round(1 - result["prob_a"], 4),
            "winner": result["winner"],
            "winner_display": _format_fighter_name(result["winner"]),
            "confidence": round(result["confidence"], 4),
            "details_a": details_a,
            "details_b": details_b,
            "vegas": vegas,
            "drivers": drivers,
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict_card", methods=["POST"])
def predict_card():
    """
    Predict a full fight card.
    POST JSON: {"fights": [{"fighter_a": "...", "fighter_b": "..."}, ...],
                "event_date": "2026-04-05", "event_name": "UFC 315"}
    """
    data = request.get_json()
    if not data or "fights" not in data:
        return jsonify({"error": "Provide fights array"}), 400

    event_date = data.get("event_date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    event_name = data.get("event_name", "UFC Event")

    models = model_state.get("models")
    if models is None:
        return jsonify({"error": "Model not loaded yet"}), 503

    results = []

    for i, fight in enumerate(data["fights"]):
        fa = fight.get("fighter_a", "")
        fb = fight.get("fighter_b", "")
        if not fa or not fb:
            results.append({"bout_num": i + 1, "error": "Missing fighter name"})
            continue

        try:
            r = predict_fight_mmaai(
                fa, fb,
                models=models,
                fighter_stats=model_state["fighter_stats"],
                feature_cols=model_state["feature_cols"],
                elo_ratings=model_state["elo_ratings"],
                elo_last_date=model_state["elo_last_date"],
                elo_extra=model_state["elo_extra"],
                event_date=event_date,
                verbose=False,
            )
            odds = _get_odds(fa, fb)
            vegas = {}
            if odds:
                vegas["odds_a"] = odds["odds_a"]
                vegas["odds_b"] = odds["odds_b"]
                imp_a = _american_to_implied(odds["odds_a"])
                imp_b = _american_to_implied(odds["odds_b"])
                if imp_a is not None and imp_b is not None:
                    total = imp_a + imp_b
                    vegas["implied_prob_a"] = round(imp_a / total, 4)
                    vegas["implied_prob_b"] = round(imp_b / total, 4)

            results.append({
                "bout_num": i + 1,
                "fighter_a": fa,
                "fighter_b": fb,
                "display_a": _format_fighter_name(fa),
                "display_b": _format_fighter_name(fb),
                "prob_a": round(r["prob_a"], 4),
                "prob_b": round(1 - r["prob_a"], 4),
                "winner": r["winner"],
                "winner_display": _format_fighter_name(r["winner"]),
                "confidence": round(r["confidence"], 4),
                "vegas": vegas,
            })
        except ValueError as e:
            results.append({"bout_num": i + 1, "error": str(e),
                            "fighter_a": fa, "fighter_b": fb})

    return jsonify({
        "event_name": event_name,
        "event_date": event_date,
        "predictions": results,
    })


def _get_top_drivers(result, models, n=5):
    """Extract top N feature drivers from LR model."""
    lr = models.get("lr")
    if lr is None:
        return []

    feat_cols = models["feat_cols"]
    scaler = models["scaler"]
    fighter_stats = model_state["fighter_stats"]

    f1, f2 = result["f1"], result["f2"]
    sf1 = fighter_stats.get(result["name_a"] if result["name_a"] == f1 else result["name_b"], {})
    sf2 = fighter_stats.get(result["name_b"] if result["name_b"] == f2 else result["name_a"], {})

    coef = lr.coef_[0]

    # Rebuild feature row
    row = {}
    for feat in feat_cols:
        if feat in SELECTED_ELO_FEATURES:
            row[feat] = 0.0
        elif feat == "weightclass_encoded":
            row[feat] = sf1.get("weightindex", 0)
        elif feat == "scheduled_rounds":
            row[feat] = 3.0
        elif feat == "days_since_last_fight_f1":
            row[feat] = sf1.get("days_since_last_fight", 0)
        elif feat.endswith("_diff"):
            col = feat[:-5]
            row[feat] = sf1.get(col, 0) - sf2.get(col, 0)
        else:
            row[feat] = 0.0

    vals = [row.get(f, 0) for f in feat_cols]
    X_arr = np.array(vals).reshape(1, -1)
    X_scaled = scaler.transform(X_arr)[0]

    contributions = [(feat_cols[i], float(vals[i]), float(X_scaled[i] * coef[i]))
                     for i in range(len(feat_cols))]
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)

    drivers = []
    for feat_name, raw_val, contrib in contributions[:n]:
        favors = f1 if contrib > 0 else f2
        drivers.append({
            "feature": feat_name,
            "raw_value": round(raw_val, 3),
            "contribution": round(contrib, 3),
            "favors": favors,
            "favors_display": _format_fighter_name(favors),
        })
    return drivers


# ── Z-Score baselines + endpoint ─────────────────────────────────────────────

def _norm_cdf(z):
    """Approximate normal CDF for z-score → percentile conversion."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _compute_wc_baselines():
    """Compute weight-class mean/std for all z-score stats. Called once at startup."""
    conn = sqlite3.connect(DB_PATH)

    # Collect all unique (table, col) pairs we need
    table_cols = {}
    for stats in ZSCORE_STAT_CONFIG.values():
        for s in stats:
            table_cols.setdefault(s["table"], []).append(s["col"])

    # For each table, get each fighter's LATEST row + weightindex
    all_data = {}
    for table, cols in table_cols.items():
        cols_str = ", ".join(f"t.{c}" for c in cols)
        if table == "final_features_fast":
            # final_features_fast already has weightindex
            q = f"""
                SELECT t.jfighter, t.weightindex, {cols_str}
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY jfighter ORDER BY DATE DESC) AS _rn
                    FROM {table}
                ) t WHERE t._rn = 1
            """
        else:
            # Join to final_features_fast for weightindex
            q = f"""
                SELECT t.jfighter, f.weightindex, {cols_str}
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY jfighter ORDER BY DATE DESC) AS _rn
                    FROM {table}
                ) t
                JOIN (
                    SELECT jfighter, weightindex, ROW_NUMBER() OVER (PARTITION BY jfighter ORDER BY DATE DESC) AS _rn
                    FROM final_features_fast
                ) f ON f.jfighter = t.jfighter AND f._rn = 1
                WHERE t._rn = 1
            """
        try:
            df = pd.read_sql_query(q, conn)
            for col in cols:
                for _, row in df.iterrows():
                    wi = row.get("weightindex")
                    val = row.get(col)
                    if wi is not None and val is not None and pd.notna(val):
                        all_data.setdefault(col, []).append((int(wi), float(val)))
        except Exception as e:
            print(f"  Warning: could not load {table} for baselines: {e}")

    conn.close()

    # Compute per-weightclass mean/std
    baselines = {}
    for col, entries in all_data.items():
        df_col = pd.DataFrame(entries, columns=["weightindex", "value"])
        for wi, grp in df_col.groupby("weightindex"):
            wi = int(wi)
            if len(grp) < 15:
                # Small weight class — use global baselines
                mean_val = df_col["value"].mean()
                std_val = df_col["value"].std()
            else:
                mean_val = grp["value"].mean()
                std_val = grp["value"].std()
            if std_val == 0 or pd.isna(std_val):
                std_val = 1.0
            baselines.setdefault(wi, {})[col] = {
                "mean": float(mean_val),
                "std": float(std_val),
            }

    model_state["wc_baselines"] = baselines
    total_wc = len(baselines)
    total_stats = sum(len(v) for v in baselines.values())
    print(f"  Z-score baselines: {total_stats} stat×WC entries across {total_wc} weight classes")


@app.route("/api/fighter/<name>/zscores")
def fighter_zscores(name):
    """Return weight-class z-scores for a fighter, grouped by category."""
    baselines = model_state.get("wc_baselines")
    if not baselines:
        return jsonify({"error": "Baselines not computed"}), 503

    conn = sqlite3.connect(DB_PATH)

    # Get fighter's weightindex
    wi_row = pd.read_sql_query(
        "SELECT weightindex FROM final_features_fast WHERE jfighter = ? ORDER BY DATE DESC LIMIT 1",
        conn, params=(name,),
    )
    if wi_row.empty:
        conn.close()
        return jsonify({"error": f"Fighter not found: {name}"}), 404
    wi = int(wi_row.iloc[0]["weightindex"])

    wc_bl = baselines.get(wi, {})

    # Collect all unique tables we need
    table_cols = {}
    for stats in ZSCORE_STAT_CONFIG.values():
        for s in stats:
            table_cols.setdefault(s["table"], []).append(s["col"])

    # Query fighter's latest row from each table
    fighter_vals = {}
    for table, cols in table_cols.items():
        cols_str = ", ".join(cols)
        try:
            row = pd.read_sql_query(
                f"SELECT {cols_str} FROM {table} WHERE jfighter = ? ORDER BY DATE DESC LIMIT 1",
                conn, params=(name,),
            )
            if not row.empty:
                for col in cols:
                    val = row.iloc[0].get(col)
                    if val is not None and pd.notna(val):
                        fighter_vals[col] = float(val)
        except Exception:
            pass

    conn.close()

    # Build response by category
    categories = {}
    for cat_name, stats in ZSCORE_STAT_CONFIG.items():
        cat_stats = []
        z_sum, z_count = 0.0, 0
        for s in stats:
            col = s["col"]
            raw = fighter_vals.get(col)
            bl = wc_bl.get(col)
            if raw is not None and bl:
                z = (raw - bl["mean"]) / bl["std"]
                if s["inverted"]:
                    z = -z
                z = max(-3.0, min(3.0, z))
                pct = round(_norm_cdf(z) * 100, 1)
                z_sum += z
                z_count += 1
                cat_stats.append({
                    "key": col,
                    "display_name": s["name"],
                    "z_score": round(z, 2),
                    "raw_value": round(raw, 4),
                    "wc_mean": round(bl["mean"], 4),
                    "percentile": pct,
                    "inverted": s["inverted"],
                })
            else:
                cat_stats.append({
                    "key": col,
                    "display_name": s["name"],
                    "z_score": None,
                    "raw_value": round(raw, 4) if raw is not None else None,
                    "wc_mean": round(bl["mean"], 4) if bl else None,
                    "percentile": None,
                    "inverted": s["inverted"],
                })
        categories[cat_name] = {
            "avg_z": round(z_sum / z_count, 2) if z_count > 0 else None,
            "stats": cat_stats,
        }

    return jsonify({
        "jfighter": name,
        "weightindex": wi,
        "categories": categories,
    })


# ── Feature importance display names ─────────────────────────────────────────
_FEAT_DISPLAY = {
    "sig_str_land_per_min_dec_avg_diff": "Sig. Strikes Landed/Min",
    "total_str_land_per_min_dec_avg_diff": "Total Strikes Landed/Min",
    "td_land_per_min_dec_avg_diff": "Takedowns Landed/Min",
    "sub_att_per_min_dec_avg_diff": "Submission Attempts/Min",
    "ctrl_per_min_dec_avg_diff": "Control Time/Min",
    "sig_str_acc_dec_avg_diff": "Sig. Strike Accuracy",
    "total_str_acc_dec_avg_diff": "Total Strike Accuracy",
    "head_acc_dec_avg_diff": "Head Strike Accuracy",
    "body_acc_dec_avg_diff": "Body Strike Accuracy",
    "leg_acc_dec_avg_diff": "Leg Strike Accuracy",
    "distance_acc_dec_avg_diff": "Distance Strike Accuracy",
    "clinch_acc_dec_avg_diff": "Clinch Strike Accuracy",
    "ground_acc_dec_avg_diff": "Ground Strike Accuracy",
    "sig_str_def_dec_avg_diff": "Sig. Strike Defense",
    "td_def_dec_avg_diff": "Takedown Defense",
    "ko_dec_avg_diff": "KO Rate",
    "ko5_dec_avg_diff": "KO Rate (R1-5)",
    "win_dec_avg_diff": "Win Rate",
    "loss_dec_avg_diff": "Loss Rate",
    "finish_rate_dec_avg_diff": "Finish Rate",
    "kd_per_min_dec_avg_diff": "Knockdowns/Min",
    "ko_eff_dec_avg_diff": "KO Efficiency",
    "td_att_per_min_dec_avg_diff": "TD Attempts/Min",
    "age_dec_avg_diff": "Age",
    "days_since_last_fight_dec_avg_diff": "Days Since Last Fight",
    "age_ratio_diff": "Age Ratio",
    "reach_ratio_dec_avg_diff": "Reach Ratio",
    "grapple_strike_mix_dec_avg_diff": "Grapple/Strike Mix",
    "str_eff_diff_dec_avg_diff": "Striking Efficiency",
    "age_diff": "Age Diff",
    "age_dec_avg_diff": "Age (Career Avg) Diff",
    "days_since_last_fight_diff": "Days Since Last Fight Diff",
    "WEIGHT_diff": "Weight Diff",
    "ufc_age_diff": "UFC Career Length Diff",
    "adjperf_sigstr_pm_dec_avg_diff": "Adj. Sig Strikes/Min Diff",
    "adjperf_totalstr_pm_dec_avg_diff": "Adj. Total Strikes/Min Diff",
    "adjperf_td_per15_dec_avg_diff": "Adj. Takedowns/15Min Diff",
    "adjperf_sub_per15_dec_avg_diff": "Adj. Sub Attempts/15Min Diff",
    "adjperf_ctrl_per_min_dec_avg_diff": "Adj. Control/Min Diff",
    "adjperf_kd_pm_dec_avg_diff": "Adj. Knockdowns/Min Diff",
    "adjperf_ko_eff_dec_avg_diff": "Adj. KO Efficiency Diff",
    "adjperf_td_att_pm_dec_avg_diff": "Adj. TD Attempts/Min Diff",
    "peak_elo_diff": "Peak Elo Rating Diff",
    "elo_win_prob": "Elo Win Probability",
    "precomp_elo_diff": "Elo Rating Diff",
    "elo_momentum_diff": "Elo Momentum",
    "avg_opp_elo_diff": "Avg Opponent Elo",
    "elo_consist_diff": "Elo Consistency",
    "elo_predictability": "Match Predictability",
    "height_ratio_diff": "Height Ratio",
    "weight_diff": "Weight Diff",
    "stance_southpaw_diff": "Southpaw Stance",
    "age_squared_diff": "Age Squared",
    "scheduled_rounds": "Scheduled Rounds",
    "age_prime_diff": "Age Prime",
    "ufc_age_diff": "UFC Career Age",
    "ufc_fight_count_diff": "UFC Fight Count",
    "win_streak_diff": "Win Streak",
    "loss_streak_diff": "Loss Streak",
    "finish_streak_diff": "Finish Streak",
    "recent_ko_rate_3_diff": "Recent KO Rate (3 fights)",
    "recent_finish_rate_3_diff": "Recent Finish Rate (3 fights)",
    "ko_in_r1_rate_dec_avg_diff": "R1 KO Rate",
    "kod_rate_dec_avg_diff": "KO'd Rate",
    "damage_efficiency_dec_avg_diff": "Damage Efficiency",
    "output_per_damage_dec_avg_diff": "Output per Damage",
    "sigstr_absorbed_pm_dec_avg_diff": "Strikes Absorbed/Min",
    "grappling_dominance_pm_dec_avg_diff": "Grappling Dominance",
    "td_to_ctrl_conversion_dec_avg_diff": "TD-to-Control Conv.",
    "r3_vs_r1_sigstr_ratio_dec_avg_diff": "Late Round Output",
    "opp_avg_win_ratio_diff": "Opposition Quality",
    "event_rolling_ema": "Event Win Rate Trend",
    "weightindex": "Weight Class",
}


def _feat_display_name(feat):
    if feat in _FEAT_DISPLAY:
        return _FEAT_DISPLAY[feat]
    # Style matchup features (not diffs, they're interactions)
    if feat in ('striking_matchup', 'grappling_matchup', 'wrestling_matchup',
                'power_matchup', 'sub_matchup', 'style_distance'):
        return feat.replace("_", " ").title()
    # Everything else is a diff — clean up and append "Diff"
    name = feat
    for suffix in ["_adjperf_dec_avg_diff", "_opp_mean_dec_avg_diff", "_dec_avg_diff", "_diff"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    # Add qualifier
    qualifier = ""
    if "adjperf" in feat:
        qualifier = " (Adj.)"
    elif "opp_mean" in feat:
        qualifier = " (vs Opp)"
    elif "rd1" in feat:
        qualifier = " (R1)"
    pretty = name.replace("_", " ").title()
    return f"{pretty}{qualifier} Diff"


@app.route("/api/model/feature_importance")
def feature_importance():
    """Return top features by absolute LR coefficient weight."""
    models = model_state.get("models")
    if models is None:
        return jsonify({"error": "Model not loaded"}), 503

    coef = models["lr"].coef_[0]
    features = models["feat_cols"]

    items = []
    for i, feat in enumerate(features):
        c = float(coef[i])
        items.append({
            "feature": feat,
            "display_name": _feat_display_name(feat),
            "coefficient": round(c, 5),
            "abs_weight": round(abs(c), 5),
            "direction": "positive" if c > 0 else "negative",
        })
    items.sort(key=lambda x: x["abs_weight"], reverse=True)
    return jsonify({"features": items[:25], "total_features": len(features)})


@app.route("/api/model/summary")
def model_summary():
    """Return model stats for the insights page."""
    models = model_state.get("models")
    if models is None:
        return jsonify({"error": "Model not loaded"}), 503

    feat_cols = models["feat_cols"]
    coef = models["lr"].coef_[0]
    active = int(sum(1 for c in coef if c != 0))

    # Categorize features
    def categorize(f):
        if 'elo' in f or 'peak' in f: return 'Elo Rating'
        if f in ['striking_matchup','grappling_matchup','wrestling_matchup',
                 'power_matchup','sub_matchup','style_distance']: return 'Style Matchup'
        if 'adjperf' in f: return 'Opponent-Adjusted Stats'
        if 'opp_mean' in f: return 'Opponent History'
        if f in ['age_diff','ufc_age_diff','days_since_last_fight_diff',
                 'WEIGHT_diff','age_dec_avg_diff','days_since_last_fight_f1']: return 'Demographics'
        if 'rd1' in f: return 'Round 1 Stats'
        if 'weightclass' in f or 'scheduled' in f: return 'Fight Context'
        return 'Fight Stats'

    cats = {}
    for i, f in enumerate(feat_cols):
        cat = categorize(f)
        cats.setdefault(cat, {"active": 0, "total": 0})
        cats[cat]["total"] += 1
        if coef[i] != 0:
            cats[cat]["active"] += 1

    categories = [{"name": k, "active": v["active"], "total": v["total"]}
                  for k, v in sorted(cats.items(), key=lambda x: x[1]["active"], reverse=True)]

    # Elo rankings
    elo_rankings = []
    elo_ratings = model_state.get("elo_ratings", {})
    elo_extra = model_state.get("elo_extra", {})
    if elo_ratings:
        ranked = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, elo in ranked:
            peak = (elo_extra or {}).get("peak_elo", {}).get(name, elo)
            elo_rankings.append({
                "name": name,
                "display_name": _format_fighter_name(name),
                "elo": round(elo),
                "peak": round(peak),
            })

    return jsonify({
        "total_features": len(feat_cols),
        "active_features": active,
        "zeroed_features": len(feat_cols) - active,
        "training_fights": len(model_state.get("df_trained", [])),
        "categories": categories,
        "elo_rankings": elo_rankings,
        "metrics": {
            "accuracy": "69.9%",
            "log_loss": "0.5976",
            "auc": "0.7419",
            "test_fights": 408,
        },
        "confidence_tiers": [
            {"range": "50-55%", "label": "Toss-up", "fights": 50, "accuracy": "56%"},
            {"range": "55-65%", "label": "Lean", "fights": 144, "accuracy": "59%"},
            {"range": "65-75%", "label": "Confident", "fights": 114, "accuracy": "78%"},
            {"range": "75%+", "label": "Strong pick", "fights": 100, "accuracy": "83%"},
        ],
    })


@app.route("/api/fighters/compare_zscores", methods=["POST"])
def compare_zscores():
    """Return z-scores for two fighters side by side."""
    data = request.get_json()
    if not data or "fighter_a" not in data or "fighter_b" not in data:
        return jsonify({"error": "Provide fighter_a and fighter_b"}), 400

    baselines = model_state.get("wc_baselines")
    if not baselines:
        return jsonify({"error": "Baselines not computed"}), 503

    conn = sqlite3.connect(DB_PATH)
    result = {}

    for key in ["fighter_a", "fighter_b"]:
        name = data[key]
        # Get weightindex
        wi_row = pd.read_sql_query(
            "SELECT weightindex FROM final_features_fast WHERE jfighter = ? ORDER BY DATE DESC LIMIT 1",
            conn, params=(name,),
        )
        if wi_row.empty:
            result[key] = {"error": f"Not found: {name}", "categories": {}}
            continue
        wi = int(wi_row.iloc[0]["weightindex"])
        wc_bl = baselines.get(wi, {})

        # Collect stats from all tables
        table_cols = {}
        for stats in ZSCORE_STAT_CONFIG.values():
            for s in stats:
                table_cols.setdefault(s["table"], []).append(s["col"])

        fighter_vals = {}
        for table, cols in table_cols.items():
            cols_str = ", ".join(cols)
            try:
                row = pd.read_sql_query(
                    f"SELECT {cols_str} FROM {table} WHERE jfighter = ? ORDER BY DATE DESC LIMIT 1",
                    conn, params=(name,),
                )
                if not row.empty:
                    for col in cols:
                        val = row.iloc[0].get(col)
                        if val is not None and pd.notna(val):
                            fighter_vals[col] = float(val)
            except Exception:
                pass

        # Build categories
        categories = {}
        for cat_name, stats in ZSCORE_STAT_CONFIG.items():
            cat_stats = []
            z_sum, z_count = 0.0, 0
            for s in stats:
                col = s["col"]
                raw = fighter_vals.get(col)
                bl = wc_bl.get(col)
                if raw is not None and bl:
                    z = (raw - bl["mean"]) / bl["std"]
                    if s["inverted"]:
                        z = -z
                    z = max(-3.0, min(3.0, z))
                    pct = round(_norm_cdf(z) * 100, 1)
                    z_sum += z
                    z_count += 1
                    cat_stats.append({
                        "key": col, "display_name": s["name"],
                        "z_score": round(z, 2), "percentile": pct,
                    })
                else:
                    cat_stats.append({
                        "key": col, "display_name": s["name"],
                        "z_score": None, "percentile": None,
                    })
            categories[cat_name] = {
                "avg_z": round(z_sum / z_count, 2) if z_count > 0 else None,
                "stats": cat_stats,
            }
        result[key] = {"jfighter": name, "weightindex": wi, "categories": categories}

    conn.close()
    return jsonify(result)


# ── Startup ──────────────────────────────────────────────────────────────────
# Train model on import so gunicorn workers have it ready
init_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
