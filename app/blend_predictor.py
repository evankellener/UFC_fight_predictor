"""Blend predictor for the Flask app: LR (elastic net) + XGBoost ensemble.

Loads artifacts from app/models/blend/ saved by scripts/train_and_save_blend.py,
and exposes a predict_fight(f1, f2, date) interface matching the existing
UFCFightPredictor contract.

For live prediction the market / SoS / interaction features default to 0 when
event context (location, card position) is unavailable. This matches how the
training pipeline handles missing rows (fillna(0)).
"""
import json
import os
import pickle
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

APP_DIR  = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
sys.path.insert(0, str(REPO_DIR / "src"))

BLEND_DIR = APP_DIR / "models" / "blend"

# Prefer slim (shipped to deployment) sqlite; fall back to full scraper DB locally.
_DB_CANDIDATES = [
    REPO_DIR / "data" / "sqlite_db" / "slim_scrapper.db",
    REPO_DIR / "data" / "sqlite_db" / "sqlite_scrapper.db",
]
SQLITE_DB = next((p for p in _DB_CANDIDATES if p.exists()), _DB_CANDIDATES[0])


class BlendPredictor:
    """LR + XGB 50/50 blend with ~68% walk-forward accuracy and +4% past-year ROI."""

    def __init__(self, blend_dir: Path = BLEND_DIR, verbose: bool = True):
        self.blend_dir = Path(blend_dir)
        if verbose:
            print(f"[BlendPredictor] loading artifacts from {self.blend_dir}")

        with open(self.blend_dir / "lr.pkl", "rb") as f:
            self.lr = pickle.load(f)
        with open(self.blend_dir / "lr_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        self.xgb = XGBClassifier()
        self.xgb.load_model(str(self.blend_dir / "xgb.json"))

        with open(self.blend_dir / "feat_lists.json") as f:
            lists = json.load(f)
        self.lr_cols  = lists["lr_cols"]
        self.xgb_cols = lists["xgb_cols"]
        self.blend_w  = float(lists.get("blend_weight_xgb", 0.5))

        # Per-fighter latest row (for lookup + display)
        self.latest = pd.read_csv(
            self.blend_dir / "fighter_latest_row.csv",
            low_memory=False,
        )
        self.latest["DATE"] = pd.to_datetime(self.latest["DATE"], errors="coerce")

        # Full training frame (for historical matchup lookup)
        self.train_df = pd.read_csv(
            self.blend_dir / "training_df.csv", low_memory=False
        )
        self.train_df["DATE"] = pd.to_datetime(self.train_df["DATE"], errors="coerce")

        # Per-fighter absolute stats (last_fight_date, dob, stance, SoS, current_elo)
        abs_path = self.blend_dir / "fighter_abs_stats.json"
        if abs_path.exists():
            with open(abs_path) as f:
                self.abs_stats = json.load(f)
        else:
            self.abs_stats = {}
            if verbose:
                print("[BlendPredictor] WARN: fighter_abs_stats.json not found — "
                      "as-of-date recomputation disabled. Run scripts/build_fighter_abs_stats.py")

        if verbose:
            print(
                f"[BlendPredictor] ready: {len(self.latest)} fighters indexed, "
                f"{len(self.lr_cols)} LR cols, {len(self.xgb_cols)} XGB cols"
            )

    # ── public helpers ──────────────────────────────────────────────────────

    def get_available_fighters(self):
        """Return sorted list of fighters we have any record of."""
        return sorted(self.latest["jfighter"].dropna().unique().tolist())

    def get_fighter_stats(self, fighter_name):
        """Return a fighter profile dict: record, current Elo, last fight, stance, etc.

        Source of truth: the rescraped sqlite (`ufc_winlossko` has per-bout W/L).
        Fast enough to hit on every profile request (one SELECT per call).
        """
        jf = self._to_jfighter(fighter_name)
        abs_ = self.abs_stats.get(jf, {})

        # Pull fight-by-fight record from the authoritative sqlite table.
        try:
            with sqlite3.connect(str(SQLITE_DB)) as c:
                rows = c.execute(
                    "SELECT DATE, win, loss, BOUT, jbout "
                    "FROM ufc_winlossko WHERE jfighter = ? ORDER BY DATE",
                    (jf,),
                ).fetchall()
        except Exception as e:
            rows = []

        wins = sum(1 for r in rows if int(r[1] or 0) == 1)
        losses = sum(1 for r in rows if int(r[2] or 0) == 1)
        total = len(rows)
        draws_nc = total - wins - losses
        last_date = rows[-1][0] if rows else abs_.get("last_fight_date")

        stats = {
            "fighter": fighter_name,
            "jfighter": jf,
            "record": f"{wins}-{losses}" + (f"-{draws_nc}" if draws_nc > 0 else ""),
            "wins": wins,
            "losses": losses,
            "draws_nc": draws_nc,
            "total_ufc_fights": total,
            "last_fight_date": last_date,
            "current_elo": round(abs_.get("current_elo", 0.0), 1),
            "peak_elo": None,  # filled below if available
            "stance": abs_.get("stance", "").title() or None,
            "nationality": abs_.get("nationality"),
            "weightindex": abs_.get("weightindex"),
            "dob": abs_.get("dob"),
            "sos_last5": round(abs_.get("sos_last5", 0.0), 1),
            "form_winrate5": round(abs_.get("form_winrate5", 0.0), 3),
        }
        return stats if (total or abs_) else None

    def get_fighter_fights(self, fighter_name):
        """Return chronological fight history from the authoritative sqlite table.

        One row per fight: date, opponent, result, event, method-ish fields.
        """
        jf = self._to_jfighter(fighter_name)
        try:
            with sqlite3.connect(str(SQLITE_DB)) as c:
                rows = c.execute(
                    "SELECT DATE, EVENT, BOUT, win, loss, ko, kod, subw, subwd, "
                    "udec, udecd, mdec, mdecd, sdec, sdecd, fight_time_minutes "
                    "FROM ufc_winlossko WHERE jfighter = ? ORDER BY DATE DESC",
                    (jf,),
                ).fetchall()
        except Exception:
            return []

        fights = []
        for r in rows:
            (date, event, bout, win, loss, ko, kod, subw, subwd,
             udec, udecd, mdec, mdecd, sdec, sdecd, dur) = r
            # Parse opponent from BOUT string "X vs. Y"
            parts = [p.strip() for p in (bout or "").replace("  ", " ").split("vs.")]
            me_name = fighter_name.replace(" ", "").lower()
            opp = (parts[1] if len(parts) == 2 and parts[0].replace(" ", "").lower() == me_name
                   else parts[0] if len(parts) == 2 else "Unknown")
            # Method (check both won-by and lost-by flags)
            if int(ko or 0) or int(kod or 0):       method = "KO/TKO"
            elif int(subw or 0) or int(subwd or 0): method = "SUB"
            elif int(udec or 0) or int(udecd or 0): method = "U-DEC"
            elif int(mdec or 0) or int(mdecd or 0): method = "M-DEC"
            elif int(sdec or 0) or int(sdecd or 0): method = "S-DEC"
            else:                                   method = "DEC/Other"
            fights.append({
                "date": date,
                "event": event or "Unknown Event",
                "opponent": opp,
                "result": "Win" if int(win or 0) == 1 else ("Loss" if int(loss or 0) == 1 else "NC/Draw"),
                "method": method,
                "duration_min": round(float(dur or 0), 2),
            })
        return fights

    def predict_fight(self, fighter1_name, fighter2_name, fight_date=None):
        """Predict win probability for fighter1 beating fighter2.

        Returns a dict with keys:
          success, fighter1, fighter2, prob_f1, prob_f2, predicted_winner,
          lr_prob, xgb_prob, blend_prob, method.
        """
        try:
            jf1 = self._to_jfighter(fighter1_name)
            jf2 = self._to_jfighter(fighter2_name)

            # 1) Try historical matchup lookup (most faithful: full feature set)
            row = self._find_historical_row(jf1, jf2, fight_date)
            method = "historical"

            if row is None:
                # 2) Live: build diff row from per-fighter latest known stats
                row = self._build_live_row(jf1, jf2, event_date=fight_date)
                method = "live (as-of-date)"

            if row is None:
                return {
                    "success": False,
                    "error": f"One or both fighters unknown: {fighter1_name}, {fighter2_name}",
                }

            # Align to the column orders the models expect
            lr_X  = self.scaler.transform([[row.get(c, 0.0) for c in self.lr_cols]])
            xgb_X = np.array([[row.get(c, 0.0) for c in self.xgb_cols]], dtype=float)

            p_lr  = float(self.lr.predict_proba(lr_X)[0, 1])
            p_xgb = float(self.xgb.predict_proba(xgb_X)[0, 1])
            p_bl  = self.blend_w * p_xgb + (1 - self.blend_w) * p_lr

            # Caller asked about fighter1_name.  The training convention is that
            # "jfighter" is whoever's perspective the row is from.  In both modes
            # we built the row so that jfighter == jf1, so p_bl is P(jf1 wins).
            winner = fighter1_name if p_bl >= 0.5 else fighter2_name
            return {
                "success": True,
                "fighter1": fighter1_name,
                "fighter2": fighter2_name,
                "fighter1_win_probability": round(p_bl, 3),
                "fighter2_win_probability": round(1 - p_bl, 3),
                "predicted_winner": winner,
                "confidence": round(float(max(p_bl, 1 - p_bl)), 3),
                "prob_f1": p_bl,
                "prob_f2": 1 - p_bl,
                "lr_prob":    round(p_lr, 3),
                "xgb_prob":   round(p_xgb, 3),
                "blend_prob": round(p_bl, 3),
                "method":     method,
                "model":      "LR+XGB blend",
            }

        except Exception as e:
            return {"success": False, "error": f"Prediction failed: {e}"}

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_jfighter(name):
        """Convert 'Jon Jones' -> 'JonJones' to match training convention."""
        return "".join(str(name).split())

    def _find_historical_row(self, jf1, jf2, fight_date):
        """Return the training row for jf1 (as jfighter) vs jf2, or None."""
        d = self.train_df
        mask = (d["jfighter"] == jf1) & (d["opp_jfighter"] == jf2)
        if fight_date:
            try:
                fd = pd.to_datetime(fight_date)
                mask &= (d["DATE"] == fd)
            except Exception:
                pass
        hits = d[mask]
        if hits.empty:
            # flip perspective and negate diffs? simpler: try opposite and invert prob later (not done here)
            mask2 = (d["jfighter"] == jf2) & (d["opp_jfighter"] == jf1)
            if fight_date:
                try:
                    fd = pd.to_datetime(fight_date)
                    mask2 &= (d["DATE"] == fd)
                except Exception:
                    pass
            hits = d[mask2]
            if hits.empty:
                return None
            # Invert diffs (row is from f2 perspective)
            r = hits.iloc[-1].to_dict()
            for c in self.xgb_cols:
                if c.endswith("_diff") or c.startswith("ix_") or c in ("home_advantage_diff",):
                    r[c] = -r.get(c, 0.0)
            # probabilistic: flip win too
            if "win" in r:
                r["win"] = 1 - int(r["win"])
            return r
        return hits.iloc[-1].to_dict()

    def _build_live_row(self, jf1, jf2, event_date=None):
        """Construct a feature row for a (f1, f2, event_date) matchup.

        Two-stage build:
          (1) Diff each fighter's latest snapshotted values to approximate
              decayed rolling stats (valid as long as neither has fought since
              the last artifact rebuild).
          (2) Override the calendar-dependent features with values recomputed
              as-of `event_date`: age, days-since-last-fight, SoS diffs,
              stance mismatch, and all interactions that depend on them.
        """
        r1 = self.latest[self.latest["jfighter"] == jf1]
        r2 = self.latest[self.latest["jfighter"] == jf2]
        if r1.empty or r2.empty:
            return None
        r1 = r1.iloc[0]; r2 = r2.iloc[0]

        # Stage 1: baseline diff from snapshots
        out = {}
        for c in set(self.lr_cols + self.xgb_cols):
            v1 = r1.get(c, 0.0); v2 = r2.get(c, 0.0)
            try:
                v1 = float(v1); v2 = float(v2)
            except (TypeError, ValueError):
                v1, v2 = 0.0, 0.0
            if np.isnan(v1) or np.isinf(v1): v1 = 0.0
            if np.isnan(v2) or np.isinf(v2): v2 = 0.0
            if c.endswith("_diff") or c.startswith("ix_"):
                out[c] = 0.5 * (v1 - v2)
            else:
                out[c] = 0.5 * (v1 + v2)

        # Stage 2: override calendar-dependent features if we have abs_stats + date
        self._apply_as_of_date_overrides(out, jf1, jf2, event_date)
        return out

    def _apply_as_of_date_overrides(self, row, jf1, jf2, event_date):
        """Recompute calendar-dependent features at `event_date` using abs_stats."""
        a1 = self.abs_stats.get(jf1); a2 = self.abs_stats.get(jf2)
        if not a1 or not a2:
            return  # abs-stats missing → keep snapshot values

        try:
            evt = pd.to_datetime(event_date) if event_date else pd.Timestamp.today().normalize()
        except Exception:
            evt = pd.Timestamp.today().normalize()

        # Days-since-last-fight, computed as of event date
        def days_since(abs_stats):
            try:
                last = pd.to_datetime(abs_stats["last_fight_date"])
                return max(0, (evt - last).days)
            except Exception:
                return 0

        layoff1 = days_since(a1); layoff2 = days_since(a2)
        row["days_since_last_fight_diff"] = float(layoff1 - layoff2)
        # keep the f1-only feature if the model uses it
        if "days_since_last_fight_f1" in row:
            row["days_since_last_fight_f1"] = float(layoff1)

        # Age at event date
        def age_on(abs_stats):
            try:
                dob = pd.to_datetime(abs_stats["dob"])
                return max(0.0, (evt - dob).days / 365.25)
            except Exception:
                return 0.0

        age1 = age_on(a1); age2 = age_on(a2)
        if age1 and age2:
            row["age_diff"] = float(age1 - age2)
            row["age_ratio_diff"] = float((age1 / age2) - (age2 / age1))
            # Gaussian age-peak (same formula as src/predict_event.compute_age_prime)
            import math
            def age_prime(a, peak=27.0, width=3.0):
                return math.exp(-((a - peak) ** 2) / (2 * width ** 2))
            row["age_prime_diff"] = age_prime(age1) - age_prime(age2) if "age_prime_diff" in row else row.get("age_prime_diff", 0.0)

        # Strength-of-schedule / form / trajectory (straight diffs of abs values)
        sos_map = {
            "sos_last3_diff":      ("sos_last3",      0.0),
            "sos_last5_diff":      ("sos_last5",      0.0),
            "sos_trajectory_diff": ("sos_trajectory", 0.0),
            "form_winrate3_diff":  ("form_winrate3",  0.0),
            "form_winrate5_diff":  ("form_winrate5",  0.0),
            "elo_trajectory_diff": ("elo_trajectory", 0.0),
            "career_fights_diff":  ("career_fights",  0),
        }
        for diff_col, (abs_key, default) in sos_map.items():
            row[diff_col] = float(a1.get(abs_key, default) - a2.get(abs_key, default))

        # Stance mismatch + southpaw advantage (from stance strings)
        def stance_code(s):
            s = (s or "").lower()
            return {"orthodox": 1, "southpaw": 2, "switch": 3}.get(s, 0)

        s1 = stance_code(a1.get("stance")); s2 = stance_code(a2.get("stance"))
        row["stance_mismatch"] = int(s1 != s2 and s1 > 0 and s2 > 0)
        row["southpaw_advantage_diff"] = int(s1 == 2) - int(s2 == 2)

        # Recompute interactions that depend on features we just refreshed
        g = lambda k: row.get(k, 0.0)
        row["ix_age_x_elo"]        = g("age_diff") * g("elo_win_prob")
        row["ix_age_x_streak"]     = g("age_diff") * g("win_streak_entering_diff")
        row["ix_elo_x_streak"]     = g("precomp_elo_diff") * g("win_streak_entering_diff")
        row["ix_age_x_fights12m"]  = g("age_diff") * g("fights_last_12m_diff")
        row["ix_reach_x_stance"]   = g("reach_ratio_diff") * g("stance_mismatch")
        row["ix_elo_x_layoff"]     = g("precomp_elo_diff") * g("days_since_last_fight_diff")
        row["ix_age_x_layoff"]     = g("age_diff") * g("days_since_last_fight_diff")
        row["ix_home_x_main"]      = g("home_advantage_diff") * g("is_main_event")
        row["ix_age_x_main"]       = g("age_diff") * g("is_main_event")
        row["ix_elo_x_age_ratio"]  = g("elo_win_prob") * g("age_ratio_diff")
        row["ix_elo_x_card"]       = g("elo_win_prob") * g("card_position_norm_career_diff")
        row["ix_sos_x_age"]        = g("sos_last5_diff") * g("age_diff")
        row["ix_sos_x_elo"]        = g("sos_last5_diff") * g("elo_win_prob")
        row["ix_form_x_layoff"]    = g("form_winrate5_diff") * g("days_since_last_fight_diff")
        row["ix_traj_x_age"]       = g("elo_trajectory_diff") * g("age_diff")


# ── Smoke test when run directly ────────────────────────────────────────────
if __name__ == "__main__":
    bp = BlendPredictor()
    pairs = [
        ("Jon Jones", "Stipe Miocic"),
        ("Islam Makhachev", "Arman Tsarukyan"),
        ("Ilia Topuria", "Max Holloway"),
    ]
    for a, b in pairs:
        r = bp.predict_fight(a, b)
        if r["success"]:
            print(f"{a:<22} vs {b:<22}  p({a})={r['blend_prob']:.3f}  "
                  f"winner={r['predicted_winner']} ({r['method']})")
        else:
            print(f"{a} vs {b}: ERROR — {r.get('error')}")
