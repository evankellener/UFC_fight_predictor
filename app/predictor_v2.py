"""Predictor v2 — clean inference path for the 202-feature final model.

Unlike the legacy `BlendPredictor`, this predictor:
  1. Works from per-fighter ABSOLUTE feature rows (not pre-diffed), so
     constructing `f1 - f2` diffs at inference is direct and symmetric.
  2. Computes all calendar-dependent features as-of the requested event_date.
  3. Computes Tier 1c recency directly from `ufc_winlossko`.
  4. Computes both UFC-only and expanded Elo from cached trajectories.
  5. Computes striking + grappling (Tier 2b) Elo from cached trajectories.

Feature construction matches `scripts/run_threshold_sweep_both_elos.py`
exactly — verified by `scripts/verify_predictor_v2.py`.

Artifacts expected under `app/models/blend_v2/` (built by
`scripts/build_predictor_v2_artifacts.py`):
  - fighter_mma_history.parquet  : per-fighter per-fight abs stats
  - fighter_elo_ufc.json         : UFC-only Elo trajectory
  - fighter_elo_exp.json         : Expanded Elo trajectory
  - fighter_style_elo.json       : striking + grappling Elo trajectory
  - fighter_bios.json            : DOB, stance, weightindex per fighter
  - lr.pkl, lr_scaler.pkl        : trained model + scaler
  - lr_imputer.pkl               : SimpleImputer (fitted on train)
  - feat_cols.json               : ordered feature list

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§11):
  §1  All lookups use `DATE < event_date` (strict) to find the most-recent
      prior state. Never the fight-at-event_date row itself.
  §2  Elo uses sigmoid decay from last_fight_date to event_date, like
      training. compute_elo's precomp_elo is BEFORE the fight by construction.
  §4  Scaler + imputer fitted on train only (loaded from disk; never re-fit).
  §6  LR hyperparams frozen in training artifact.
"""
from __future__ import annotations

import bisect
import json
import math
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
ARTIFACTS = APP_DIR / "models" / "blend_v2"

# Match training config
LOGISTIC_SCALE = 449.205
ELO_DECAY_MAX  = 0.25
ELO_DECAY_MID  = 730.0
ELO_DECAY_STEEP = 80.0
BASE_ELO = 1500.0

ELO_COLS_BASE = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
                 "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
TIER_1C = ["win_streak_entering_diff", "coming_off_loss_diff", "fights_last_12m_diff"]
STYLE_COLS = ["striking_elo_diff", "grappling_elo_diff"]

# Weight-class history features (docs/feature_wc_history.md)
WC_HIST_COLS = ["wc_native_winrate_diff", "wc_native_fights_diff",
                "wc_native_ko_rate_diff", "days_since_this_wc_diff"]
# Non-diff flag on the fight as a whole (symmetric under swap)
WC_PAIR_COLS = ["cross_division_flag"]
WC_SHRINK_ALPHA = 3.0     # Beta-Binomial shrinkage strength
LAM_WC = 0.13             # Decay per year (matches pipeline convention)
DAYS_SINCE_NEVER = 9999   # Sentinel when never fought at this wc


def _sigmoid_decay(days_inactive):
    """Sigmoid decay fraction applied to (elo - 1500).
    Matches compute_elo's decay: a fighter inactive for `days_inactive` has
    their rating partially pulled toward BASE_ELO.
    """
    if days_inactive <= 0:
        return 0.0
    # From src/elo_feature.py compute_elo: sigmoid(days) in [0, decay_max]
    return ELO_DECAY_MAX / (1.0 + math.exp(-(days_inactive - ELO_DECAY_MID) / ELO_DECAY_STEEP))


def _to_jf(name: str) -> str:
    """Strip spaces — match training convention (AlexPereira, not 'Alex Pereira')."""
    return "".join(str(name).split())


class PredictorV2:
    """Clean inference for the 202-feature LR model.

    Usage:
        p = PredictorV2()
        result = p.predict("Jon Jones", "Stipe Miocic", event_date="2024-11-16")
    """

    def __init__(self, artifact_dir: Path = ARTIFACTS, db_path: str | Path | None = None,
                 verbose: bool = True):
        self.dir = Path(artifact_dir)
        self.verbose = verbose

        # Trained model
        with open(self.dir / "lr.pkl", "rb") as f:
            self.lr = pickle.load(f)
        with open(self.dir / "lr_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(self.dir / "lr_imputer.pkl", "rb") as f:
            self.imputer = pickle.load(f)
        self.feat_cols: list[str] = json.loads((self.dir / "feat_cols.json").read_text())

        # Caches
        self.mma_hist = pd.read_parquet(self.dir / "fighter_mma_history.parquet")
        self.mma_hist["DATE"] = pd.to_datetime(self.mma_hist["DATE"])
        # Build index: jfighter → sorted DataFrame of their per-fight rows
        self.mma_by_fighter = {jf: grp.sort_values("DATE").reset_index(drop=True)
                                for jf, grp in self.mma_hist.groupby("jfighter")}

        self.elo_ufc = json.loads((self.dir / "fighter_elo_ufc.json").read_text())
        self.elo_exp = json.loads((self.dir / "fighter_elo_exp.json").read_text())
        self.style   = json.loads((self.dir / "fighter_style_elo.json").read_text())
        self.bios    = json.loads((self.dir / "fighter_bios.json").read_text())

        # DB path for recency lookup (ufc_winlossko + event dates)
        self.db_path = str(db_path) if db_path else str(
            REPO_DIR / "data" / "sqlite_db" / "sqlite_scrapper.db"
        )
        # Prefer pre-computed winlossko cache (JSON artifact) over live DB query.
        # The full DB (sqlite_scrapper.db) is gitignored; slim_scrapper.db on
        # Render doesn't have ufc_winlossko. Fall back to DB if cache missing.
        wl_json = self.dir / "fighter_winlossko.json"
        if wl_json.exists():
            raw = json.loads(wl_json.read_text())
            # Rehydrate: {jf: [[iso_date_str, win_int], ...]} → {jf: [(np.datetime64, int), ...]}
            self._wl_history = {
                jf: [(np.datetime64(d), int(w)) for d, w in entries]
                for jf, entries in raw.items()
            }
            if self.verbose:
                print(f"[PredictorV2] loaded winlossko cache ({len(self._wl_history)} fighters)")
        else:
            # Fallback: live DB query (won't work on Render with slim DB)
            self._wl_history = self._load_winlossko_history()

        # Weight-class history cache (docs/feature_wc_history.md)
        # {jf: [(np.datetime64, weightindex_int, win_int, ko_int), ...]}
        wc_json = self.dir / "fighter_wc_history.json"
        if wc_json.exists():
            raw = json.loads(wc_json.read_text())
            self._wc_history = {
                jf: [(np.datetime64(d), int(wc), int(w), int(k))
                     for d, wc, w, k in entries]
                for jf, entries in raw.items()
            }
            if self.verbose:
                print(f"[PredictorV2] loaded wc history cache ({len(self._wc_history)} fighters)")
        else:
            self._wc_history = {}
            if self.verbose:
                print(f"[PredictorV2] no wc history cache found — wc features will be 0/neutral")

        if self.verbose:
            print(f"[PredictorV2] loaded {len(self.feat_cols)} features, "
                  f"{len(self.mma_by_fighter)} fighters with MMA history")

    # ── Public API ──────────────────────────────────────────────────────
    def predict(self, fighter_a: str, fighter_b: str, event_date: str | pd.Timestamp,
                scheduled_rounds: int = 3, symmetrize: bool = True) -> dict:
        """Predict P(fighter_a wins) on given event_date.

        symmetrize=True (default) averages both argument orderings to eliminate
        asymmetry from features that depend on f1 specifically
        (days_since_last_fight_f1, age_ratio_diff). Training used a fixed
        orientation (red-corner=f1); at inference the caller provides arbitrary
        order, so without symmetrization p(A|a,b) != 1-p(B|b,a).

        Returns:
            {
              "fighter_a": str, "fighter_b": str,
              "p_fighter_a": float, "p_fighter_b": float,
              "predicted_winner": str, "confidence": float,
              "feature_row": dict (orientation A, for debugging),
              "n_features_filled": int,
              "p_orient_a": float (raw p(a) with a=f1),
              "p_orient_b": float (1 - p(b=f1))
            }
        """
        # Oriented prediction (raw, without symmetrization)
        r_a = self._predict_oriented(fighter_a, fighter_b, event_date, scheduled_rounds)
        if not r_a.get("success"):
            return r_a

        if not symmetrize:
            return r_a

        # Flip and combine
        r_b = self._predict_oriented(fighter_b, fighter_a, event_date, scheduled_rounds)
        if not r_b.get("success"):
            return r_a  # fall back to one-sided if flip fails

        p_a_orient_a = r_a["p_fighter_a"]       # p(a) with a=f1
        p_a_orient_b = 1.0 - r_b["p_fighter_a"] # p(a) with b=f1
        p_a = 0.5 * (p_a_orient_a + p_a_orient_b)
        p_b = 1.0 - p_a

        return {
            "success": True,
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "p_fighter_a": round(p_a, 4),
            "p_fighter_b": round(p_b, 4),
            "predicted_winner": fighter_a if p_a >= 0.5 else fighter_b,
            "confidence": round(max(p_a, p_b), 4),
            "event_date": r_a["event_date"],
            "feature_row": r_a["feature_row"],
            "n_features_filled": r_a["n_features_filled"],
            "p_orient_a": round(p_a_orient_a, 4),
            "p_orient_b": round(p_a_orient_b, 4),
            "asymmetry_pp": round(abs(p_a_orient_a - p_a_orient_b) * 100, 2),
        }

    def _predict_oriented(self, fighter_a: str, fighter_b: str,
                          event_date: str | pd.Timestamp,
                          scheduled_rounds: int = 3) -> dict:
        """Single-orientation prediction (treats fighter_a as f1). Used internally
        by the symmetrizing predict()."""
        jf_a = _to_jf(fighter_a); jf_b = _to_jf(fighter_b)
        evt_ts = pd.to_datetime(event_date)

        # Caller's fighter_a = f1 (matches training red-corner convention).
        jf1, jf2, flip = jf_a, jf_b, False

        row = self._build_feature_row(jf1, jf2, evt_ts, scheduled_rounds)
        if row is None:
            return {"success": False,
                    "error": f"Missing data for {fighter_a} or {fighter_b}"}

        X = np.array([[row.get(c, 0.0) for c in self.feat_cols]], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_imp = self.imputer.transform(X)
        X_s = self.scaler.transform(X_imp)
        p_f1 = float(self.lr.predict_proba(X_s)[0, 1])

        p_a = (1 - p_f1) if flip else p_f1
        p_b = 1 - p_a
        winner = fighter_a if p_a >= 0.5 else fighter_b
        return {
            "success": True,
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "p_fighter_a": round(p_a, 4),
            "p_fighter_b": round(p_b, 4),
            "predicted_winner": winner,
            "confidence": round(max(p_a, p_b), 4),
            "event_date": str(evt_ts.date()),
            "feature_row": row,
            "n_features_filled": sum(1 for c in self.feat_cols if c in row and row[c] != 0),
        }

    # ── Feature row construction ────────────────────────────────────────
    def _build_feature_row(self, jf1: str, jf2: str, evt_ts: pd.Timestamp,
                           scheduled_rounds: int) -> dict | None:
        """Build a feature row for (jf1, jf2, evt_ts) matching training exactly.

        Returns dict keyed on feature names, or None if either fighter missing.
        """
        # Get prior row for each fighter
        a1 = self._latest_prior_mma(jf1, evt_ts)
        a2 = self._latest_prior_mma(jf2, evt_ts)
        if a1 is None or a2 is None:
            return None

        row: dict = {}

        # ── MMA-AI diff features ──────────────────────────────────────
        # For each `_diff` feature in feat_cols, strip suffix and diff.
        for feat in self.feat_cols:
            if feat.endswith("_ufc") or feat.endswith("_exp"):
                continue
            if feat in TIER_1C or feat in STYLE_COLS:
                continue
            if feat in ("weightclass_encoded", "scheduled_rounds",
                        "days_since_last_fight_f1"):
                continue
            if feat.endswith("_diff"):
                base = feat[:-5]   # strip "_diff"
                v1 = a1.get(base, 0.0) or 0.0
                v2 = a2.get(base, 0.0) or 0.0
                try:
                    row[feat] = float(v1) - float(v2)
                except Exception:
                    row[feat] = 0.0

        # ── Calendar-dependent overrides ──────────────────────────────
        # Age at event date (overrides age_dec_avg_diff only if we want strict
        # training parity — but training used age_dec_avg, which is decayed-avg
        # of age per fight. a1[age_dec_avg] is the fighter's dec-avg-age at
        # their latest prior fight, already correct per training logic. Keep it.
        # We DO override `age_diff` (raw, current age) and `age_ratio_diff`.
        b1, b2 = self.bios.get(jf1, {}), self.bios.get(jf2, {})
        age1 = self._age_on(b1.get("dob"), evt_ts)
        age2 = self._age_on(b2.get("dob"), evt_ts)
        if "age_diff" in self.feat_cols and age1 and age2:
            row["age_diff"] = float(age1 - age2)
        if "age_ratio_diff" in self.feat_cols and age1 and age2:
            row["age_ratio_diff"] = float((age1 / age2) - 1.0)
        # Reach ratio from bios
        r1 = b1.get("reach"); r2 = b2.get("reach")
        if "reach_ratio_diff" in self.feat_cols and r1 and r2 and r2 > 0:
            row["reach_ratio_diff"] = float((r1 / r2) - 1.0)
        # Weight diff
        w1 = b1.get("weight"); w2 = b2.get("weight")
        if "WEIGHT_diff" in self.feat_cols and w1 is not None and w2 is not None:
            row["WEIGHT_diff"] = float(w1 - w2)

        # UFC age (years since debut) overrides ufc_age_diff
        if "ufc_age_diff" in self.feat_cols:
            u1 = self._ufc_age_on(jf1, evt_ts)
            u2 = self._ufc_age_on(jf2, evt_ts)
            row["ufc_age_diff"] = float(u1 - u2)

        # Days since last fight
        dsl1 = self._days_since_last_fight(jf1, evt_ts)
        dsl2 = self._days_since_last_fight(jf2, evt_ts)
        if "days_since_last_fight_diff" in self.feat_cols:
            row["days_since_last_fight_diff"] = float(dsl1 - dsl2)
        if "days_since_last_fight_f1" in self.feat_cols:
            row["days_since_last_fight_f1"] = float(dsl1)

        # ── Elo features (UFC-only source) ────────────────────────────
        elo1_ufc = self._elo_at(jf1, evt_ts, self.elo_ufc)
        elo2_ufc = self._elo_at(jf2, evt_ts, self.elo_ufc)
        self._fill_elo_features(row, "ufc", elo1_ufc, elo2_ufc)

        # ── Elo features (Expanded source) ────────────────────────────
        elo1_exp = self._elo_at(jf1, evt_ts, self.elo_exp)
        elo2_exp = self._elo_at(jf2, evt_ts, self.elo_exp)
        self._fill_elo_features(row, "exp", elo1_exp, elo2_exp)

        # ── Style Elos ────────────────────────────────────────────────
        s1 = self._style_at(jf1, evt_ts)
        s2 = self._style_at(jf2, evt_ts)
        if "striking_elo_diff" in self.feat_cols:
            row["striking_elo_diff"] = float(s1["striking"] - s2["striking"])
        if "grappling_elo_diff" in self.feat_cols:
            row["grappling_elo_diff"] = float(s1["grappling"] - s2["grappling"])

        # ── Tier 1c recency ───────────────────────────────────────────
        rec1 = self._recency_state(jf1, evt_ts)
        rec2 = self._recency_state(jf2, evt_ts)
        if "win_streak_entering_diff" in self.feat_cols:
            row["win_streak_entering_diff"] = float(rec1["streak"] - rec2["streak"])
        if "coming_off_loss_diff" in self.feat_cols:
            row["coming_off_loss_diff"] = float(rec1["coming_off_loss"] - rec2["coming_off_loss"])
        if "fights_last_12m_diff" in self.feat_cols:
            row["fights_last_12m_diff"] = float(rec1["fights_12m"] - rec2["fights_12m"])

        # ── Static features ──────────────────────────────────────────
        if "weightclass_encoded" in self.feat_cols:
            # Use the most recent known weightindex for f1 (training convention
            # took fighter1's weightindex).
            _wc = a1.get("weightindex", 0)
            try:
                row["weightclass_encoded"] = 0.0 if _wc is None or (
                    isinstance(_wc, float) and math.isnan(_wc)) else float(_wc)
            except (TypeError, ValueError):
                row["weightclass_encoded"] = 0.0
        if "scheduled_rounds" in self.feat_cols:
            row["scheduled_rounds"] = float(scheduled_rounds)

        # ── Weight-class history features (docs/feature_wc_history.md) ───
        # Infer the fight's weight class from f1's most-recent division
        # (same convention as weightclass_encoded). Default 0 if unknown.
        raw_wc = a1.get("weightindex", 0)
        try:
            current_wc = 0 if raw_wc is None or (isinstance(raw_wc, float) and math.isnan(raw_wc)) else int(raw_wc)
        except (TypeError, ValueError):
            current_wc = 0
        wc1 = self._wc_history_state(jf1, evt_ts, current_wc)
        wc2 = self._wc_history_state(jf2, evt_ts, current_wc)
        if "wc_native_winrate_diff" in self.feat_cols:
            row["wc_native_winrate_diff"] = float(wc1["winrate"] - wc2["winrate"])
        if "wc_native_fights_diff" in self.feat_cols:
            row["wc_native_fights_diff"] = float(wc1["fights"] - wc2["fights"])
        if "wc_native_ko_rate_diff" in self.feat_cols:
            row["wc_native_ko_rate_diff"] = float(wc1["ko_rate"] - wc2["ko_rate"])
        if "days_since_this_wc_diff" in self.feat_cols:
            row["days_since_this_wc_diff"] = float(wc1["days_since"] - wc2["days_since"])
        if "cross_division_flag" in self.feat_cols:
            # 1 if either fighter's most-recent-prior-division != current_wc
            crossed = int((wc1["modal_wc"] not in (0, current_wc))
                          or (wc2["modal_wc"] not in (0, current_wc)))
            row["cross_division_flag"] = float(crossed)

        # Fill any missing cols with 0 (imputer will handle if still NaN)
        for c in self.feat_cols:
            if c not in row:
                row[c] = 0.0

        return row

    # ── Internal: per-fighter state-at-date lookups ─────────────────────
    def _latest_prior_mma(self, jf: str, evt_ts: pd.Timestamp) -> dict | None:
        """Row for jf with DATE <= evt_ts (prefer same-date row if fight is in history).

        Why `<=` not `<`: each history row's dec_avg features are computed from
        STRICTLY-PRIOR fights by the pipeline. The row AT date T already reflects
        "fighter's state going INTO the fight on T" — exactly what we need to
        match training. For novel future fights, there's no same-date row and
        this falls back to the most recent prior row (slightly stale approximation).
        """
        frame = self.mma_by_fighter.get(jf)
        if frame is None or len(frame) == 0:
            return None
        prior = frame[frame["DATE"] <= evt_ts]
        if len(prior) == 0:
            return None
        return prior.iloc[-1].to_dict()

    def _elo_at(self, jf: str, evt_ts: pd.Timestamp, elo_cache: dict) -> dict:
        """Fighter's 5 Elo features as-of event date.

        Two cases:
          (A) If evt_ts matches a fight in the trajectory (same date), use that
              fight's own precomp_elo — it already reflects training's decay.
              No additional decay.
          (B) If evt_ts is AFTER all known fights (novel future prediction),
              use final_state + sigmoid decay from last fight to evt_ts.

        This mirrors the training pipeline: precomp is the rating the model
        USED for that fight, including any inactivity decay baked in.
        """
        traj = elo_cache.get("trajectory", {}).get(jf, [])
        final = elo_cache.get("final_state", {}).get(jf, {})
        evt_str = str(evt_ts.date())

        # Case A: event_date matches a training fight — use its precomp directly
        exact = [e for e in traj if e["date"] == evt_str]
        if exact:
            e = exact[0]
            return dict(
                precomp_elo=float(e["precomp_elo"]),
                elo_momentum=float(e["elo_momentum"]),
                peak_elo=float(e["peak_elo"]),
                avg_opp_elo=float(e["avg_opp_elo"]),
                elo_consist=float(e["elo_consist"]),
            )

        # Case B or pre-event lookup
        prior_entries = [e for e in traj if e["date"] < evt_str]
        if not prior_entries:
            # Debut — BASE_ELO, no decay (matches compute_elo default)
            return dict(precomp_elo=BASE_ELO, elo_momentum=0.0,
                        peak_elo=BASE_ELO, avg_opp_elo=BASE_ELO, elo_consist=0.0)

        last = prior_entries[-1]
        last_fight_date = pd.to_datetime(last["date"])
        # POST-last-fight rating = precomp of the NEXT trajectory entry, or final_state
        next_traj = [e for e in traj if e["date"] > evt_str]
        if next_traj:
            # There's a future fight > evt_ts in history. Use ITS precomp, which
            # reflects all training-time decay up through last_fight_date → next_fight.
            # This is only approximate for evt_ts between the two; training-time
            # decay was computed for (next_fight - last_fight_date), not (evt_ts - last_fight_date).
            post_elo = next_traj[0]["precomp_elo"]
            post_momentum = next_traj[0]["elo_momentum"]
            post_peak = next_traj[0]["peak_elo"]
            post_avg_opp = next_traj[0]["avg_opp_elo"]
            post_consist = next_traj[0]["elo_consist"]
            # Don't re-decay — next_traj's precomp already has decay baked in.
            return dict(precomp_elo=float(post_elo),
                        elo_momentum=float(post_momentum),
                        peak_elo=float(post_peak),
                        avg_opp_elo=float(post_avg_opp),
                        elo_consist=float(post_consist))
        # Truly novel — evt_ts is after all known fights. Apply decay manually.
        post_elo = final.get("current_elo", BASE_ELO)
        days_inactive = (evt_ts - last_fight_date).days
        decay_frac = _sigmoid_decay(days_inactive)
        decayed_elo = post_elo - decay_frac * (post_elo - BASE_ELO)
        return dict(
            precomp_elo=float(decayed_elo),
            elo_momentum=float(last["elo_momentum"]),
            peak_elo=float(final.get("peak_elo", post_elo)),
            avg_opp_elo=float(last["avg_opp_elo"]),
            elo_consist=float(last["elo_consist"]),
        )

    def _fill_elo_features(self, row: dict, suffix: str,
                           e1: dict, e2: dict) -> None:
        """Populate 6 Elo diff features with a given suffix (_ufc or _exp)."""
        for c in ELO_COLS_BASE:
            feat = f"{c}_{suffix}"
            if feat not in self.feat_cols:
                continue
            if c == "elo_win_prob":
                diff = e1["precomp_elo"] - e2["precomp_elo"]
                row[feat] = 1.0 / (1.0 + 10 ** (-diff / LOGISTIC_SCALE))
            elif c == "precomp_elo_diff":
                row[feat] = e1["precomp_elo"] - e2["precomp_elo"]
            elif c == "elo_momentum_diff":
                row[feat] = e1["elo_momentum"] - e2["elo_momentum"]
            elif c == "peak_elo_diff":
                row[feat] = e1["peak_elo"] - e2["peak_elo"]
            elif c == "avg_opp_elo_diff":
                row[feat] = e1["avg_opp_elo"] - e2["avg_opp_elo"]
            elif c == "elo_consist_diff":
                row[feat] = e1["elo_consist"] - e2["elo_consist"]

    def _style_at(self, jf: str, evt_ts: pd.Timestamp) -> dict:
        """Striking + grappling Elo AS-OF event_date. Matches training: each
        trajectory entry is the pre-fight rating going INTO that fight.

        If evt_ts matches a fight in the trajectory, return that entry's rating
        (training-time value for that fight). Otherwise find the next-future entry
        (which also has correct post-all-prior-fights rating), or final_state.
        """
        traj = self.style.get("trajectory", {}).get(jf, [])
        final = self.style.get("final_state", {}).get(jf, {})
        evt_str = str(evt_ts.date())
        # Exact match (event is in training history)
        exact = [e for e in traj if e["date"] == evt_str]
        if exact:
            return dict(striking=float(exact[0]["striking"]),
                         grappling=float(exact[0]["grappling"]))
        prior = [e for e in traj if e["date"] < evt_str]
        if not prior:
            return dict(striking=BASE_ELO, grappling=BASE_ELO)
        next_entries = [e for e in traj if e["date"] > evt_str]
        if next_entries:
            return dict(striking=float(next_entries[0]["striking"]),
                         grappling=float(next_entries[0]["grappling"]))
        return dict(striking=float(final.get("striking", BASE_ELO)),
                     grappling=float(final.get("grappling", BASE_ELO)))

    # ── Recency (Tier 1c) from UFC winlossko ────────────────────────────
    def _load_winlossko_history(self) -> dict:
        """Returns {jfighter: [(date, won_int), ...] sorted ascending}."""
        conn = sqlite3.connect(self.db_path)
        hist = pd.read_sql(
            "SELECT w.jfighter, e.DATE, w.win FROM ufc_winlossko w "
            "JOIN ufc_event_details e ON e.jevent = w.jevent", conn)
        conn.close()
        hist["DATE"] = pd.to_datetime(hist["DATE"])
        hist = hist.sort_values(["jfighter", "DATE"]).dropna(subset=["DATE"])
        out = {}
        for jf, grp in hist.groupby("jfighter"):
            out[jf] = list(zip(grp["DATE"].values, grp["win"].astype(int).values))
        return out

    def _recency_state(self, jf: str, evt_ts: pd.Timestamp) -> dict:
        """Tier 1c recency features for jf as-of evt_ts (strict prior)."""
        h = self._wl_history.get(jf, [])
        evt64 = np.datetime64(evt_ts)
        prior = [(d, w) for (d, w) in h if d < evt64]
        if not prior:
            return dict(streak=0, coming_off_loss=0, fights_12m=0)
        # Streak: consecutive wins from most recent
        streak = 0
        for _, w in reversed(prior):
            if w == 1:
                streak += 1
            else:
                break
        coming_off_loss = 1 if prior[-1][1] == 0 else 0
        # Fights in last 365 days
        one_year = np.timedelta64(365, "D")
        fights_12m = sum(1 for d, _ in prior if (evt64 - d) <= one_year)
        return dict(streak=streak, coming_off_loss=coming_off_loss, fights_12m=fights_12m)

    # ── Bio-derived helpers ─────────────────────────────────────────────
    @staticmethod
    def _age_on(dob_str, evt_ts):
        if not dob_str: return 0.0
        try:
            dob = pd.to_datetime(dob_str)
            return max(0.0, (evt_ts - dob).days / 365.25)
        except Exception:
            return 0.0

    def _ufc_age_on(self, jf, evt_ts):
        """Years since debut UFC fight."""
        h = self._wl_history.get(jf, [])
        if not h: return 0.0
        evt64 = np.datetime64(evt_ts)
        prior = [d for d, _ in h if d < evt64]
        if not prior: return 0.0
        debut = pd.Timestamp(prior[0])
        return max(0.0, (evt_ts - debut).days / 365.25)

    def _days_since_last_fight(self, jf, evt_ts):
        h = self._wl_history.get(jf, [])
        evt64 = np.datetime64(evt_ts)
        prior = [d for d, _ in h if d < evt64]
        if not prior: return 365.0  # matches training default
        last = pd.Timestamp(prior[-1])
        return max(0.0, float((evt_ts - last).days))

    # ── Weight-class history (docs/feature_wc_history.md) ──────────────
    def _wc_history_state(self, jf: str, evt_ts: pd.Timestamp,
                          current_wc: int) -> dict:
        """Division-specific history for jf as-of evt_ts.

        Returns dict with:
          winrate    — Beta-shrunk decayed win rate at current_wc
          fights     — count of prior bouts at current_wc
          ko_rate    — decayed KO rate at current_wc (shrunk toward 0)
          days_since — days since last bout at current_wc (or DAYS_SINCE_NEVER)
          modal_wc   — most recent prior division (0 if no priors)
        """
        h = self._wc_history.get(jf, [])
        evt64 = np.datetime64(evt_ts)
        # All strictly-prior bouts (any division)
        prior_all = [(d, wc, w, k) for d, wc, w, k in h if d < evt64]
        if not prior_all:
            return dict(winrate=0.5, fights=0, ko_rate=0.0,
                        days_since=DAYS_SINCE_NEVER, modal_wc=0)

        # Modal (most-frequent) prior division — captures career-level home
        # division. Used for cross_division_flag. E.g. Spann has 12 LHW + 2 HW,
        # modal=LHW; a HW fight flags cross-division.
        from collections import Counter
        modal_wc = int(Counter(wc for _, wc, _, _ in prior_all).most_common(1)[0][0])

        # Filter to bouts at current_wc
        prior_wc = [(d, w, k) for d, wc, w, k in prior_all if wc == current_wc]
        if not prior_wc:
            return dict(winrate=0.5, fights=0, ko_rate=0.0,
                        days_since=DAYS_SINCE_NEVER, modal_wc=modal_wc)

        # Decay weights: exp(-λ * years_ago)
        evt_ts_pd = pd.Timestamp(evt_ts)
        weights = np.array([
            math.exp(-LAM_WC * (evt_ts_pd - pd.Timestamp(d)).days / 365.25)
            for d, _, _ in prior_wc
        ])
        wins = np.array([w for _, w, _ in prior_wc], dtype=float)
        kos = np.array([k for _, _, k in prior_wc], dtype=float)
        w_sum = float(weights.sum())

        if w_sum <= 0:
            return dict(winrate=0.5, fights=len(prior_wc), ko_rate=0.0,
                        days_since=DAYS_SINCE_NEVER, modal_wc=modal_wc)

        # Beta-Binomial shrinkage:
        #   winrate_smooth = (Σw*y + 0.5*α) / (Σw + α)
        decayed_win_num = float((weights * wins).sum())
        decayed_ko_num  = float((weights * kos).sum())
        winrate = (decayed_win_num + 0.5 * WC_SHRINK_ALPHA) / (w_sum + WC_SHRINK_ALPHA)
        ko_rate = decayed_ko_num / (w_sum + WC_SHRINK_ALPHA)  # prior 0 for KO

        last_at_wc = pd.Timestamp(prior_wc[-1][0])
        days_since = float((evt_ts_pd - last_at_wc).days)

        return dict(winrate=winrate, fights=len(prior_wc),
                    ko_rate=ko_rate, days_since=days_since,
                    modal_wc=modal_wc)


# ── Smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = PredictorV2()
    pairs = [
        ("Jon Jones", "Stipe Miocic", "2024-11-16"),
        ("Islam Makhachev", "Arman Tsarukyan", "2024-06-01"),
        ("Ilia Topuria", "Max Holloway", "2024-10-26"),
    ]
    for a, b, d in pairs:
        r = p.predict(a, b, d)
        if r["success"]:
            print(f"{a:<20} vs {b:<20}  {d}  p({a})={r['p_fighter_a']:.3f}  "
                  f"winner={r['predicted_winner']}  "
                  f"filled {r['n_features_filled']}/{len(p.feat_cols)}")
        else:
            print(f"{a} vs {b} {d}: ERROR — {r.get('error')}")
