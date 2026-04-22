"""Build striking Elo + grappling Elo — per-fight precomp ratings.

Different from the main Elo in that updates are based on per-fight STAT
margins, not W/L:
  Striking winner: fighter with more sig-strikes-landed in the bout
  Grappling winner: fighter with more (takedowns landed + ctrl-time/60)

Writes data/tmp/style_elo_features.csv keyed by (DATE, jbout, jfighter):
  striking_elo_diff   = striking_elo(jfighter) − striking_elo(opp) BEFORE the fight
  grappling_elo_diff  = grappling_elo(jfighter) − grappling_elo(opp) BEFORE the fight

Leakage guardrails (LEAKAGE_REFERENCE.md):
  §2, §5  precomp (rating BEFORE the fight); bouts processed in chronological
          order; each update uses only stats from that specific bout.
  §3  Per-fight "winner" derivation uses only stats FROM THAT BOUT — no
      future info. Completely bout-local.
"""
import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"
BASE_ELO = 1500
K_STRIKE = 20
K_GRAPPLE = 20
LOGISTIC_SCALE = 400.0


def _expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / LOGISTIC_SCALE))


def _update(r_a, r_b, actual_a, k):
    exp_a = _expected(r_a, r_b)
    return r_a + k * (actual_a - exp_a), r_b + k * ((1 - actual_a) - (1 - exp_a))


def main():
    print("="*70)
    print("Build striking + grappling Elo from per-fight smoothed stats")
    print("="*70)

    # ── Load bouts + per-fighter per-bout smoothed stats ────────────────
    conn = sqlite3.connect(DB)
    bouts = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    print(f"Bouts: {len(bouts):,}")

    stats = pd.read_sql("""
        SELECT jevent, jbout, jfighter,
               sigstracc, tdacc, ctrl, subatt
        FROM ufc_fighter_match_stats_smooth
    """, conn)
    conn.close()
    print(f"Per-fight smoothed stats rows: {len(stats):,}")

    # Pivot to per-bout: f1_*, f2_* using the bout's f1/f2
    # Join stats onto bouts (each bout has two rows in stats)
    b = bouts.merge(stats, on=["jevent", "jbout"], how="inner")
    # Now b has 2 rows per bout (one per fighter). Pivot.
    f1_rows = b[b["jfighter"] == b["f1"]][["jevent", "jbout", "DATE", "f1", "f2",
                                             "sigstracc", "tdacc", "ctrl", "subatt"]].rename(
        columns={"sigstracc": "f1_sigstr", "tdacc": "f1_td",
                 "ctrl": "f1_ctrl", "subatt": "f1_sub"})
    f2_rows = b[b["jfighter"] == b["f2"]][["jevent", "jbout",
                                             "sigstracc", "tdacc", "ctrl", "subatt"]].rename(
        columns={"sigstracc": "f2_sigstr", "tdacc": "f2_td",
                 "ctrl": "f2_ctrl", "subatt": "f2_sub"})
    merged = f1_rows.merge(f2_rows, on=["jevent", "jbout"], how="inner")
    merged = merged.sort_values(["DATE", "jevent", "jbout"]).reset_index(drop=True)
    print(f"Bouts with both fighters' stats: {len(merged):,}")

    # ── Compute actual-score (0..1) for striking and grappling per bout ──
    # Use logistic on margin so small margins → near 0.5 (draw-ish)
    def margin_to_score(m, scale=5.0):
        # scale chosen so a ~5-strike or ~1-takedown margin maps near 0.75 actual
        return 1.0 / (1.0 + np.exp(-m / scale))

    merged["strk_margin"] = merged["f1_sigstr"] - merged["f2_sigstr"]
    merged["strk_actual_f1"] = margin_to_score(merged["strk_margin"], scale=5.0)

    # Grappling score: takedowns + control time in minutes + sub attempts
    merged["grp_f1"] = merged["f1_td"] + (merged["f1_ctrl"] / 60.0) + 0.3 * merged["f1_sub"]
    merged["grp_f2"] = merged["f2_td"] + (merged["f2_ctrl"] / 60.0) + 0.3 * merged["f2_sub"]
    merged["grp_margin"] = merged["grp_f1"] - merged["grp_f2"]
    merged["grp_actual_f1"] = margin_to_score(merged["grp_margin"], scale=1.5)

    # ── Run Elo chains chronologically ──────────────────────────────────
    strike_r = defaultdict(lambda: BASE_ELO)
    grapple_r = defaultdict(lambda: BASE_ELO)
    out_rows = []

    for r in merged.itertuples():
        # PRECOMP diffs (BEFORE this bout's update)
        s_f1 = strike_r[r.f1]; s_f2 = strike_r[r.f2]
        g_f1 = grapple_r[r.f1]; g_f2 = grapple_r[r.f2]

        # Emit BOTH fighter perspectives (mirror)
        out_rows.append({
            "DATE": r.DATE, "jbout": r.jbout, "jfighter": r.f1,
            "striking_elo_diff":  s_f1 - s_f2,
            "grappling_elo_diff": g_f1 - g_f2,
        })
        out_rows.append({
            "DATE": r.DATE, "jbout": r.jbout, "jfighter": r.f2,
            "striking_elo_diff":  s_f2 - s_f1,
            "grappling_elo_diff": g_f2 - g_f1,
        })

        # Update ratings AFTER emitting precomp
        if not (np.isnan(r.strk_actual_f1) or np.isnan(r.grp_actual_f1)):
            new_s_f1, new_s_f2 = _update(s_f1, s_f2, r.strk_actual_f1, K_STRIKE)
            new_g_f1, new_g_f2 = _update(g_f1, g_f2, r.grp_actual_f1, K_GRAPPLE)
            strike_r[r.f1] = new_s_f1;  strike_r[r.f2] = new_s_f2
            grapple_r[r.f1] = new_g_f1; grapple_r[r.f2] = new_g_f2

    out = pd.DataFrame(out_rows)
    out.to_csv(DT / "style_elo_features.csv", index=False)
    print(f"\nWrote {len(out):,} rows × {out.shape[1]} cols -> {DT/'style_elo_features.csv'}")

    print("\nRating distribution (final):")
    s_vals = list(strike_r.values())
    g_vals = list(grapple_r.values())
    print(f"  Striking Elo:   mean={np.mean(s_vals):.1f}  "
          f"p10={np.percentile(s_vals,10):.0f}  p90={np.percentile(s_vals,90):.0f}  "
          f"max={max(s_vals):.0f}")
    print(f"  Grappling Elo:  mean={np.mean(g_vals):.1f}  "
          f"p10={np.percentile(g_vals,10):.0f}  p90={np.percentile(g_vals,90):.0f}  "
          f"max={max(g_vals):.0f}")

    print(f"\nFeature std-dev sanity:")
    print(f"  striking_elo_diff:   {out['striking_elo_diff'].std():.2f}  "
          f"(range {out['striking_elo_diff'].min():.0f} to {out['striking_elo_diff'].max():.0f})")
    print(f"  grappling_elo_diff:  {out['grappling_elo_diff'].std():.2f}  "
          f"(range {out['grappling_elo_diff'].min():.0f} to {out['grappling_elo_diff'].max():.0f})")


if __name__ == "__main__":
    main()
