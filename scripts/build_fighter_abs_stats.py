"""Build per-fighter absolute stats cache for live as-of-date predictions.

Saves app/models/blend/fighter_abs_stats.json with for each fighter:
  last_fight_date, dob, nationality, stance, weightindex,
  sos_last3, sos_last5, sos_trajectory, form_winrate3, form_winrate5,
  elo_trajectory, career_fights, current_elo

These are ABSOLUTE (per-fighter) values. At prediction time the app will
compute diffs vs the opponent AND freshly recompute calendar-dependent
features (age, days-since-last-fight) as of the requested event date.
"""
import json, sqlite3, sys
import numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, "src")
from elo_feature import compute_elo

DATA_TMP = Path("data/tmp")
OUT = Path("app/models/blend/fighter_abs_stats.json")
SCRAPER_DB = "data/sqlite_db/sqlite_scrapper.db"
APP_DB     = "data/sqlite_db/app.db"

# ── Fighter profile tables ──
sc = sqlite3.connect(SCRAPER_DB); ac = sqlite3.connect(APP_DB)
tott = pd.read_sql("SELECT jfighter, DOB, STANCE, weightindex FROM ufc_fighter_tott", sc)
tott["DOB"] = pd.to_datetime(tott["DOB"], errors="coerce")
tott_map = {r.jfighter: {"dob": r.DOB, "stance": (r.STANCE or "").strip().lower(),
                         "weightindex": int(r.weightindex) if pd.notna(r.weightindex) else 0}
            for r in tott.itertuples()}
nat = pd.read_sql("SELECT jfighter, country FROM ufc_fighter_nationality", ac)
nat_map = dict(zip(nat["jfighter"], nat["country"]))

# ── Compute Elo (so we have per-fight pre-elos for SoS + current rating) ──
bouts = pd.read_csv(DATA_TMP/"elo_bouts.csv", parse_dates=["DATE"])
elo_df, ratings, _last_date, extra = compute_elo(
    bouts, K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923, decay_max=0.25,
    decay_midpoint=730.0, decay_steepness=80.0, logistic_scale=449.205,
    opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2,
)
# Per-fighter fight history: (DATE, opp_pre_elo, won, own_pre_elo)
rows = []
for r in elo_df.itertuples():
    rows.append((r.DATE, r.f1, r.precomp_elo_f2, int(r.f1_win == 1), r.precomp_elo_f1))
    rows.append((r.DATE, r.f2, r.precomp_elo_f1, int(r.f1_win != 1), r.precomp_elo_f2))
hist = pd.DataFrame(rows, columns=["DATE","jfighter","opp_elo","won","own_elo"])
hist = hist.sort_values(["jfighter","DATE"])

out = {}
for jf, g in hist.groupby("jfighter"):
    dates = list(g["DATE"])
    opp_elos = list(g["opp_elo"])
    wons = list(g["won"])
    own_elos = list(g["own_elo"])
    n = len(dates)
    if n == 0: continue

    last3 = opp_elos[-3:]; last5 = opp_elos[-5:]
    sos3 = float(np.mean(last3)); sos5 = float(np.mean(last5))
    if n >= 6:
        sos_traj = sos3 - float(np.mean(opp_elos[-6:-3]))
        elo_traj = own_elos[-1] - own_elos[-6]
    else:
        sos_traj = 0.0; elo_traj = 0.0
    wr3 = float(np.mean(wons[-3:])); wr5 = float(np.mean(wons[-5:]))
    career_fights = int(n)
    current_elo = float(ratings.get(jf, 1500.0))

    prof = tott_map.get(jf, {})
    peak_elo = float(extra.get("peak_elo", {}).get(jf, current_elo))
    out[jf] = {
        "last_fight_date": dates[-1].strftime("%Y-%m-%d"),
        "dob": prof.get("dob").strftime("%Y-%m-%d") if pd.notna(prof.get("dob")) else None,
        "nationality": nat_map.get(jf),
        "stance": prof.get("stance", ""),
        "weightindex": prof.get("weightindex", 0),
        "current_elo": current_elo,
        "peak_elo": peak_elo,
        "sos_last3": sos3,
        "sos_last5": sos5,
        "sos_trajectory": sos_traj,
        "form_winrate3": wr3,
        "form_winrate5": wr5,
        "elo_trajectory": elo_traj,
        "career_fights": career_fights,
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f: json.dump(out, f)
print(f"Wrote {len(out)} fighters to {OUT}")
print(f"Sample (Jon Jones): {json.dumps(out.get('JonJones', {}), indent=2)}")
