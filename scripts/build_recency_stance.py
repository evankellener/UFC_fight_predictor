"""Leakage-safe recency/streak/stance features (no external geo deps).

Writes data/tmp/recency_stance_features.csv keyed by (DATE, jbout, jfighter).

Features:
  win_streak_entering_diff  — # consecutive wins immediately before this fight
  coming_off_loss_diff      — 1 if previous fight was a loss (else 0)
  fights_last_12m_diff      — count of fights in the 365 days before this fight
  stance_mismatch           — 1 if one orthodox + one southpaw (static, symmetric)
  southpaw_advantage_diff   — +1 if jfighter is southpaw vs orthodox; -1 if opposite

Leakage guardrails (LEAKAGE_REFERENCE.md):
  §3  All career aggregates use strict d < fight_date on per-fighter timeline.
  §2  No EMAs/rolling — just bounded-window counts.
  §7  No odds.
"""
import sqlite3, pandas as pd, numpy as np
from pathlib import Path

DT = Path("data/tmp")
DB = "data/sqlite_db/sqlite_scrapper.db"

base = pd.read_csv(DT / "mmaai_features.csv",
                   usecols=["DATE", "jbout", "jfighter", "opp_jfighter", "win"],
                   parse_dates=["DATE"])
base["win"] = base["win"].astype(int)
print(f"Base rows: {len(base):,}")

# ── Per-fighter timeline (STRICTLY PRIOR fights per §3) ─────────────────────
# Build from ufc_winlossko — each row = one fighter's result in a specific bout
conn = sqlite3.connect(DB)
hist = pd.read_sql("""
    SELECT w.jfighter, e.DATE, w.win
    FROM ufc_winlossko w
    JOIN ufc_event_details e ON e.jevent = w.jevent
""", conn)
hist["DATE"] = pd.to_datetime(hist["DATE"])
hist = hist.sort_values(["jfighter", "DATE"]).reset_index(drop=True)
print(f"Per-fighter history rows: {len(hist):,}")

fighter_hist = {f: list(zip(grp["DATE"].values, grp["win"].astype(int).values))
                for f, grp in hist.groupby("jfighter")}

def pre_fight_stats(jf, dt):
    """(win_streak, coming_off_loss, fights_12m) using only fights with d < dt."""
    h = fighter_hist.get(jf, [])
    prior = [(d, w) for d, w in h if d < np.datetime64(dt)]
    if not prior:
        return 0, 0, 0
    # Win streak: count consecutive wins from most recent fight backward
    streak = 0
    for _, w in reversed(prior):
        if w == 1:
            streak += 1
        else:
            break
    coming_off_loss = 1 if prior[-1][1] == 0 else 0
    one_year = np.timedelta64(365, "D")
    fights_12m = sum(1 for d, _ in prior if (np.datetime64(dt) - d) <= one_year)
    return streak, coming_off_loss, fights_12m

# ── Stance map (static per fighter) ─────────────────────────────────────────
tott = pd.read_sql("SELECT jfighter, STANCE FROM ufc_fighter_tott", conn)
conn.close()
stance_map = {r.jfighter: (r.STANCE or "").strip().lower() for r in tott.itertuples()}

def is_southpaw(jf):
    s = stance_map.get(jf, "")
    return 1 if "southpaw" in s else 0

def is_orthodox(jf):
    s = stance_map.get(jf, "")
    return 1 if "orthodox" in s else 0

# ── Build output ────────────────────────────────────────────────────────────
rows = []
for r in base.itertuples():
    s_jf  = pre_fight_stats(r.jfighter,    r.DATE)
    s_opp = pre_fight_stats(r.opp_jfighter, r.DATE)
    jf_sp  = is_southpaw(r.jfighter);  jf_or  = is_orthodox(r.jfighter)
    opp_sp = is_southpaw(r.opp_jfighter); opp_or = is_orthodox(r.opp_jfighter)
    rows.append({
        "DATE": r.DATE, "jbout": r.jbout, "jfighter": r.jfighter,
        "win_streak_entering_diff": s_jf[0] - s_opp[0],
        "coming_off_loss_diff":     s_jf[1] - s_opp[1],
        "fights_last_12m_diff":     s_jf[2] - s_opp[2],
        "stance_mismatch":          int((jf_sp and opp_or) or (jf_or and opp_sp)),
        "southpaw_advantage_diff":  jf_sp - opp_sp,
    })

df = pd.DataFrame(rows)
df.to_csv(DT / "recency_stance_features.csv", index=False)
print(f"\nWrote {len(df):,} rows × {df.shape[1]} cols -> {DT/'recency_stance_features.csv'}")
print("\nSanity:")
for c in df.columns:
    if c in ("DATE", "jbout", "jfighter"): continue
    nz = (df[c] != 0).sum()
    print(f"  {c:<30} nonzero={nz}/{len(df)}  mean={df[c].mean():+.4f}")
