"""Build fighter_wc_history.json — per-fight (date, weightindex, win, ko)
keyed by jfighter. Used by PredictorV2 to compute division-specific history
features at inference time without needing the full DB.

See: docs/feature_wc_history.md
"""
import json, sqlite3, pandas as pd
from pathlib import Path

DB = "data/sqlite_db/sqlite_scrapper.db"
OUT = Path("app/models/blend_v2/fighter_wc_history.json")


def main():
    conn = sqlite3.connect(DB)
    # Per-fight: weightindex from ufc_fight_results (matchmaking data),
    # win/ko from ufc_winlossko (outcome data; used only for PRIOR fights
    # at inference/training — never the current fight's own row).
    hist = pd.read_sql("""
        SELECT w.jfighter,
               e.DATE,
               w.jbout,
               fr.weightindex,
               w.win,
               w.ko
          FROM ufc_winlossko w
          JOIN ufc_event_details e ON e.jevent = w.jevent
          LEFT JOIN ufc_fight_results fr
            ON fr.jevent = w.jevent AND fr.jbout = w.jbout
    """, conn)
    conn.close()

    hist["DATE"] = pd.to_datetime(hist["DATE"])
    # Drop rows without weightindex (can't build wc history without it)
    before = len(hist)
    hist = hist.dropna(subset=["weightindex", "DATE"])
    hist["weightindex"] = hist["weightindex"].astype(int)
    hist["win"] = hist["win"].astype(int)
    hist["ko"] = hist["ko"].astype(int)
    hist = hist.sort_values(["jfighter", "DATE"])
    print(f"Loaded {len(hist):,} fighter-bout rows "
          f"(dropped {before - len(hist):,} with missing weightindex/DATE)")

    # Serialize: {jfighter: [[iso_date, weightindex, win, ko], ...]}
    out = {}
    for jf, grp in hist.groupby("jfighter"):
        out[jf] = [
            [pd.Timestamp(d).strftime("%Y-%m-%d"),
             int(wc), int(w), int(k)]
            for d, wc, w, k in zip(
                grp["DATE"].values,
                grp["weightindex"].values,
                grp["win"].values,
                grp["ko"].values)
        ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {OUT}")
    print(f"  fighters: {len(out):,}")
    print(f"  total bouts: {sum(len(v) for v in out.values()):,}")
    print(f"  size: {OUT.stat().st_size / 1024:.1f} KB")

    # Integrity check — Buchecha vs Spann (the motivating case)
    for jf in ["MarcusBuchecha", "RyanSpann"]:
        entries = out.get(jf, [])
        wcs = [e[1] for e in entries]
        wins = [e[2] for e in entries]
        print(f"  {jf}: {len(entries)} bouts, weightindices={set(wcs)}, "
              f"wins={sum(wins)}/{len(wins)}")


if __name__ == "__main__":
    main()
