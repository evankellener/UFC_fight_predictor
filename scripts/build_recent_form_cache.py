"""Build fighter_recent_form.json — per-fighter per-bout (date, win, ko, subw, fight_time_minutes).

Used by PredictorV2 to compute last-3-fights recent-form features without
needing DB access at inference. Same pattern as fighter_winlossko.json
but with extra outcome columns (ko, subw, fight_time).

See: docs/feature_era_rolling_baselines.md and
docs/additional_model_bumps.md Tier A entry for recent-form.
"""
import json, sqlite3
from pathlib import Path
import pandas as pd

DB = "data/sqlite_db/sqlite_scrapper.db"
OUT = Path("app/models/blend_v2/fighter_recent_form.json")


def main():
    conn = sqlite3.connect(DB)
    hist = pd.read_sql("""
        SELECT w.jfighter, e.DATE, w.win, w.ko, w.subw, w.fight_time_minutes
        FROM ufc_winlossko w
        JOIN ufc_event_details e ON e.jevent = w.jevent
    """, conn)
    conn.close()

    hist["DATE"] = pd.to_datetime(hist["DATE"])
    hist = hist.sort_values(["jfighter", "DATE"]).dropna(subset=["DATE"])
    # Fill NaN for integer columns with 0
    for col in ("win", "ko", "subw"):
        hist[col] = hist[col].fillna(0).astype(int)
    hist["fight_time_minutes"] = hist["fight_time_minutes"].fillna(0).astype(float)

    out = {}
    for jf, grp in hist.groupby("jfighter"):
        out[jf] = [
            [pd.Timestamp(d).strftime("%Y-%m-%d"),
             int(w), int(k), int(s), float(t)]
            for d, w, k, s, t in zip(
                grp["DATE"].values, grp["win"].values, grp["ko"].values,
                grp["subw"].values, grp["fight_time_minutes"].values)
        ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {OUT}")
    print(f"  fighters: {len(out):,}")
    print(f"  total bouts: {sum(len(v) for v in out.values()):,}")
    print(f"  size: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
