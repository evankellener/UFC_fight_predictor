"""Export a slim sqlite DB containing only the tables the Flask app needs,
so we can commit it to git for deployment (Render, etc) without shipping
the multi-GB scraper artifacts.

Overwrites data/sqlite_db/sqlite_scrapper.db in place? NO — writes to
data/sqlite_db/slim_scrapper.db. Flask app paths are updated to prefer
this slim version when it exists.

Run after a fresh scrape/rescrape to refresh deployment data.
"""
import sqlite3
import os
from pathlib import Path

SRC      = Path("data/sqlite_db/sqlite_scrapper.db")
SRC_APP  = Path("data/sqlite_db/app.db")
DST      = Path("data/sqlite_db/slim_scrapper.db")

# Tables the Flask apps (src/app.py production site + app/app.py BlendPredictor) read.
# KEEP_FROM_SCRAPER comes from sqlite_scrapper.db. KEEP_FROM_APP comes from app.db.
KEEP_FROM_SCRAPER = [
    "ufc_winlossko",         # fight-by-fight W/L (fighter profiles, history)
    "ufc_fighter_tott",      # DOB, stance, weightindex
    "ufc_event_details",     # event DATE + LOCATION
    "ufc_fight_results",     # method + round + time fields
    "ufc_fighter_details",   # fighter metadata (URL, nickname)
    "final_features_fast",   # AdjPerf z-scores for radar charts (src/app.py)
]
KEEP_FROM_APP = [
    "ufc_fighter_nationality",  # flags + home-advantage
    "ufc_fight_odds",           # odds display
    "ufc_card_position",        # main-event + career card position
]
KEEP = KEEP_FROM_SCRAPER  # kept for backcompat readability

if DST.exists():
    DST.unlink()

dst = sqlite3.connect(str(DST))

def copy_tables(src_path, tables, label):
    if not src_path.exists():
        print(f"[{label}] source not found: {src_path} — skipping")
        return
    src = sqlite3.connect(str(src_path))
    for t in tables:
        ddl = src.execute("SELECT sql FROM sqlite_master WHERE name=?", (t,)).fetchone()
        if not ddl:
            print(f"  [{label}] {t}: MISSING — skipping")
            continue
        # drop any existing copy in dst then re-create
        dst.execute(f'DROP TABLE IF EXISTS "{t}"')
        dst.execute(ddl[0])
        cols = [c[1] for c in src.execute(f'PRAGMA table_info("{t}")')]
        placeholders = ",".join("?" * len(cols))
        rows = list(src.execute(f'SELECT * FROM "{t}"'))
        dst.executemany(f'INSERT INTO "{t}" VALUES ({placeholders})', rows)
        print(f"  [{label}] {t}: {len(rows):>6,} rows")
    src.close()

copy_tables(SRC,     KEEP_FROM_SCRAPER, "scraper")
copy_tables(SRC_APP, KEEP_FROM_APP,     "app    ")

dst.commit()
dst.isolation_level = None
dst.execute("VACUUM")
dst.close()

sz_mb = DST.stat().st_size / 1024 / 1024
print(f"\nWrote {DST} — {sz_mb:.2f} MB")
