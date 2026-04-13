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

SRC = Path("data/sqlite_db/sqlite_scrapper.db")
DST = Path("data/sqlite_db/slim_scrapper.db")

# Tables the Flask app (BlendPredictor + legacy fallbacks) actually reads
KEEP = [
    "ufc_winlossko",        # fight-by-fight W/L (fighter profiles, history)
    "ufc_fighter_tott",     # DOB, stance, weightindex
    "ufc_event_details",    # event DATE + LOCATION (for market features lookup)
    "ufc_fight_results",    # method fields
    "ufc_fighter_details",  # fighter metadata (URL, nickname)
]

if DST.exists():
    DST.unlink()

src = sqlite3.connect(str(SRC))
dst = sqlite3.connect(str(DST))

for t in KEEP:
    ddl = src.execute(f"SELECT sql FROM sqlite_master WHERE name=?", (t,)).fetchone()
    if not ddl:
        print(f"  {t}: MISSING in source — skipping")
        continue
    dst.execute(ddl[0])
    cols = [c[1] for c in src.execute(f'PRAGMA table_info("{t}")')]
    placeholders = ",".join("?" * len(cols))
    rows = list(src.execute(f'SELECT * FROM "{t}"'))
    dst.executemany(f'INSERT INTO "{t}" VALUES ({placeholders})', rows)
    print(f"  {t}: {len(rows):>6,} rows")

dst.commit()
dst.isolation_level = None
dst.execute("VACUUM")
dst.close()
src.close()

sz_mb = DST.stat().st_size / 1024 / 1024
print(f"\nWrote {DST} — {sz_mb:.2f} MB")
