"""Regenerate Elo feature set including pre-UFC fight history.

Uses `src/elo_feature.py::load_bouts_expanded()` which pulls BOTH:
  - UFC fights from ufc_winlossko (full metadata, used for prediction test)
  - Non-UFC fights from ufc_complete_fight_history (Pride, Strikeforce, WEC,
    DWCS, Bellator, etc.) — used ONLY for Elo warm-up, not as test targets

The mechanism: a UFC debutant who fought 8 times in DWCS will now enter with a
real Elo rating (from those DWCS fights) instead of defaulting to 1500. Same
for 1- and 2-fight UFC fighters whose opponents had prior non-UFC careers.

Expected effect: ROI and metrics should IMPROVE primarily for threshold=1 and
threshold=2 populations (where Elo initialization matters most). Threshold=3
shouldn't change much — those fighters already have meaningful UFC Elo.

Writes:
  data/tmp/elo_bouts_expanded.csv     — expanded bout list (UFC + non-UFC)

Leakage guardrails:
  §2/§5  compute_elo processes bouts chronologically; precomp_elo is BEFORE
         each fight. Adding non-UFC bouts doesn't change the per-fight
         precomp-only invariant; each bout still only uses strictly earlier
         bouts for rating.
  Non-UFC bouts are used ONLY for warm-up; they're not in the test set
  (test comes from UFC-only fight rows in mmaai_features.csv).
"""
import sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import elo_feature
# Force use of full scraper DB (has ufc_complete_fight_history with 11,223
# pre-UFC bouts). The module default is app.db (slim, Flask-only).
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
from elo_feature import load_bouts_expanded

DT = Path("data/tmp")

print("="*70)
print("Building expanded Elo bout list (UFC + pre-UFC fight history)")
print("="*70)
b = load_bouts_expanded()
out = DT / "elo_bouts_expanded.csv"
b.to_csv(out, index=False)
print(f"\nSaved {len(b):,} bouts to {out}")
print(f"  UFC:     {(b['source'] == 'ufc').sum():,}")
print(f"  non-UFC: {(b['source'] == 'non_ufc').sum():,}")

# Sanity: for a few known UFC debutants, how many pre-UFC bouts do we have?
print("\nPre-UFC bout counts for selected fighters (sanity check):")
for name in ["BoNickal", "KayceeChase", "IliaTopuria", "AlexandrePantoja",
            "JaredCannonier", "MichaelChandler", "JohnnyWalker"]:
    non_ufc = b[(b["source"] == "non_ufc") &
                 ((b["f1"] == name) | (b["f2"] == name))]
    ufc = b[(b["source"] == "ufc") & ((b["f1"] == name) | (b["f2"] == name))]
    print(f"  {name:20s}  pre-UFC={len(non_ufc):3d}  UFC={len(ufc):3d}")
