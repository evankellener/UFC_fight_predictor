"""Rebuild only fighter_mma_history.parquet (Part 1 of artifacts)."""
import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
from build_predictor_v2_artifacts import build_mma_history
build_mma_history()
