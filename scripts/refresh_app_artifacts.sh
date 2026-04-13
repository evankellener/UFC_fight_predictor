#!/usr/bin/env bash
# Rebuild all Flask-app artifacts from the current data/ state.
# Run after a fresh rescrape + mmaai_features rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[1/3] Training LR + XGB blend and saving artifacts..."
python scripts/train_and_save_blend.py
echo ""
echo "[2/3] Building per-fighter absolute-stats cache for live predictions..."
python scripts/build_fighter_abs_stats.py
echo ""
echo "[3/3] Exporting slim sqlite for deployment (commit-safe)..."
python scripts/export_slim_db.py
echo ""
echo "Done."
echo "To deploy: git add app/models/blend/ data/sqlite_db/slim_scrapper.db && git commit && git push"
echo "Render will auto-build on push."
