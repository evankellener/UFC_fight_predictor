"""Resume predictor_v2 artifacts build — only Parts 3 + 4 (bios + train LR).
Parts 1 + 2 already succeeded."""
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from build_predictor_v2_artifacts import build_bios, train_final_lr, OUT
import json

print("── Building bios (robust)...")
bios = build_bios()
with open(OUT / "fighter_bios.json", "w") as f:
    json.dump(bios, f, default=str)
print(f"  saved to {OUT/'fighter_bios.json'}")

print("\n── Training final LR...")
train_final_lr()

print("\nDONE")
for p in sorted(OUT.glob("*")):
    print(f"  {p.name:<40s}  {p.stat().st_size / 1024:>8.1f} KB")
