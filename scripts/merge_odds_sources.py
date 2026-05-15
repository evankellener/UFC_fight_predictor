"""Merge fightodds.io closing lines (2015-2026) with odds_table.csv (2020+).

Strategy:
  - fo_closing is primary (Pinnacle-preferred, devigged, higher quality)
  - odds_table fills in fights not covered by fo_closing
  - Output: data/tmp/odds_table_extended.csv  (same schema as odds_table.csv)
  - Also copies to odds_table.csv with backup of original

Schema of output (matches odds_table.csv):
  DATE, BOUT, FIGHTER, prob_norm, odds
"""
import pandas as pd
from pathlib import Path
import shutil

REPO = Path(__file__).parent.parent
TMP  = REPO / "data" / "tmp"

FO_FILE  = TMP / "fo_closing.csv"
OT_FILE  = TMP / "odds_table.csv"
OUT_FILE = TMP / "odds_table_extended.csv"


def norm(s):
    return "".join(str(s).lower().split())


def main():
    fo = pd.read_csv(FO_FILE, parse_dates=["DATE"])
    ot = pd.read_csv(OT_FILE, parse_dates=["DATE"])

    print(f"fo_closing: {len(fo)} rows  ({fo['DATE'].min().date()} → {fo['DATE'].max().date()})")
    print(f"odds_table: {len(ot)} rows  ({ot['DATE'].min().date()} → {ot['DATE'].max().date()})")

    # Build fo as the primary table (in odds_table.csv schema)
    fo_out = pd.DataFrame({
        "DATE":      fo["DATE"],
        "BOUT":      fo["BOUT"],
        "FIGHTER":   fo["FIGHTER_DISPLAY"],   # use display name for compatibility
        "prob_norm": fo["prob_norm"],
        "odds":      fo["odds"],
        "source":    "fo",
    })

    # Identify which (DATE, fighter_norm) are already covered by fo
    fo_out["_fkey"] = fo_out["FIGHTER"].map(norm)
    fo_keys = set(zip(fo_out["DATE"].dt.date, fo_out["_fkey"]))

    # From ot, keep only rows NOT covered by fo
    ot["_fkey"] = ot["FIGHTER"].map(norm)
    ot_gap = ot[~ot.apply(lambda r: (r["DATE"].date(), r["_fkey"]) in fo_keys, axis=1)].copy()
    ot_gap = ot_gap.assign(source="ot")

    print(f"\nfo rows used:     {len(fo_out)}")
    print(f"ot gap-fill rows: {len(ot_gap)}  "
          f"({ot_gap['DATE'].min().date() if len(ot_gap) else 'N/A'} "
          f"→ {ot_gap['DATE'].max().date() if len(ot_gap) else 'N/A'})")

    # Combine
    cols = ["DATE", "BOUT", "FIGHTER", "prob_norm", "odds", "source"]
    combined = pd.concat([fo_out[cols], ot_gap[cols + ["_fkey"]].drop(columns=["_fkey"])[cols]],
                         ignore_index=True)
    combined = combined.sort_values(["DATE", "BOUT", "FIGHTER"]).reset_index(drop=True)

    print(f"\nCombined: {len(combined)} rows  "
          f"({combined['DATE'].min().date()} → {combined['DATE'].max().date()})")

    # Stats
    fo_pct = (combined["source"] == "fo").mean() * 100
    ot_pct = (combined["source"] == "ot").mean() * 100
    print(f"  fo (Pinnacle-preferred): {fo_pct:.1f}%")
    print(f"  ot (The-Odds-API):       {ot_pct:.1f}%")

    # Coverage by year
    combined["year"] = combined["DATE"].dt.year
    by_year = combined.groupby("year").agg(
        rows=("FIGHTER", "count"),
        fo_rows=("source", lambda x: (x == "fo").sum()),
    )
    by_year["fights"] = by_year["rows"] // 2
    by_year["fo_pct"] = (by_year["fo_rows"] / by_year["rows"] * 100).round(1)
    print("\nCoverage by year:")
    print(by_year[["fights", "fo_pct"]].to_string())

    # Save extended file (without source column)
    out = combined.drop(columns=["source", "year"], errors="ignore")
    out.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}  ({len(out)} rows)")

    # Backup original odds_table and replace with extended
    backup = OT_FILE.with_name("odds_table_theodds_backup.csv")
    if not backup.exists():
        shutil.copy(OT_FILE, backup)
        print(f"Backed up original to: {backup.name}")

    shutil.copy(OUT_FILE, OT_FILE)
    print(f"Replaced odds_table.csv with extended version.")


if __name__ == "__main__":
    main()
