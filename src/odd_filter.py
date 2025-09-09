import pandas as pd
import numpy as np
from typing import Dict, Optional

def filter_sportsbook_odds(
    input_csv: str,
    output_csv: str,
    thresholds: Optional[Dict[str, float]] = None
) -> None:
    """
    Filter sportsbook odds columns in a CSV file by setting out-of-bounds values to NaN.
    Args:
        input_csv: Path to input CSV file.
        output_csv: Path to save filtered CSV file.
        thresholds: Dict mapping column names to max absolute value allowed.
    """
    if thresholds is None:
        thresholds = {
            'draftkings_odds': 5000,
            'fanduel_odds':    3500,
            'betmgm_odds':     5000,
            'bet365_odds':     5000,
            'bovada_odds':     5000,
        }
    df = pd.read_csv(input_csv)
    for col, max_abs in thresholds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col].abs() > max_abs, col] = np.nan
    print("Removed out-of-bounds entries:")
    for col in thresholds:
        if col in df.columns:
            total = len(df)
            valid = df[col].notna().sum()
            removed = total - valid
            print(f"  {col}: {removed} rows removed")
    df.to_csv(output_csv, index=False)
    print(f"\nFiltered data written to {output_csv}")

if __name__ == "__main__":
    filter_sportsbook_odds(
        input_csv="final_with_odds.csv",
        output_csv="final_with_odds_filtered.csv"
    ) 