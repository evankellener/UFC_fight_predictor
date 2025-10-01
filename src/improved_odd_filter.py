import pandas as pd
import numpy as np
from typing import Dict, Optional

def improved_filter_sportsbook_odds(
    input_csv: str,
    output_csv: str,
    thresholds: Optional[Dict[str, float]] = None,
    handle_missing_odds: str = "average_available"
) -> None:
    """
    Improved filter for sportsbook odds that handles missing values intelligently.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to save filtered CSV file
        thresholds: Dict mapping column names to max absolute value allowed
        handle_missing_odds: How to handle missing odds:
            - "average_available": Average only available odds for each fight
            - "drop_row": Drop entire row if any odds are missing (original behavior)
            - "keep_nan": Keep NaN values as-is
    """
    if thresholds is None:
        thresholds = {
            'draftkings_odds': 5000,
            'fanduel_odds': 3500,
            'betmgm_odds': 5000,
            'bet365_odds': 5000,
            'bovada_odds': 5000,
        }
    
    df = pd.read_csv(input_csv)
    odds_columns = list(thresholds.keys())
    
    print("=== IMPROVED ODDS FILTERING ===")
    print(f"Processing {len(df)} rows")
    print(f"Handling missing odds: {handle_missing_odds}")
    
    # Step 1: Filter out-of-bounds values
    for col, max_abs in thresholds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            before_count = df[col].notna().sum()
            df.loc[df[col].abs() > max_abs, col] = np.nan
            after_count = df[col].notna().sum()
            removed = before_count - after_count
            print(f"  {col}: {removed} out-of-bounds values set to NaN")
    
    # Step 2: Handle missing odds based on strategy
    if handle_missing_odds == "average_available":
        print("\n=== AVERAGING AVAILABLE ODDS ===")
        
        # Group by fight (assuming DATE, EVENT, BOUT identify unique fights)
        fight_identifiers = ['DATE', 'EVENT', 'BOUT']
        if all(col in df.columns for col in fight_identifiers):
            # Calculate average odds for each fight using only available odds
            df['avg_odds_calculated'] = df[odds_columns].mean(axis=1, skipna=True)
            
            # Count fights with different numbers of available odds
            for i in range(1, len(odds_columns) + 1):
                count = df[odds_columns].notna().sum(axis=1).eq(i).sum()
                if count > 0:
                    print(f"  {count} fights with {i} available odds")
            
            # Show some examples
            print("\n=== EXAMPLES OF ODDS AVERAGING ===")
            sample_fights = df.groupby(fight_identifiers).first().head(3)
            for idx, fight in sample_fights.iterrows():
                available_odds = [fight[col] for col in odds_columns if pd.notna(fight[col])]
                avg_odds = fight['avg_odds_calculated']
                print(f"  Fight: {fight.get('BOUT', 'Unknown')}")
                print(f"    Available odds: {available_odds}")
                print(f"    Calculated average: {avg_odds:.1f}")
                print()
        
    elif handle_missing_odds == "drop_row":
        print("\n=== DROPPING ROWS WITH MISSING ODDS ===")
        before_count = len(df)
        df = df.dropna(subset=odds_columns)
        after_count = len(df)
        dropped = before_count - after_count
        print(f"  Dropped {dropped} rows with missing odds")
        print(f"  Remaining rows: {after_count}")
    
    # Step 3: Final statistics
    print(f"\n=== FINAL STATISTICS ===")
    for col in odds_columns:
        if col in df.columns:
            total = len(df)
            valid = df[col].notna().sum()
            missing = total - valid
            print(f"  {col}: {valid}/{total} valid ({missing} missing)")
    
    # Save the filtered data
    df.to_csv(output_csv, index=False)
    print(f"\nFiltered data written to {output_csv}")
    
    return df

def compare_filtering_strategies(input_csv: str):
    """Compare different filtering strategies on the same data"""
    print("=== COMPARING FILTERING STRATEGIES ===")
    
    strategies = {
        "average_available": "Average only available odds",
        "drop_row": "Drop rows with any missing odds",
        "keep_nan": "Keep NaN values as-is"
    }
    
    results = {}
    
    for strategy, description in strategies.items():
        print(f"\n--- {description} ---")
        output_file = f"temp_filtered_{strategy}.csv"
        
        try:
            df = improved_filter_sportsbook_odds(
                input_csv=input_csv,
                output_csv=output_file,
                handle_missing_odds=strategy
            )
            
            # Calculate some statistics
            odds_columns = ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']
            total_rows = len(df)
            rows_with_all_odds = df[odds_columns].notna().all(axis=1).sum()
            rows_with_some_odds = df[odds_columns].notna().any(axis=1).sum()
            
            results[strategy] = {
                'total_rows': total_rows,
                'rows_with_all_odds': rows_with_all_odds,
                'rows_with_some_odds': rows_with_some_odds,
                'data_retention': rows_with_some_odds / total_rows if total_rows > 0 else 0
            }
            
            print(f"  Total rows: {total_rows}")
            print(f"  Rows with all odds: {rows_with_all_odds}")
            print(f"  Rows with some odds: {rows_with_some_odds}")
            print(f"  Data retention: {results[strategy]['data_retention']:.1%}")
            
        except Exception as e:
            print(f"  Error with strategy {strategy}: {e}")
    
    return results

if __name__ == "__main__":
    # Compare strategies
    compare_filtering_strategies("src/final_with_odds.csv")
    
    # Apply the improved filter with averaging strategy
    improved_filter_sportsbook_odds(
        input_csv="src/final_with_odds.csv",
        output_csv="src/final_with_odds_improved.csv",
        handle_missing_odds="average_available"
    )
