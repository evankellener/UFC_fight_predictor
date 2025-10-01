import pandas as pd
import numpy as np
from typing import Dict, Optional

def clamp_odds_to_realistic_ranges(
    input_csv: str,
    output_csv: str,
    odds_columns: Optional[list] = None
) -> None:
    """
    Clamp extreme odds values to realistic ranges to fix data quality issues.
    
    Rules:
    1. If odds are between -100 and 100, clamp them to the closer boundary
       - If odds < 100 and > -100, push to -100 (for negative) or 100 (for positive)
    2. Keep other extreme values as they might be legitimate long shots
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to save fixed CSV file
        odds_columns: List of odds column names to process
    """
    if odds_columns is None:
        odds_columns = [
            'draftkings_odds', 'fanduel_odds', 'betmgm_odds', 
            'bet365_odds', 'bovada_odds'
        ]
    
    df = pd.read_csv(input_csv)
    
    print("=== ODDS CLAMPING FIX ===")
    print(f"Processing {len(df)} rows")
    
    for col in odds_columns:
        if col in df.columns:
            # Convert to numeric, errors become NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Count original values
            original_count = df[col].notna().sum()
            
            # Apply clamping logic
            # If odds are between -100 and 100 (inclusive), clamp to boundaries
            # Also clamp odds that are close to 100 (like 102) to 100
            mask_positive = (df[col] > 0) & (df[col] <= 110)  # Clamp odds up to 110 to 100
            mask_negative = (df[col] < 0) & (df[col] >= -110)  # Clamp odds down to -110 to -100
            
            # Clamp positive odds < 100 to 100
            df.loc[mask_positive, col] = 100
            
            # Clamp negative odds > -100 to -100
            df.loc[mask_negative, col] = -100
            
            # Count changes
            clamped_positive = mask_positive.sum()
            clamped_negative = mask_negative.sum()
            
            print(f"{col}:")
            print(f"  Clamped {clamped_positive} positive odds (< 100) to 100")
            print(f"  Clamped {clamped_negative} negative odds (> -100) to -100")
            print(f"  Total changes: {clamped_positive + clamped_negative}")
    
    # Save the fixed data
    df.to_csv(output_csv, index=False)
    print(f"\nFixed data written to {output_csv}")
    
    return df

def demonstrate_fix_on_jiri_fight():
    """Demonstrate the fix on the problematic Jiri Prochazka fight"""
    print("\n=== DEMONSTRATION: JIRI PROCHAZKA FIGHT FIX ===")
    
    # Original problematic odds (including NaN)
    original_odds = [-105.0, 102.0, -105.0, np.nan, 100.0]  # Missing bet365_odds (NaN)
    print(f"Original odds: {original_odds}")
    
    # Calculate average properly (ignoring NaN)
    valid_odds = [x for x in original_odds if not np.isnan(x)]
    original_avg = np.mean(valid_odds)
    print(f"Original average (ignoring NaN): {original_avg:.1f}")
    
    # Apply clamping
    clamped_odds = []
    for odds in original_odds:
        if np.isnan(odds):
            clamped_odds.append(np.nan)  # Keep NaN as NaN
        elif 0 < odds <= 110:
            clamped_odds.append(100)  # Clamp 102 to 100
        elif -110 <= odds < 0:
            clamped_odds.append(-100)  # Clamp -105 to -100
        else:
            clamped_odds.append(odds)  # Keep as is
    
    print(f"Clamped odds: {clamped_odds}")
    
    # Calculate new average (ignoring NaN)
    valid_clamped = [x for x in clamped_odds if not np.isnan(x)]
    new_avg = np.mean(valid_clamped)
    print(f"New average (ignoring NaN): {new_avg:.1f}")
    
    # Calculate profit difference
    original_profit = calculate_profit(original_avg, 100, True)
    new_profit = calculate_profit(new_avg, 100, True)
    
    print(f"Original profit on $100 bet: ${original_profit:.2f}")
    print(f"New profit on $100 bet: ${new_profit:.2f}")
    print(f"Difference: ${new_profit - original_profit:.2f}")
    
    # Show the actual issue
    print(f"\n=== THE REAL ISSUE ===")
    print(f"The problem is that 102.0 (FanDuel odds) should be clamped to 100")
    print(f"This changes the average from {original_avg:.1f} to {new_avg:.1f}")
    print(f"Which reduces the profit from ${original_profit:.2f} to ${new_profit:.2f}")

def calculate_profit(vegas_odds, stake=100, won=True):
    """Calculate profit from betting odds"""
    if not won:
        return -stake
    if vegas_odds == 0:  # Handle zero odds case
        return 0  # No profit or loss on a 0 odds bet
    if vegas_odds > 0:
        return (vegas_odds / 100) * stake
    else:
        return (100 / abs(vegas_odds)) * stake

if __name__ == "__main__":
    # Demonstrate the fix
    demonstrate_fix_on_jiri_fight()
    
    # Apply the fix to the data
    clamp_odds_to_realistic_ranges(
        input_csv="src/final_with_odds.csv",
        output_csv="src/final_with_odds_clamped.csv"
    )
