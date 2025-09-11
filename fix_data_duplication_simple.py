#!/usr/bin/env python3
"""
Fix UFC fight data by:
1. Duplicating each fight row and swapping fighters
2. Swapping basic stats between fighters
3. Ensuring consistent win/loss data
"""

import pandas as pd
import numpy as np
import os

def swap_fighter_data(row):
    """Create a swapped version of a fight row"""
    swapped = row.copy()
    
    # Swap fighter names
    swapped['FIGHTER'] = row['opp_FIGHTER']
    swapped['opp_FIGHTER'] = row['FIGHTER']
    
    # Swap win result (if fighter 1 won, then fighter 2 lost)
    try:
        win_val = int(row['win'])
        swapped['win'] = str(1 - win_val)
    except (ValueError, TypeError):
        # Handle non-numeric values (like header rows)
        swapped['win'] = row['win']
    
    # Swap bout counts
    swapped['precomp_boutcount'] = row['opp_precomp_boutcount']
    swapped['opp_precomp_boutcount'] = row['precomp_boutcount']
    swapped['postcomp_boutcount'] = row['opp_postcomp_boutcount']
    swapped['opp_postcomp_boutcount'] = row['postcomp_boutcount']
    
    # Swap ages
    swapped['age'] = row['opp_age']
    swapped['opp_age'] = row['age']
    
    # Swap reach
    swapped['REACH'] = row['opp_REACH']
    swapped['opp_REACH'] = row['REACH']
    
    # Swap Elo ratings
    swapped['precomp_elo'] = row['opp_precomp_elo']
    swapped['opp_precomp_elo'] = row['precomp_elo']
    swapped['postcomp_elo'] = row['opp_postcomp_elo']
    swapped['opp_postcomp_elo'] = row['postcomp_elo']
    
    # Swap strike Elo
    swapped['precomp_strike_elo'] = row['opp_precomp_strike_elo']
    swapped['opp_precomp_strike_elo'] = row['precomp_strike_elo']
    swapped['postcomp_strike_elo'] = row['opp_postcomp_strike_elo']
    swapped['opp_postcomp_strike_elo'] = row['postcomp_strike_elo']
    
    # Swap grapple Elo
    swapped['precomp_grapple_elo'] = row['opp_precomp_grapple_elo']
    swapped['opp_precomp_grapple_elo'] = row['precomp_grapple_elo']
    swapped['postcomp_grapple_elo'] = row['opp_postcomp_grapple_elo']
    swapped['opp_postcomp_grapple_elo'] = row['postcomp_grapple_elo']
    
    # Swap takedown stats
    swapped['precomp_tdavg'] = row['opp_precomp_tdavg']
    swapped['opp_precomp_tdavg'] = row['precomp_tdavg']
    swapped['precomp_tdavg3'] = row['opp_precomp_tdavg3']
    swapped['opp_precomp_tdavg3'] = row['precomp_tdavg3']
    swapped['precomp_tdavg5'] = row['opp_precomp_tdavg5']
    swapped['opp_precomp_tdavg5'] = row['precomp_tdavg5']
    
    swapped['precomp_tddef'] = row['opp_precomp_tddef']
    swapped['opp_precomp_tddef'] = row['precomp_tddef']
    swapped['precomp_tddef3'] = row['opp_precomp_tddef3']
    swapped['opp_precomp_tddef3'] = row['precomp_tddef3']
    swapped['precomp_tddef5'] = row['opp_precomp_tddef5']
    swapped['opp_precomp_tddef5'] = row['precomp_tddef5']
    
    # Swap accuracy stats
    swapped['precomp_totalacc_perc'] = row['opp_precomp_totalacc_perc']
    swapped['opp_precomp_totalacc_perc'] = row['precomp_totalacc_perc']
    swapped['precomp_totalacc_perc3'] = row['opp_precomp_totalacc_perc3']
    swapped['opp_precomp_totalacc_perc3'] = row['precomp_totalacc_perc3']
    swapped['precomp_totalacc_perc5'] = row['opp_precomp_totalacc_perc5']
    swapped['opp_precomp_totalacc_perc5'] = row['precomp_totalacc_perc5']
    
    # Swap strike defense
    swapped['precomp_strdef'] = row['opp_precomp_strdef']
    swapped['opp_precomp_strdef'] = row['precomp_strdef']
    swapped['precomp_strdef3'] = row['opp_precomp_strdef3']
    swapped['opp_precomp_strdef3'] = row['precomp_strdef3']
    swapped['precomp_strdef5'] = row['opp_precomp_strdef5']
    swapped['opp_precomp_strdef5'] = row['precomp_strdef5']
    
    # Swap age ratios
    swapped['age_ratio_difference'] = row['opp_age_ratio_difference']
    swapped['opp_age_ratio_difference'] = row['age_ratio_difference']
    
    # Swap winsum
    swapped['precomp_winsum3'] = row['opp_precomp_winsum3']
    swapped['opp_precomp_winsum3'] = row['precomp_winsum3']
    
    # Swap weight stats
    swapped['weightindex'] = row['opp_weightindex']
    swapped['opp_weightindex'] = row['weightindex']
    swapped['weight_of_fight'] = row['opp_weight_of_fight']
    swapped['opp_weight_of_fight'] = row['weight_of_fight']
    
    # Swap distance accuracy
    swapped['precomp_distacc_perc'] = row['opp_precomp_distacc_perc']
    swapped['opp_precomp_distacc_perc'] = row['precomp_distacc_perc']
    
    # Swap takedown accuracy
    swapped['precomp_tdacc_perc3'] = row['opp_precomp_tdacc_perc3']
    swapped['opp_precomp_tdacc_perc3'] = row['precomp_tdacc_perc3']
    swapped['precomp_tdacc_perc5'] = row['opp_precomp_tdacc_perc5']
    swapped['opp_precomp_tdacc_perc5'] = row['precomp_tdacc_perc5']
    
    # Swap leg accuracy
    swapped['precomp_legacc_perc3'] = row['opp_precomp_legacc_perc3']
    swapped['opp_precomp_legacc_perc3'] = row['precomp_legacc_perc3']
    
    return swapped

def process_fight_data(input_file, output_file):
    """Process the fight data with duplication and swapping"""
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Original dataset size: {len(df)} rows")
    
    # Create swapped rows
    print("Creating swapped fight rows...")
    swapped_rows = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        # Add original row
        swapped_rows.append(row)
        
        # Add swapped row
        swapped_row = swap_fighter_data(row)
        swapped_rows.append(swapped_row)
    
    # Create new dataframe
    print("Creating new dataset...")
    new_df = pd.DataFrame(swapped_rows)
    print(f"New dataset size: {len(new_df)} rows")
    
    # Save the new dataset
    print(f"Saving to {output_file}...")
    new_df.to_csv(output_file, index=False)
    print("Done!")
    
    return new_df

def verify_ruffy_data(df):
    """Verify Mauricio Ruffy's data after processing"""
    print("\n=== Verifying Mauricio Ruffy's data ===")
    ruffy_fights = df[(df['FIGHTER'].str.contains('Ruffy', case=False, na=False)) | 
                      (df['opp_FIGHTER'].str.contains('Ruffy', case=False, na=False))]
    
    print(f"Total Ruffy fights: {len(ruffy_fights)}")
    
    for idx, row in ruffy_fights.iterrows():
        print(f"\nFight: {row['FIGHTER']} vs {row['opp_FIGHTER']}")
        print(f"Date: {row['DATE']}")
        print(f"win column: {row['win']}")
        print(f"precomp_boutcount: {row['precomp_boutcount']}")
        print(f"opp_precomp_boutcount: {row['opp_precomp_boutcount']}")
        print(f"postcomp_boutcount: {row['postcomp_boutcount']}")
        print(f"opp_postcomp_boutcount: {row['opp_postcomp_boutcount']}")
        
        if row['win'] == '1':
            print(f"Result: {row['FIGHTER']} WON")
        else:
            print(f"Result: {row['opp_FIGHTER']} WON")

if __name__ == "__main__":
    # Process the data
    input_file = "data/final.csv"
    output_file = "data/final_with_swapped.csv"
    
    new_df = process_fight_data(input_file, output_file)
    
    # Verify Ruffy's data
    verify_ruffy_data(new_df)
    
    print(f"\nDataset saved to {output_file}")
    print(f"Original size: {len(pd.read_csv(input_file))} rows")
    print(f"New size: {len(new_df)} rows")
    print(f"Expected size: {len(pd.read_csv(input_file)) * 2} rows")
