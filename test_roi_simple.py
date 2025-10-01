#!/usr/bin/env python3
"""
Test script to run the simple ROI calculation
"""

import sys
import os
sys.path.append('src')

from roi_calculator_simple import calculate_roi_simple

def main():
    print("Testing Simple ROI Calculation...")
    
    # Check if files exist
    odds_table_path = 'data/tmp/odds_table.csv'
    vegas_data_path = 'src/final_with_odds_filtered.csv'
    
    if not os.path.exists(odds_table_path):
        print(f"ERROR: {odds_table_path} not found!")
        return
    
    if not os.path.exists(vegas_data_path):
        print(f"ERROR: {vegas_data_path} not found!")
        return
    
    # Run the simple ROI calculation
    try:
        roi_df = calculate_roi_simple(
            odds_table_path=odds_table_path,
            vegas_data_path=vegas_data_path,
            stake=100
        )
        
        if not roi_df.empty:
            print(f"\nSUCCESS: ROI calculation completed!")
            print(f"Final dataset shape: {roi_df.shape}")
            print(f"Columns: {list(roi_df.columns)}")
        else:
            print("WARNING: ROI calculation returned empty dataset")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
