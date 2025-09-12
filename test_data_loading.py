#!/usr/bin/env python3
"""Test script to verify data loading works in production environment"""

import os
import sys
import pandas as pd

def test_data_loading():
    """Test if data files can be loaded properly"""
    print("Testing data loading...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test different data file paths
    data_paths = [
        'data/final.csv',
        '../data/final.csv',
        'data/tmp/final_min_fight1.csv',
        '../data/tmp/final_min_fight1.csv'
    ]
    
    for path in data_paths:
        print(f"\nTesting path: {path}")
        if os.path.exists(path):
            print(f"✓ File exists: {path}")
            try:
                df = pd.read_csv(path, low_memory=False)
                print(f"✓ Successfully loaded: {len(df)} rows")
                print(f"  Columns: {len(df.columns)}")
                print(f"  Sample fighters: {df['FIGHTER'].unique()[:5].tolist()}")
                return True
            except Exception as e:
                print(f"✗ Error loading {path}: {e}")
        else:
            print(f"✗ File not found: {path}")
    
    print("\nNo data file could be loaded successfully!")
    return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
