#!/usr/bin/env python3
"""
Simple test for odds outlier filtering
"""

import pandas as pd
import numpy as np

def filter_odds_outliers(df, odds_column, method='iqr', threshold=1.5):
    """
    Filter out statistical outliers in odds data.
    """
    # Remove NaN values
    odds_data = df[odds_column].dropna()
    
    if method == 'iqr':
        Q1 = odds_data.quantile(0.25)
        Q3 = odds_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Filter outliers
        filtered_df = df[(df[odds_column] >= lower_bound) & 
                       (df[odds_column] <= upper_bound)]
        
    elif method == 'zscore':
        z_scores = np.abs((odds_data - odds_data.mean()) / odds_data.std())
        filtered_df = df[z_scores <= threshold]
    
    print(f"Original fights: {len(df)}")
    print(f"After outlier filtering: {len(filtered_df)}")
    print(f"Removed {len(df) - len(filtered_df)} outlier fights")
    print(f"Bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")
    
    return filtered_df

def test_with_real_data():
    """Test with your actual odds data"""
    
    # Load your data
    print("Loading odds data...")
    df = pd.read_csv('src/final_with_odds.csv')
    
    # Get the odds columns
    odds_cols = ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bovada_odds']
    
    # Calculate average odds
    df['avg_vegas_odds'] = df[odds_cols].mean(axis=1, skipna=True)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Odds statistics:")
    print(f"Min: {df['avg_vegas_odds'].min()}")
    print(f"Max: {df['avg_vegas_odds'].max()}")
    print(f"Mean: {df['avg_vegas_odds'].mean():.2f}")
    print(f"Std: {df['avg_vegas_odds'].std():.2f}")
    
    # Show the most extreme values
    print(f"\nTop 10 highest odds:")
    print(df.nlargest(10, 'avg_vegas_odds')[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds']])
    
    print(f"\nTop 10 lowest odds:")
    print(df.nsmallest(10, 'avg_vegas_odds')[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds']])
    
    # Test filtering
    print("\n" + "="*60)
    print("Testing IQR outlier filtering...")
    filtered_df = filter_odds_outliers(df, 'avg_vegas_odds', method='iqr', threshold=2.0)
    
    # Show what was removed
    removed_df = df[~df.index.isin(filtered_df.index)]
    print(f"\nRemoved fights with extreme odds:")
    print(removed_df[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds']].sort_values('avg_vegas_odds', ascending=False))

if __name__ == "__main__":
    test_with_real_data() 