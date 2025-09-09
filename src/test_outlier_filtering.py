#!/usr/bin/env python3
"""
Test script for odds outlier filtering
"""

import pandas as pd
import numpy as np
from ensemble_model_best import FightOutcomeModel

def test_outlier_filtering():
    """Test the outlier filtering functionality"""
    
    # Create sample data with outliers
    np.random.seed(42)
    n_fights = 1000
    
    # Generate realistic odds data (mostly between -500 and +500)
    normal_odds = np.random.normal(0, 150, n_fights - 3)
    normal_odds = np.clip(normal_odds, -500, 500)
    
    # Add some extreme outliers
    outlier_odds = [3000, -2500, 2000]
    
    all_odds = np.concatenate([normal_odds, outlier_odds])
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'DATE': pd.date_range('2024-01-01', periods=n_fights),
        'EVENT': ['UFC Event'] * n_fights,
        'BOUT': [f'Fight_{i}' for i in range(n_fights)],
        'FIGHTER': [f'Fighter_{i}' for i in range(n_fights)],
        'avg_vegas_odds': all_odds,
        'win': np.random.choice([0, 1], n_fights)
    })
    
    print("Sample odds statistics:")
    print(f"Min: {df['avg_vegas_odds'].min()}")
    print(f"Max: {df['avg_vegas_odds'].max()}")
    print(f"Mean: {df['avg_vegas_odds'].mean():.2f}")
    print(f"Std: {df['avg_vegas_odds'].std():.2f}")
    
    # Test outlier filtering
    model = FightOutcomeModel("dummy_path")  # We only need the filtering method
    
    print("\n" + "="*50)
    print("Testing IQR outlier filtering...")
    filtered_df = model.filter_odds_outliers(df, 'avg_vegas_odds', method='iqr', threshold=1.5)
    
    print("\n" + "="*50)
    print("Testing Z-score outlier filtering...")
    filtered_df_z = model.filter_odds_outliers(df, 'avg_vegas_odds', method='zscore', threshold=3.0)
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"Original fights: {len(df)}")
    print(f"After IQR filtering: {len(filtered_df)}")
    print(f"After Z-score filtering: {len(filtered_df_z)}")
    
    # Show the outliers that were removed
    outliers_iqr = df[~df.index.isin(filtered_df.index)]
    outliers_z = df[~df.index.isin(filtered_df_z.index)]
    
    print(f"\nOutliers removed by IQR method:")
    print(outliers_iqr[['avg_vegas_odds']].sort_values('avg_vegas_odds', ascending=False))
    
    print(f"\nOutliers removed by Z-score method:")
    print(outliers_z[['avg_vegas_odds']].sort_values('avg_vegas_odds', ascending=False))

if __name__ == "__main__":
    test_outlier_filtering() 