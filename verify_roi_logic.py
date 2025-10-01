#!/usr/bin/env python3
"""
Verify ROI calculation logic step by step
"""

import pandas as pd
import numpy as np

def verify_roi_logic():
    print("=== ROI LOGIC VERIFICATION ===")
    
    # Load data
    df_model = pd.read_csv('data/tmp/odds_table.csv', parse_dates=['DATE'])
    df_vegas = pd.read_csv('src/final_with_odds_filtered.csv', parse_dates=['DATE'])
    
    # Fix timezone
    try:
        df_vegas['DATE'] = df_vegas['DATE'].dt.tz_convert(None)
    except:
        pass
    
    print("Step 1: Data loaded")
    print(f"Model odds: {df_model.shape[0]} rows")
    print(f"Vegas data: {df_vegas.shape[0]} rows")
    
    # Merge data
    df = pd.merge(
        df_model[['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'odds']],
        df_vegas[['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'win', 'draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']],
        on=['DATE', 'EVENT', 'BOUT', 'FIGHTER'],
        how='inner'
    )
    
    print(f"\nStep 2: Merged data: {df.shape[0]} rows")
    
    # Calculate average vegas odds
    df['avg_vegas_odds'] = df[['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']].mean(axis=1, skipna=True)
    
    print("\nStep 3: Average vegas odds calculated")
    print("Sample data:")
    print(df[['FIGHTER', 'odds', 'avg_vegas_odds', 'win']].head())
    
    # Filter realistic odds
    df = df[(df['avg_vegas_odds'] >= -500) & (df['avg_vegas_odds'] <= 500)]
    df = df.dropna(subset=['avg_vegas_odds', 'win', 'odds'])
    
    print(f"\nStep 4: After filtering: {df.shape[0]} rows")
    
    # Select model's favorite per fight (lowest odds = highest probability)
    picks = df.groupby('BOUT').apply(lambda x: x.loc[x['odds'].idxmin()]).reset_index(drop=True)
    
    print(f"\nStep 5: Selected {len(picks)} unique fights")
    print("Strategy: Bet on fighter with LOWEST model odds (highest probability)")
    print("Using VEGAS odds to calculate profit/loss")
    
    # Calculate profit correctly
    def calculate_profit(vegas_odds, stake=100, won=True):
        if not won:
            return -stake
        if vegas_odds > 0:
            return (vegas_odds / 100) * stake
        else:
            return (100 / abs(vegas_odds)) * stake
    
    picks['profit'] = picks.apply(lambda row: calculate_profit(row['avg_vegas_odds'], 100, row['win'] == 1), axis=1)
    
    # Calculate cumulative metrics
    picks = picks.sort_values('DATE')
    picks['cum_profit'] = picks['profit'].cumsum()
    picks['cum_stake'] = (picks.index + 1) * 100
    picks['cum_roi'] = picks['cum_profit'] / picks['cum_stake']
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total fights: {len(picks)}")
    print(f"Total stake: ${len(picks) * 100}")
    print(f"Total profit: ${picks['profit'].sum():.2f}")
    print(f"Final ROI: {picks['profit'].sum() / (len(picks) * 100) * 100:.2f}%")
    print(f"Win rate: {picks['win'].mean() * 100:.2f}%")
    
    # Check some individual calculations
    print(f"\n=== SAMPLE PROFIT CALCULATIONS ===")
    sample = picks[['FIGHTER', 'avg_vegas_odds', 'win', 'profit']].head(10)
    for idx, row in sample.iterrows():
        vegas_odds = row['avg_vegas_odds']
        won = row['win']
        profit = row['profit']
        if won:
            if vegas_odds > 0:
                expected = (vegas_odds / 100) * 100
            else:
                expected = (100 / abs(vegas_odds)) * 100
        else:
            expected = -100
        print(f"{row['FIGHTER']:20} | Vegas: {vegas_odds:8.1f} | Won: {won} | Profit: ${profit:8.2f} | Expected: ${expected:8.2f}")
    
    # Check ROI trends
    print(f"\n=== ROI TREND ANALYSIS ===")
    print("First 10 fights ROI progression:")
    for i in range(min(10, len(picks))):
        print(f"Fight {i+1}: Cumulative ROI = {picks['cum_roi'].iloc[i]:.2%}")
    
    print(f"\nLast 10 fights ROI progression:")
    for i in range(max(0, len(picks)-10), len(picks)):
        print(f"Fight {i+1}: Cumulative ROI = {picks['cum_roi'].iloc[i]:.2%}")
    
    # Monthly analysis
    picks['month'] = picks['DATE'].dt.to_period('M')
    monthly_roi = picks.groupby('month').apply(lambda x: x['profit'].sum() / (len(x) * 100))
    
    print(f"\n=== MONTHLY ROI ANALYSIS ===")
    for month, roi in monthly_roi.items():
        print(f"{month}: {roi:.2%}")
    
    return picks

if __name__ == "__main__":
    picks = verify_roi_logic()
