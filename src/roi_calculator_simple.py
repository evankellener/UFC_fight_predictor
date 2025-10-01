#!/usr/bin/env python3
"""
Simple ROI Calculator for UFC Fight Predictions (no matplotlib dependency)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def american_odds_to_probability(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def calculate_profit_correct(vegas_odds, stake=100, won=True):
    """
    Correctly calculate profit from American odds
    - vegas_odds: American odds (e.g., -150, +200)
    - stake: amount bet (default $100)
    - won: whether the bet won
    """
    if not won:
        return -stake
    
    if vegas_odds == 0:  # Handle zero odds case
        return 0  # No profit or loss on a 0 odds bet
    
    if vegas_odds > 0:
        # Positive odds: win (odds/100) * stake
        return (vegas_odds / 100) * stake
    else:
        # Negative odds: win (100/abs(odds)) * stake
        return (100 / abs(vegas_odds)) * stake

def filter_realistic_odds(df, vegas_cols, min_odds=-500, max_odds=500):
    """
    Filter out unrealistic odds but keep reasonable range
    """
    print(f"Original dataset size: {len(df)}")
    
    # Filter each vegas column
    for col in vegas_cols:
        if col in df.columns:
            before = len(df)
            df = df[(df[col] >= min_odds) & (df[col] <= max_odds) | df[col].isna()]
            after = len(df)
            print(f"Filtered {col}: {before} -> {after} rows")
    
    print(f"After odds filtering: {len(df)}")
    return df

def calculate_roi_simple(odds_table_path, vegas_data_path, vegas_cols=None, stake=100):
    """
    Simple ROI calculation with proper odds handling
    """
    print("=== SIMPLE ROI CALCULATION ===")
    
    # 1. Load data
    print("Loading data...")
    df_model = pd.read_csv(odds_table_path, parse_dates=['DATE'])
    df_vegas = pd.read_csv(vegas_data_path, parse_dates=['DATE'])
    
    # Handle timezone
    try:
        df_vegas['DATE'] = df_vegas['DATE'].dt.tz_convert(None)
    except:
        pass
    
    print(f"Model odds data: {len(df_model)} rows")
    print(f"Vegas data: {len(df_vegas)} rows")
    
    # 2. Set default vegas columns
    vegas_cols = vegas_cols or ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']
    
    # 3. Merge data
    key_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER']
    df = pd.merge(
        df_model[key_cols + ['odds']],
        df_vegas[key_cols + ['win'] + vegas_cols],
        on=key_cols,
        how='inner'
    )
    print(f"After merge: {len(df)} rows")
    
    # 4. Calculate average vegas odds (only from available odds)
    available_vegas_cols = [col for col in vegas_cols if col in df.columns]
    df['avg_vegas_odds'] = df[available_vegas_cols].mean(axis=1, skipna=True)
    
    # 5. Filter realistic odds
    df = filter_realistic_odds(df, available_vegas_cols, min_odds=-500, max_odds=500)
    
    # 6. Remove rows with missing critical data
    df = df.dropna(subset=['avg_vegas_odds', 'win', 'odds'])
    print(f"After removing missing data: {len(df)} rows")
    
    if len(df) == 0:
        print("ERROR: No data remaining after filtering!")
        return pd.DataFrame()
    
    # 7. Select model's favorite (lowest odds = highest probability)
    picks = df.groupby('BOUT').apply(lambda x: x.loc[x['odds'].idxmin()]).reset_index(drop=True)
    print(f"Unique fights selected: {len(picks)}")
    
    # 8. Calculate profit using corrected formula
    picks['stake'] = stake
    picks['profit'] = picks.apply(
        lambda row: calculate_profit_correct(row['avg_vegas_odds'], stake, row['win'] == 1), 
        axis=1
    )
    
    # 9. Calculate cumulative metrics
    picks = picks.sort_values('DATE')
    picks['cum_profit'] = picks['profit'].cumsum()
    picks['cum_stake'] = picks['stake'].cumsum()
    picks['cum_roi'] = picks['cum_profit'] / picks['cum_stake']
    
    # 10. Calculate additional metrics
    picks['implied_prob_vegas'] = picks['avg_vegas_odds'].apply(american_odds_to_probability)
    picks['implied_prob_model'] = picks['odds'].apply(american_odds_to_probability)
    picks['edge'] = picks['implied_prob_model'] - picks['implied_prob_vegas']
    
    # 11. Print summary statistics
    print("\n=== ROI SUMMARY ===")
    print(f"Total fights: {len(picks)}")
    print(f"Total stake: ${picks['cum_stake'].iloc[-1]:,.2f}")
    print(f"Total profit: ${picks['cum_profit'].iloc[-1]:,.2f}")
    print(f"Overall ROI: {picks['cum_roi'].iloc[-1]:.2%}")
    print(f"Win rate: {picks['win'].mean():.2%}")
    print(f"Average edge: {picks['edge'].mean():.3f}")
    
    # 12. Best and worst nights
    night_stats = picks.groupby('DATE').agg({
        'profit': 'sum',
        'win': 'mean',
        'edge': 'mean'
    }).round(3)
    
    if not night_stats.empty:
        best_night = night_stats['profit'].idxmax()
        worst_night = night_stats['profit'].idxmin()
        
        print(f"\nBest night: {best_night.date()} | Profit: ${night_stats.loc[best_night, 'profit']:.2f} | Win rate: {night_stats.loc[best_night, 'win']:.1%}")
        print(f"Worst night: {worst_night.date()} | Profit: ${night_stats.loc[worst_night, 'profit']:.2f} | Win rate: {night_stats.loc[worst_night, 'win']:.1%}")
    
    # 13. Performance by edge ranges
    print("\n=== PERFORMANCE BY EDGE RANGES ===")
    edge_ranges = [
        (-np.inf, -0.05, "Very Negative Edge"),
        (-0.05, -0.02, "Negative Edge"),
        (-0.02, 0.02, "Neutral Edge"),
        (0.02, 0.05, "Positive Edge"),
        (0.05, np.inf, "Very Positive Edge")
    ]
    
    for min_edge, max_edge, label in edge_ranges:
        mask = (picks['edge'] >= min_edge) & (picks['edge'] < max_edge)
        subset = picks[mask]
        
        if len(subset) > 0:
            win_rate = subset['win'].mean()
            avg_profit = subset['profit'].mean()
            total_profit = subset['profit'].sum()
            count = len(subset)
            
            print(f"{label:20}: {count:3d} bets | Win: {win_rate:.1%} | Avg Profit: ${avg_profit:6.2f} | Total: ${total_profit:8.2f}")
    
    # 14. Best and worst individual bets
    print(f"\n=== TOP 5 BEST BETS ===")
    best_bets = picks.nlargest(5, 'profit')[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds', 'odds', 'win', 'profit', 'edge']]
    print(best_bets.to_string(index=False))
    
    print(f"\n=== TOP 5 WORST BETS ===")
    worst_bets = picks.nsmallest(5, 'profit')[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds', 'odds', 'win', 'profit', 'edge']]
    print(worst_bets.to_string(index=False))
    
    # 15. Monthly breakdown
    print(f"\n=== MONTHLY PERFORMANCE ===")
    picks['month'] = picks['DATE'].dt.to_period('M')
    monthly = picks.groupby('month').agg({
        'win': ['count', 'mean'],
        'profit': ['sum', 'mean'],
        'edge': 'mean'
    }).round(3)
    monthly.columns = ['Bets', 'Win_Rate', 'Total_Profit', 'Avg_Profit', 'Avg_Edge']
    print(monthly.to_string())
    
    return picks

if __name__ == "__main__":
    # Example usage
    roi_df = calculate_roi_simple(
        odds_table_path='../data/tmp/odds_table.csv',
        vegas_data_path='final_with_odds_filtered.csv',
        stake=100
    )
