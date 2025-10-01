#!/usr/bin/env python3
"""
Simple ROI Analysis for UFC Fight Predictions (no external dependencies)
Use this in your notebook to replace the existing ROI calculation
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
    """Correctly calculate profit from American odds"""
    if not won:
        return -stake
    
    if vegas_odds == 0:  # Handle zero odds case
        return 0  # No profit or loss on a 0 odds bet
    
    if vegas_odds > 0:
        return (vegas_odds / 100) * stake
    else:
        return (100 / abs(vegas_odds)) * stake

def comprehensive_roi_analysis(odds_table_path, vegas_data_path, vegas_cols=None, stake=100):
    """
    Comprehensive ROI analysis with proper calculations and detailed reporting
    """
    print("=== COMPREHENSIVE ROI ANALYSIS ===")
    
    # 1. Load and merge data
    df_model = pd.read_csv(odds_table_path, parse_dates=['DATE'])
    df_vegas = pd.read_csv(vegas_data_path, parse_dates=['DATE'])
    
    try:
        df_vegas['DATE'] = df_vegas['DATE'].dt.tz_convert(None)
    except:
        pass
    
    vegas_cols = vegas_cols or ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']
    
    # Merge data
    key_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER']
    df = pd.merge(
        df_model[key_cols + ['odds']],
        df_vegas[key_cols + ['win'] + vegas_cols],
        on=key_cols,
        how='inner'
    )
    
    # Calculate average vegas odds
    available_vegas_cols = [col for col in vegas_cols if col in df.columns]
    df['avg_vegas_odds'] = df[available_vegas_cols].mean(axis=1, skipna=True)
    
    # Filter realistic odds (less aggressive than original)
    print(f"Original dataset: {len(df)} rows")
    for col in available_vegas_cols:
        if col in df.columns:
            before = len(df)
            df = df[(df[col] >= -500) & (df[col] <= 500) | df[col].isna()]
            after = len(df)
            print(f"Filtered {col}: {before} -> {after} rows")
    
    # Remove missing data
    df = df.dropna(subset=['avg_vegas_odds', 'win', 'odds'])
    print(f"After removing missing data: {len(df)} rows")
    
    if len(df) == 0:
        print("ERROR: No data remaining after filtering!")
        return pd.DataFrame()
    
    # Select model's favorite per fight
    picks = df.groupby('BOUT').apply(lambda x: x.loc[x['odds'].idxmin()]).reset_index(drop=True)
    
    # Calculate profits correctly
    picks['stake'] = stake
    picks['profit'] = picks.apply(
        lambda row: calculate_profit_correct(row['avg_vegas_odds'], stake, row['win'] == 1), 
        axis=1
    )
    
    # Calculate cumulative metrics
    picks = picks.sort_values('DATE')
    picks['cum_profit'] = picks['profit'].cumsum()
    picks['cum_stake'] = picks['stake'].cumsum()
    picks['cum_roi'] = picks['cum_profit'] / picks['cum_stake']
    
    # Calculate edge metrics
    picks['implied_prob_vegas'] = picks['avg_vegas_odds'].apply(american_odds_to_probability)
    picks['implied_prob_model'] = picks['odds'].apply(american_odds_to_probability)
    picks['edge'] = picks['implied_prob_model'] - picks['implied_prob_vegas']
    
    # Print comprehensive summary
    print(f"\n=== ROI SUMMARY ===")
    print(f"Total fights analyzed: {len(picks)}")
    print(f"Total stake: ${picks['cum_stake'].iloc[-1]:,.2f}")
    print(f"Total profit: ${picks['cum_profit'].iloc[-1]:,.2f}")
    print(f"Overall ROI: {picks['cum_roi'].iloc[-1]:.2%}")
    print(f"Win rate: {picks['win'].mean():.2%}")
    print(f"Average edge: {picks['edge'].mean():.3f}")
    print(f"Average profit per bet: ${picks['profit'].mean():.2f}")
    
    # Performance by edge ranges
    print(f"\n=== PERFORMANCE BY EDGE RANGES ===")
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
    
    # Monthly performance
    print(f"\n=== MONTHLY PERFORMANCE ===")
    picks['month'] = picks['DATE'].dt.to_period('M')
    monthly = picks.groupby('month').agg({
        'win': ['count', 'mean'],
        'profit': ['sum', 'mean'],
        'edge': 'mean'
    }).round(3)
    monthly.columns = ['Bets', 'Win_Rate', 'Total_Profit', 'Avg_Profit', 'Avg_Edge']
    print(monthly.to_string())
    
    # Best and worst bets
    print(f"\n=== TOP 5 BEST BETS ===")
    best_bets = picks.nlargest(5, 'profit')[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds', 'odds', 'win', 'profit', 'edge']]
    print(best_bets.to_string(index=False))
    
    print(f"\n=== TOP 5 WORST BETS ===")
    worst_bets = picks.nsmallest(5, 'profit')[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds', 'odds', 'win', 'profit', 'edge']]
    print(worst_bets.to_string(index=False))
    
    # Simple statistical analysis (without scipy)
    print(f"\n=== STATISTICAL ANALYSIS ===")
    
    # Basic statistics
    profit_mean = picks['profit'].mean()
    profit_std = picks['profit'].std()
    profit_median = picks['profit'].median()
    
    print(f"Profit statistics:")
    print(f"  Mean: ${profit_mean:.2f}")
    print(f"  Median: ${profit_median:.2f}")
    print(f"  Std Dev: ${profit_std:.2f}")
    
    # Simple t-test approximation
    if profit_std > 0:
        t_stat = profit_mean / (profit_std / np.sqrt(len(picks)))
        print(f"  T-statistic: {t_stat:.3f}")
        
        if abs(t_stat) > 2:
            print("✅ ROI appears statistically significant (|t| > 2)")
        else:
            print("❌ ROI may not be statistically significant (|t| <= 2)")
    
    # Sharpe ratio (risk-adjusted return)
    if profit_std > 0:
        sharpe_ratio = profit_mean / profit_std
        print(f"  Sharpe ratio: {sharpe_ratio:.3f}")
    
    # Win streak analysis
    picks['win_streak'] = (picks['win'] != picks['win'].shift()).cumsum()
    streak_lengths = picks.groupby('win_streak')['win'].count()
    max_win_streak = streak_lengths[picks.groupby('win_streak')['win'].first() == 1].max()
    max_loss_streak = streak_lengths[picks.groupby('win_streak')['win'].first() == 0].max()
    
    print(f"  Max win streak: {max_win_streak}")
    print(f"  Max loss streak: {max_loss_streak}")
    
    return picks

if __name__ == "__main__":
    # Example usage
    roi_df = comprehensive_roi_analysis(
        odds_table_path='../data/tmp/odds_table.csv',
        vegas_data_path='final_with_odds_filtered.csv',
        stake=100
    )
