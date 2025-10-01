#!/usr/bin/env python3
"""
Final Comprehensive ROI Analysis for UFC Fight Predictions
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
    
    # Statistical significance test
    print(f"\n=== STATISTICAL ANALYSIS ===")
    from scipy import stats
    
    # Test if ROI is significantly different from 0
    t_stat, p_value = stats.ttest_1samp(picks['profit'], 0)
    print(f"T-test for profit > 0: t={t_stat:.3f}, p={p_value:.3f}")
    
    if p_value < 0.05:
        print("✅ ROI is statistically significant (p < 0.05)")
    else:
        print("❌ ROI is not statistically significant (p >= 0.05)")
    
    # Sharpe ratio (risk-adjusted return)
    if picks['profit'].std() > 0:
        sharpe_ratio = picks['profit'].mean() / picks['profit'].std()
        print(f"Sharpe ratio: {sharpe_ratio:.3f}")
    
    return picks

# Function to create simple visualizations (if matplotlib is available)
def create_simple_plots(picks):
    """Create simple plots if matplotlib is available"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative profit
        axes[0,0].plot(picks['DATE'], picks['cum_profit'])
        axes[0,0].set_title('Cumulative Profit Over Time')
        axes[0,0].set_ylabel('Cumulative Profit ($)')
        axes[0,0].grid(True)
        
        # Cumulative ROI
        axes[0,1].plot(picks['DATE'], picks['cum_roi'] * 100)
        axes[0,1].set_title('Cumulative ROI Over Time')
        axes[0,1].set_ylabel('Cumulative ROI (%)')
        axes[0,1].grid(True)
        
        # Profit distribution
        axes[1,0].hist(picks['profit'], bins=20, alpha=0.7)
        axes[1,0].set_title('Profit Distribution')
        axes[1,0].set_xlabel('Profit ($)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(x=0, color='red', linestyle='--')
        
        # Edge vs Profit scatter
        scatter = axes[1,1].scatter(picks['edge'], picks['profit'], c=picks['win'], cmap='RdYlGn', alpha=0.6)
        axes[1,1].set_title('Edge vs Profit')
        axes[1,1].set_xlabel('Edge')
        axes[1,1].set_ylabel('Profit ($)')
        axes[1,1].axhline(y=0, color='red', linestyle='--')
        axes[1,1].axvline(x=0, color='red', linestyle='--')
        
        plt.tight_layout()
        plt.savefig('roi_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available - skipping plots")

if __name__ == "__main__":
    # Example usage
    roi_df = comprehensive_roi_analysis(
        odds_table_path='../data/tmp/odds_table.csv',
        vegas_data_path='final_with_odds_filtered.csv',
        stake=100
    )
    
    # Create plots if possible
    if not roi_df.empty:
        create_simple_plots(roi_df)
