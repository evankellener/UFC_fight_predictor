#!/usr/bin/env python3
"""
Fixed ROI Calculator for UFC Fight Predictions
Addresses issues with odds handling, profit calculations, and data filtering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    - min_odds: minimum American odds (default -500)
    - max_odds: maximum American odds (default +500)
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

def calculate_roi_fixed(odds_table_path, vegas_data_path, vegas_cols=None, stake=100):
    """
    Fixed ROI calculation with proper odds handling and profit calculation
    """
    print("=== FIXED ROI CALCULATION ===")
    
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
    
    best_night = night_stats['profit'].idxmax()
    worst_night = night_stats['profit'].idxmin()
    
    print(f"\nBest night: {best_night.date()} | Profit: ${night_stats.loc[best_night, 'profit']:.2f} | Win rate: {night_stats.loc[best_night, 'win']:.1%}")
    print(f"Worst night: {worst_night.date()} | Profit: ${night_stats.loc[worst_night, 'profit']:.2f} | Win rate: {night_stats.loc[worst_night, 'win']:.1%}")
    
    # 13. Create comprehensive visualizations
    create_roi_visualizations(picks)
    
    return picks

def create_roi_visualizations(picks):
    """Create comprehensive ROI visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Cumulative Profit Over Time
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(picks['DATE'], picks['cum_profit'], linewidth=2, color='green', alpha=0.8)
    ax1.fill_between(picks['DATE'], picks['cum_profit'], alpha=0.3, color='green')
    ax1.set_title('Cumulative Profit Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Profit ($)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 2. Cumulative ROI Over Time
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(picks['DATE'], picks['cum_roi'] * 100, linewidth=2, color='blue', alpha=0.8)
    ax2.fill_between(picks['DATE'], picks['cum_roi'] * 100, alpha=0.3, color='blue')
    ax2.set_title('Cumulative ROI Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative ROI (%)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 3. Profit per Fight
    ax3 = plt.subplot(3, 3, 3)
    colors = ['green' if p > 0 else 'red' for p in picks['profit']]
    ax3.bar(range(len(picks)), picks['profit'], color=colors, alpha=0.7)
    ax3.set_title('Profit per Fight', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Profit ($)')
    ax3.set_xlabel('Fight Number')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 4. Win Rate by Month
    ax4 = plt.subplot(3, 3, 4)
    picks['month'] = picks['DATE'].dt.to_period('M')
    monthly_stats = picks.groupby('month').agg({
        'win': 'mean',
        'profit': 'sum',
        'edge': 'mean'
    })
    
    ax4.bar(range(len(monthly_stats)), monthly_stats['win'] * 100, 
            color='skyblue', alpha=0.7)
    ax4.set_title('Win Rate by Month', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_xlabel('Month')
    ax4.set_xticks(range(len(monthly_stats)))
    ax4.set_xticklabels([str(m) for m in monthly_stats.index], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Edge Distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(picks['edge'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax5.set_title('Edge Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Edge (Model Prob - Vegas Prob)')
    ax5.set_ylabel('Frequency')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax5.grid(True, alpha=0.3)
    
    # 6. Profit vs Edge Scatter
    ax6 = plt.subplot(3, 3, 6)
    scatter = ax6.scatter(picks['edge'], picks['profit'], 
                         c=picks['win'], cmap='RdYlGn', alpha=0.6)
    ax6.set_title('Profit vs Edge', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Edge')
    ax6.set_ylabel('Profit ($)')
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Win (1=Yes, 0=No)')
    
    # 7. Rolling ROI (30-fight window)
    ax7 = plt.subplot(3, 3, 7)
    window = 30
    if len(picks) >= window:
        rolling_roi = picks['profit'].rolling(window=window).sum() / (window * 100)
        ax7.plot(picks['DATE'], rolling_roi * 100, linewidth=2, color='orange')
        ax7.set_title(f'Rolling ROI ({window}-fight window)', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Rolling ROI (%)')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 8. Odds Comparison
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(picks['avg_vegas_odds'], picks['odds'], alpha=0.6)
    ax8.plot([picks['avg_vegas_odds'].min(), picks['avg_vegas_odds'].max()], 
             [picks['avg_vegas_odds'].min(), picks['avg_vegas_odds'].max()], 
             'r--', alpha=0.7, label='Perfect correlation')
    ax8.set_title('Model Odds vs Vegas Odds', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Vegas Odds')
    ax8.set_ylabel('Model Odds')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Monthly Profit
    ax9 = plt.subplot(3, 3, 9)
    monthly_profit = picks.groupby('month')['profit'].sum()
    colors = ['green' if p > 0 else 'red' for p in monthly_profit]
    ax9.bar(range(len(monthly_profit)), monthly_profit, color=colors, alpha=0.7)
    ax9.set_title('Monthly Profit', fontsize=14, fontweight='bold')
    ax9.set_ylabel('Profit ($)')
    ax9.set_xlabel('Month')
    ax9.set_xticks(range(len(monthly_profit)))
    ax9.set_xticklabels([str(m) for m in monthly_profit.index], rotation=45)
    ax9.grid(True, alpha=0.3)
    ax9.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('roi_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed analysis
    create_detailed_analysis(picks)

def create_detailed_analysis(picks):
    """Create detailed analysis tables and additional insights"""
    
    print("\n=== DETAILED ANALYSIS ===")
    
    # 1. Performance by edge ranges
    print("\nPerformance by Edge Ranges:")
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
    
    # 2. Best and worst individual bets
    print(f"\nTop 5 Best Bets:")
    best_bets = picks.nlargest(5, 'profit')[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds', 'odds', 'win', 'profit', 'edge']]
    print(best_bets.to_string(index=False))
    
    print(f"\nTop 5 Worst Bets:")
    worst_bets = picks.nsmallest(5, 'profit')[['DATE', 'BOUT', 'FIGHTER', 'avg_vegas_odds', 'odds', 'win', 'profit', 'edge']]
    print(worst_bets.to_string(index=False))
    
    # 3. Monthly breakdown
    print(f"\nMonthly Performance:")
    monthly = picks.groupby('month').agg({
        'win': ['count', 'mean'],
        'profit': ['sum', 'mean'],
        'edge': 'mean'
    }).round(3)
    monthly.columns = ['Bets', 'Win_Rate', 'Total_Profit', 'Avg_Profit', 'Avg_Edge']
    print(monthly.to_string())

if __name__ == "__main__":
    # Example usage
    roi_df = calculate_roi_fixed(
        odds_table_path='../data/tmp/odds_table.csv',
        vegas_data_path='final_with_odds_filtered.csv',
        stake=100
    )
