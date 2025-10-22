#!/usr/bin/env python3
"""
Run XGBoost ROI calculation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_model_best import FightOutcomeModel

print("💰 Running XGBoost ROI Calculation")
print("=" * 80)
print()

# Initialize model
print("📂 Loading data...")
fight_model = FightOutcomeModel("data/tmp/final.csv", random_seed=42)
print()

# Calculate ROI
print("🎲 Calculating ROI using XGBoost odds...")
print()

roi_df = fight_model.calculate_roi(
    odds_table_path='data/tmp/xgboost_odds_table.csv',
    vegas_data_path='final_with_odds_clamped.csv'
)

print()
print("=" * 80)
print("📊 ROI RESULTS")
print("=" * 80)
print()

if roi_df is not None and len(roi_df) > 0:
    print(f"Total bets placed: {len(roi_df)}")
    print(f"Wins: {roi_df['win'].sum()}")
    print(f"Losses: {(1 - roi_df['win']).sum()}")
    print(f"Win rate: {roi_df['win'].mean():.2%}")
    print()
    
    if 'profit' in roi_df.columns:
        total_profit = roi_df['profit'].sum()
        total_stake = len(roi_df) * 100  # $100 per bet
        roi_pct = (total_profit / total_stake) * 100
        
        print(f"💵 Financial Results:")
        print(f"   Total stake: ${total_stake:,.2f}")
        print(f"   Total profit/loss: ${total_profit:,.2f}")
        print(f"   ROI: {roi_pct:.2f}%")
        print()
        
        # Save results
        output_path = 'data/tmp/xgboost_roi_results.csv'
        roi_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")
    else:
        print("⚠️  Profit column not found in results")
        print(f"   Available columns: {', '.join(roi_df.columns)}")
else:
    print("❌ No ROI results returned")

print()
print("=" * 80)
print("✅ XGBoost ROI Calculation Complete!")
print("=" * 80)

