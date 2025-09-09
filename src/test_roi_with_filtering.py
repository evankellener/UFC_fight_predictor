#!/usr/bin/env python3
"""
Test ROI calculation with outlier filtering
"""

from ensemble_model_best import FightOutcomeModel
import pandas as pd

def test_roi_with_filtering():
    """Test ROI calculation with outlier filtering"""
    
    # Initialize the model
    print("Initializing model...")
    model = FightOutcomeModel('../data/tmp/final.csv')
    
    # Train logistic regression before generating odds table
    print("Tuning logistic regression...")
    model.tune_logistic_regression()
    
    # Generate odds table
    print("Generating odds table...")
    model.generate_odds_table()
    
    # Calculate ROI with outlier filtering
    print("\nCalculating ROI with outlier filtering...")
    picks = model.calculate_roi(
        odds_table_path='../data/tmp/odds_table.csv',
        vegas_data_path='final_with_odds.csv',
        stake=100
    )
    
    print(f"\nFinal Results:")
    print(f"Total fights analyzed: {len(picks)}")
    print(f"Final ROI: {picks['cum_roi'].iloc[-1]:.4%}")
    print(f"Total profit: ${picks['cum_profit'].iloc[-1]:.2f}")
    print(f"Total stake: ${picks['cum_stake'].iloc[-1]:.2f}")
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"Average profit per fight: ${picks['profit'].mean():.2f}")
    print(f"Win rate: {(picks['win'] == 1).mean():.2%}")
    print(f"Number of winning bets: {(picks['win'] == 1).sum()}")
    print(f"Number of losing bets: {(picks['win'] == 0).sum()}")
    
    return picks

if __name__ == "__main__":
    picks = test_roi_with_filtering() 