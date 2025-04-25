import pandas as pd
import os
import numpy as np
import pickle
from elo_feature_enhancer import EloFeatureEnhancer

def test_fighter_consistency(fighter_name):
    """Test the consistency of a single fighter's stats"""
    
    # Path to the stored fighter stats
    stats_path = os.path.join(os.getcwd(), 'data/tmp/recent_fighter_stats.pkl')
    
    # Load the stats if they exist
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            fighter_stats = pickle.load(f)
            
        if fighter_name in fighter_stats:
            print(f"Found stored stats for {fighter_name}")
            fighter_data = fighter_stats[fighter_name]
            
            # Print key ELO metrics
            print("\nKey ELO metrics:")
            elo_metrics = [key for key in fighter_data.keys() if ('elo' in key.lower() and 'precomp_' in key)]
            for metric in sorted(elo_metrics):
                print(f"{metric}: {fighter_data[metric]}")
            
            # Print key performance metrics
            print("\nKey performance metrics:")
            perf_metrics = [
                'precomp_sigstr_pm', 'precomp_tdavg', 'precomp_sapm', 
                'precomp_subavg', 'precomp_tddef', 'precomp_sigstr_perc'
            ]
            for metric in perf_metrics:
                if metric in fighter_data:
                    print(f"{metric}: {fighter_data[metric]}")
        else:
            print(f"No stored stats found for {fighter_name}")
    else:
        print(f"No stored fighter stats found at {stats_path}")

def test_enhanced_evaluate_generalization():
    """
    Load actual fight data and check if our enhanced evaluation would improve results
    This is just a simulation that doesn't run the full model
    """
    # Load interleaved data with ELO values
    data_path = os.path.join(os.getcwd(), 'data/tmp/interleaved_with_elo.csv')
    if not os.path.exists(data_path):
        print(f"Could not find data at {data_path}")
        return
        
    # Load recent fighter stats
    stats_path = os.path.join(os.getcwd(), 'data/tmp/recent_fighter_stats.pkl')
    if not os.path.exists(stats_path):
        print(f"Could not find stored fighter stats at {stats_path}")
        return
        
    # Load the data
    data = pd.read_csv(data_path, low_memory=False)
    
    # Load the stored fighter stats
    with open(stats_path, 'rb') as f:
        recent_fighter_stats = pickle.load(f)
    
    # Get the last year of fights for simulation
    data['DATE'] = pd.to_datetime(data['DATE'])
    latest_date = data['DATE'].max()
    one_year_ago = latest_date - pd.DateOffset(years=1)
    test_fights = data[data['DATE'] > one_year_ago]
    
    # Count how many of the test fights we could evaluate with stored stats
    fighters_with_stats = set(recent_fighter_stats.keys())
    
    # Check how many fights in test set have both fighters in our stored stats
    test_fight_pairs = list(zip(test_fights['FIGHTER'], test_fights['opp_FIGHTER']))
    covered_fights = sum(1 for f1, f2 in test_fight_pairs if f1 in fighters_with_stats and f2 in fighters_with_stats)
    
    total_test_fights = len(test_fights)
    print(f"Test fights in the last year: {total_test_fights}")
    print(f"Fights where both fighters have stored stats: {covered_fights} ({covered_fights/total_test_fights*100:.1f}%)")
    
    # Compare with current approach
    # Current approach uses previous fight data from the filtered dataset
    filtered_data_path = os.path.join(os.getcwd(), 'data/tmp/final.csv')
    if os.path.exists(filtered_data_path):
        filtered_data = pd.read_csv(filtered_data_path, low_memory=False)
        filtered_fighters = set(filtered_data['FIGHTER'].unique())
        
        old_covered = sum(1 for f1, f2 in test_fight_pairs 
                          if f1 in filtered_fighters and f2 in filtered_fighters)
        
        print(f"Fights covered by old approach: {old_covered} ({old_covered/total_test_fights*100:.1f}%)")
        print(f"Additional fights covered by new approach: {covered_fights - old_covered}")
        
        if covered_fights > old_covered:
            print("✅ The new approach provides better coverage for generalization testing")
        else:
            print("❌ The new approach does not improve coverage")
    else:
        print(f"Could not find filtered data at {filtered_data_path}")

if __name__ == "__main__":
    print("Testing fighter stats consistency and coverage")
    print("="*50)
    
    # Test a few fighters
    test_fighter_consistency("Alexandre Pantoja")
    test_fighter_consistency("Jon Jones")
    
    print("\n" + "="*50)
    print("Testing enhanced evaluation coverage")
    print("="*50)
    test_enhanced_evaluate_generalization()