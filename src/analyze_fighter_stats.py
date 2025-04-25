import pandas as pd
import os
import numpy as np

class FighterStatsAnalyzer:
    """
    A class to analyze a specific fighter's stats and check consistency
    between postcomp values and precomp values of next fights.
    """
    
    def __init__(self, data_path):
        """
        Initialize with the path to the data file.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path, low_memory=False)
        print(f"Loaded data with {len(self.df)} rows")
        
        # Extract common metrics - focus on the most important ones
        self.common_metrics = [
            'elo', 'elo_diff', 'elo_change_3', 'elo_change_5',
            'sigstr_pm', 'tdavg', 'sapm', 'subavg', 'tddef', 
            'sigstr_perc', 'strdef', 'tdacc_perc', 'winsum', 'losssum'
        ]
        
    def analyze_fighter_stats(self, fighter_name):
        """
        Analyze a specific fighter's stats to check consistency
        between postcomp and precomp values.
        
        Args:
            fighter_name (str): Name of the fighter to analyze
        
        Returns:
            pd.DataFrame: DataFrame with comparison results
        """
        # Filter data for the specified fighter
        fighter_data = self.df[self.df['FIGHTER'] == fighter_name].copy()
        
        if len(fighter_data) == 0:
            print(f"No data found for fighter: {fighter_name}")
            return None
            
        print(f"Found {len(fighter_data)} fights for {fighter_name}")
        
        # Sort by date to analyze fights chronologically
        fighter_data['DATE'] = pd.to_datetime(fighter_data['DATE'])
        fighter_data.sort_values('DATE', inplace=True)
        
        # Create a results table
        result_rows = []
        
        # Iterate through fights and compare postcomp with next fight's precomp
        for i in range(len(fighter_data) - 1):
            current_fight = fighter_data.iloc[i]
            next_fight = fighter_data.iloc[i+1]
            
            current_date = current_fight['DATE']
            next_date = next_fight['DATE']
            current_opponent = current_fight.get('opp_FIGHTER', 'Unknown')
            next_opponent = next_fight.get('opp_FIGHTER', 'Unknown')
            
            # Focus on important metrics rather than all of them
            for base_metric in self.common_metrics:
                postcomp_col = f'postcomp_{base_metric}'
                precomp_col = f'precomp_{base_metric}'
                
                if postcomp_col in current_fight.index and precomp_col in next_fight.index:
                    postcomp_value = current_fight[postcomp_col]
                    precomp_value = next_fight[precomp_col]
                    
                    # Try to convert to numeric for comparison
                    try:
                        postcomp_value = pd.to_numeric(postcomp_value, errors='coerce')
                        precomp_value = pd.to_numeric(precomp_value, errors='coerce')
                        
                        # If either value is NaN, skip
                        if pd.isna(postcomp_value) or pd.isna(precomp_value):
                            continue
                            
                        # Calculate difference
                        diff = abs(postcomp_value - precomp_value)
                        match = diff < 0.01  # Allow small differences due to floating point
                        
                        result_rows.append({
                            'Fight Date': current_date,
                            'Next Fight Date': next_date,
                            'Opponent': current_opponent,
                            'Next Opponent': next_opponent,
                            'Metric': base_metric,
                            'Postcomp Value': postcomp_value,
                            'Next Precomp Value': precomp_value,
                            'Difference': diff,
                            'Match': match
                        })
                    except Exception as e:
                        print(f"Error processing {base_metric}: {e}")
        
        if not result_rows:
            print(f"No comparable fights found for {fighter_name}")
            return None
            
        # Create DataFrame from results
        results_df = pd.DataFrame(result_rows)
        
        # Calculate match percentage
        match_percentage = results_df['Match'].mean() * 100
        print(f"Match percentage: {match_percentage:.2f}%")
        
        # Show mismatched metrics
        mismatched = results_df[results_df['Match'] == False]
        if len(mismatched) > 0:
            print(f"Found {len(mismatched)} mismatched metrics out of {len(results_df)} ({len(mismatched)/len(results_df)*100:.2f}%)")
            # Sort by difference magnitude
            if 'Difference' in mismatched.columns:
                mismatched = mismatched.sort_values('Difference', ascending=False)
            print("\nTop mismatches (largest differences first):")
            display_cols = ['Fight Date', 'Next Fight Date', 'Opponent', 'Next Opponent', 'Metric', 
                           'Postcomp Value', 'Next Precomp Value', 'Difference']
            print(mismatched[display_cols].head(10).to_string())
        else:
            print("All metrics match perfectly.")
        
        # Analyze fight intervals
        if len(fighter_data) > 1:
            fight_dates = fighter_data['DATE']
            date_diffs = [(fight_dates.iloc[i+1] - fight_dates.iloc[i]).days for i in range(len(fight_dates)-1)]
            avg_interval = sum(date_diffs) / len(date_diffs)
            max_interval = max(date_diffs)
            print(f"\nFight interval analysis:")
            print(f"Average days between fights: {avg_interval:.1f}")
            print(f"Maximum days between fights: {max_interval}")
            
            # Check if there's correlation between long intervals and mismatches
            if len(mismatched) > 0:
                print("\nChecking correlation between fight intervals and mismatches...")
                mismatched_dates = set(mismatched['Fight Date'])
                intervals_with_mismatches = []
                intervals_without_mismatches = []
                
                for i in range(len(fighter_data) - 1):
                    current_date = fighter_data['DATE'].iloc[i]
                    interval = (fighter_data['DATE'].iloc[i+1] - current_date).days
                    
                    if current_date in mismatched_dates:
                        intervals_with_mismatches.append(interval)
                    else:
                        intervals_without_mismatches.append(interval)
                
                if intervals_with_mismatches and intervals_without_mismatches:
                    avg_interval_with_mismatches = sum(intervals_with_mismatches) / len(intervals_with_mismatches)
                    avg_interval_without_mismatches = sum(intervals_without_mismatches) / len(intervals_without_mismatches)
                    
                    print(f"Average interval with mismatches: {avg_interval_with_mismatches:.1f} days")
                    print(f"Average interval without mismatches: {avg_interval_without_mismatches:.1f} days")
                    
                    if avg_interval_with_mismatches > avg_interval_without_mismatches:
                        print("Longer intervals between fights appear to correlate with more mismatches.")
            
        return results_df
    
    def analyze_multiple_fighters(self, fighter_names):
        """
        Analyze stats for a list of fighters.
        
        Args:
            fighter_names (list): List of fighter names to analyze
        
        Returns:
            dict: Dictionary with analysis results for each fighter
        """
        results = {}
        match_percentages = {}
        
        for fighter in fighter_names:
            print(f"\n{'='*50}")
            print(f"Analyzing {fighter}...")
            result_df = self.analyze_fighter_stats(fighter)
            
            if result_df is not None:
                results[fighter] = result_df
                match_percentages[fighter] = result_df['Match'].mean() * 100
        
        # Print summary
        print("\n\n" + "="*80)
        print("SUMMARY OF FIGHTER STAT CONSISTENCY")
        print("="*80)
        
        for fighter, percentage in sorted(match_percentages.items(), key=lambda x: x[1]):
            print(f"{fighter}: {percentage:.2f}% match")
            
        overall_match = sum(match_percentages.values()) / len(match_percentages) if match_percentages else 0
        print(f"\nOverall match percentage: {overall_match:.2f}%")
        
        return results

# Example usage
if __name__ == "__main__":
    # Use absolute path
    data_path = "/Users/evankellener/Desktop/UFC_fight_predictor/data/tmp/interleaved_with_elo.csv"
    analyzer = FighterStatsAnalyzer(data_path)
    
    # Famous fighters to analyze
    fighters_to_analyze = [
        "Alexandre Pantoja",
        "Jon Jones",
        "Khabib Nurmagomedov",
        "Israel Adesanya",
        "Alex Pereira"
    ]
    
    all_results = analyzer.analyze_multiple_fighters(fighters_to_analyze)