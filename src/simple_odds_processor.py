#!/usr/bin/env python3
"""
Simple Odds Processor
Processes existing odds data with improved filtering and clamping
Works without external API dependencies
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

class SimpleOddsProcessor:
    """
    A simple odds processor that applies data quality fixes to existing odds data
    """
    
    def __init__(self):
        self.main_bookmakers = ['draftkings', 'fanduel', 'betmgm', 'bet365', 'bovada']
        
    def clamp_odds_to_realistic_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clamp extreme odds values to realistic ranges to fix data quality issues.
        
        Rules:
        1. If odds are between -110 and 110, clamp them to -100 or 100
        2. Keep other extreme values as they might be legitimate long shots
        """
        odds_columns = [col for col in df.columns if col.endswith('_odds')]
        
        print("=== APPLYING ODDS CLAMPING ===")
        
        for col in odds_columns:
            if col in df.columns:
                # Convert to numeric, errors become NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Apply clamping logic
                # If odds are between -110 and 110 (inclusive), clamp to boundaries
                mask_positive = (df[col] > 0) & (df[col] <= 110)
                mask_negative = (df[col] < 0) & (df[col] >= -110)
                
                # Clamp positive odds <= 110 to 100
                df.loc[mask_positive, col] = 100
                
                # Clamp negative odds >= -110 to -100
                df.loc[mask_negative, col] = -100
                
                # Count changes
                clamped_positive = mask_positive.sum()
                clamped_negative = mask_negative.sum()
                
                if clamped_positive + clamped_negative > 0:
                    print(f"  {col}: Clamped {clamped_positive + clamped_negative} values")
        
        return df
    
    def improved_filter_sportsbook_odds(self, df: pd.DataFrame, 
                                      thresholds: Optional[Dict[str, float]] = None,
                                      handle_missing_odds: str = "average_available") -> pd.DataFrame:
        """
        Improved filter for sportsbook odds that handles missing values intelligently.
        """
        if thresholds is None:
            thresholds = {
                'draftkings_odds': 5000,
                'fanduel_odds': 3500,
                'betmgm_odds': 5000,
                'bet365_odds': 5000,
                'bovada_odds': 5000,
            }
        
        odds_columns = list(thresholds.keys())
        
        print("=== APPLYING IMPROVED ODDS FILTERING ===")
        print(f"Handling missing odds: {handle_missing_odds}")
        
        # Step 1: Filter out-of-bounds values
        for col, max_abs in thresholds.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                before_count = df[col].notna().sum()
                df.loc[df[col].abs() > max_abs, col] = np.nan
                after_count = df[col].notna().sum()
                removed = before_count - after_count
                if removed > 0:
                    print(f"  {col}: {removed} out-of-bounds values set to NaN")
        
        # Step 2: Handle missing odds based on strategy
        if handle_missing_odds == "average_available":
            print("\n=== AVERAGING AVAILABLE ODDS ===")
            
            # Calculate average odds for each fight using only available odds
            df['avg_odds_calculated'] = df[odds_columns].mean(axis=1, skipna=True)
            
            # Count fights with different numbers of available odds
            for i in range(1, len(odds_columns) + 1):
                count = df[odds_columns].notna().sum(axis=1).eq(i).sum()
                if count > 0:
                    print(f"  {count} fights with {i} available odds")
            
        elif handle_missing_odds == "drop_row":
            print("\n=== DROPPING ROWS WITH MISSING ODDS ===")
            before_count = len(df)
            df = df.dropna(subset=odds_columns)
            after_count = len(df)
            dropped = before_count - after_count
            print(f"  Dropped {dropped} rows with missing odds")
            print(f"  Remaining rows: {after_count}")
        
        # Step 3: Final statistics
        print(f"\n=== FINAL STATISTICS ===")
        for col in odds_columns:
            if col in df.columns:
                total = len(df)
                valid = df[col].notna().sum()
                missing = total - valid
                print(f"  {col}: {valid}/{total} valid ({missing} missing)")
        
        return df
    
    def process_odds_data(self, input_csv: str, output_csv: str = None) -> pd.DataFrame:
        """
        Process existing odds data with all data quality fixes
        
        Args:
            input_csv: Path to input CSV file with odds data
            output_csv: Path to save processed CSV file
            
        Returns:
            DataFrame with processed odds data
        """
        print("=== SIMPLE ODDS PROCESSOR ===")
        print(f"Loading data from: {input_csv}")
        
        # Load data
        df = pd.read_csv(input_csv, parse_dates=['DATE'])
        print(f"Processing {len(df)} fights")
        
        # Apply data quality fixes
        print("\n=== APPLYING DATA QUALITY FIXES ===")
        
        # Apply odds clamping
        df = self.clamp_odds_to_realistic_ranges(df)
        
        # Apply improved filtering
        df = self.improved_filter_sportsbook_odds(df, handle_missing_odds="average_available")
        
        # Save results
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nProcessed data saved to: {output_csv}")
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total fights: {len(df)}")
        odds_cols = [col for col in df.columns if col.endswith('_odds')]
        fights_with_odds = df[odds_cols].notna().any(axis=1).sum()
        print(f"Fights with odds data: {fights_with_odds}")
        
        return df
    
    def analyze_roi(self, df: pd.DataFrame, stake: float = 100) -> Dict:
        """
        Analyze ROI for the processed odds data
        """
        print("\n=== ROI ANALYSIS ===")
        
        # Calculate average vegas odds
        odds_columns = [col for col in df.columns if col.endswith('_odds')]
        df['avg_vegas_odds'] = df[odds_columns].mean(axis=1, skipna=True)
        
        # Remove rows with missing critical data
        df_clean = df.dropna(subset=['avg_vegas_odds', 'win'])
        
        if len(df_clean) == 0:
            return {'error': 'No valid data for ROI analysis'}
        
        # Simulate model picks (pick fighter with most favorable odds)
        model_picks = []
        for bout in df_clean['BOUT'].unique():
            bout_data = df_clean[df_clean['BOUT'] == bout]
            if len(bout_data) == 2:  # Only complete fights
                bout_data = bout_data.copy()
                bout_data['odds_abs'] = abs(bout_data['avg_vegas_odds'])
                pick = bout_data.loc[bout_data['odds_abs'].idxmin()]
                model_picks.append(pick)
        
        if len(model_picks) == 0:
            return {'error': 'No valid picks for ROI analysis'}
        
        model_picks_df = pd.DataFrame(model_picks)
        
        # Calculate profit for each bet
        def calculate_profit(vegas_odds, stake=100, won=True):
            if not won:
                return -stake
            if abs(vegas_odds) < 0.01:  # Handle near-zero odds
                return 0
            if vegas_odds > 0:
                return (vegas_odds / 100) * stake
            else:
                return (100 / abs(vegas_odds)) * stake
        
        model_picks_df['profit'] = model_picks_df.apply(
            lambda row: calculate_profit(row['avg_vegas_odds'], stake, row['win'] == 1), 
            axis=1
        )
        
        # Calculate ROI metrics
        total_stake = len(model_picks_df) * stake
        total_profit = model_picks_df['profit'].sum()
        roi = total_profit / total_stake if total_stake > 0 else 0
        win_rate = model_picks_df['win'].mean()
        
        results = {
            'total_bets': len(model_picks_df),
            'total_stake': total_stake,
            'total_profit': total_profit,
            'roi': roi,
            'win_rate': win_rate,
            'wins': model_picks_df['win'].sum(),
            'losses': (model_picks_df['win'] == 0).sum()
        }
        
        print(f"Total bets: {results['total_bets']}")
        print(f"Total stake: ${results['total_stake']:,.2f}")
        print(f"Total profit: ${results['total_profit']:,.2f}")
        print(f"ROI: {results['roi']:.2%}")
        print(f"Win rate: {results['win_rate']:.2%}")
        print(f"Wins: {results['wins']}, Losses: {results['losses']}")
        
        return results

def main():
    """Example usage of the SimpleOddsProcessor"""
    processor = SimpleOddsProcessor()
    
    # Process existing odds data with data quality fixes
    df = processor.process_odds_data(
        input_csv='final_with_odds.csv',
        output_csv='final_with_odds_processed.csv'
    )
    
    # Analyze ROI
    roi_results = processor.analyze_roi(df)
    
    return df, roi_results

if __name__ == "__main__":
    main()
