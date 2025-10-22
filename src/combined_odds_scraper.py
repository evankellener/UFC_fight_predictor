#!/usr/bin/env python3
"""
Combined Odds Scraper Class
Integrates odds API fetching with improved filtering and clamping
"""

import re
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

class CombinedOddsScraper:
    """
    A comprehensive odds scraper that fetches odds from API and applies data quality fixes
    """
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv('ODDS_API_KEY')
        self.api_key = api_key
        self.sport = 'mma_mixed_martial_arts'
        self.regions = 'us'
        self.markets = 'h2h'
        self.odds_format = 'american'
        self.date_fmt = 'iso'
        self.lookback_days = 546
        self.main_bookmakers = ['draftkings', 'fanduel', 'betmgm', 'bet365', 'bovada']
        self.fuzzy_threshold = 0.8
        
        self.hist_url = f'https://api.the-odds-api.com/v4/historical/sports/{self.sport}/odds'
        
    def normalize(self, name: str) -> str:
        """Normalize fighter names for matching"""
        return re.sub(r'\W+', '', (name or '').lower())
    
    def similar(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a, b).ratio()
    
    def fetch_snapshot(self, ts_iso: str) -> List[Dict]:
        """Fetch odds snapshot for a specific timestamp"""
        try:
            r = requests.get(self.hist_url, params={
                'apiKey': self.api_key,
                'regions': self.regions,
                'markets': self.markets,
                'oddsFormat': self.odds_format,
                'dateFormat': self.date_fmt,
                'date': ts_iso,
            })
            r.raise_for_status()
            return r.json().get('data', [])
        except Exception as e:
            print(f"Error fetching data for {ts_iso}: {e}")
            return []
    
    def find_best_event(self, fn: str, on: str, row_date, ev_list: List[Dict]) -> Optional[Dict]:
        """Find the best-matching event for fighter vs opponent on row_date (±1 day)"""
        candidates = []
        for ev in ev_list:
            ev_date = ev['commence_dt'].date()
            if abs((ev_date - row_date).days) > 1:
                continue

            home, away = ev['home'], ev['away']
            # exact both-way match
            if {fn, on} == {home, away}:
                return ev

            # fuzzy both-way match
            sim1 = self.similar(fn, home) + self.similar(on, away)
            sim2 = self.similar(fn, away) + self.similar(on, home)
            if max(sim1, sim2) / 2 >= self.fuzzy_threshold:
                candidates.append((max(sim1, sim2) / 2, ev))

        # return highest-scoring fuzzy candidate, if any
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
        return None
    
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
                
                # Count original values
                original_count = df[col].notna().sum()
                
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
                
                print(f"  {col}: Clamped {clamped_positive + clamped_negative} values")
        
        return df
    
    def improved_filter_sportsbook_odds(self, df: pd.DataFrame, 
                                      thresholds: Optional[Dict[str, float]] = None,
                                      handle_missing_odds: str = "average_available") -> pd.DataFrame:
        """
        Improved filter for sportsbook odds that handles missing values intelligently.
        
        Args:
            df: DataFrame with odds data
            thresholds: Dict mapping column names to max absolute value allowed
            handle_missing_odds: How to handle missing odds:
                - "average_available": Average only available odds for each fight
                - "drop_row": Drop entire row if any odds are missing
                - "keep_nan": Keep NaN values as-is
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
    
    def scrape_odds(self, input_csv: str, output_csv: str = None) -> pd.DataFrame:
        """
        Main method to scrape odds and apply all data quality fixes
        
        Args:
            input_csv: Path to input CSV file with fight data
            output_csv: Path to save processed CSV file
            
        Returns:
            DataFrame with processed odds data
        """
        print("=== COMBINED ODDS SCRAPER ===")
        print(f"Loading data from: {input_csv}")
        
        # Load data
        df = pd.read_csv(input_csv, parse_dates=['DATE'])
        df['DATE'] = pd.to_datetime(df['DATE'], utc=True)
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=self.lookback_days)
        df = df[df['DATE'] >= cutoff].copy()
        
        print(f"Processing {len(df)} fights from {df['DATE'].min().date()} to {df['DATE'].max().date()}")
        
        # Prepare for matching
        df['f_norm'] = df['FIGHTER'].apply(self.normalize)
        df['o_norm'] = df['opp_FIGHTER'].apply(self.normalize)
        df['date_str'] = df['DATE'].dt.strftime('%Y-%m-%d')
        
        # 1) Fetch and build ev_list
        print("\n=== FETCHING ODDS DATA ===")
        raw = []
        for d in sorted(df['date_str'].unique()):
            base = datetime.fromisoformat(d)
            for delta in (0, 1):
                ts = (base + timedelta(days=delta)).strftime('%Y-%m-%dT00:00:00Z')
                print(f"  Fetching {ts}")
                raw.extend(self.fetch_snapshot(ts))
        
        # Dedupe events
        seen = {}
        for e in raw:
            seen[e['id']] = e
        ev_list = []
        for e in seen.values():
            ct = pd.to_datetime(e['commence_time'], utc=True)
            ev_list.append({
                'commence_dt': ct,
                'home': self.normalize(e['home_team']),
                'away': self.normalize(e['away_team']),
                'bookmakers': e.get('bookmakers', []),
            })
        
        print(f"Found {len(ev_list)} unique events")
        
        # 2) Prepare output cols
        for bk in self.main_bookmakers:
            df[f"{bk}_odds"] = None
        
        # 3) Row-by-row match with fuzzy fallback
        print("\n=== MATCHING FIGHTS TO ODDS ===")
        matches = 0
        for idx, row in df.iterrows():
            fn, on = row['f_norm'], row['o_norm']
            row_date = row['DATE'].date()
            ev = self.find_best_event(fn, on, row_date, ev_list)
            if not ev:
                continue
            
            matches += 1
            for bm in ev['bookmakers']:
                key = bm.get('key')
                if key not in self.main_bookmakers:
                    continue
                h2h = next((m for m in bm['markets'] if m['key'] == 'h2h'), None)
                if not h2h:
                    continue
                # pull this row's fighter price
                price = next((o['price'] for o in h2h['outcomes'] 
                              if self.normalize(o['name']) == fn), None)
                df.at[idx, f"{key}_odds"] = price
        
        print(f"Matched {matches} fights to odds data")
        
        # Clean up temporary columns
        df.drop(columns=['f_norm', 'o_norm', 'date_str'], inplace=True)
        
        # 4) Apply data quality fixes
        print("\n=== APPLYING DATA QUALITY FIXES ===")
        
        # Apply odds clamping
        df = self.clamp_odds_to_realistic_ranges(df)
        
        # Apply improved filtering
        df = self.improved_filter_sportsbook_odds(df, handle_missing_odds="average_available")
        
        # 5) Save results
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nProcessed data saved to: {output_csv}")
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total fights: {len(df)}")
        print(f"Fights with odds data: {df[self.main_bookmakers].notna().any(axis=1).sum()}")
        
        return df
    
    def analyze_roi(self, df: pd.DataFrame, stake: float = 100) -> Dict:
        """
        Analyze ROI for the processed odds data
        
        Args:
            df: DataFrame with processed odds data
            stake: Betting stake per fight
            
        Returns:
            Dictionary with ROI analysis results
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
    """Example usage of the CombinedOddsScraper"""
    scraper = CombinedOddsScraper()
    
    # Scrape odds with data quality fixes
    df = scraper.scrape_odds(
        input_csv='../data/tmp/final.csv',
        output_csv='final_with_odds_processed.csv'
    )
    
    # Analyze ROI
    roi_results = scraper.analyze_roi(df)
    
    return df, roi_results

if __name__ == "__main__":
    main()
