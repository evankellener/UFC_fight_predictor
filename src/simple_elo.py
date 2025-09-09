import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class SimpleElo:
    def __init__(self, base_rating=1500, k=32, decay_rate=2):
        """
        Initialize a simple Elo calculator with specified parameters.
        
        Args:
            base_rating: Starting Elo rating for new fighters (default: 1500)
            k: K-factor determining magnitude of rating changes (default: 32)
            decay_rate: Points deducted per month of inactivity (default: 2)
        """
        self.rating_dict = {}  # Stores the Elo rating of each fighter
        self.last_fight_date = {}  # Stores the date of the last fight for each fighter
        self.base_rating = base_rating
        self.k = k
        self.decay_rate = decay_rate
        self.elo_history = {}  # Track Elo history for visualization
        self.post_fight_elo = {}  # Store postcomp_elo for each fighter to ensure continuity
    
    def add_player(self, name, date):
        """Add a new player to the Elo system with base rating."""
        if name not in self.rating_dict:
            self.rating_dict[name] = self.base_rating
            self.last_fight_date[name] = date
            self.elo_history[name] = []  # Initialize Elo history
    
    def calculate_elo(self, winner, loser, date, is_title_fight=False):
        """
        Calculate and update the Elo ratings after a match.
        
        Args:
            winner: Name of the winning fighter
            loser: Name of the losing fighter
            date: Date of the fight (YYYY-MM-DD format)
            is_title_fight: Whether this is a title fight (default: False)
        """
        # Ensure both fighters are in the system
        if winner not in self.rating_dict:
            self.add_player(winner, date)
        if loser not in self.rating_dict:
            self.add_player(loser, date)
            
        # Get current ratings
        winner_rating = self.rating_dict[winner]
        loser_rating = self.rating_dict[loser]

        # Calculate the expected scores
        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))

        # Adjust K factor for title fights (15% bonus for title fights)
        k_factor = self.k
        if is_title_fight:
            k_factor = k_factor * 1.15
        
        # Calculate base ELO changes
        elo_change = k_factor * (1 - expected_winner)
        
        # Add modest upset adjustment (10%) if the winner was expected to lose
        if expected_winner < 0.5:
            upset_bonus = elo_change * 0.1
            elo_change += upset_bonus
            
        # Update ratings based on match results
        new_winner_rating = winner_rating + elo_change
        new_loser_rating = loser_rating - k_factor * expected_loser
        
        # Ensure minimum ELO floor
        new_winner_rating = max(new_winner_rating, 1200)
        new_loser_rating = max(new_loser_rating, 1200)

        # Store updated ratings
        self.rating_dict[winner] = new_winner_rating
        self.rating_dict[loser] = new_loser_rating

        # Update the date of the last fight
        self.last_fight_date[winner] = date
        self.last_fight_date[loser] = date
        
        # Store history
        self.elo_history.setdefault(winner, []).append((date, new_winner_rating))
        self.elo_history.setdefault(loser, []).append((date, new_loser_rating))
    
    def process_fights(self, data):
        """
        Process all fights in the dataset and calculate Elo ratings.
        
        Args:
            data: DataFrame containing fight data with columns:
                  DATE, FIGHTER, opp_FIGHTER, result (1 if fighter won, 0 if lost)
        
        Returns:
            DataFrame with added Elo rating columns and rolling averages
        """
        if data is None or data.empty:
            raise ValueError("Data is empty or not provided.")
        
        # Reset the Elo rating state to ensure consistent calculation
        self.rating_dict = {}
        self.last_fight_date = {}
        self.elo_history = {}
        self.post_fight_elo = {}  # Reset post-fight Elo tracking
        
        # Create a copy of the data to avoid modifying the original
        data = data.copy()
        
        # Sort by date to ensure chronological processing
        data = data.sort_values(by='DATE', ascending=True)
        
        # Initialize Elo columns
        for col in ['precomp_elo', 'opp_precomp_elo', 'postcomp_elo', 'opp_postcomp_elo']:
            if col not in data.columns:
                data[col] = np.nan
        
        # Ensure DATE is properly formatted as a string
        data['DATE'] = data['DATE'].astype(str)
        
        # Create a unique fight ID to avoid processing the same fight twice
        data['fight_id'] = data['DATE'] + '_' + data['FIGHTER'].astype(str) + '_vs_' + data['opp_FIGHTER'].astype(str)
        
        # Keep track of processed fights
        processed_fights = set()
        
        # Create dictionaries to track fighter's most recent Elo
        fighter_last_elo = {}
        
        # Process all fights to calculate Elo ratings chronologically
        for idx, row in data.iterrows():
            fighter = str(row['FIGHTER'])
            opponent = str(row['opp_FIGHTER'])
            date = str(row['DATE'])
            fight_id = row['fight_id']
            
            # Skip if this fight has already been processed
            reverse_fight_id = date + '_' + opponent + '_vs_' + fighter
            if fight_id in processed_fights or reverse_fight_id in processed_fights:
                # Get values from our post_fight_elo tracking dictionary for perfect continuity
                orig_fight_rows = data[(data['DATE'] == date) & (data['FIGHTER'] == opponent) & (data['opp_FIGHTER'] == fighter)]
                
                if not orig_fight_rows.empty:
                    orig_fight_row = orig_fight_rows.iloc[0]
                    
                    # For duplicate fights, we need to ensure EXACT mirroring of values
                    # Fighter's precomp_elo must match opponent's opp_precomp_elo in original fight
                    # Fighter's postcomp_elo must match opponent's opp_postcomp_elo in original fight
                    
                    # Set exact values from the original fight
                    data.loc[idx, 'precomp_elo'] = orig_fight_row['opp_precomp_elo']
                    data.loc[idx, 'opp_precomp_elo'] = orig_fight_row['precomp_elo']
                    data.loc[idx, 'postcomp_elo'] = orig_fight_row['opp_postcomp_elo']
                    data.loc[idx, 'opp_postcomp_elo'] = orig_fight_row['postcomp_elo']
                    
                    # Double check results are consistent (winners gain, losers lose)
                    if row.get('result', 0) == 1:  # Current fighter won
                        # Ensure fighter (winner) has higher postcomp_elo than precomp_elo
                        fighter_pre = float(orig_fight_row['opp_precomp_elo'])
                        fighter_post = float(orig_fight_row['opp_postcomp_elo'])
                        
                        # If values are inconsistent, enforce proper gain/loss
                        if fighter_post <= fighter_pre:
                            # Fix the inconsistency - winner must gain points
                            data.loc[idx, 'postcomp_elo'] = fighter_pre + 16
                            
                            # Also fix the original fight record for consistency
                            data.loc[orig_fight_rows.index, 'opp_postcomp_elo'] = fighter_pre + 16
                            
                            # Update tracking dictionary
                            self.post_fight_elo[fighter] = fighter_pre + 16
                    else:  # Current fighter lost
                        # Ensure fighter (loser) has lower postcomp_elo than precomp_elo
                        fighter_pre = float(orig_fight_row['opp_precomp_elo'])
                        fighter_post = float(orig_fight_row['opp_postcomp_elo'])
                        
                        # If values are inconsistent, enforce proper gain/loss
                        if fighter_post >= fighter_pre:
                            # Fix the inconsistency - loser must lose points
                            data.loc[idx, 'postcomp_elo'] = fighter_pre - 16
                            
                            # Also fix the original fight record for consistency
                            data.loc[orig_fight_rows.index, 'opp_postcomp_elo'] = fighter_pre - 16
                            
                            # Update tracking dictionary
                            self.post_fight_elo[fighter] = fighter_pre - 16
                else:
                    # If we can't find the original fight (should not happen), use post_fight_elo
                    fighter_precomp_elo = self.post_fight_elo.get(fighter, self.base_rating)
                    opp_precomp_elo = self.post_fight_elo.get(opponent, self.base_rating)
                    
                    # Set pre-fight values
                    data.loc[idx, 'precomp_elo'] = fighter_precomp_elo
                    data.loc[idx, 'opp_precomp_elo'] = opp_precomp_elo
                    
                    # Calculate post-fight values - always ensure winners gain points, losers lose
                    if row.get('result', 0) == 1:  # Current fighter won
                        fighter_post = fighter_precomp_elo + 16
                        opp_post = opp_precomp_elo - 16
                    else:  # Current fighter lost
                        fighter_post = fighter_precomp_elo - 16
                        opp_post = opp_precomp_elo + 16
                        
                    data.loc[idx, 'postcomp_elo'] = fighter_post
                    data.loc[idx, 'opp_postcomp_elo'] = opp_post
                    
                    # Update tracking dictionary for perfect continuity
                    self.post_fight_elo[fighter] = fighter_post
                    self.post_fight_elo[opponent] = opp_post
                
                # This fight has been processed
                continue
                
            # Determine winner and loser
            if row.get('result', 0) == 1:
                winner, loser = fighter, opponent
            else:
                winner, loser = opponent, fighter

            # Set pre-fight Elo ratings based on fighter's history
            # If it's their first fight, use exactly base_rating (1500)
            # Otherwise use their last fight's postcomp_elo from our tracking dictionary
            
            # Check if this is the fighter's first ever fight
            fighter_first_fight = fighter not in self.post_fight_elo and fighter not in self.rating_dict
            # Check if this is the opponent's first ever fight
            opponent_first_fight = opponent not in self.post_fight_elo and opponent not in self.rating_dict
            
            # Get appropriate ELO values
            fighter_elo = self.base_rating if fighter_first_fight else self.post_fight_elo.get(fighter, self.base_rating)
            opponent_elo = self.base_rating if opponent_first_fight else self.post_fight_elo.get(opponent, self.base_rating)
            
            # Store the pre-fight Elo ratings
            data.loc[idx, 'precomp_elo'] = fighter_elo
            data.loc[idx, 'opp_precomp_elo'] = opponent_elo
            
            # Initialize or update the rating_dict with current pre-fight Elo
            self.rating_dict[fighter] = fighter_elo
            self.rating_dict[opponent] = opponent_elo
            
            # Record fight date for decay calculations
            if fighter not in self.last_fight_date:
                self.add_player(fighter, date)
            else:
                # Apply decay since last fight
                try:
                    last_date = datetime.strptime(self.last_fight_date[fighter], "%Y-%m-%d")
                    current_date = datetime.strptime(date, "%Y-%m-%d")
                    
                    months_inactive = (current_date - last_date).days / 30
                    if months_inactive > 0:
                        decay = min(self.decay_rate * months_inactive, 50)  # Cap the decay at 50 points
                        self.rating_dict[fighter] = max(self.rating_dict[fighter] - decay, 1200)  # Minimum 1200 rating
                        # Update pre-fight Elo after decay
                        data.loc[idx, 'precomp_elo'] = self.rating_dict[fighter]
                except Exception as e:
                    # If date parsing fails, don't apply decay
                    pass
                
                self.last_fight_date[fighter] = date
                
            if opponent not in self.last_fight_date:
                self.add_player(opponent, date)
            else:
                # Apply decay since last fight
                try:
                    last_date = datetime.strptime(self.last_fight_date[opponent], "%Y-%m-%d")
                    current_date = datetime.strptime(date, "%Y-%m-%d")
                    
                    months_inactive = (current_date - last_date).days / 30
                    if months_inactive > 0:
                        decay = min(self.decay_rate * months_inactive, 50)  # Cap the decay at 50 points
                        self.rating_dict[opponent] = max(self.rating_dict[opponent] - decay, 1200)  # Minimum 1200 rating
                        # Update pre-fight Elo after decay
                        data.loc[idx, 'opp_precomp_elo'] = self.rating_dict[opponent]
                except Exception as e:
                    # If date parsing fails, don't apply decay
                    pass
                
                self.last_fight_date[opponent] = date
            
            # Store current pre-fight values (after any decay)
            pre_fight_fighter_elo = self.rating_dict[fighter]
            pre_fight_opponent_elo = self.rating_dict[opponent]
            
            # Determine if this is a title fight (based on column or text in bout description if available)
            is_title_fight = False
            if 'BOUT' in row and isinstance(row['BOUT'], str):
                is_title_fight = 'title' in row['BOUT'].lower() or 'championship' in row['BOUT'].lower()
            elif 'bout' in row and isinstance(row['bout'], str):
                is_title_fight = 'title' in row['bout'].lower() or 'championship' in row['bout'].lower()
            
            # Calculate new Elo ratings based on fight outcome
            expected_winner_score = 1 / (1 + 10 ** ((pre_fight_opponent_elo - pre_fight_fighter_elo) / 400))
            expected_loser_score = 1 / (1 + 10 ** ((pre_fight_fighter_elo - pre_fight_opponent_elo) / 400))
            
            # Adjust K factor for title fights (15% bonus for title fights)
            k_factor = self.k
            if is_title_fight:
                k_factor = k_factor * 1.15
            
            # Calculate Elo changes for winner and loser
            if winner == fighter:
                # Fighter won - ALWAYS gain points
                elo_change = k_factor * (1 - expected_winner_score)
                if expected_winner_score < 0.5:  # upset bonus
                    elo_change += elo_change * 0.1
                
                # Ensure elo_change is positive (winner must gain points)
                elo_change = max(elo_change, 16)  # Minimum 16 points gain for winner
                loser_change = max(k_factor * expected_loser_score, 16)  # Minimum 16 points loss for loser
                
                # Update ratings
                new_fighter_elo = pre_fight_fighter_elo + elo_change
                new_opponent_elo = pre_fight_opponent_elo - loser_change
                
                # Ensure minimum ELO floor
                new_fighter_elo = max(new_fighter_elo, 1200)
                new_opponent_elo = max(new_opponent_elo, 1200)
                
                # Update rating dictionary and history
                self.rating_dict[fighter] = new_fighter_elo
                self.rating_dict[opponent] = new_opponent_elo
                
                # Store post-fight ratings in dataframe
                data.loc[idx, 'postcomp_elo'] = new_fighter_elo
                data.loc[idx, 'opp_postcomp_elo'] = new_opponent_elo
                
                # Update post-fight Elo for exact continuity between fights
                self.post_fight_elo[fighter] = new_fighter_elo
                self.post_fight_elo[opponent] = new_opponent_elo
                
            else:
                # Opponent won - ALWAYS gain points
                elo_change = k_factor * (1 - expected_winner_score)
                if expected_winner_score > 0.5:  # upset bonus for opponent
                    elo_change += elo_change * 0.1
                
                # Ensure elo_change is positive (winner must gain points)
                elo_change = max(elo_change, 16)  # Minimum 16 points gain for winner
                loser_change = max(k_factor * expected_loser_score, 16)  # Minimum 16 points loss for loser
                
                # Update ratings
                new_opponent_elo = pre_fight_opponent_elo + elo_change
                new_fighter_elo = pre_fight_fighter_elo - loser_change
                
                # Ensure minimum ELO floor
                new_fighter_elo = max(new_fighter_elo, 1200)
                new_opponent_elo = max(new_opponent_elo, 1200)
                
                # Update rating dictionary and history
                self.rating_dict[fighter] = new_fighter_elo
                self.rating_dict[opponent] = new_opponent_elo
                
                # Store post-fight ratings in dataframe
                data.loc[idx, 'postcomp_elo'] = new_fighter_elo
                data.loc[idx, 'opp_postcomp_elo'] = new_opponent_elo
                
                # Update post-fight Elo for exact continuity between fights
                self.post_fight_elo[fighter] = new_fighter_elo
                self.post_fight_elo[opponent] = new_opponent_elo
            
            # Update Elo history
            self.elo_history.setdefault(fighter, []).append((date, self.rating_dict[fighter]))
            self.elo_history.setdefault(opponent, []).append((date, self.rating_dict[opponent]))
            
            # Mark this fight as processed
            processed_fights.add(fight_id)
            
            # Update opponent rows for the same fight (the "other perspective" of the same fight)
            opp_rows = data[(data['DATE'] == date) & (data['FIGHTER'] == opponent) & (data['opp_FIGHTER'] == fighter)]
            if not opp_rows.empty:
                # For opponent rows, ensure exact mirror values for perfect consistency
                # The opponent's precomp_elo should exactly match our opp_precomp_elo
                data.loc[opp_rows.index, 'precomp_elo'] = pre_fight_opponent_elo
                data.loc[opp_rows.index, 'opp_precomp_elo'] = pre_fight_fighter_elo
                
                # The opponent's postcomp_elo should exactly match our opp_postcomp_elo
                data.loc[opp_rows.index, 'postcomp_elo'] = new_opponent_elo
                data.loc[opp_rows.index, 'opp_postcomp_elo'] = new_fighter_elo
                
                # Double check for winner gain / loser loss consistency
                if winner == fighter:
                    # Our fighter won, opponent lost - ensure gain/loss is correct
                    if new_opponent_elo >= pre_fight_opponent_elo:
                        # Inconsistency: loser shouldn't gain points
                        fixed_opp_elo = pre_fight_opponent_elo - 16
                        data.loc[opp_rows.index, 'postcomp_elo'] = fixed_opp_elo
                        data.loc[idx, 'opp_postcomp_elo'] = fixed_opp_elo
                        self.post_fight_elo[opponent] = fixed_opp_elo
                    
                    if new_fighter_elo <= pre_fight_fighter_elo:
                        # Inconsistency: winner shouldn't lose points
                        fixed_fighter_elo = pre_fight_fighter_elo + 16
                        data.loc[opp_rows.index, 'opp_postcomp_elo'] = fixed_fighter_elo
                        data.loc[idx, 'postcomp_elo'] = fixed_fighter_elo
                        self.post_fight_elo[fighter] = fixed_fighter_elo
                else:
                    # Our fighter lost, opponent won - ensure gain/loss is correct
                    if new_opponent_elo <= pre_fight_opponent_elo:
                        # Inconsistency: winner shouldn't lose points
                        fixed_opp_elo = pre_fight_opponent_elo + 16
                        data.loc[opp_rows.index, 'postcomp_elo'] = fixed_opp_elo
                        data.loc[idx, 'opp_postcomp_elo'] = fixed_opp_elo
                        self.post_fight_elo[opponent] = fixed_opp_elo
                    
                    if new_fighter_elo >= pre_fight_fighter_elo:
                        # Inconsistency: loser shouldn't gain points
                        fixed_fighter_elo = pre_fight_fighter_elo - 16
                        data.loc[opp_rows.index, 'opp_postcomp_elo'] = fixed_fighter_elo
                        data.loc[idx, 'postcomp_elo'] = fixed_fighter_elo
                        self.post_fight_elo[fighter] = fixed_fighter_elo
        
        # Convert Elo columns to numeric
        for col in ['precomp_elo', 'opp_precomp_elo', 'postcomp_elo', 'opp_postcomp_elo']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Initialize diff columns if they don't exist
        for col in ['elo_prev_pre', 'elo_prev_post', 'opp_elo_prev_pre', 'opp_elo_prev_post']:
            if col not in data.columns:
                data[col] = 0
                
        # Calculate differences in ELO ratings for consecutive fights, proper chronological order
        # First, calculate differences for each fighter chronologically
        for fighter in data['FIGHTER'].unique():
            fighter_data = data[data['FIGHTER'] == fighter].sort_values(by='DATE')
            if len(fighter_data) <= 1:
                continue
                
            for i in range(1, len(fighter_data)):
                idx = fighter_data.index[i]
                prev_idx = fighter_data.index[i-1]
                
                data.loc[idx, 'elo_prev_pre'] = float(fighter_data.iloc[i]['precomp_elo'] - fighter_data.iloc[i-1]['precomp_elo'])
                data.loc[idx, 'elo_prev_post'] = float(fighter_data.iloc[i]['postcomp_elo'] - fighter_data.iloc[i-1]['postcomp_elo'])
        
        # Do the same for opponents
        for fighter in data['opp_FIGHTER'].unique():
            fighter_data = data[data['opp_FIGHTER'] == fighter].sort_values(by='DATE')
            if len(fighter_data) <= 1:
                continue
                
            for i in range(1, len(fighter_data)):
                idx = fighter_data.index[i]
                prev_idx = fighter_data.index[i-1]
                
                data.loc[idx, 'opp_elo_prev_pre'] = float(fighter_data.iloc[i]['opp_precomp_elo'] - fighter_data.iloc[i-1]['opp_precomp_elo'])
                data.loc[idx, 'opp_elo_prev_post'] = float(fighter_data.iloc[i]['opp_postcomp_elo'] - fighter_data.iloc[i-1]['opp_postcomp_elo'])

        # Convert Elo columns to numeric to ensure consistent data types
        for col in ['precomp_elo', 'opp_precomp_elo', 'postcomp_elo', 'opp_postcomp_elo']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        # Initialize diff columns if they don't exist
        for col in ['elo_prev_pre', 'elo_prev_post', 'opp_elo_prev_pre', 'opp_elo_prev_post']:
            if col not in data.columns:
                data[col] = 0.0
            else:
                data[col] = data[col].astype(float)
            
        # Initialize all the rolling columns if they don't exist
        roll_cols = [
            'precomp_elo_change_3', 'precomp_elo_change_5', 
            'postcomp_elo_change_3', 'postcomp_elo_change_5',
            'opp_precomp_elo_change_3', 'opp_precomp_elo_change_5',
            'opp_postcomp_elo_change_3', 'opp_postcomp_elo_change_5'
        ]
        
        # Create columns with proper data type (float)
        for col in roll_cols:
            if col not in data.columns:
                data[col] = 0.0
            else:
                data[col] = data[col].astype(float)

        # Calculate rolling sums for the last 3 and 5 fights using proper chronological dataframes
        for fighter in data['FIGHTER'].unique():
            fighter_data = data[data['FIGHTER'] == fighter].sort_values(by='DATE')
            if len(fighter_data) < 1:
                continue
                
            # Calculate rolling sums
            fighter_data['rolling_3_pre'] = fighter_data['elo_prev_pre'].rolling(3, min_periods=1).sum()
            fighter_data['rolling_5_pre'] = fighter_data['elo_prev_pre'].rolling(5, min_periods=1).sum()
            fighter_data['rolling_3_post'] = fighter_data['elo_prev_post'].rolling(3, min_periods=1).sum()
            fighter_data['rolling_5_post'] = fighter_data['elo_prev_post'].rolling(5, min_periods=1).sum()
            
            # Update the original dataframe with proper type conversion
            for idx, row in fighter_data.iterrows():
                data.loc[idx, 'precomp_elo_change_3'] = float(row['rolling_3_pre'])
                data.loc[idx, 'precomp_elo_change_5'] = float(row['rolling_5_pre'])
                data.loc[idx, 'postcomp_elo_change_3'] = float(row['rolling_3_post'])
                data.loc[idx, 'postcomp_elo_change_5'] = float(row['rolling_5_post'])
        
        # Do the same for opponents
        for fighter in data['opp_FIGHTER'].unique():
            fighter_data = data[data['opp_FIGHTER'] == fighter].sort_values(by='DATE')
            if len(fighter_data) < 1:
                continue
                
            # Calculate rolling sums
            fighter_data['rolling_3_pre'] = fighter_data['opp_elo_prev_pre'].rolling(3, min_periods=1).sum()
            fighter_data['rolling_5_pre'] = fighter_data['opp_elo_prev_pre'].rolling(5, min_periods=1).sum()
            fighter_data['rolling_3_post'] = fighter_data['opp_elo_prev_post'].rolling(3, min_periods=1).sum()
            fighter_data['rolling_5_post'] = fighter_data['opp_elo_prev_post'].rolling(5, min_periods=1).sum()
            
            # Update the original dataframe with proper type conversion
            for idx, row in fighter_data.iterrows():
                data.loc[idx, 'opp_precomp_elo_change_3'] = float(row['rolling_3_pre'])
                data.loc[idx, 'opp_precomp_elo_change_5'] = float(row['rolling_5_pre'])
                data.loc[idx, 'opp_postcomp_elo_change_3'] = float(row['rolling_3_post'])
                data.loc[idx, 'opp_postcomp_elo_change_5'] = float(row['rolling_5_post'])
        
        # Drop temporary columns
        if 'fight_id' in data.columns:
            data = data.drop(columns=['fight_id'])
        
        # Fill NaN values in elo columns with defaults to avoid errors
        for col in roll_cols + ['precomp_elo', 'opp_precomp_elo', 'postcomp_elo', 'opp_postcomp_elo', 
                               'elo_prev_pre', 'elo_prev_post', 'opp_elo_prev_pre', 'opp_elo_prev_post']:
            if col in data.columns:
                data[col] = data[col].fillna(0)
            
        return data
    
    def query_fighter_elo(self, fighter):
        """
        Plot the Elo rating history for a specific fighter.
        
        Args:
            fighter: Name of the fighter to query
        """
        if fighter not in self.elo_history or not self.elo_history[fighter]:
            print(f"No Elo data found for {fighter}.")
            return
        
        history = self.elo_history[fighter]
        dates, elos = zip(*sorted(history))
        
        plt.figure(figsize=(10, 5))
        plt.plot(dates, elos, marker='o', linestyle='-', label=f'{fighter} Elo')
        plt.xlabel('Date')
        plt.ylabel('Elo Rating')
        plt.title(f'Elo Rating Progression for {fighter}')
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def top_k_fighters(self, k=10):
        """
        Return the top K fighters by Elo rating.
        
        Args:
            k: Number of top fighters to return (default: 10)
            
        Returns:
            List of (fighter_name, elo_rating) tuples sorted by rating
        """
        sorted_fighters = sorted(self.rating_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_fighters[:k]