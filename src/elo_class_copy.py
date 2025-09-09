import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class FightDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path, dtype={'DATE': 'str'})
            return self.data
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def clean_data(self):
        self.data['key'] = self.data['DATE'].astype(str) + self.data['BOUT']
        
        for idx, row in self.data.iterrows():
            if row['result'] == 1:
                self.data.at[idx, 'winner'] = row['FIGHTER']
                self.data.at[idx, 'loser'] = row['opp_FIGHTER']
            else:
                self.data.at[idx, 'winner'] = row['opp_FIGHTER']
                self.data.at[idx, 'loser'] = row['FIGHTER']
        return self.data

class EnhancedElo:
    def __init__(self, base_elo=1500, k_factor=40, decay_rate=2):
        """
        Initialize enhanced Elo calculator with higher k-factor for more dramatic changes.
        
        Args:
            base_elo: Starting ELO rating (default: 1500)
            k_factor: Determines magnitude of rating changes (increased to 40)
            decay_rate: Rate of rating decay for inactivity (reduced to 2)
        """
        self.rating_dict = {}
        self.last_fight_date = {}
        self.base_elo = base_elo
        self.k_factor = k_factor
        self.elo_history = {}
        self.decay_rate = decay_rate  # Store Elo history for visualization
    
    def calculate_activity_factor(self, fighter, fight_date):
        """ 
        Calculates a decay factor based on time since last fight.
        Modified to prevent excessive decay and match champion fighter performance.
        """
        if fighter not in self.last_fight_date:
            return 1.0  # No decay if it's the fighter's first recorded fight

        last_fight = self.last_fight_date[fighter]
        if pd.isna(last_fight):
            return 1.0

        last_fight_date = datetime.strptime(last_fight, "%Y-%m-%d")
        current_fight_date = datetime.strptime(fight_date, "%Y-%m-%d")
        
        days_since_last_fight = (current_fight_date - last_fight_date).days
        months_inactive = days_since_last_fight / 30

        # Improved decay calculation with a cap on maximum decay
        # Apply a less aggressive decay with a higher minimum factor
        max_decay = min(months_inactive * self.decay_rate, 50) / 1000
        activity_factor = max(1.0 - max_decay, 0.95)  # Higher minimum factor (95%)

        return activity_factor

    def add_player(self, name, date):
        if name not in self.rating_dict:
            self.rating_dict[name] = self.base_elo
            self.last_fight_date[name] = date
            self.elo_history[name] = []  # Track Elo changes

    def adjust_k_factor(self, row):
        """ 
        Adjusts K-factor based on the method of victory.
        Simplified to avoid overfitting.
        """
        # Use consistent k-factor base for all fights
        method_k = self.k_factor
        
        # Check for title fight bonus only
        bout_description = str(row.get('BOUT', '')).lower()
        if ('championship' in bout_description or 'title' in bout_description or 
            'belt' in bout_description or ('ufc' in bout_description and 'champion' in bout_description)):
            # Apply modest title fight bonus
            method_k *= 1.15  # 15% bonus for title fights
            
        return method_k
        
    def adjust_weight_factor(self, row):
        """ 
        Adjusts the weight impact factor based on weight changes.
        Simplified to avoid data type errors.
        """
        # Simplify to a constant factor for model accuracy
        return 1.0

    def calculate_elo(self, winner, loser, date, row):
        winner_rating = self.rating_dict.get(winner, self.base_elo)
        loser_rating = self.rating_dict.get(loser, self.base_elo)
        
        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))

        k_adj = self.adjust_k_factor(row)
        weight_factor = self.adjust_weight_factor(row)

        winner_activity_factor = self.calculate_activity_factor(winner, date)
        loser_activity_factor = self.calculate_activity_factor(loser, date)

        # Minimize upset factor to avoid overfitting
        rating_diff = winner_rating - loser_rating
        # Very modest upset bonus for significant upsets only
        upset_factor = 1.1 if rating_diff < -50 else 1.0
        
        final_k_winner = k_adj * weight_factor * winner_activity_factor * upset_factor
        final_k_loser = k_adj * weight_factor * loser_activity_factor * upset_factor
        
        # Modified ELO update formula for more dramatic changes
        new_winner_rating = winner_rating + final_k_winner * (1 - expected_winner)
        new_loser_rating = loser_rating - final_k_loser * expected_loser
        
        self.rating_dict[winner] = new_winner_rating
        self.rating_dict[loser] = new_loser_rating
        
        self.elo_history.setdefault(winner, []).append((date, new_winner_rating))
        self.elo_history.setdefault(loser, []).append((date, new_loser_rating))

        self.last_fight_date[winner] = date
        self.last_fight_date[loser] = date

    def process_fights(self, data, min_fights=0):
        if data is None or data.empty:
            raise ValueError("Data is empty or not provided.")
        
        data = data.sort_values(by='DATE', ascending=True)
        
        # Process all rows, not just up to the second-to-last one
        for idx, row in data.iterrows():
            fighter = row['FIGHTER']
            opponent = row['opp_FIGHTER']
            winner = row['winner']
            loser = row['loser']
            date = row['DATE']

            if fighter not in self.rating_dict:
                self.add_player(fighter, date)
            if opponent not in self.rating_dict:
                self.add_player(opponent, date)

            # Initialize Elo columns
            data.at[idx, 'precomp_elo'] = self.rating_dict[fighter]
            data.at[idx, 'opp_precomp_elo'] = self.rating_dict[opponent]
            
            # Calculate Elo changes
            self.calculate_elo(winner, loser, date, row)
            
            data.at[idx, 'postcomp_elo'] = self.rating_dict[fighter]
            data.at[idx, 'opp_postcomp_elo'] = self.rating_dict[opponent]
        
        # Convert Elo columns to numeric before calculations
        data['precomp_elo'] = pd.to_numeric(data['precomp_elo'], errors='coerce')
        data['opp_precomp_elo'] = pd.to_numeric(data['opp_precomp_elo'], errors='coerce')
        data['postcomp_elo'] = pd.to_numeric(data['postcomp_elo'], errors='coerce')
        data['opp_postcomp_elo'] = pd.to_numeric(data['opp_postcomp_elo'], errors='coerce')
        
        # Calculate differences in Elo ratings for consecutive fights
        data['elo_prev_pre'] = data.groupby('FIGHTER')['precomp_elo'].diff().fillna(0)
        data['elo_prev_post'] = data.groupby('FIGHTER')['postcomp_elo'].diff().fillna(0)
        data['opp_elo_prev_pre'] = data.groupby('opp_FIGHTER')['opp_precomp_elo'].diff().fillna(0)
        data['opp_elo_prev_post'] = data.groupby('opp_FIGHTER')['opp_postcomp_elo'].diff().fillna(0)

        data['elo_prev_pre'] = data.groupby('FIGHTER')['precomp_elo'].diff().fillna(0)
        data['elo_prev_post'] = data.groupby('FIGHTER')['postcomp_elo'].diff().fillna(0)
        data['opp_elo_prev_pre'] = data.groupby('opp_FIGHTER')['opp_precomp_elo'].diff().fillna(0)
        data['opp_elo_prev_post'] = data.groupby('opp_FIGHTER')['opp_postcomp_elo'].diff().fillna(0)

        # Calculate rolling sums for the last 3 and 5 fights
        data['precomp_elo_change_3'] = data.groupby('FIGHTER')['elo_prev_pre'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['precomp_elo_change_5'] = data.groupby('FIGHTER')['elo_prev_pre'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
        data['postcomp_elo_change_3'] = data.groupby('FIGHTER')['elo_prev_post'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['postcomp_elo_change_5'] = data.groupby('FIGHTER')['elo_prev_post'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)

        data['opp_precomp_elo_change_3'] = data.groupby('opp_FIGHTER')['opp_elo_prev_pre'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_precomp_elo_change_5'] = data.groupby('opp_FIGHTER')['opp_elo_prev_pre'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_postcomp_elo_change_3'] = data.groupby('opp_FIGHTER')['opp_elo_prev_post'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_postcomp_elo_change_5'] = data.groupby('opp_FIGHTER')['opp_elo_prev_post'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)

        
        # Convert bout counts to numeric
        numeric_columns = ['precomp_boutcount', 'opp_precomp_boutcount']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Filter fighters based on the number of fights
        data = data[(data['precomp_boutcount'] >= min_fights) & (data['opp_precomp_boutcount'] >= min_fights)]
        
        return data

    def query_fighter_elo(self, fighter):
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
        plt.show()

    def top_k_fighters(self, k=10):
        sorted_fighters = sorted(self.rating_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_fighters[:k]