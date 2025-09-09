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
            if row['WINNER'] == row['FIGHTER']:
                self.data.at[idx, 'winner'] = row['FIGHTER']
                self.data.at[idx, 'loser'] = row['opp_FIGHTER']
            else:
                self.data.at[idx, 'winner'] = row['opp_FIGHTER']
                self.data.at[idx, 'loser'] = row['FIGHTER']

        return self.data

class EnhancedElo:
    def __init__(self, base_elo=1500, k_factor=40, decay_rate=2):
        self.rating_dict = {}
        self.last_fight_date = {}
        self.base_elo = base_elo
        self.k_factor = k_factor
        self.elo_history = {}
        self.decay_rate = decay_rate

    def calculate_activity_factor(self, fighter, fight_date):
        if fighter not in self.last_fight_date:
            return 1.0

        last_fight = self.last_fight_date[fighter]
        if pd.isna(last_fight):
            return 1.0

        last_fight_date = datetime.strptime(last_fight, "%Y-%m-%d")
        current_fight_date = datetime.strptime(fight_date, "%Y-%m-%d")

        days_since_last_fight = (current_fight_date - last_fight_date).days
        months_inactive = days_since_last_fight / 30
        max_decay = min(months_inactive * self.decay_rate, 50) / 1000
        activity_factor = max(1.0 - max_decay, 0.95)

        return activity_factor

    def add_player(self, name, date):
        if name not in self.rating_dict:
            self.rating_dict[name] = self.base_elo
            self.last_fight_date[name] = date
            self.elo_history[name] = []

    def adjust_k_factor(self, row):
        method_k = self.k_factor
        bout_description = str(row.get('BOUT', '')).lower()
        if ('championship' in bout_description or 'title' in bout_description or 
            'belt' in bout_description or ('ufc' in bout_description and 'champion' in bout_description)):
            method_k *= 1.15
        return method_k

    def adjust_weight_factor(self, row):
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

        rating_diff = winner_rating - loser_rating
        upset_factor = 1.1 if rating_diff < -50 else 1.0

        final_k_winner = k_adj * weight_factor * winner_activity_factor * upset_factor
        final_k_loser = k_adj * weight_factor * loser_activity_factor * upset_factor

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
        
        data = data.sort_values(by=['DATE', 'BOUT'])
        fighter_elo_history = {}

        for idx, row in data.iterrows():
            fighter = row['FIGHTER']
            opponent = row['opp_FIGHTER']
            winner = row['winner']
            loser = row['loser']
            date = row['DATE']

            self.add_player(fighter, date)
            self.add_player(opponent, date)

            fighter_pre_elo = self.rating_dict[fighter]
            opponent_pre_elo = self.rating_dict[opponent]

            self.calculate_elo(winner, loser, date, row)

            fighter_post_elo = self.rating_dict[fighter]
            opponent_post_elo = self.rating_dict[opponent]

            fighter_elo_history.setdefault(fighter, []).append({
                'date': date, 'bout': row['BOUT'], 'pre': fighter_pre_elo, 'post': fighter_post_elo, 'idx': idx
            })
            fighter_elo_history.setdefault(opponent, []).append({
                'date': date, 'bout': row['BOUT'], 'pre': opponent_pre_elo, 'post': opponent_post_elo, 'idx': idx
            })

        for fighter, history in fighter_elo_history.items():
            history.sort(key=lambda x: x['date'])
        
        '''

        for fighter, history in fighter_elo_history.items():
            for i in range(0, len(history)):
                if i != 0:
                    prev = history[i - 1]
                curr = history[i]
                data.at[curr['idx'], 'precomp_elo'] = prev['post']
                curr['pre'] = prev['post']
        '''
        for fighter, history in fighter_elo_history.items():
            for fight in history:
                idx = fight['idx']
                data.at[idx, 'precomp_elo'] = fight['pre']
                data.at[idx, 'postcomp_elo'] = fight['post']

        # Add opponent ELOs for each row
        for fighter, history in fighter_elo_history.items():
            for fight in history:
                idx = fight['idx']
                fighter_name = data.at[idx, 'FIGHTER']
                opponent_name = data.at[idx, 'opp_FIGHTER']
                data.at[idx, 'opp_precomp_elo'] = data.loc[idx, 'precomp_elo'] if opponent_name == fighter_name else self.rating_dict.get(opponent_name, self.base_elo)
                data.at[idx, 'opp_postcomp_elo'] = data.loc[idx, 'postcomp_elo'] if opponent_name == fighter_name else self.rating_dict.get(opponent_name, self.base_elo)

        # Convert Elo columns to numeric before calculations
        data['precomp_elo'] = pd.to_numeric(data['precomp_elo'], errors='coerce')
        data['opp_precomp_elo'] = pd.to_numeric(data['opp_precomp_elo'], errors='coerce')
        data['postcomp_elo'] = pd.to_numeric(data['postcomp_elo'], errors='coerce')
        data['opp_postcomp_elo'] = pd.to_numeric(data['opp_postcomp_elo'], errors='coerce')

        # Calculate differences in Elo ratings for consecutive fights
        data['precomp_elo_prev'] = data.groupby('FIGHTER')['precomp_elo'].diff().fillna(0)
        data['postcomp_elo_prev'] = data.groupby('FIGHTER')['postcomp_elo'].diff().fillna(0)
        data['opp_precomp_elo_prev'] = data.groupby('opp_FIGHTER')['opp_precomp_elo'].diff().fillna(0)
        data['opp_postcomp_elo_prev'] = data.groupby('opp_FIGHTER')['opp_postcomp_elo'].diff().fillna(0)

        # Calculate rolling sums for the last 3 and 5 fights
        data['precomp_elo_change_3'] = data.groupby('FIGHTER')['precomp_elo_prev'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['precomp_elo_change_5'] = data.groupby('FIGHTER')['precomp_elo_prev'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
        data['postcomp_elo_change_3'] = data.groupby('FIGHTER')['postcomp_elo_prev'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['postcomp_elo_change_5'] = data.groupby('FIGHTER')['postcomp_elo_prev'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)

        data['opp_precomp_elo_change_3'] = data.groupby('opp_FIGHTER')['opp_precomp_elo_prev'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_precomp_elo_change_5'] = data.groupby('opp_FIGHTER')['opp_precomp_elo_prev'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_postcomp_elo_change_3'] = data.groupby('opp_FIGHTER')['opp_postcomp_elo_prev'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_postcomp_elo_change_5'] = data.groupby('opp_FIGHTER')['opp_postcomp_elo_prev'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)

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
