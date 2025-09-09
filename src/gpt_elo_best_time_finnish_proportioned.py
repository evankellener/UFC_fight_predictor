import pandas as pd
from collections import defaultdict

class EnhancedElo:
    def __init__(self, k_factor=32, base_elo=1500):
        self.k = k_factor
        self.base_elo = base_elo
        self.elo_dict = {}
        self.last_fight_date = {}
        self.finish_factor = defaultdict(float)
        
    def get_elo(self, fighter):
        return self.elo_dict.get(fighter, self.base_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 220))

    def update_elo(self, rating_a, rating_b, score_a, k_a):
        expected_a = self.expected_score(rating_a, rating_b)
        return rating_a + k_a * (score_a - expected_a)

    def calculate_finish_factor(self, fighter, row, result):
        """Dynamically adjust finish factor based on performance"""
        current_factor = self.finish_factor.get(fighter, 1.0)
        
        # Check if fighter finished the fight
        if result == 1 and (row['ko'] == 1 or row['subw'] == 1):
            # Increase factor for finishes, but cap at 1.5
            new_factor = min(1.5, current_factor + 0.05)
        elif result == 0 and (row['kod'] == 1 or row['subwd'] == 1):
            # Decrease factor for being finished
            new_factor = max(0.7, current_factor - 0.1)
        else:
            # Gradual regression to mean
            if current_factor > 1.0:
                new_factor = max(1.0, current_factor - 0.01)
            elif current_factor < 1.0:
                new_factor = min(1.0, current_factor + 0.02)
            else:
                new_factor = 1.0
        
        return new_factor

    def process_fights(self, df):
        df = df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values(by='DATE').reset_index(drop=True)

        df["precomp_elo"] = self.base_elo
        df["opp_precomp_elo"] = self.base_elo
        df["postcomp_elo"] = self.base_elo
        df["opp_postcomp_elo"] = self.base_elo

        for i, row in df.iterrows():
            fighter = row['FIGHTER']
            opponent = row['opp_FIGHTER']
            result = 1 if str(row['result']).strip().lower() == 'win' else 0

            # Get current Elo ratings
            fighter_elo = self.get_elo(fighter)
            opponent_elo = self.get_elo(opponent)
            
            # Calculate dynamic finish factor
            fighter_finish_factor = self.calculate_finish_factor(fighter, row, result)
            opponent_finish_factor = self.calculate_finish_factor(opponent, row, 1 - result)
            
            # Apply finish factors to K
            k_fighter = self.k * fighter_finish_factor
            k_opponent = self.k * opponent_finish_factor
            
            # Store pre-fight Elo
            df.at[i, "precomp_elo"] = fighter_elo
            df.at[i, "opp_precomp_elo"] = opponent_elo
            
            # Update Elo ratings
            fighter_new = self.update_elo(fighter_elo, opponent_elo, result, k_fighter)
            opponent_new = self.update_elo(opponent_elo, fighter_elo, 1 - result, k_opponent)
            
            # Store post-fight Elo
            df.at[i, "postcomp_elo"] = fighter_new
            df.at[i, "opp_postcomp_elo"] = opponent_new
            
            # Update system state
            self.elo_dict[fighter] = fighter_new
            self.elo_dict[opponent] = opponent_new
            self.finish_factor[fighter] = fighter_finish_factor
            self.finish_factor[opponent] = opponent_finish_factor
            self.last_fight_date[fighter] = row['DATE']
            self.last_fight_date[opponent] = row['DATE']

        return df

    def top_n_fighters(self, n=10):
        sorted_fighters = sorted(self.elo_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {n} Fighters by Elo:")
        for fighter, elo in sorted_fighters[:n]:
            print(f"{fighter}: {elo:.2f}")
        return sorted_fighters[:n]