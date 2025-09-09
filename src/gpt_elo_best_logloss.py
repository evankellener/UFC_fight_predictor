import pandas as pd
import numpy as np
from collections import defaultdict

class EnhancedElo:
    def __init__(self, k_factor=40, base_elo=1500):
        self.k = k_factor
        self.base_elo = base_elo
        self.elo_dict = {}
        self.streak_dict = defaultdict(int)
        self.last_fight_dict = {}

    def get_elo(self, fighter):
        return self.elo_dict.get(fighter, self.base_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 220))

    def update_elo_custom_k(self, rating_a, rating_b, score_a, custom_k):
        expected_a = self.expected_score(rating_a, rating_b)
        return rating_a + custom_k * (score_a - expected_a)

    def update_elo(self, rating_a, rating_b, score_a, method_weight):
        custom_k = self.k * method_weight
        return self.update_elo_custom_k(rating_a, rating_b, score_a, custom_k)

    def hybrid_decay(self, elo, days_inactive, base_decay=0.978, max_decay=0.95):
        if days_inactive <= 365:
            return elo
        # Linearly scale decay factor between base_decay and max_decay
        decay_scale = (min(days_inactive, 730) - 365) / 365
        decay_factor = base_decay - (base_decay - max_decay) * decay_scale
        return elo * decay_factor

    def process_fights(self, df):
        df = df.copy()
        df = df.sort_values(by='DATE', ascending=True).reset_index(drop=True)

        df["precomp_elo"] = 0
        df["opp_precomp_elo"] = 0
        df["postcomp_elo"] = 0
        df["opp_postcomp_elo"] = 0
        df["elo_confidence"] = 0

        for i, row in df.iterrows():
            fighter = row['FIGHTER']
            opponent = row['opp_FIGHTER']
            result_raw = row['result']
            result = int(result_raw) if not isinstance(result_raw, str) else int(result_raw.strip().lower() == 'win')

            fighter_elo = self.get_elo(fighter)
            opponent_elo = self.get_elo(opponent)
            fight_date = row['DATE']

            # Apply hybrid layoff decay
            if fighter in self.last_fight_dict:
                days_inactive = (fight_date - self.last_fight_dict[fighter]).days
                fighter_elo = self.hybrid_decay(fighter_elo, days_inactive)
            if opponent in self.last_fight_dict:
                days_inactive = (fight_date - self.last_fight_dict[opponent]).days
                opponent_elo = self.hybrid_decay(opponent_elo, days_inactive)

            # Method weight
            if (row.get("ko") == 1 or row.get("ko")):
                method_weight = 1.18
            elif (row.get("subw") == 1 or row.get("subwd")):
                method_weight = 2.07
            elif (row.get("udec") == 1 or row.get("udecd")):
                method_weight = 1.0
            elif (row.get("sdec") == 1 or row.get("sdecd")):
                method_weight = 0.02
            elif (row.get("mdec") == 1 or row.get("mdecd")):
                method_weight = 0.27
            else:
                method_weight = 1.0

            # Streak bonus
            streak_bonus_fighter = 1 + 0.05 * self.streak_dict[fighter]
            streak_bonus_opponent = 1 + 0.05 * self.streak_dict[opponent]

            k_fighter = self.k * method_weight * streak_bonus_fighter
            k_opponent = self.k * method_weight * streak_bonus_opponent

            df.at[i, "precomp_elo"] = fighter_elo
            df.at[i, "opp_precomp_elo"] = opponent_elo

            prob_win = self.expected_score(fighter_elo, opponent_elo)
            df.at[i, "elo_confidence"] = prob_win

            fighter_new = self.update_elo_custom_k(fighter_elo, opponent_elo, result, k_fighter)
            opponent_new = self.update_elo_custom_k(opponent_elo, fighter_elo, 1 - result, k_opponent)

            df.at[i, "postcomp_elo"] = fighter_new
            df.at[i, "opp_postcomp_elo"] = opponent_new

            self.elo_dict[fighter] = fighter_new
            self.elo_dict[opponent] = opponent_new

            # Update streaks
            if result == 1:
                self.streak_dict[fighter] += 1
                self.streak_dict[opponent] = 0
            else:
                self.streak_dict[opponent] += 1
                self.streak_dict[fighter] = 0

            # Update last fight dates
            self.last_fight_dict[fighter] = fight_date
            self.last_fight_dict[opponent] = fight_date

        return df

    def top_n_fighters(self, n=10):
        sorted_fighters = sorted(self.elo_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {n} Fighters by Elo:")
        for fighter, elo in sorted_fighters[:n]:
            print(f"{fighter}: {elo:.2f}")
        return sorted_fighters[:n]
