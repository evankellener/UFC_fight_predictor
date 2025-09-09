import pandas as pd
from collections import defaultdict
import numpy as np

class EnhancedElo:
    def __init__(self, k_factor=40, base_elo=1500, decay_rate=0.9995):
        self.k = k_factor
        self.base_elo = base_elo
        self.decay_rate = decay_rate  # For time-based decay
        self.elo_dict = {}
        self.last_fight_dict = {}
        self.streak_dict = defaultdict(int)
        self.fight_count = defaultdict(int)

    def get_elo(self, fighter, current_date):
        last_date = self.last_fight_dict.get(fighter, current_date)
        base_elo = self.base_elo
        raw_elo = self.elo_dict.get(fighter, base_elo)
        days_since = (current_date - last_date).days

        decay = self.decay_rate ** days_since
        decayed_elo = base_elo + (raw_elo - base_elo) * decay
        return decayed_elo

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_elo_custom_k(self, rating_a, rating_b, score_a, custom_k):
        expected_a = self.expected_score(rating_a, rating_b)
        return rating_a + custom_k * (score_a - expected_a)

    def process_fights(self, df):
        df = df.copy()
        df = df.sort_values(by='DATE', ascending=True).reset_index(drop=True)

        df["precomp_elo"] = 0
        df["opp_precomp_elo"] = 0
        df["postcomp_elo"] = 0
        df["opp_postcomp_elo"] = 0

        for i, row in df.iterrows():
            fighter = row['FIGHTER']
            opponent = row['opp_FIGHTER']
            fight_date = row['DATE']
            result_raw = row['result']
            result = int(result_raw) if not isinstance(result_raw, str) else int(result_raw.strip().lower() == 'win')

            fighter_elo = self.get_elo(fighter, fight_date)
            opponent_elo = self.get_elo(opponent, fight_date)

            # Method weight
            if (df.at[i, "ko"] == 1 or df.at[i, "ko"]):
                method_weight = 1.18
            elif (df.at[i, "subw"] == 1 or df.at[i, "subwd"]):
                method_weight = 2.07
            elif (df.at[i, "udec"] == 1 or df.at[i, "udecd"]):
                method_weight = 1.0
            elif (df.at[i, "sdec"] == 1 or df.at[i, "sdecd"]):
                method_weight = 0.02
            elif (df.at[i, "mdec"] == 1 or df.at[i, "mdecd"]):
                method_weight = 0.27
            else:
                method_weight = 1.0

            # Win streak multiplier
            streak_bonus = 1 + 0.05 * self.streak_dict[fighter]

            # Experience scaling (1 / sqrt(total_fights))
            exp_fighter = max(1, self.fight_count[fighter])
            exp_opp = max(1, self.fight_count[opponent])
            fighter_exp_scale = 1 / np.sqrt(exp_fighter)
            opp_exp_scale = 1 / np.sqrt(exp_opp)

            # Title fight bonus (assumes presence of 'title_fight' column — default to 0)
            importance = row.get('title_fight', 0)
            importance_scale = 2 if importance == 1 else 1

            k_fighter = self.k * method_weight * streak_bonus * fighter_exp_scale * importance_scale
            k_opp = self.k * method_weight * opp_exp_scale * importance_scale

            df.at[i, "precomp_elo"] = fighter_elo
            df.at[i, "opp_precomp_elo"] = opponent_elo

            fighter_new = self.update_elo_custom_k(fighter_elo, opponent_elo, result, k_fighter)
            opponent_new = self.update_elo_custom_k(opponent_elo, fighter_elo, 1 - result, k_opp)

            df.at[i, "postcomp_elo"] = fighter_new
            df.at[i, "opp_postcomp_elo"] = opponent_new

            # Update Elo and metadata
            self.elo_dict[fighter] = fighter_new
            self.elo_dict[opponent] = opponent_new

            self.last_fight_dict[fighter] = fight_date
            self.last_fight_dict[opponent] = fight_date

            self.fight_count[fighter] += 1
            self.fight_count[opponent] += 1

            if result == 1:
                self.streak_dict[fighter] += 1
                self.streak_dict[opponent] = 0
            else:
                self.streak_dict[opponent] += 1
                self.streak_dict[fighter] = 0

        return df

    def top_n_fighters(self, n=10):
        sorted_fighters = sorted(self.elo_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {n} Fighters by Elo:")
        for fighter, elo in sorted_fighters[:n]:
            print(f"{fighter}: {elo:.2f}")
        return sorted_fighters[:n]
