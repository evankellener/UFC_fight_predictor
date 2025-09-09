import pandas as pd
from collections import defaultdict


class EnhancedElo:
    def __init__(self, k_factor=40, base_elo=1500):
        self.k = k_factor
        self.base_elo = base_elo
        self.elo_dict = {}
        self.streak_dict = defaultdict(int)
        self.last_fight_dict = {}
        self.relative_weight_delta_affected = 0

    def get_elo(self, fighter):
        return self.elo_dict.get(fighter, self.base_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 220))

    def update_elo_custom_k(self, rating_a, rating_b, score_a, custom_k):
        expected_a = self.expected_score(rating_a, rating_b)
        return rating_a + custom_k * (score_a - expected_a)

    def relative_weight_modifier(self, f_nat, f_opp, fight_class):
        try:
            if pd.isna(fight_class):
                return 1.0
            delta_f = float(f_nat) - float(fight_class)
            delta_o = float(f_opp) - float(fight_class)
            relative_diff = delta_f - delta_o
            if abs(relative_diff) >= 1:
                self.relative_weight_delta_affected += 1
                return max(0.75, 1 - 0.08 * abs(relative_diff))
            return 1.0
        except:
            return 1.0

    def round_modifier(self, finish_round, time_format):
        try:
            round_str = str(time_format).strip().lower()
            max_rounds = 3
            if '5' in round_str:
                max_rounds = 5
            elif '1' in round_str:
                max_rounds = 1
            if pd.isna(finish_round) or int(finish_round) > max_rounds:
                return 1.0
            progress = int(finish_round) / max_rounds
            return 1 + (1 - progress) * 0.25  # up to 25% bonus for R1 finish
        except:
            return 1.0

    def process_fights(self, df):
        df = df.copy()
        df = df.sort_values(by='DATE').reset_index(drop=True)

        df["precomp_elo"] = 0
        df["opp_precomp_elo"] = 0
        df["postcomp_elo"] = 0
        df["opp_postcomp_elo"] = 0

        for i, row in df.iterrows():
            fighter = row['FIGHTER']
            opponent = row['opp_FIGHTER']
            result_raw = row['result']
            result = int(result_raw) if not isinstance(result_raw, str) else int(result_raw.strip().lower() == 'win')

            fighter_elo = self.get_elo(fighter)
            opponent_elo = self.get_elo(opponent)

            fight_date = row['DATE']
            if fighter in self.last_fight_dict:
                if (fight_date - self.last_fight_dict[fighter]).days > 365:
                    fighter_elo *= 0.978
            if opponent in self.last_fight_dict:
                if (fight_date - self.last_fight_dict[opponent]).days > 365:
                    opponent_elo *= 0.978

            if (row.get("ko") == 1 or row.get("ko")):
                method_weight = 1.3
            elif (row.get("subw") == 1 or row.get("subwd")):
                method_weight = 1.6
            elif (row.get("udec") == 1 or row.get("udecd")):
                method_weight = 1.0
            elif (row.get("sdec") == 1 or row.get("sdecd")):
                method_weight = 0.6
            elif (row.get("mdec") == 1 or row.get("mdecd")):
                method_weight = 0.8
            else:
                method_weight = 1.0

            streak_bonus_fighter = 1 + 0.15 * self.streak_dict[fighter]
            streak_bonus_opponent = 1 + 0.15 * self.streak_dict[opponent]

            weight_mod = self.relative_weight_modifier(
                row.get('precomp_weight_avg3'),
                row.get('opp_precomp_weight_avg3'),
                row.get('weight_of_fight')
            )

            #first round fighter bonus
            if row.get("round") == 1:
                method_weight *= 1.15
            #first round opponent bonus
            if row.get("opp_round") == 1:
                method_weight *= 1.15

            round_mod = self.round_modifier(row.get("round"), row.get("time_format"))

            k_fighter = self.k * method_weight * streak_bonus_fighter * weight_mod * round_mod
            k_opponent = self.k * method_weight * streak_bonus_opponent * weight_mod * round_mod

            df.at[i, "precomp_elo"] = fighter_elo
            df.at[i, "opp_precomp_elo"] = opponent_elo

            fighter_new = self.update_elo_custom_k(fighter_elo, opponent_elo, result, k_fighter)
            opponent_new = self.update_elo_custom_k(opponent_elo, fighter_elo, 1 - result, k_opponent)

            df.at[i, "postcomp_elo"] = fighter_new
            df.at[i, "opp_postcomp_elo"] = opponent_new

            self.elo_dict[fighter] = fighter_new
            self.elo_dict[opponent] = opponent_new

            if result == 1:
                self.streak_dict[fighter] += 1
                self.streak_dict[opponent] = 0
            else:
                self.streak_dict[opponent] += 1
                self.streak_dict[fighter] = 0

            self.last_fight_dict[fighter] = fight_date
            self.last_fight_dict[opponent] = fight_date

        return df

    def top_n_fighters(self, n=10):
        sorted_fighters = sorted(self.elo_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {n} Fighters by Elo:")
        for fighter, elo in sorted_fighters[:n]:
            print(f"{fighter}: {elo:.2f}")
        return sorted_fighters[:n]
