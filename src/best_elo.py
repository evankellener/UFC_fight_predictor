import pandas as pd
from collections import defaultdict
import datetime
from datetime import timedelta

class EnhancedElo:
    def __init__(self, k_factor=34.6508, base_elo=1500):
        self.k = k_factor
        self.base_elo = base_elo
        self.elo_dict = {}
        self.streak_dict = defaultdict(int)
        self.last_fight_dict = {}
        self.relative_weight_delta_affected = 0

    def get_elo(self, fighter):
        return self.elo_dict.get(fighter, self.base_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 472.6467))

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
                return max(0.75, 1 - 0.1246 * abs(relative_diff))
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

    def get_experience_modifier(self, fighter_bouts, opponent_bouts):
        """Adjust K-factor based on experience differential."""
        if pd.isna(fighter_bouts) or pd.isna(opponent_bouts):
            return 1.0
        try:
            f_bouts = int(fighter_bouts)
            o_bouts = int(opponent_bouts)
            exp_diff = f_bouts - o_bouts
            if abs(exp_diff) < 5:
                return 1.0
            modifier = 1.0 - (exp_diff * 0.0604/ 10)
            return max(0.85, min(1.15, modifier))
        except:
            return 1.0

    def process_fights(self, df, decay_rate=0.9828):
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
                if (fight_date - self.last_fight_dict[fighter]).days > 274:
                    fighter_elo *= decay_rate
            if opponent in self.last_fight_dict:
                if (fight_date - self.last_fight_dict[opponent]).days > 274:
                    opponent_elo *= decay_rate

            if (row.get("ko") == 1 or row.get("ko")):
                method_weight = 1.0537
            elif (row.get("subw") == 1 or row.get("subwd")):
                method_weight = 1.2374
            elif (row.get("udec") == 1 or row.get("udecd")):
                method_weight = 1.3953
            elif (row.get("sdec") == 1 or row.get("sdecd")):
                method_weight = 0.5647
            elif (row.get("mdec") == 1 or row.get("mdecd")):
                method_weight = 0.5886
            else:
                method_weight = 1.0

            streak_bonus_fighter = 1 + 0.2881 * self.streak_dict[fighter]
            streak_bonus_opponent = 1 + 0.2881 * self.streak_dict[opponent]

            weight_mod = self.relative_weight_modifier(
                row.get('precomp_weight_avg3'),
                row.get('opp_precomp_weight_avg3'),
                row.get('weight_of_fight')
            )

            #first round fighter bonus
            if row.get("round") == 1:
                method_weight *= 0.1730
            #first round opponent bonus
            if row.get("opp_round") == 1:
                method_weight *= 0.1730

            round_mod = self.round_modifier(row.get("round"), row.get("time_format"))

            # Experience differential modifier
            exp_mod_fighter = self.get_experience_modifier(row.get('precomp_boutcount'), row.get('opp_precomp_boutcount'))
            exp_mod_opponent = self.get_experience_modifier(row.get('opp_precomp_boutcount'), row.get('precomp_boutcount'))

            k_fighter = self.k * method_weight * streak_bonus_fighter * weight_mod * round_mod * exp_mod_fighter
            k_opponent = self.k * method_weight * streak_bonus_opponent * weight_mod * round_mod * exp_mod_opponent

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

    def test_accuracy(self, df):
        df = df.copy()
        df = df.sort_values(by='DATE').reset_index(drop=True)
        #filter to data to only contain the last 365 days]
        df = df[df['DATE'] >= (pd.Timestamp.now() - timedelta(days=365))]

        #for only fights where precomp_boutcount and opp_precomp_boutcount are > 1
        #calculate the accuracy of elo testing how mnay time precomp_elo is greater than opp_precomp_elo where 'win' is 1
        # additionally if opp_precomp_elo is greater than precomp_elo and win is 0, then it is correct
        df = df[(df['precomp_boutcount'] > 1) & (df['opp_precomp_boutcount'] > 1)]
        correct = 0
        total = 0
        for _, row in df.iterrows():
            if (row['precomp_elo'] > row['opp_precomp_elo'] and row['win'] == 1) or (row['opp_precomp_elo'] > row['precomp_elo'] and row['win'] == 0):
                correct += 1
            total += 1

        return correct / total