import pandas as pd
from collections import defaultdict
import numpy as np
from datetime import datetime

class StrikingElo:
    def __init__(self, base_elo=1500, k_factor=20):
        self.base_elo = base_elo
        self.k = k_factor
        self.elo_dict = defaultdict(lambda: self.base_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def process_fights(self, df):
        df = df.sort_values("DATE").copy()
        numeric_cols = [
            "precomp_sigstr_pm", "precomp_sapm",
            "opp_precomp_sigstr_pm", "opp_precomp_sapm"
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        fighter_elo_log = {}
        opponent_elo_log = {}
        fighter_post_elo_log = {}
        opponent_post_elo_log = {}

        for _, row in df.iterrows():
            fighter = row["FIGHTER"]
            opponent = row["opp_FIGHTER"]
            date = row["DATE"]

            elo_f = self.elo_dict[fighter]
            elo_o = self.elo_dict[opponent]
            expected_f = self.expected_score(elo_f, elo_o)

            landed = row["precomp_sigstr_pm"]
            absorbed = row["precomp_sapm"]
            opp_landed = row["opp_precomp_sigstr_pm"]
            opp_absorbed = row["opp_precomp_sapm"]
            if any(pd.isna(x) for x in [landed, absorbed, opp_landed, opp_absorbed]):
                continue

            f_eff = landed - absorbed
            o_eff = opp_landed - opp_absorbed

            strength_weight = 1 + (elo_o - 1500) / 1000
            f_eff *= strength_weight
            o_eff *= strength_weight

            if isinstance(date, str):
                date = pd.to_datetime(date)
            days_since = (datetime.today() - date).days
            recency_weight = np.exp(-days_since / 1000)

            perf_score = 0.5 if f_eff == o_eff else 1.0 if f_eff > o_eff else 0.0
            score_f = recency_weight * perf_score
            score_o = recency_weight * (1 - perf_score)

            # Store pre-fight Elo
            fighter_elo_log[(fighter, date)] = elo_f
            opponent_elo_log[(opponent, date)] = elo_o

            # Update Elo
            self.elo_dict[fighter] += self.k * (score_f - expected_f)
            self.elo_dict[opponent] += self.k * (score_o - (1 - expected_f))

            # Store post-fight Elo
            fighter_post_elo_log[(fighter, date)] = self.elo_dict[fighter]
            opponent_post_elo_log[(opponent, date)] = self.elo_dict[opponent]

        # Insert Elo values into the DataFrame
        df["precomp_strike_elo"] = df.apply(lambda row: fighter_elo_log.get((row["FIGHTER"], row["DATE"]), np.nan), axis=1)
        df["opp_precomp_strike_elo"] = df.apply(lambda row: opponent_elo_log.get((row["opp_FIGHTER"], row["DATE"]), np.nan), axis=1)
        df["postcomp_strike_elo"] = df.apply(lambda row: fighter_post_elo_log.get((row["FIGHTER"], row["DATE"]), np.nan), axis=1)
        df["opp_postcomp_strike_elo"] = df.apply(lambda row: opponent_post_elo_log.get((row["opp_FIGHTER"], row["DATE"]), np.nan), axis=1)

        return df

    
    def top_n_fighters(self, n=10):
        #print top N fighters based on current Elo ratings
        sorted_fighters = sorted(self.elo_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {n} Fighters by Elo:")
        for i, (fighter, elo) in enumerate(sorted_fighters[:n], start=1):
            print(f"{i}. {fighter}: {elo:.2f}")