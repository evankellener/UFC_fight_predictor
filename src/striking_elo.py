import pandas as pd
from collections import defaultdict
import numpy as np

class StrikingElo:
    def __init__(self, base_elo=1500, k_factor=20):
        self.base_elo = base_elo
        self.k = k_factor
        self.elo_dict = defaultdict(lambda: self.base_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def process_fights(self, df):
        df = df.sort_values("DATE").copy()

        # Ensure numeric types for required columns
        numeric_cols = [
            "sigstracc", "sigstrabs",
            "opp_sigstracc", "opp_sigstrabs"
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
            expected_o = 1 - expected_f

            landed = row["sigstracc"]
            absorbed = row["sigstrabs"]
            opp_landed = row["opp_sigstracc"]
            opp_absorbed = row["opp_sigstrabs"]

            if any(pd.isna(x) for x in [landed, absorbed, opp_landed, opp_absorbed]):
                continue

            f_eff = landed - absorbed
            o_eff = opp_landed - opp_absorbed

            if f_eff == o_eff:
                score_f, score_o = 0.5, 0.5
            elif f_eff > o_eff:
                score_f, score_o = 1.0, 0.0
            else:
                score_f, score_o = 0.0, 1.0

            # Log pre-fight Elo
            fighter_elo_log[(fighter, date)] = elo_f
            opponent_elo_log[(opponent, date)] = elo_o

            # Update Elo ratings
            self.elo_dict[fighter] += self.k * (score_f - expected_f)
            self.elo_dict[opponent] += self.k * (score_o - expected_o)

            # Log post-fight Elo
            fighter_post_elo_log[(fighter, date)] = self.elo_dict[fighter]
            opponent_post_elo_log[(opponent, date)] = self.elo_dict[opponent]

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

            
    
