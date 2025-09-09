import pandas as pd
from collections import defaultdict
import numpy as np

class NewStrikingElo:
    def __init__(self, base_elo=1500, k_off=20, k_def=20):
        self.base_elo = base_elo
        self.k_off = k_off
        self.k_def = k_def
        # Separate offense and defense Elo dictionaries
        self.off_elo = defaultdict(lambda: self.base_elo)
        self.def_elo = defaultdict(lambda: self.base_elo)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def process_fights(self, df):
        df = df.sort_values("DATE").copy()

        # Ensure numeric types
        numeric_cols = [
            "sigstracc", "sigstrabs",
            "opp_sigstracc", "opp_sigstrabs"
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Logs for pre/post Elo
        pre_off_log = {}
        pre_def_log = {}
        post_off_log = {}
        post_def_log = {}
        opp_pre_off_log = {}
        opp_pre_def_log = {}
        opp_post_off_log = {}
        opp_post_def_log = {}

        for _, row in df.iterrows():
            fighter = row["FIGHTER"]
            opponent = row["opp_FIGHTER"]
            date = row["DATE"]

            # Current Elo ratings
            elo_off_f = self.off_elo[fighter]
            elo_def_f = self.def_elo[fighter]
            elo_off_o = self.off_elo[opponent]
            elo_def_o = self.def_elo[opponent]

            # Log pre-fight
            pre_off_log[(fighter, date)] = elo_off_f
            pre_def_log[(fighter, date)] = elo_def_f
            opp_pre_off_log[(opponent, date)] = elo_off_o
            opp_pre_def_log[(opponent, date)] = elo_def_o

            # Expected scores: offense vs. defense and vice versa
            exp_off_f = self.expected_score(elo_off_f, elo_def_o)
            exp_def_f = self.expected_score(elo_def_f, elo_off_o)
            exp_off_o = self.expected_score(elo_off_o, elo_def_f)
            exp_def_o = self.expected_score(elo_def_o, elo_off_f)

            # Performance metrics (using significant strikes)
            landed = row["sigstracc"]
            attempts = row["sigstrabs"]
            opp_landed = row["opp_sigstracc"]
            opp_attempts = row["opp_sigstrabs"]

            # Skip if missing or zero attempts
            if any(pd.isna(x) for x in [landed, attempts, opp_landed, opp_attempts]) or attempts == 0 or opp_attempts == 0:
                continue

            perf_off_f = landed / attempts
            perf_def_f = 1 - (opp_landed / opp_attempts)
            perf_off_o = opp_landed / opp_attempts
            perf_def_o = 1 - (landed / attempts)

            # Update Elo ratings
            self.off_elo[fighter] += self.k_off * (perf_off_f - exp_off_f)
            self.def_elo[fighter] += self.k_def * (perf_def_f - exp_def_f)
            self.off_elo[opponent] += self.k_off * (perf_off_o - exp_off_o)
            self.def_elo[opponent] += self.k_def * (perf_def_o - exp_def_o)

            # Log post-fight
            post_off_log[(fighter, date)] = self.off_elo[fighter]
            post_def_log[(fighter, date)] = self.def_elo[fighter]
            opp_post_off_log[(opponent, date)] = self.off_elo[opponent]
            opp_post_def_log[(opponent, date)] = self.def_elo[opponent]

        # Assign logs back to dataframe
        df['pre_off_elo'] = df.apply(lambda r: pre_off_log.get((r['FIGHTER'], r['DATE'])), axis=1)
        df['pre_def_elo'] = df.apply(lambda r: pre_def_log.get((r['FIGHTER'], r['DATE'])), axis=1)
        df['post_off_elo'] = df.apply(lambda r: post_off_log.get((r['FIGHTER'], r['DATE'])), axis=1)
        df['post_def_elo'] = df.apply(lambda r: post_def_log.get((r['FIGHTER'], r['DATE'])), axis=1)
        df['opp_pre_off_elo'] = df.apply(lambda r: opp_pre_off_log.get((r['opp_FIGHTER'], r['DATE'])), axis=1)
        df['opp_pre_def_elo'] = df.apply(lambda r: opp_pre_def_log.get((r['opp_FIGHTER'], r['DATE'])), axis=1)
        df['opp_post_off_elo'] = df.apply(lambda r: opp_post_off_log.get((r['opp_FIGHTER'], r['DATE'])), axis=1)
        df['opp_post_def_elo'] = df.apply(lambda r: opp_post_def_log.get((r['opp_FIGHTER'], r['DATE'])), axis=1)

        return df

    def top_n_offense(self, n=10):
        sorted_off = sorted(self.off_elo.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {n} Fighters by Offense Elo:")
        for i, (f, e) in enumerate(sorted_off[:n], 1):
            print(f"{i}. {f}: {e:.2f}")

    def top_n_defense(self, n=10):
        sorted_def = sorted(self.def_elo.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {n} Fighters by Defense Elo:")
        for i, (f, e) in enumerate(sorted_def[:n], 1):
            print(f"{i}. {f}: {e:.2f}")
