import pandas as pd
import numpy as np
from datetime import datetime

class SimpleGlicko:
    def __init__(self, base_rating=1500, base_rd=350, tau=0.5):
        """
        Minimal Glicko-1 rating system for head-to-head sports.
        Args:
            base_rating: Starting rating for new fighters (default: 1500)
            base_rd: Initial rating deviation (default: 350)
            tau: System constant for RD increase between rating periods (default: 0.5)
        """
        self.rating_dict = {}  # fighter: (rating, RD)
        self.last_fight_date = {}  # fighter: last fight date
        self.base_rating = base_rating
        self.base_rd = base_rd
        self.tau = tau
        self.glicko_history = {}  # fighter: [(date, rating, RD)]

    def add_player(self, name, date):
        if name not in self.rating_dict:
            self.rating_dict[name] = (self.base_rating, self.base_rd)
            self.last_fight_date[name] = date
            self.glicko_history[name] = []

    def _g(self, RD):
        return 1 / np.sqrt(1 + 3 * (RD ** 2) / (np.pi ** 2 * 400 ** 2))

    def _E(self, rating, opp_rating, opp_RD):
        g = self._g(opp_RD)
        return 1 / (1 + 10 ** (-g * (rating - opp_rating) / 400))

    def _update_rd(self, old_rd, days_inactive):
        # Increase RD for inactivity (simple model)
        return min(np.sqrt(old_rd ** 2 + self.tau ** 2 * days_inactive), 350)

    def process_fights(self, data):
        if data is None or data.empty:
            raise ValueError("Data is empty or not provided.")
        data = data.copy()
        data = data.sort_values(by='DATE', ascending=True)
        for col in ['precomp_glicko', 'opp_precomp_glicko', 'postcomp_glicko', 'opp_postcomp_glicko']:
            if col not in data.columns:
                data[col] = np.nan
        for idx, row in data.iterrows():
            fighter = str(row['FIGHTER'])
            opponent = str(row['opp_FIGHTER'])
            date = str(row['DATE'])
            result = row.get('result', 0)
            if isinstance(result, str):
                result = 1 if result.strip().lower() == 'win' else 0
            # Add new fighters if needed
            if fighter not in self.rating_dict:
                self.add_player(fighter, date)
            if opponent not in self.rating_dict:
                self.add_player(opponent, date)
            rating_f, rd_f = self.rating_dict[fighter]
            rating_o, rd_o = self.rating_dict[opponent]
            # Inactivity update
            try:
                last_date_f = datetime.strptime(self.last_fight_date[fighter], "%Y-%m-%d")
                curr_date = datetime.strptime(date, "%Y-%m-%d")
                days_inactive = (curr_date - last_date_f).days
                rd_f = self._update_rd(rd_f, days_inactive)
            except:
                pass
            try:
                last_date_o = datetime.strptime(self.last_fight_date[opponent], "%Y-%m-%d")
                curr_date = datetime.strptime(date, "%Y-%m-%d")
                days_inactive = (curr_date - last_date_o).days
                rd_o = self._update_rd(rd_o, days_inactive)
            except:
                pass
            # Store pre-fight values
            data.loc[idx, 'precomp_glicko'] = rating_f
            data.loc[idx, 'opp_precomp_glicko'] = rating_o
            # Glicko update for both fighters
            for p, opp, res, rd_p, rd_opp in [
                (fighter, opponent, result, rd_f, rd_o),
                (opponent, fighter, 1 - result, rd_o, rd_f)
            ]:
                rating, rd = self.rating_dict[p]
                g = self._g(rd_opp)
                E = self._E(rating, self.rating_dict[opp][0], rd_opp)
                v = 1 / (g ** 2 * E * (1 - E))
                delta = v * g * (res - E)
                new_rating = rating + delta
                new_rd = max(30, 1 / np.sqrt(1 / rd ** 2 + 1 / v))
                self.rating_dict[p] = (new_rating, new_rd)
                self.glicko_history.setdefault(p, []).append((date, new_rating, new_rd))
                if p == fighter:
                    data.loc[idx, 'postcomp_glicko'] = new_rating
                else:
                    data.loc[idx, 'opp_postcomp_glicko'] = new_rating
            self.last_fight_date[fighter] = date
            self.last_fight_date[opponent] = date
        return data

    def query_fighter_glicko(self, fighter):
        if fighter not in self.glicko_history or not self.glicko_history[fighter]:
            print(f"No Glicko data found for {fighter}.")
            return
        import matplotlib.pyplot as plt
        history = self.glicko_history[fighter]
        dates, ratings, rds = zip(*sorted(history))
        plt.figure(figsize=(10, 5))
        plt.plot(dates, ratings, marker='o', label=f'{fighter} Glicko')
        plt.fill_between(dates, np.array(ratings) - np.array(rds), np.array(ratings) + np.array(rds), alpha=0.2, label='RD')
        plt.xlabel('Date')
        plt.ylabel('Glicko Rating')
        plt.title(f'Glicko Rating Progression for {fighter}')
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    #assess the predictive power of the glicko rating by comparing a fighter's glicko rating to their opponent's glicko rating and seeing if the higher glicko predics the win

    def top_k_fighters(self, k=10):
        sorted_fighters = sorted(self.rating_dict.items(), key=lambda x: x[1][0], reverse=True)
        return [(f, r[0], r[1]) for f, r in sorted_fighters[:k]] 