#!/usr/bin/env python3
import pandas as pd

def main():
    # 1) Load your interleaved fights, parse DATE, and normalize to YYYY-MM-DD
    inter = pd.read_csv('../data/tmp/interleaved_clean.csv', parse_dates=['DATE'])
    inter['date'] = inter['DATE'].dt.strftime('%Y-%m-%d')

    # 2) Pull just the three matching keys and drop duplicates
    fights = (
        inter[['date', 'FIGHTER', 'opp_FIGHTER']]
        .drop_duplicates()
        .rename(columns={
            'FIGHTER':      'home_team',
            'opp_FIGHTER':  'away_team'
        })
    )

    # 3) Load the per-bookmaker odds, parse and normalize its date column
    odds = pd.read_csv('filtered_odds_by_bookmaker.csv', parse_dates=['date'])
    odds['date'] = odds['date'].dt.strftime('%Y-%m-%d')

    # 4) Inner-join on date, home_team, away_team
    matched = pd.merge(
        odds,
        fights,
        on=['date', 'home_team', 'away_team'],
        how='inner'
    )

    # 5) Save the filtered file
    matched.to_csv('filtered_odds_by_bookmaker_matched.csv', index=False)
    print(f"Kept {len(matched)} rows — wrote filtered_odds_by_bookmaker_matched.csv")

if __name__ == '__main__':
    main()
