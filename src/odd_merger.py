#!/usr/bin/env python3
import json
import pandas as pd

def main():
    # 1) Load your interleaved fight data
    inter = pd.read_csv('interleaved_clean.csv', parse_dates=['DATE'])
    # Normalize date to string YYYY-MM-DD for merging
    inter['date'] = inter['DATE'].dt.strftime('%Y-%m-%d')
    
    # 2) Load your aggregated odds JSON
    #    Replace with the correct filename you wrote out in the previous step
    with open('odds_2025_with_aftermidnight.json') as f:
        odds_list = json.load(f)
    odds = pd.DataFrame(odds_list)
    
    # 3) Normalize the odds DataFrame
    #    - extract date from 'commence_time'
    odds['date'] = pd.to_datetime(odds['commence_time']).dt.strftime('%Y-%m-%d')
    #    - rename home/away to match interleaved column names
    odds = odds.rename(columns={
        'home_team': 'FIGHTER',
        'away_team': 'opp_FIGHTER'
    })
    
    # 4) Pivot odds so each bookmaker×outcome is its own column
    #    (if you already have a pivoted CSV, skip this step and load that CSV instead)
    flat_rows = []
    for _, row in odds.iterrows():
        for book in row.get('bookmakers', []):
            if book.get('key') and 'markets' in book:
                for m in book['markets']:
                    if m.get('key') == 'h2h':
                        for out in m.get('outcomes', []):
                            flat_rows.append({
                                'date':       row['date'],
                                'FIGHTER':    row['FIGHTER'],
                                'opp_FIGHTER':row['opp_FIGHTER'],
                                'bookmaker':  book['key'],
                                'outcome':    out['name'],
                                'price':      out['price'],
                            })
    flat = pd.DataFrame(flat_rows)
    
    odds_pivot = flat.pivot_table(
        index=['date','FIGHTER','opp_FIGHTER'],
        columns=['bookmaker','outcome'],
        values='price'
    )
    # flatten MultiIndex into single-level columns
    odds_pivot.columns = [
        f"{bk}_{oc}" for bk, oc in odds_pivot.columns
    ]
    odds_pivot = odds_pivot.reset_index()
    
    # 5) Merge the pivoted odds into your interleaved DataFrame
    merged = pd.merge(
        inter,
        odds_pivot,
        on=['date','FIGHTER','opp_FIGHTER'],
        how='left'
    )
    
    # 6) Save the result
    merged.to_csv('interleaved_with_odds.csv', index=False)
    print(f"Saved merged file with {len(merged)} rows to 'interleaved_with_odds.csv'")

if __name__ == '__main__':
    main()
