#!/usr/bin/env python3
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from difflib import SequenceMatcher

API_KEY       = 'a8506b7492befca11590b71ba388575f'  # ← your Odds API key here
SPORT         = 'mma_mixed_martial_arts'
REGIONS       = 'us'
MARKETS       = 'h2h'
ODDS_FORMAT   = 'american'
DATE_FMT      = 'iso'
LOOKBACK_DAYS = 546

HIST_URL = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds'
MAIN_BOOKMAKERS = ['draftkings','fanduel','betmgm','bet365','bovada']
FUZZY_THRESHOLD = 0.8

def normalize(name):
    return re.sub(r'\W+', '', (name or '').lower())

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def fetch_snapshot(ts_iso):
    r = requests.get(HIST_URL, params={
        'apiKey': API_KEY, 'regions': REGIONS,
        'markets': MARKETS, 'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FMT,  'date': ts_iso,
    })
    r.raise_for_status()
    return r.json().get('data', [])

def find_best_event(fn, on, row_date, ev_list):
    """Return the best-matching event for fighter vs opponent on row_date (±1 day)."""
    candidates = []
    for ev in ev_list:
        ev_date = ev['commence_dt'].date()
        if abs((ev_date - row_date).days) > 1:
            continue

        home, away = ev['home'], ev['away']
        # exact both-way match
        if {fn, on} == {home, away}:
            return ev

        # fuzzy both-way match
        sim1 = similar(fn, home) + similar(on, away)
        sim2 = similar(fn, away) + similar(on, home)
        if max(sim1, sim2) / 2 >= FUZZY_THRESHOLD:
            candidates.append((max(sim1, sim2) / 2, ev))

    # return highest-scoring fuzzy candidate, if any
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]

    return None

def main():
    df = pd.read_csv('../data/tmp/final.csv', parse_dates=['DATE'])
    df['DATE'] = pd.to_datetime(df['DATE'], utc=True)
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df['DATE'] >= cutoff].copy()

    df['f_norm'] = df['FIGHTER'].apply(normalize)
    df['o_norm'] = df['opp_FIGHTER'].apply(normalize)
    df['date_str'] = df['DATE'].dt.strftime('%Y-%m-%d')

    # 1) Fetch and build ev_list
    raw = []
    for d in sorted(df['date_str'].unique()):
        base = datetime.fromisoformat(d)
        for delta in (0,1):
            ts = (base + timedelta(days=delta)).strftime('%Y-%m-%dT00:00:00Z')
            print(f"Fetching {ts}")
            raw.extend(fetch_snapshot(ts))

    # dedupe
    seen = {}
    for e in raw:
        seen[e['id']] = e
    ev_list = []
    for e in seen.values():
        ct = pd.to_datetime(e['commence_time'], utc=True)
        ev_list.append({
            'commence_dt': ct,
            'home': normalize(e['home_team']),
            'away': normalize(e['away_team']),
            'bookmakers': e.get('bookmakers', []),
        })

    # 2) Prepare output cols
    for bk in MAIN_BOOKMAKERS:
        df[f"{bk}_odds"] = None

    # 3) Row-by-row match with fuzzy fallback
    for idx, row in df.iterrows():
        fn, on = row['f_norm'], row['o_norm']
        row_date = row['DATE'].date()
        ev = find_best_event(fn, on, row_date, ev_list)
        if not ev:
            continue

        for bm in ev['bookmakers']:
            key = bm.get('key')
            if key not in MAIN_BOOKMAKERS:
                continue
            h2h = next((m for m in bm['markets'] if m['key']=='h2h'), None)
            if not h2h:
                continue
            # pull this row’s fighter price
            price = next((o['price'] for o in h2h['outcomes'] 
                          if normalize(o['name']) == fn), None)
            df.at[idx, f"{key}_odds"] = price

    df.drop(columns=['f_norm','o_norm','date_str'], inplace=True)
    df.to_csv('final_with_odds.csv', index=False)
    print("Done! → final_with_odds.csv")

if __name__ == "__main__":
    main()
