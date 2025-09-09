#!/usr/bin/env python3
import requests
import json
import pandas as pd

API_KEY     = 'a8506b7492befca11590b71ba388575f'  # ← your key here
SPORT       = 'mma_mixed_martial_arts'
REGIONS     = 'us'
MARKETS     = 'h2h'
ODDS_FORMAT = 'american'
DATE_FMT    = 'iso'
DATE        = '2025-03-22'

URL = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds'

def main():
    # Build ISO-8601 timestamp at midnight UTC for your target date
    ts = pd.to_datetime(DATE).strftime('%Y-%m-%dT00:00:00Z')
    
    params = {
        'apiKey':     API_KEY,
        'regions':    REGIONS,
        'markets':    MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FMT,
        'date':       ts,                  # must be ISO-8601 :contentReference[oaicite:0]{index=0}
    }
    
    resp = requests.get(URL, params=params)
    resp.raise_for_status()
    snapshot = resp.json()
    
    # Keep only events whose commence_time matches 2025-03-01
    events = [
        ev for ev in snapshot.get('data', [])
        if ev.get('commence_time', '')[:10] == DATE
    ]
    
    # Write out to JSON
    out_file = f'odds_{DATE}.json'
    with open(out_file, 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"Fetched snapshot '{snapshot.get('timestamp')}' and saved {len(events)} events to '{out_file}'")

if __name__ == '__main__':
    main()
