#!/usr/bin/env python3
"""
new_odds_scrapper.py

Scrapes fight odds tables from BestFightOdds event pages for events in the past two years.
Usage:
  python new_odds_scrapper.py path/to/eventdetails.csv
The CSV must include a date column (e.g. "Date" or "EventDate") and a URL column (e.g. "URL").
"""
import sys
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller

# ----------------------------------------------------------------------------
# 1) Auto-install matching chromedriver
# ----------------------------------------------------------------------------
chromedriver_autoinstaller.install()

# ----------------------------------------------------------------------------
# 2) Configure headless Chrome
# ----------------------------------------------------------------------------
chrome_opts = Options()
chrome_opts.add_argument("--headless")
chrome_opts.add_argument("--no-sandbox")
chrome_opts.add_argument("--disable-dev-shm-usage")
chrome_opts.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=chrome_opts)


def scrape_event_odds(event_url: str) -> pd.DataFrame:
    """
    Loads the event URL in headless Chrome, waits for the first table,
    and returns the first table as a DataFrame.
    """
    driver.get(event_url)
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "table"))
    )
    html = driver.page_source
    dfs = pd.read_html(html)
    if not dfs:
        raise RuntimeError(f"No odds table found at {event_url}")
    return dfs[0]


def main():
    # 1) Read CSV path
    if len(sys.argv) != 2:
        print("Usage: python new_odds_scrapper.py path/to/eventdetails.csv")
        sys.exit(1)
    csv_path = sys.argv[1]

    # 2) Load events
    events = pd.read_csv(csv_path)
    # auto-detect date and URL columns
    date_col = next((c for c in events.columns if 'date' in c.lower()), None)
    url_col = next((c for c in events.columns if 'url' in c.lower()), None)
    if date_col is None or url_col is None:
        print("CSV must contain a date column and a URL column.")
        sys.exit(1)

    # 3) Parse dates and filter last 2 years
    events[date_col] = pd.to_datetime(events[date_col])
    cutoff = pd.Timestamp.now() - relativedelta(years=2)
    recent = events[events[date_col] >= cutoff]
    print(f"🔎 Found {len(recent)} events since {cutoff.date()} to scrape.")

    # 4) Scrape each event with progress bar
    all_odds = []
    for _, row in tqdm(recent.iterrows(), total=len(recent), desc="Events"):
        evt_name = row.get('Event') or row.get('event_name') or ''
        ev_url = row[url_col]
        try:
            df = scrape_event_odds(ev_url)
            df['event'] = evt_name
            df['event_url'] = ev_url
            df['scrape_date'] = datetime.now()
            all_odds.append(df)
        except Exception as e:
            print(f"❌ [{evt_name}] {e}")

    # 5) Combine and save
    if all_odds:
        combined = pd.concat(all_odds, ignore_index=True)
        combined.to_csv('combined_odds.csv', index=False)
        print("💾 Saved combined odds to combined_odds.csv")
    else:
        print("No odds tables were scraped.")

    driver.quit()


if __name__ == '__main__':
    main()
