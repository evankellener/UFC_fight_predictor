# Combined Odds Scraper Demo
# This code can be added to the notebook under "## Odds Scrapper"

import sys
import os
sys.path.append('../src')

from combined_odds_scraper import CombinedOddsScraper
import pandas as pd

# Initialize the combined scraper
scraper = CombinedOddsScraper()

# Example 1: Scrape odds for recent fights with data quality fixes
print("=== SCRAPING ODDS WITH DATA QUALITY FIXES ===")
df_processed = scraper.scrape_odds(
    input_csv='../data/tmp/final.csv',
    output_csv='../src/final_with_odds_processed.csv'
)

# Example 2: Analyze ROI for the processed data
print("\n=== ANALYZING ROI ===")
roi_results = scraper.analyze_roi(df_processed, stake=100)

# Example 3: Show sample of processed data
print("\n=== SAMPLE PROCESSED DATA ===")
odds_cols = [col for col in df_processed.columns if col.endswith('_odds')]
sample_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'win'] + odds_cols
print(df_processed[sample_cols].head(10))

# Example 4: Monthly ROI analysis
print("\n=== MONTHLY ROI ANALYSIS ===")
df_processed['DATE'] = pd.to_datetime(df_processed['DATE'])
df_processed['month'] = df_processed['DATE'].dt.to_period('M')

monthly_roi = []
for month in df_processed['month'].unique():
    month_data = df_processed[df_processed['month'] == month]
    month_results = scraper.analyze_roi(month_data, stake=100)
    monthly_roi.append({
        'month': str(month),
        'roi': month_results['roi'],
        'bets': month_results['total_bets'],
        'profit': month_results['total_profit']
    })

monthly_df = pd.DataFrame(monthly_roi)
print(monthly_df.sort_values('roi', ascending=False).head(10))

print("\n=== SCRAPER COMPLETE ===")
print("The combined scraper has:")
print("✓ Fetched odds from API")
print("✓ Applied odds clamping to fix extreme values")
print("✓ Used improved filtering to handle missing odds")
print("✓ Calculated realistic ROI metrics")
print("✓ Provided monthly analysis")
