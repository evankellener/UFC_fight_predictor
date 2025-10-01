# Copy this code into your notebook under "## Odds Scrapper"

# Import the combined scraper
import sys
sys.path.append('../src')
from combined_odds_scraper import CombinedOddsScraper

# Initialize the scraper
scraper = CombinedOddsScraper()

# Scrape odds with all data quality fixes applied
print("=== COMBINED ODDS SCRAPER ===")
print("This scraper combines:")
print("• Odds API fetching")
print("• Odds clamping (fixes extreme values like 102 → 100)")
print("• Improved filtering (handles missing odds intelligently)")
print("• ROI analysis with realistic calculations")

# Run the scraper
df_with_odds = scraper.scrape_odds(
    input_csv='../data/tmp/final.csv',
    output_csv='../src/final_with_odds_processed.csv'
)

# Analyze ROI
roi_results = scraper.analyze_roi(df_with_odds, stake=100)

print(f"\n=== RESULTS ===")
print(f"✅ Processed {len(df_with_odds)} fights")
print(f"✅ ROI: {roi_results['roi']:.2%}")
print(f"✅ Win rate: {roi_results['win_rate']:.2%}")
print(f"✅ Total profit: ${roi_results['total_profit']:,.2f}")

# The df_with_odds DataFrame is now ready for your model training!
