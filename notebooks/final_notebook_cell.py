# Copy this code into your notebook under "## Odds Scrapper"

# Import the simple odds processor
import sys
sys.path.append('../src')
from simple_odds_processor import SimpleOddsProcessor

# Initialize the processor
processor = SimpleOddsProcessor()

# Process existing odds data with all data quality fixes
print("=== COMBINED ODDS SCRAPER ===")
print("This processor combines:")
print("• Odds clamping (fixes extreme values like 102 → 100)")
print("• Improved filtering (handles missing odds intelligently)")
print("• ROI analysis with realistic calculations")

# Process the data
df_with_odds = processor.process_odds_data(
    input_csv='../src/final_with_odds.csv',
    output_csv='../src/final_with_odds_processed.csv'
)

# Analyze ROI
roi_results = processor.analyze_roi(df_with_odds, stake=100)

print(f"\n=== RESULTS ===")
print(f"✅ Processed {len(df_with_odds)} fights")
print(f"✅ ROI: {roi_results['roi']:.2%}")
print(f"✅ Win rate: {roi_results['win_rate']:.2%}")
print(f"✅ Total profit: ${roi_results['total_profit']:,.2f}")

# Show sample of processed data
print(f"\n=== SAMPLE PROCESSED DATA ===")
odds_cols = [col for col in df_with_odds.columns if col.endswith('_odds')]
sample_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'win'] + odds_cols
print(df_with_odds[sample_cols].head())

# Monthly ROI analysis
print(f"\n=== MONTHLY ROI ANALYSIS ===")
df_with_odds['DATE'] = pd.to_datetime(df_with_odds['DATE'])
df_with_odds['month'] = df_with_odds['DATE'].dt.to_period('M')

monthly_roi = []
for month in df_with_odds['month'].unique():
    month_data = df_with_odds[df_with_odds['month'] == month]
    month_results = processor.analyze_roi(month_data, stake=100)
    monthly_roi.append({
        'month': str(month),
        'roi': month_results['roi'],
        'bets': month_results['total_bets'],
        'profit': month_results['total_profit']
    })

monthly_df = pd.DataFrame(monthly_roi)
print("Top 10 months by ROI:")
print(monthly_df.sort_values('roi', ascending=False).head(10))

print("\n=== SCRAPER COMPLETE ===")
print("The df_with_odds DataFrame is now ready for your model training!")
print("All data quality issues have been fixed:")
print("✓ Extreme odds values clamped to realistic ranges")
print("✓ Missing odds handled intelligently (averaged available odds)")
print("✓ ROI calculations are now realistic and sustainable")
