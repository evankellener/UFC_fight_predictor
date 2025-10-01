# Replace your existing ROI calculation in the notebook with this:

# Import the fixed ROI analysis
import sys
sys.path.append('src')
from roi_analysis_final import comprehensive_roi_analysis

# Run the corrected ROI calculation
roi_df = comprehensive_roi_analysis(
    odds_table_path='../data/tmp/odds_table.csv',
    vegas_data_path='final_with_odds_filtered.csv',
    stake=100
)

# The roi_df now contains all the corrected calculations with proper:
# - Profit calculations
# - Edge analysis  
# - Statistical significance testing
# - Monthly performance breakdown
# - Best/worst bet analysis

print(f"\n✅ ROI Analysis Complete!")
print(f"Final ROI: {roi_df['cum_roi'].iloc[-1]:.2%}")
print(f"Total Profit: ${roi_df['cum_profit'].iloc[-1]:,.2f}")
print(f"Win Rate: {roi_df['win'].mean():.2%}")
