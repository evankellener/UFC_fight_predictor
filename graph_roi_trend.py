"""
Graph ROI Trend Over Time
Visualize per-event ROI over the past year to detect trends
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

print("="*80)
print("ROI TREND ANALYSIS - Past Year")
print("="*80)
print()

# Load the 1-year test results
results_file = 'data/tmp/roi_full_1_year_test.csv'
df = pd.read_csv(results_file)
df['DATE'] = pd.to_datetime(df['DATE'])

print(f"Loaded {len(df)} picks from {df['DATE'].min().date()} to {df['DATE'].max().date()}")
print()

# Calculate per-event statistics
print("Calculating per-event ROI...")
event_stats = df.groupby(['DATE', 'EVENT']).agg({
    'stake': 'sum',
    'profit': 'sum',
    'won': ['sum', 'count']
}).reset_index()

event_stats.columns = ['DATE', 'EVENT', 'total_stake', 'total_profit', 'wins', 'bets']
event_stats['roi_pct'] = (event_stats['total_profit'] / event_stats['total_stake']) * 100
event_stats['win_rate'] = event_stats['wins'] / event_stats['bets']
event_stats['cum_profit'] = event_stats['total_profit'].cumsum()
event_stats['cum_stake'] = event_stats['total_stake'].cumsum()
event_stats['cum_roi'] = (event_stats['cum_profit'] / event_stats['cum_stake']) * 100

# Sort by date
event_stats = event_stats.sort_values('DATE').reset_index(drop=True)

print(f"Total events: {len(event_stats)}")
print(f"Average bets per event: {event_stats['bets'].mean():.1f}")
print()

# Calculate trend line
x = np.arange(len(event_stats))
y = event_stats['roi_pct'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("="*80)
print("TREND ANALYSIS")
print("="*80)
print(f"Slope: {slope:.4f} percentage points per event")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")
print()

if p_value < 0.05:
    if slope > 0:
        print("📈 STATISTICALLY SIGNIFICANT UPWARD TREND")
        print(f"   ROI improving by ~{slope:.2f}% per event")
    else:
        print("📉 STATISTICALLY SIGNIFICANT DOWNWARD TREND")
        print(f"   ROI declining by ~{abs(slope):.2f}% per event")
else:
    print("➡️  NO STATISTICALLY SIGNIFICANT TREND")
    print("   ROI fluctuations appear to be random variance")

print()

# Monthly breakdown
event_stats['Month'] = event_stats['DATE'].dt.to_period('M')
monthly_stats = event_stats.groupby('Month').agg({
    'bets': 'sum',
    'wins': 'sum',
    'total_stake': 'sum',
    'total_profit': 'sum'
}).reset_index()
monthly_stats['roi_pct'] = (monthly_stats['total_profit'] / monthly_stats['total_stake']) * 100
monthly_stats['win_rate'] = monthly_stats['wins'] / monthly_stats['bets']

print("="*80)
print("MONTHLY BREAKDOWN")
print("="*80)
print(f"{'Month':<12s} {'Bets':>6s} {'Win%':>8s} {'ROI':>10s} {'Profit':>12s}")
print("-"*55)

for _, row in monthly_stats.iterrows():
    print(f"{str(row['Month']):<12s} {row['bets']:>6d} {row['win_rate']*100:>7.1f}% {row['roi_pct']:>9.2f}% ${row['total_profit']:>10.2f}")

print()

# Best and worst events
best_event = event_stats.loc[event_stats['roi_pct'].idxmax()]
worst_event = event_stats.loc[event_stats['roi_pct'].idxmin()]

print("="*80)
print("BEST AND WORST EVENTS")
print("="*80)
print(f"🔥 Best Event: {best_event['EVENT'][:50]}")
print(f"   Date: {best_event['DATE'].date()}")
print(f"   ROI: {best_event['roi_pct']:+.2f}% ({best_event['wins']}/{best_event['bets']} wins)")
print(f"   Profit: ${best_event['total_profit']:+.2f}")
print()
print(f"❌ Worst Event: {worst_event['EVENT'][:50]}")
print(f"   Date: {worst_event['DATE'].date()}")
print(f"   ROI: {worst_event['roi_pct']:+.2f}% ({worst_event['wins']}/{worst_event['bets']} wins)")
print(f"   Profit: ${worst_event['total_profit']:+.2f}")
print()

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ROI Trend Analysis - Past Year', fontsize=16, fontweight='bold')

# Plot 1: Per-Event ROI with trend line
ax1 = axes[0, 0]
ax1.scatter(event_stats.index, event_stats['roi_pct'], alpha=0.6, s=100, 
            c=event_stats['roi_pct'], cmap='RdYlGn', vmin=-50, vmax=100)
ax1.plot(event_stats.index, intercept + slope * x, 'r--', linewidth=2, 
         label=f'Trend: {slope:+.2f}% per event (p={p_value:.3f})')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax1.axhline(y=event_stats['roi_pct'].mean(), color='blue', linestyle='--', 
            linewidth=1, label=f'Mean: {event_stats["roi_pct"].mean():.1f}%')
ax1.set_xlabel('Event Number (Chronological)', fontsize=12)
ax1.set_ylabel('ROI per Event (%)', fontsize=12)
ax1.set_title('Per-Event ROI with Trend Line', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative ROI over time
ax2 = axes[0, 1]
ax2.plot(event_stats.index, event_stats['cum_roi'], linewidth=2, color='green')
ax2.fill_between(event_stats.index, 0, event_stats['cum_roi'], alpha=0.3, color='green')
ax2.set_xlabel('Event Number', fontsize=12)
ax2.set_ylabel('Cumulative ROI (%)', fontsize=12)
ax2.set_title('Cumulative ROI Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add final ROI annotation
final_roi = event_stats['cum_roi'].iloc[-1]
ax2.annotate(f'Final: {final_roi:.1f}%', 
             xy=(len(event_stats)-1, final_roi),
             xytext=(10, -20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Plot 3: Monthly ROI comparison
ax3 = axes[1, 0]
monthly_labels = [str(m) for m in monthly_stats['Month']]
colors = ['green' if roi > 0 else 'red' for roi in monthly_stats['roi_pct']]
bars = ax3.bar(range(len(monthly_stats)), monthly_stats['roi_pct'], color=colors, alpha=0.7)
ax3.set_xticks(range(len(monthly_stats)))
ax3.set_xticklabels(monthly_labels, rotation=45, ha='right')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('ROI (%)', fontsize=12)
ax3.set_title('Monthly ROI Performance', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, monthly_stats['roi_pct'])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
             fontsize=9, fontweight='bold')

# Plot 4: Rolling 5-event ROI
ax4 = axes[1, 1]
window_size = 5
rolling_roi = event_stats['roi_pct'].rolling(window=window_size, min_periods=1).mean()
ax4.plot(event_stats.index, rolling_roi, linewidth=2, color='blue', label=f'{window_size}-Event Rolling Avg')
ax4.plot(event_stats.index, event_stats['roi_pct'], alpha=0.3, color='gray', label='Per-Event ROI')
ax4.axhline(y=event_stats['roi_pct'].mean(), color='red', linestyle='--', 
            linewidth=1, label=f'Overall Mean: {event_stats["roi_pct"].mean():.1f}%')
ax4.set_xlabel('Event Number', fontsize=12)
ax4.set_ylabel('ROI (%)', fontsize=12)
ax4.set_title(f'{window_size}-Event Rolling Average ROI', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roi_trend_analysis.png', dpi=150, bbox_inches='tight')
print("="*80)
print("VISUALIZATION SAVED")
print("="*80)
print("✅ Chart saved: roi_trend_analysis.png")
print()

# Additional analysis: First half vs Second half
midpoint = len(event_stats) // 2
first_half = event_stats.iloc[:midpoint]
second_half = event_stats.iloc[midpoint:]

first_half_roi = (first_half['total_profit'].sum() / first_half['total_stake'].sum()) * 100
second_half_roi = (second_half['total_profit'].sum() / second_half['total_stake'].sum()) * 100

print("="*80)
print("FIRST HALF vs SECOND HALF COMPARISON")
print("="*80)
print(f"First Half  ({first_half['DATE'].min().date()} to {first_half['DATE'].max().date()}):")
print(f"  Events: {len(first_half)}")
print(f"  Bets: {first_half['bets'].sum()}")
print(f"  ROI: {first_half_roi:+.2f}%")
print(f"  Win Rate: {first_half['wins'].sum() / first_half['bets'].sum() * 100:.1f}%")
print()
print(f"Second Half ({second_half['DATE'].min().date()} to {second_half['DATE'].max().date()}):")
print(f"  Events: {len(second_half)}")
print(f"  Bets: {second_half['bets'].sum()}")
print(f"  ROI: {second_half_roi:+.2f}%")
print(f"  Win Rate: {second_half['wins'].sum() / second_half['bets'].sum() * 100:.1f}%")
print()
print(f"Difference: {second_half_roi - first_half_roi:+.2f}% ROI")

if second_half_roi < first_half_roi:
    print("📉 ROI is LOWER in second half - confirms degradation")
else:
    print("📈 ROI is HIGHER in second half - model improving!")

print()

# Volatility analysis
print("="*80)
print("VOLATILITY ANALYSIS")
print("="*80)
print(f"Mean ROI per event: {event_stats['roi_pct'].mean():.2f}%")
print(f"Std Dev: {event_stats['roi_pct'].std():.2f}%")
print(f"Min ROI: {event_stats['roi_pct'].min():.2f}%")
print(f"Max ROI: {event_stats['roi_pct'].max():.2f}%")
print(f"Median ROI: {event_stats['roi_pct'].median():.2f}%")
print()

positive_events = (event_stats['roi_pct'] > 0).sum()
negative_events = (event_stats['roi_pct'] < 0).sum()
print(f"Positive ROI events: {positive_events} ({positive_events/len(event_stats)*100:.1f}%)")
print(f"Negative ROI events: {negative_events} ({negative_events/len(event_stats)*100:.1f}%)")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print("Key Takeaways:")
if p_value < 0.05:
    if slope < 0:
        print(f"  ⚠️  Statistically significant DOWNWARD trend ({slope:.3f}% per event)")
        print(f"     This confirms ROI degradation over time")
    else:
        print(f"  ✅ Statistically significant UPWARD trend ({slope:.3f}% per event)")
        print(f"     Model is actually improving!")
else:
    print(f"  ➡️  No significant trend detected (p={p_value:.3f})")
    print(f"     ROI fluctuations are likely normal variance")

print()
print(f"  Overall Mean ROI: {event_stats['roi_pct'].mean():.2f}%")
print(f"  Volatility (Std Dev): {event_stats['roi_pct'].std():.2f}%")
print(f"  Win Rate (Events): {positive_events}/{len(event_stats)} ({positive_events/len(event_stats)*100:.1f}%)")
print()
print("="*80)

