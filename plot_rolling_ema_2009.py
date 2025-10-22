"""
Plot rolling_ema over time from 2009-present
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/tmp/final.csv', low_memory=False)
df['DATE'] = pd.to_datetime(df['DATE'])
df['win_numeric'] = pd.to_numeric(df['win'], errors='coerce')
df['precomp_boutcount'] = pd.to_numeric(df['precomp_boutcount'], errors='coerce')
df['opp_precomp_boutcount'] = pd.to_numeric(df['opp_precomp_boutcount'], errors='coerce')

# Version B filter
df = df[(df['precomp_boutcount'] >= 1) & (df['opp_precomp_boutcount'] >= 1)].copy()
df = df.sort_values('DATE').reset_index(drop=True)

# Calculate rolling EMA
rolling_ema = df['win_numeric'].ewm(span=200, min_periods=20).mean()
df['precomp_rolling_ema'] = rolling_ema.shift(1)
df = df.dropna(subset=['precomp_rolling_ema'])

# Filter to 2009+
df_modern = df[df['DATE'] >= '2009-01-01'].copy()

print(f"Plotting {len(df_modern)} rows from 2009-present")
print(f"Date range: {df_modern['DATE'].min().date()} to {df_modern['DATE'].max().date()}")
print(f"EMA range: {df_modern['precomp_rolling_ema'].min():.4f} to {df_modern['precomp_rolling_ema'].max():.4f}")
print()

# Create figure with multiple views
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Raw rolling_ema over time
ax1 = axes[0]
ax1.plot(df_modern['DATE'], df_modern['precomp_rolling_ema'], alpha=0.6, linewidth=0.5)
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='0.50 baseline')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Rolling EMA', fontsize=12)
ax1.set_title('Rolling EMA Over Time (2009-Present, Version B Data)', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.45, 0.55)

# Plot 2: Monthly aggregated
df_modern['year_month'] = df_modern['DATE'].dt.to_period('M')
monthly = df_modern.groupby('year_month').agg({
    'precomp_rolling_ema': 'mean',
    'win_numeric': 'mean',
    'DATE': 'first'
}).reset_index()
monthly['DATE'] = monthly['DATE'].dt.date

ax2 = axes[1]
ax2.plot(monthly['DATE'], monthly['precomp_rolling_ema'], label='Rolling EMA (monthly avg)', linewidth=2, alpha=0.8)
ax2.plot(monthly['DATE'], monthly['win_numeric'], label='Actual win rate (monthly)', linewidth=1, alpha=0.6, color='orange')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='0.50 baseline')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Rate', fontsize=12)
ax2.set_title('Monthly Aggregated: EMA vs Actual Win Rate', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.40, 0.60)

# Plot 3: Distribution histogram
ax3 = axes[2]
ax3.hist(df_modern['precomp_rolling_ema'], bins=50, alpha=0.7, edgecolor='black')
ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='0.50 baseline')
ax3.axvline(x=df_modern['precomp_rolling_ema'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df_modern["precomp_rolling_ema"].mean():.4f}')
ax3.set_xlabel('Rolling EMA Value', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of Rolling EMA Values', fontsize=14, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rolling_ema_2009_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved: rolling_ema_2009_analysis.png")
print()

# Statistics
print("="*80)
print("STATISTICS (2009-Present)")
print("="*80)
print(f"Mean:   {df_modern['precomp_rolling_ema'].mean():.6f}")
print(f"Median: {df_modern['precomp_rolling_ema'].median():.6f}")
print(f"StdDev: {df_modern['precomp_rolling_ema'].std():.6f}")
print(f"Min:    {df_modern['precomp_rolling_ema'].min():.6f}")
print(f"Max:    {df_modern['precomp_rolling_ema'].max():.6f}")
print(f"Range:  {df_modern['precomp_rolling_ema'].max() - df_modern['precomp_rolling_ema'].min():.6f}")
print()

# Quantiles
print("Quantiles:")
for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    val = df_modern['precomp_rolling_ema'].quantile(q)
    print(f"  {q*100:5.1f}%: {val:.6f}")
print()

# Test set stats
cutoff = df_modern['DATE'].max() - pd.Timedelta(days=365)
test_df = df_modern[df_modern['DATE'] >= cutoff]
print("="*80)
print("TEST SET (Last 12 months)")
print("="*80)
print(f"Rows: {len(test_df)}")
print(f"EMA Mean:   {test_df['precomp_rolling_ema'].mean():.6f}")
print(f"EMA StdDev: {test_df['precomp_rolling_ema'].std():.6f}")
print(f"EMA Range:  {test_df['precomp_rolling_ema'].min():.6f} to {test_df['precomp_rolling_ema'].max():.6f}")
print()

