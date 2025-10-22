"""
XGBoost ROI Calculator with rolling_ema Feature

This script:
1. Loads data with rolling_ema feature
2. Trains XGBoost model with champion hyperparameters
3. Generates odds table (predicted probabilities)
4. Calculates ROI using Vegas odds
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from datetime import timedelta
import sys

print("=" * 80)
print("XGBoost ROI Calculator with rolling_ema Feature")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load Configuration and Data
# ============================================================================

print("STEP 1: Loading Configuration and Data")
print("-" * 80)

# Load champion XGBoost config
with open('xgboost_ga_results_1760303427.json') as f:
    config = json.load(f)

baseline_features = config['features']
print(f"✓ Loaded champion config: {len(baseline_features)} baseline features")
print(f"✓ Hyperparameters loaded")
print()

# Load data with rolling_ema
df = pd.read_csv('data/tmp/final_with_rolling_ema.csv', low_memory=False)
df['DATE'] = pd.to_datetime(df['DATE'])
print(f"✓ Loaded data: {df.shape}")
print(f"✓ Date range: {df['DATE'].min()} to {df['DATE'].max()}")
print()

# ============================================================================
# STEP 2: Prepare Data (Exact Champion Model Logic)
# ============================================================================

print("STEP 2: Preparing Data (Champion Model Logic)")
print("-" * 80)

# Apply exact filtering as champion model
df = df[df['DATE'] >= '2009-01-01']
df = df[df['sex'].astype(str) == '2']
print(f"✓ After date/sex filter: {len(df)} fights")

# Create differential Elo features if needed
if 'precomp_elo_diff' not in df.columns:
    df['precomp_elo_diff'] = df['precomp_elo'] - df['opp_precomp_elo']
if 'precomp_strike_elo_diff' not in df.columns:
    df['precomp_strike_elo_diff'] = df['precomp_strike_elo'] - df['opp_precomp_strike_elo']
if 'precomp_grapple_elo_diff' not in df.columns:
    df['precomp_grapple_elo_diff'] = df['precomp_grapple_elo'] - df['opp_precomp_grapple_elo']

# Handle win column
df = df.dropna(subset=['win'])
df['win'] = pd.to_numeric(df['win']).astype(int)

# Null filtering using baseline features only
thresh = int(0.7 * len(baseline_features))
null_counts = df[baseline_features].isnull().sum(axis=1)
df = df[null_counts <= thresh]
print(f"✓ After null filter: {len(df)} fights")

# Bout count filtering
df = df[(df['precomp_boutcount'] >= 1) & (df['opp_precomp_boutcount'] >= 1)]
print(f"✓ After bout count filter: {len(df)} fights")
print()

# ============================================================================
# STEP 3: Train/Test Split (Time Series)
# ============================================================================

print("STEP 3: Train/Test Split")
print("-" * 80)

# Use last 365 days as test set (same as validation)
latest = df['DATE'].max()
cutoff = latest - timedelta(days=365)

train = df[df['DATE'] < cutoff]
test = df[df['DATE'] >= cutoff]

print(f"✓ Cutoff date: {cutoff.date()}")
print(f"✓ Train set: {len(train)} fights ({train['DATE'].min().date()} to {train['DATE'].max().date()})")
print(f"✓ Test set: {len(test)} fights ({test['DATE'].min().date()} to {test['DATE'].max().date()})")
print()

# ============================================================================
# STEP 4: Train XGBoost Model with rolling_ema
# ============================================================================

print("STEP 4: Training XGBoost Model")
print("-" * 80)

# Features: baseline + rolling_ema
all_features = baseline_features + ['rolling_ema']
print(f"✓ Total features: {len(all_features)} ({len(baseline_features)} baseline + rolling_ema)")

# Prepare data
X_train = train[all_features]
y_train = train['win']
X_test = test[all_features]
y_test = test['win']

# Impute and scale
imputer = SimpleImputer(strategy='median')
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_scaled = scaler.transform(imputer.transform(X_test))

# Train model
print("Training XGBoost...")
model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    early_stopping_rounds=20,
    **config['hyperparams']
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)

print(f"✓ Model trained successfully")
print(f"✓ Test Accuracy: {acc*100:.2f}%")
print(f"✓ Test Log Loss: {ll:.4f}")
print()

# ============================================================================
# STEP 5: Generate Odds Table (Full Dataset)
# ============================================================================

print("STEP 5: Generating Odds Table")
print("-" * 80)

# Prepare full dataset for predictions
X_full = df[all_features]
X_full_scaled = scaler.transform(imputer.transform(X_full))

# Get predictions
y_pred_proba_full = model.predict_proba(X_full_scaled)[:, 1]

# Create odds table
odds_df = df[['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'win']].copy()
odds_df['predicted_prob'] = y_pred_proba_full

# Convert probability to odds (similar to logistic regression)
# American odds format
def prob_to_american_odds(prob):
    """Convert probability to American odds"""
    if prob >= 0.5:
        # Favorite (negative odds)
        return -(prob / (1 - prob)) * 100
    else:
        # Underdog (positive odds)
        return ((1 - prob) / prob) * 100

odds_df['odds'] = odds_df['predicted_prob'].apply(prob_to_american_odds)

# Save odds table
output_path = 'data/tmp/xgboost_ema_odds_table.csv'
odds_df.to_csv(output_path, index=False)

print(f"✓ Generated odds for {len(odds_df)} fights")
print(f"✓ Probability range: {odds_df['predicted_prob'].min():.3f} to {odds_df['predicted_prob'].max():.3f}")
print(f"✓ Odds table saved to: {output_path}")
print()

# Show sample predictions
print("Sample Predictions:")
sample = odds_df.tail(10)
for idx, row in sample.iterrows():
    result = "✅ WIN" if row['win'] == 1 else "❌ LOSS"
    fighter = str(row['FIGHTER'])[:30]
    print(f"   {fighter:30s} | Prob: {row['predicted_prob']:.1%} | Odds: {row['odds']:+7.0f} | {result}")
print()

# ============================================================================
# STEP 6: Calculate ROI
# ============================================================================

print("=" * 80)
print("STEP 6: ROI Calculation")
print("=" * 80)
print()

def calculate_roi_xgboost(odds_table_path, vegas_data_path, vegas_cols=None, stake=100):
    """
    Calculate ROI using XGBoost odds table and Vegas data
    Adapted from FightOutcomeModel.calculate_roi
    """
    # Load data
    df_model = pd.read_csv(odds_table_path, parse_dates=['DATE'])
    df_vegas = pd.read_csv(vegas_data_path, parse_dates=['DATE'])
    
    # Remove timezone
    try:
        df_vegas['DATE'] = df_vegas['DATE'].dt.tz_convert(None)
    except:
        pass
    try:
        df_model['DATE'] = df_model['DATE'].dt.tz_convert(None)
    except:
        pass
    
    print(f"✓ Model odds: {len(df_model)} rows")
    print(f"✓ Vegas data: {len(df_vegas)} rows")
    print(f"✓ Date range: {df_model['DATE'].min().date()} to {df_model['DATE'].max().date()}")
    print()
    
    # Merge keys
    key_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER']
    vegas_cols = vegas_cols or ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']
    
    # Merge
    df = pd.merge(
        df_model,
        df_vegas[key_cols + vegas_cols + ['win']],
        on=key_cols,
        how='inner',
        suffixes=('_model', '_actual')
    )
    
    print(f"✓ Merged data: {len(df)} rows")
    
    # Calculate average Vegas odds
    df['avg_vegas_odds'] = df[vegas_cols].mean(axis=1, skipna=True)
    
    # Filter out missing data
    df_clean = df.dropna(subset=['avg_vegas_odds', 'win_actual'])
    print(f"✓ After filtering: {len(df_clean)} fights with Vegas odds")
    print()
    
    if len(df_clean) == 0:
        print("❌ No valid data for ROI calculation")
        return None
    
    # Make picks: For each bout, pick the fighter with higher predicted probability
    picks = []
    for bout_id in df_clean['BOUT'].unique():
        bout_data = df_clean[df_clean['BOUT'] == bout_id]
        if len(bout_data) == 2:  # Complete bout
            # Pick fighter with higher predicted probability
            pick_idx = bout_data['predicted_prob'].idxmax()
            pick = bout_data.loc[pick_idx].copy()
            picks.append(pick)
    
    if len(picks) == 0:
        print("❌ No valid picks")
        return None
    
    picks_df = pd.DataFrame(picks)
    print(f"✓ Made picks for {len(picks_df)} fights")
    print()
    
    # Calculate profit for each pick
    def calculate_profit(vegas_odds, stake=100, won=True):
        if not won:
            return -stake
        if abs(vegas_odds) < 0.01:
            return 0
        if vegas_odds > 0:
            return (vegas_odds / 100) * stake
        else:
            return (100 / abs(vegas_odds)) * stake
    
    picks_df['profit'] = picks_df.apply(
        lambda row: calculate_profit(row['avg_vegas_odds'], stake, row['win_actual'] == 1),
        axis=1
    )
    
    # Calculate cumulative metrics
    picks_df = picks_df.sort_values('DATE')
    picks_df['cum_profit'] = picks_df['profit'].cumsum()
    picks_df['cum_stake'] = (np.arange(len(picks_df)) + 1) * stake
    picks_df['cum_roi'] = picks_df['cum_profit'] / picks_df['cum_stake']
    
    return picks_df

# Try to calculate ROI
try:
    print("📊 Calculating ROI with Vegas odds...")
    print()
    
    roi_df = calculate_roi_xgboost(
        odds_table_path=output_path,
        vegas_data_path='final_with_odds_clamped.csv'
    )
    
    if roi_df is not None and len(roi_df) > 0:
        # Calculate metrics
        total_stake = len(roi_df) * 100
        total_profit = roi_df['profit'].sum()
        roi_pct = (total_profit / total_stake) * 100
        win_rate = roi_df['win_actual'].mean()
        
        print("=" * 80)
        print("💰 ROI RESULTS")
        print("=" * 80)
        print()
        print(f"Total Bets:    {len(roi_df)}")
        print(f"Wins:          {roi_df['win_actual'].sum()}")
        print(f"Losses:        {len(roi_df) - roi_df['win_actual'].sum()}")
        print(f"Win Rate:      {win_rate*100:.2f}%")
        print()
        print(f"Total Stake:   ${total_stake:,.2f}")
        print(f"Total Profit:  ${total_profit:,.2f}")
        print(f"ROI:           {roi_pct:.2f}%")
        print()
        
        # Show performance by confidence level
        roi_df['prob_bucket'] = pd.cut(roi_df['predicted_prob'], 
                                        bins=[0, 0.6, 0.7, 0.8, 1.0],
                                        labels=['<60%', '60-70%', '70-80%', '>80%'])
        
        print("Performance by Confidence Level:")
        print("-" * 80)
        for bucket in ['<60%', '60-70%', '70-80%', '>80%']:
            bucket_data = roi_df[roi_df['prob_bucket'] == bucket]
            if len(bucket_data) > 0:
                bucket_wins = bucket_data['win_actual'].sum()
                bucket_wr = bucket_data['win_actual'].mean()
                bucket_profit = bucket_data['profit'].sum()
                bucket_roi = (bucket_profit / (len(bucket_data) * 100)) * 100
                print(f"{bucket:8s} | Bets: {len(bucket_data):4d} | Wins: {bucket_wins:4d} | WR: {bucket_wr:.1%} | ROI: {bucket_roi:+7.2f}%")
        
        print()
        
        # Save results
        roi_output = 'data/tmp/xgboost_ema_roi_results.csv'
        roi_df.to_csv(roi_output, index=False)
        print(f"✓ Saved ROI results to: {roi_output}")
        
        # Show sample picks
        print()
        print("Sample Picks (Last 10):")
        print("-" * 80)
        sample = roi_df.tail(10)
        for idx, row in sample.iterrows():
            result = "✅ WIN" if row['win_actual'] == 1 else "❌ LOSS"
            fighter = str(row['FIGHTER'])[:25]
            print(f"{fighter:25s} | Prob: {row['predicted_prob']:.1%} | Vegas: {row['avg_vegas_odds']:+6.0f} | Profit: ${row['profit']:+7.2f} | {result}")
        
except FileNotFoundError as e:
    print(f"❌ Vegas odds file not found: {e}")
    print()
    print("To calculate ROI, ensure 'final_with_odds_clamped.csv' exists")
    print("You can still use the odds table for predictions:")
    print(f"  {output_path}")
    
except Exception as e:
    print(f"❌ Error calculating ROI: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("✅ XGBoost ROI Calculator Complete!")
print("=" * 80)
print()
print("Files created:")
print(f"  1. Odds table: {output_path}")
if 'roi_output' in locals():
    print(f"  2. ROI results: {roi_output}")
print()

