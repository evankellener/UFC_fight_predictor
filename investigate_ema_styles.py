"""
Investigate rolling_ema interactions with style-specific features
"""
import sys
sys.path.insert(0, '/Users/evankellener/Desktop/UFC_fight_predictor/src')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from datetime import timedelta
import json

# Load champion config
with open('/Users/evankellener/Desktop/UFC_fight_predictor/xgboost_ga_results_1760303427.json') as f:
    config = json.load(f)

champion_features = config['features']

print("="*80)
print("INVESTIGATING ROLLING_EMA × STYLE FEATURES")
print("="*80)
print()

# Load and prepare data
df = pd.read_csv('/Users/evankellener/Desktop/UFC_fight_predictor/data/tmp/final.csv', low_memory=False)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[df['DATE'] >= '2009-01-01']
df = df[df['sex'].astype(str) == '2']

# Create diff features
if 'precomp_elo_diff' not in df.columns:
    df['precomp_elo_diff'] = df['precomp_elo'] - df['opp_precomp_elo']
if 'precomp_strike_elo_diff' not in df.columns:
    df['precomp_strike_elo_diff'] = df['precomp_strike_elo'] - df['opp_precomp_strike_elo']
if 'precomp_grapple_elo_diff' not in df.columns:
    df['precomp_grapple_elo_diff'] = df['precomp_grapple_elo'] - df['opp_precomp_grapple_elo']

df = df.dropna(subset=['win'])
df['win'] = pd.to_numeric(df['win']).astype(int)

# Filter
thresh = int(0.7 * len(champion_features))
null_counts = df[champion_features].isnull().sum(axis=1)
df = df[null_counts <= thresh]
df = df[(df['precomp_boutcount'] >= 1) & (df['opp_precomp_boutcount'] >= 1)]

# Add rolling_ema
df = df.sort_values('DATE').reset_index(drop=True)
df['win_numeric'] = pd.to_numeric(df['win'], errors='coerce')
df['rolling_ema'] = df['win_numeric'].ewm(span=200, min_periods=20).mean().shift(1)
df = df.dropna(subset=['rolling_ema'])

# Train/test split
latest = df['DATE'].max()
cutoff = latest - timedelta(days=365)
train = df[df['DATE'] < cutoff]
test = df[df['DATE'] >= cutoff]

# Prepare features
features_with_ema = champion_features + ['rolling_ema']

# Impute and scale
imp = SimpleImputer(strategy='median')
scaler = RobustScaler()

X_train = train[features_with_ema]
y_train = train['win']
X_test = test[features_with_ema]
y_test = test['win']

X_train_scaled = scaler.fit_transform(imp.fit_transform(X_train))
X_test_scaled = scaler.transform(imp.transform(X_test))

# Train model
model = xgb.XGBClassifier(
    random_state=42, 
    n_jobs=-1, 
    eval_metric='logloss',
    early_stopping_rounds=20,
    **config['hyperparams']
)
model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

print(f"Model trained. Test accuracy: {model.score(X_test_scaled, y_test):.4f}")
print()

# ============================================================================
# INVESTIGATION 1: Verify the prediction swing on MULTIPLE fights
# ============================================================================
print("="*80)
print("1. VERIFYING PREDICTION SWINGS ON MULTIPLE SAMPLE FIGHTS")
print("="*80)
print()

ema_feature_idx = features_with_ema.index('rolling_ema')
ema_values = np.linspace(0.40, 0.60, 11)

# Test on 5 random fights
np.random.seed(42)
sample_indices = np.random.choice(len(X_test_scaled), 5, replace=False)

for fight_num, sample_idx in enumerate(sample_indices):
    sample_fight = X_test_scaled[sample_idx:sample_idx+1].copy()
    sample_features = X_test.iloc[sample_idx]
    
    print(f"Fight {fight_num + 1}:")
    print(f"  Original Elo diff: {sample_features['precomp_elo_diff']:.1f}")
    print(f"  Original rolling_ema: {sample_features['rolling_ema']:.4f}")
    print(f"  Original prediction: {model.predict_proba(sample_fight)[0, 1]:.4f}")
    
    predictions = []
    for ema_val in ema_values:
        modified_sample = sample_fight.copy()
        # Don't re-scale - just use the raw values
        # Actually, we need to be more careful here
        # Get the original scaled value and see the range
        modified_sample[0, ema_feature_idx] = (ema_val - imp.statistics_[ema_feature_idx]) / scaler.scale_[ema_feature_idx]
        pred = model.predict_proba(modified_sample)[0, 1]
        predictions.append(pred)
    
    print(f"  Predictions when varying EMA from 0.40 to 0.60:")
    for i, (ema_val, pred) in enumerate(zip(ema_values, predictions)):
        if i % 2 == 0:  # Print every other one to save space
            print(f"    EMA {ema_val:.2f}: {pred:.4f}")
    
    swing = max(predictions) - min(predictions)
    print(f"  Swing: {swing:.4f} ({swing*100:.1f}%)")
    print()

# ============================================================================
# INVESTIGATION 2: Style-specific feature interactions with rolling_ema
# ============================================================================
print("="*80)
print("2. STYLE FEATURE INTERACTIONS WITH ROLLING_EMA")
print("="*80)
print()

# Define style-specific features
striking_features = [
    'precomp_strike_elo_diff',
    'precomp_legacc_perc5',
    'opp_precomp_sigstr_pm5',
    'precomp_distacc_perc',
    'opp_precomp_str_eff_diff3',
    'opp_precomp_sapm3',
    'opp_precomp_strdef5',
    'opp_precomp_headacc_perc5'
]

grappling_features = [
    'precomp_grapple_elo_diff',
    'opp_precomp_grapple_strike_mix',
    'opp_precomp_clinchacc_perc',
    'precomp_tdavg3',
    'precomp_groundacc_perc',
    'opp_precomp_ctrl_per_min',
    'precomp_ctrl_per_min',
    'opp_precomp_tdavg5',
    'opp_precomp_groundacc_perc5'
]

# Filter to only features we have
striking_features = [f for f in striking_features if f in features_with_ema]
grappling_features = [f for f in grappling_features if f in features_with_ema]

print(f"Striking features: {len(striking_features)}")
print(f"Grappling features: {len(grappling_features)}")
print()

# Test: How does increasing striking/grappling stats affect predictions at different EMA levels?
ema_levels = [0.42, 0.47, 0.50, 0.53, 0.58]
median_features = np.median(X_train_scaled, axis=0).reshape(1, -1)

print("Effect of STRIKING advantage at different rolling_ema levels:")
print("-" * 70)

strike_elo_idx = features_with_ema.index('precomp_strike_elo_diff')

for ema_val in ema_levels:
    synthetic_sample_baseline = median_features.copy()
    synthetic_sample_strong = median_features.copy()
    
    # Set rolling_ema
    ema_scaled = (ema_val - imp.statistics_[ema_feature_idx]) / scaler.scale_[ema_feature_idx]
    synthetic_sample_baseline[0, ema_feature_idx] = ema_scaled
    synthetic_sample_strong[0, ema_feature_idx] = ema_scaled
    
    # Baseline: strike_elo_diff = 0
    strike_elo_scaled_0 = (0 - imp.statistics_[strike_elo_idx]) / scaler.scale_[strike_elo_idx]
    synthetic_sample_baseline[0, strike_elo_idx] = strike_elo_scaled_0
    
    # Strong striker: strike_elo_diff = +100
    strike_elo_scaled_100 = (100 - imp.statistics_[strike_elo_idx]) / scaler.scale_[strike_elo_idx]
    synthetic_sample_strong[0, strike_elo_idx] = strike_elo_scaled_100
    
    pred_baseline = model.predict_proba(synthetic_sample_baseline)[0, 1]
    pred_strong = model.predict_proba(synthetic_sample_strong)[0, 1]
    
    marginal_effect = pred_strong - pred_baseline
    
    print(f"rolling_ema = {ema_val:.2f}: Strike Elo +100 → {marginal_effect:+.4f} ({marginal_effect*100:+.1f}%)")

print()
print("Effect of GRAPPLING advantage at different rolling_ema levels:")
print("-" * 70)

grapple_elo_idx = features_with_ema.index('precomp_grapple_elo_diff')

for ema_val in ema_levels:
    synthetic_sample_baseline = median_features.copy()
    synthetic_sample_strong = median_features.copy()
    
    # Set rolling_ema
    ema_scaled = (ema_val - imp.statistics_[ema_feature_idx]) / scaler.scale_[ema_feature_idx]
    synthetic_sample_baseline[0, ema_feature_idx] = ema_scaled
    synthetic_sample_strong[0, ema_feature_idx] = ema_scaled
    
    # Baseline: grapple_elo_diff = 0
    grapple_elo_scaled_0 = (0 - imp.statistics_[grapple_elo_idx]) / scaler.scale_[grapple_elo_idx]
    synthetic_sample_baseline[0, grapple_elo_idx] = grapple_elo_scaled_0
    
    # Strong grappler: grapple_elo_diff = +100
    grapple_elo_scaled_100 = (100 - imp.statistics_[grapple_elo_idx]) / scaler.scale_[grapple_elo_idx]
    synthetic_sample_strong[0, grapple_elo_idx] = grapple_elo_scaled_100
    
    pred_baseline = model.predict_proba(synthetic_sample_baseline)[0, 1]
    pred_strong = model.predict_proba(synthetic_sample_strong)[0, 1]
    
    marginal_effect = pred_strong - pred_baseline
    
    print(f"rolling_ema = {ema_val:.2f}: Grapple Elo +100 → {marginal_effect:+.4f} ({marginal_effect*100:+.1f}%)")

print()

# ============================================================================
# INVESTIGATION 3: Check actual rolling_ema distribution in data
# ============================================================================
print("="*80)
print("3. ACTUAL ROLLING_EMA DISTRIBUTION IN DATA")
print("="*80)
print()

print("Training set rolling_ema statistics:")
print(f"  Min: {train['rolling_ema'].min():.4f}")
print(f"  25th percentile: {train['rolling_ema'].quantile(0.25):.4f}")
print(f"  Median: {train['rolling_ema'].median():.4f}")
print(f"  75th percentile: {train['rolling_ema'].quantile(0.75):.4f}")
print(f"  Max: {train['rolling_ema'].max():.4f}")
print()

print("Test set rolling_ema statistics:")
print(f"  Min: {test['rolling_ema'].min():.4f}")
print(f"  25th percentile: {test['rolling_ema'].quantile(0.25):.4f}")
print(f"  Median: {test['rolling_ema'].median():.4f}")
print(f"  75th percentile: {test['rolling_ema'].quantile(0.75):.4f}")
print(f"  Max: {test['rolling_ema'].max():.4f}")
print()

print("Note: If most values are between 0.48-0.52, then testing at 0.40 or 0.60")
print("      is extrapolating to unseen data, which can cause weird predictions!")

print()
print("="*80)
print("CONCLUSIONS")
print("="*80)

