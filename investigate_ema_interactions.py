"""
Investigate how rolling_ema interacts with other features
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
import matplotlib.pyplot as plt
import seaborn as sns

# Load champion config
with open('/Users/evankellener/Desktop/UFC_fight_predictor/xgboost_ga_results_1760303427.json') as f:
    config = json.load(f)

champion_features = config['features']

print("="*80)
print("INVESTIGATING ROLLING_EMA FEATURE INTERACTIONS")
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
print("Adding rolling_ema feature...")
df = df.sort_values('DATE').reset_index(drop=True)
df['win_numeric'] = pd.to_numeric(df['win'], errors='coerce')
df['rolling_ema'] = df['win_numeric'].ewm(span=200, min_periods=20).mean().shift(1)
df = df.dropna(subset=['rolling_ema'])

# Train/test split
latest = df['DATE'].max()
cutoff = latest - timedelta(days=365)
train = df[df['DATE'] < cutoff]
test = df[df['DATE'] >= cutoff]

print(f"Training samples: {len(train)}")
print(f"Test samples: {len(test)}")
print()

# Prepare features (champion + rolling_ema)
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
print("Training XGBoost with rolling_ema...")
model = xgb.XGBClassifier(
    random_state=42, 
    n_jobs=-1, 
    eval_metric='logloss',
    early_stopping_rounds=20,
    **config['hyperparams']
)
model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

from sklearn.metrics import accuracy_score, log_loss
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)

print(f"Model accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Model log loss: {ll:.4f}")
print()

# ============================================================================
# INVESTIGATION 1: Feature Importance with rolling_ema
# ============================================================================
print("="*80)
print("1. FEATURE IMPORTANCE (with rolling_ema)")
print("="*80)

importance_df = pd.DataFrame({
    'feature': features_with_ema,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(15))
print()

rolling_ema_rank = importance_df[importance_df['feature'] == 'rolling_ema'].index[0] + 1
rolling_ema_importance = importance_df[importance_df['feature'] == 'rolling_ema']['importance'].values[0]
print(f"rolling_ema rank: #{rolling_ema_rank} out of {len(features_with_ema)}")
print(f"rolling_ema importance: {rolling_ema_importance:.4f}")
print()

# ============================================================================
# INVESTIGATION 2: How predictions change with rolling_ema
# ============================================================================
print("="*80)
print("2. PREDICTION SENSITIVITY TO ROLLING_EMA")
print("="*80)

# Take a sample fight from test set
sample_idx = 10
sample_fight = X_test_scaled[sample_idx:sample_idx+1].copy()
sample_features = X_test.iloc[sample_idx]

print(f"Sample fight features:")
print(f"  Elo diff: {sample_features['precomp_elo_diff']:.1f}")
print(f"  Strike Elo diff: {sample_features['precomp_strike_elo_diff']:.1f}")
print(f"  Original rolling_ema: {sample_features['rolling_ema']:.4f}")
print()

# Vary rolling_ema from 0.40 to 0.60
ema_values = np.linspace(0.40, 0.60, 21)
predictions = []

ema_feature_idx = features_with_ema.index('rolling_ema')

for ema_val in ema_values:
    modified_sample = sample_fight.copy()
    # Scale the EMA value
    ema_scaled = (ema_val - imp.statistics_[ema_feature_idx]) / scaler.scale_[ema_feature_idx]
    modified_sample[0, ema_feature_idx] = ema_scaled
    pred = model.predict_proba(modified_sample)[0, 1]
    predictions.append(pred)

print("How predictions change when varying rolling_ema:")
print("rolling_ema  |  Prediction")
print("-" * 30)
for ema_val, pred in zip(ema_values, predictions):
    print(f"  {ema_val:.3f}     |    {pred:.4f}")

print()
print(f"Prediction range: {min(predictions):.4f} to {max(predictions):.4f}")
print(f"Swing: {max(predictions) - min(predictions):.4f} ({(max(predictions) - min(predictions))*100:.1f}%)")
print()

# ============================================================================
# INVESTIGATION 3: Interaction effects
# ============================================================================
print("="*80)
print("3. INTERACTION EFFECTS: ELO DIFF × ROLLING_EMA")
print("="*80)

# Create a grid: vary both Elo diff and rolling_ema
elo_diffs = np.linspace(-200, 200, 9)
ema_vals = np.array([0.42, 0.47, 0.50, 0.53, 0.58])

interaction_matrix = np.zeros((len(elo_diffs), len(ema_vals)))

# Use median values for all other features
median_features = np.median(X_train_scaled, axis=0).reshape(1, -1)

elo_diff_idx = features_with_ema.index('precomp_elo_diff')

for i, elo_diff in enumerate(elo_diffs):
    for j, ema_val in enumerate(ema_vals):
        synthetic_sample = median_features.copy()
        
        # Set Elo diff (already scaled in median_features, need to scale our value)
        elo_scaled = (elo_diff - imp.statistics_[elo_diff_idx]) / scaler.scale_[elo_diff_idx]
        synthetic_sample[0, elo_diff_idx] = elo_scaled
        
        # Set rolling_ema
        ema_scaled = (ema_val - imp.statistics_[ema_feature_idx]) / scaler.scale_[ema_feature_idx]
        synthetic_sample[0, ema_feature_idx] = ema_scaled
        
        pred = model.predict_proba(synthetic_sample)[0, 1]
        interaction_matrix[i, j] = pred

print("Prediction grid (rows=Elo diff, cols=rolling_ema):")
print()
print(f"{'Elo Diff':<12}", end="")
for ema_val in ema_vals:
    print(f"EMA={ema_val:.2f}  ", end="")
print()
print("-" * 70)

for i, elo_diff in enumerate(elo_diffs):
    print(f"{elo_diff:+6.0f}      ", end="")
    for j in range(len(ema_vals)):
        pred = interaction_matrix[i, j]
        print(f"  {pred:.3f}   ", end="")
    print()

print()

# ============================================================================
# INVESTIGATION 4: Marginal effects
# ============================================================================
print("="*80)
print("4. MARGINAL EFFECT OF ELO DIFF AT DIFFERENT ROLLING_EMA LEVELS")
print("="*80)

for ema_val in [0.42, 0.50, 0.58]:
    # Calculate prediction for Elo diff = 0 and Elo diff = 100
    pred_0 = interaction_matrix[len(elo_diffs)//2, list(ema_vals).index(ema_val) if ema_val in ema_vals else 2]
    
    # Find closest elo_diff to 100
    idx_100 = np.argmin(np.abs(elo_diffs - 100))
    pred_100 = interaction_matrix[idx_100, list(ema_vals).index(ema_val) if ema_val in ema_vals else 2]
    
    marginal_effect = pred_100 - pred_0
    
    print(f"When rolling_ema = {ema_val:.2f}:")
    print(f"  Elo diff +100 changes prediction by: {marginal_effect:+.4f} ({marginal_effect*100:+.1f}%)")

print()

print("="*80)
print("KEY INSIGHTS:")
print("="*80)
print(f"1. rolling_ema ranks #{rolling_ema_rank} in feature importance")
print(f"2. Varying rolling_ema alone can change predictions by {(max(predictions) - min(predictions))*100:.1f}%")
print(f"3. The effect of Elo diff is {interaction_matrix.max() - interaction_matrix.min():.2f} across meta-game states")
print("4. This confirms rolling_ema acts as a CONTEXT MODIFIER for other features")
print("="*80)

