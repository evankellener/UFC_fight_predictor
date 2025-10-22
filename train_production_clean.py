"""
Train Production Model - CLEAN VERSION
- Version B filter (both fighters with 1+ fights)
- 28 champion features only (NO rolling_ema)
- Trained on ALL available data
"""
import sys
sys.path.insert(0, '/Users/evankellener/Desktop/UFC_fight_predictor/src')

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import joblib
from datetime import datetime

print("="*80)
print("TRAINING CLEAN PRODUCTION MODEL")
print("="*80)
print()

# Load data
df = pd.read_csv('data/tmp/final.csv', low_memory=False)
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

print(f"Original data: {len(df)} rows")

# Apply filters (same as FightOutcomeModel.__init__)
df = df[df['DATE'] >= '2009-01-01']
df = df[df['sex'].astype(str) == '2']
print(f"After 2009+ and sex filter: {len(df)} rows")

# Convert numeric columns
numeric_cols = [col for col in df.columns if col not in ['FIGHTER', 'EVENT', 'DATE', 'win', 'BOUT', 'sex']]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['win_numeric'] = pd.to_numeric(df['win'], errors='coerce')

# Apply Version B filter
df = df[(df['precomp_boutcount'] >= 1) & (df['opp_precomp_boutcount'] >= 1)].copy()
print(f"After Version B filter: {len(df)} rows")
print()

# Calculate differential ELO features
if 'precomp_strike_elo' in df.columns and 'opp_precomp_strike_elo' in df.columns:
    df['precomp_strike_elo_diff'] = df['precomp_strike_elo'] - df['opp_precomp_strike_elo']
if 'precomp_grapple_elo' in df.columns and 'opp_precomp_grapple_elo' in df.columns:
    df['precomp_grapple_elo_diff'] = df['precomp_grapple_elo'] - df['opp_precomp_grapple_elo']

# Champion features (28 total - NO rolling_ema)
features = [
    'precomp_elo_diff', 'precomp_strike_elo_diff', 'precomp_grapple_elo_diff',
    'precomp_legacc_perc5', 'opp_precomp_sigstr_pm5', 'opp_precomp_grapple_strike_mix',
    'opp_precomp_clinchacc_perc', 'opp_age_ratio_difference', 'opp_precomp_elo',
    'age_ratio_difference', 'precomp_distacc_perc', 'opp_precomp_winsum',
    'precomp_tdavg3', 'opp_precomp_legacc_perc3', 'opp_precomp_str_eff_diff3',
    'precomp_winsum', 'opp_precomp_sapm3', 'precomp_groundacc_perc',
    'opp_precomp_ctrl_per_min', 'opp_REACH', 'precomp_winsum5',
    'opp_precomp_strdef5', 'precomp_ctrl_per_min', 'opp_precomp_tdavg5',
    'opp_precomp_headacc_perc5', 'precomp_elo_change_5', 'opp_precomp_winsum3',
    'opp_precomp_groundacc_perc5'
]

print(f"Using {len(features)} features (NO rolling_ema)")
print()

# Check for missing features
missing = [f for f in features if f not in df.columns]
if missing:
    print(f"ERROR: Missing features: {missing}")
    sys.exit(1)

# Prepare data
df_clean = df.dropna(subset=['win_numeric'])
df_clean = df_clean.dropna(subset=features)

print(f"After cleaning: {len(df_clean)} rows")
print(f"Date range: {df_clean['DATE'].min().date()} to {df_clean['DATE'].max().date()}")
print()

# Check balance
wins = (df_clean['win_numeric'] == 1).sum()
losses = (df_clean['win_numeric'] == 0).sum()
print(f"Win=1: {wins}")
print(f"Win=0: {losses}")
print(f"Balance: {wins/(wins+losses)*100:.2f}%")
print()

# Prepare features and target
X = df_clean[features].copy()
y = df_clean['win_numeric'].copy()

print(f"Training set: {len(X)} samples, {len(features)} features")
print()

# Load champion hyperparameters
with open('xgboost_ga_results_1760303427.json', 'r') as f:
    champion_config = json.load(f)
    hp = champion_config['hyperparams']

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': hp['max_depth'],
    'learning_rate': hp['learning_rate'],
    'n_estimators': hp['n_estimators'],
    'min_child_weight': hp['min_child_weight'],
    'subsample': hp['subsample'],
    'colsample_bytree': hp['colsample_bytree'],
    'gamma': hp['gamma'],
    'reg_alpha': hp['reg_alpha'],
    'reg_lambda': hp['reg_lambda'],
    'random_state': 42
}

print("="*80)
print("TRAINING XGBOOST")
print("="*80)
print(f"Hyperparameters:")
for k, v in params.items():
    print(f"  {k}: {v}")
print()

model = xgb.XGBClassifier(**params)
model.fit(X, y, verbose=False)

# Training accuracy
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

from sklearn.metrics import accuracy_score, log_loss
train_acc = accuracy_score(y, y_pred)
train_ll = log_loss(y, y_proba)

print("="*80)
print("TRAINING RESULTS")
print("="*80)
print(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Train Log Loss: {train_ll:.4f}")
print()

# Feature importance
feature_importance = model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': feature_importance.get(k, 0)}
    for k in features
]).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print("-"*80)
for i, row in importance_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30} {row['importance']:.4f}")
print()

# Save model
model_path = 'saved_models/production_clean_xgboost.joblib'
joblib.dump(model, model_path)
print(f"✅ Model saved: {model_path}")

# Save features list
features_path = 'saved_models/production_clean_features.json'
with open(features_path, 'w') as f:
    json.dump({'features': features}, f, indent=2)
print(f"✅ Features saved: {features_path}")

# Save metadata
metadata = {
    'model_type': 'XGBoost',
    'training_date': datetime.now().isoformat(),
    'num_features': len(features),
    'features': features,
    'hyperparameters': params,
    'data_filters': [
        'DATE >= 2009-01-01',
        'sex == 2',
        'precomp_boutcount >= 1',
        'opp_precomp_boutcount >= 1'
    ],
    'training_rows': len(X),
    'train_accuracy': float(train_acc),
    'train_log_loss': float(train_ll),
    'date_range': {
        'start': df_clean['DATE'].min().isoformat(),
        'end': df_clean['DATE'].max().isoformat()
    },
    'data_balance': {
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(wins/(wins+losses))
    }
}

metadata_path = 'saved_models/production_clean_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")
print()

print("="*80)
print("PRODUCTION MODEL READY")
print("="*80)
print()
print("Model characteristics:")
print(f"  • Clean data (Version B filter)")
print(f"  • 28 features (NO rolling_ema)")
print(f"  • Trained on {len(X):,} fights")
print(f"  • {df_clean['DATE'].min().year}-{df_clean['DATE'].max().year} date range")
print(f"  • {train_acc*100:.2f}% training accuracy")
print()
print("Ready for production predictions!")
print("="*80)

