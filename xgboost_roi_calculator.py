#!/usr/bin/env python3
"""
XGBoost ROI Calculator
Uses the champion XGBoost GA model to generate odds and calculate ROI
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import json
from ensemble_model_best import FightOutcomeModel
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

print("💰 XGBoost ROI Calculator")
print("=" * 80)
print("Using champion XGBoost GA model to generate odds and calculate ROI")
print("=" * 80)
print()

# Load XGBoost champion configuration
print("📂 Loading XGBoost champion model...")
with open('xgboost_ga_results_1760303427.json', 'r') as f:
    xgb_config = json.load(f)

features = xgb_config['features']
hyperparams = xgb_config['hyperparams']

print(f"   Features: {len(features)}")
print(f"   Hyperparameters: {hyperparams}")
print()

# Initialize model
print("🏗️  Initializing FightOutcomeModel...")
fight_model = FightOutcomeModel("data/tmp/final.csv", random_seed=42)

print(f"   Train set: {len(fight_model.train_df)} fights")
print(f"   Test set: {len(fight_model.test_df)} fights")
print()

# Prepare data
print("🔧 Preparing data...")
train_df = fight_model.train_df.copy()
test_df = fight_model.test_df.copy()

# Impute and scale
imp = SimpleImputer(strategy='median')
X_train = imp.fit_transform(train_df[features])
X_test = imp.transform(test_df[features])

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = train_df['win']
y_test = test_df['win']

print(f"   Training features shape: {X_train_scaled.shape}")
print(f"   Test features shape: {X_test_scaled.shape}")
print()

# Train XGBoost model
print("🌲 Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    **hyperparams
)

xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print("   ✅ Model trained!")
print()

# Generate predictions and odds
print("📊 Generating odds table...")

# Get probabilities for all data (train + test combined for full odds table)
all_df = pd.concat([train_df, test_df], ignore_index=True)
X_all = imp.transform(all_df[features])
X_all_scaled = scaler.transform(X_all)

# Predict probabilities
probabilities = xgb_model.predict_proba(X_all_scaled)[:, 1]

# Create odds table - match format expected by calculate_roi
odds_df = pd.DataFrame({
    'EVENT': all_df['EVENT'] if 'EVENT' in all_df.columns else all_df['jevent'] if 'jevent' in all_df.columns else 'Unknown',
    'BOUT': all_df['BOUT'] if 'BOUT' in all_df.columns else all_df['jbout'] if 'jbout' in all_df.columns else range(len(all_df)),
    'jfighter': all_df['jfighter'] if 'jfighter' in all_df.columns else range(len(all_df)),
    'FIGHTER': all_df['FIGHTER'] if 'FIGHTER' in all_df.columns else all_df['fighter_name'] if 'fighter_name' in all_df.columns else 'Unknown',
    'DATE': all_df['DATE'] if 'DATE' in all_df.columns else all_df['date'] if 'date' in all_df.columns else pd.NaT,
    'win': all_df['win'],
    'predicted_prob': probabilities,
    'odds': [round(1 / p, 2) if p > 0.01 else 100.0 for p in probabilities]  # Renamed to 'odds' to match expected format
})

print(f"   ✅ Created odds table with {len(odds_df)} fights")
print(f"   Columns: {', '.join(odds_df.columns)}")

# Save odds table
output_path = 'data/tmp/xgboost_odds_table.csv'
odds_df.to_csv(output_path, index=False)

print(f"   ✅ Odds table saved to: {output_path}")
print(f"   Rows: {len(odds_df)}")
print()

# Show sample odds
print("📋 Sample XGBoost Odds (last 10 fights):")
print()

sample = odds_df.tail(10)

for idx, row in sample.iterrows():
    result = "✅ WIN" if row['win'] == 1 else "❌ LOSS"
    fighter = str(row['FIGHTER'])[:30]
    print(f"   {fighter:30s} | Prob: {row['predicted_prob']:.1%} | Odds: {row['odds']:.2f} | {result}")

print()

# Calculate ROI if Vegas odds available
print("=" * 80)
print("💰 ROI Calculation")
print("=" * 80)
print()

try:
    # Try to load Vegas odds
    vegas_df = pd.read_csv('final_with_odds_clamped.csv')
    print(f"✅ Loaded Vegas odds: {len(vegas_df)} fights")
    print()
    
    # Calculate ROI using fight_model's method
    print("🎲 Calculating ROI...")
    roi_df = fight_model.calculate_roi(
        odds_table_path=output_path,
        vegas_data_path='final_with_odds_clamped.csv'
    )
    
    if roi_df is not None and len(roi_df) > 0:
        print()
        print("📈 ROI Results:")
        print(f"   Total bets: {len(roi_df)}")
        print(f"   Wins: {roi_df['win'].sum()}")
        print(f"   Win rate: {roi_df['win'].mean():.2%}")
        
        if 'profit' in roi_df.columns:
            total_profit = roi_df['profit'].sum()
            total_stake = len(roi_df) * 100  # Assuming $100 per bet
            roi = (total_profit / total_stake) * 100
            
            print(f"   Total profit: ${total_profit:.2f}")
            print(f"   Total stake: ${total_stake:.2f}")
            print(f"   ROI: {roi:.2f}%")
        
        # Save ROI results
        roi_output = 'data/tmp/xgboost_roi_results.csv'
        roi_df.to_csv(roi_output, index=False)
        print(f"   💾 Saved to: {roi_output}")
    
except FileNotFoundError as e:
    print(f"⚠️  Vegas odds not found: {e}")
    print("   Skipping ROI calculation")
except Exception as e:
    print(f"⚠️  Error calculating ROI: {e}")
    print("   You can manually calculate ROI using:")
    print(f"   fight_model.calculate_roi(odds_table_path='{output_path}', vegas_data_path='final_with_odds_clamped.csv')")

print()
print("=" * 80)
print("✅ XGBoost ROI Calculator Complete!")
print("=" * 80)
print()
print("💡 Next steps:")
print("   1. Use xgboost_odds_table.csv for your predictions")
print("   2. Compare ROI vs LogReg ROI")
print("   3. Run backward rolling backtest:")
print()
print("   ```python")
print("   fight_model.backward_rolling_backtest_roi(")
print("       vegas_data_path='final_with_odds_clamped.csv',")
print("       stake=100,")
print("       training_years=15,")
print("       test_period=0.5,")
print("       num_periods=12")
print("   )")
print("   ```")
print()

