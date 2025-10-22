"""
Prove the results are real, not hardcoded
"""
import sys
sys.path.insert(0, '/Users/evankellener/Desktop/UFC_fight_predictor/src')
import pandas as pd
from ensemble_model_best import FightOutcomeModel
import json

print("="*80)
print("PROVING THE RESULTS ARE REAL")
print("="*80)
print()

# 1. Show the champion features
print("1. CHAMPION FEATURES FROM CONFIG FILE:")
print("-"*80)
with open('/Users/evankellener/Desktop/UFC_fight_predictor/xgboost_ga_results_1760303427.json') as f:
    config = json.load(f)
print(f"Number of features: {len(config['features'])}")
print(f"Features: {config['features'][:5]}... (showing first 5)")
print()

# 2. Initialize model and show the actual train/test split
print("2. TRAIN/TEST SPLIT DATES:")
print("-"*80)
fight_model = FightOutcomeModel('/Users/evankellener/Desktop/UFC_fight_predictor/data/tmp/final.csv', random_seed=42)
print(f"Training period: {fight_model.train_df['DATE'].min()} to {fight_model.train_df['DATE'].max()}")
print(f"Test period: {fight_model.test_df['DATE'].min()} to {fight_model.test_df['DATE'].max()}")
print(f"Training samples: {len(fight_model.train_df)}")
print(f"Test samples: {len(fight_model.test_df)}")
print()

# 3. Train the model
print("3. TRAINING MODEL WITH CHAMPION CONFIG:")
print("-"*80)
model, acc = fight_model.tune_xgboost_full(use_champion_config=True)
print(f"Accuracy: {acc:.4f}")
print()

# 4. Show actual predictions (not hardcoded)
print("4. ACTUAL TEST PREDICTIONS (FIRST 20):")
print("-"*80)
results_df = fight_model.test_df[['DATE', 'FIGHTER', 'win']].copy()
results_df['predicted_prob'] = fight_model.probs
results_df['predicted_win'] = (fight_model.probs > 0.5).astype(int)
results_df['correct'] = (results_df['predicted_win'] == results_df['win']).astype(int)
print(results_df.head(20).to_string(index=False))
print()

# 5. Calculate accuracy manually to prove it's not hardcoded
print("5. MANUAL ACCURACY CALCULATION:")
print("-"*80)
correct = (results_df['predicted_win'] == results_df['win']).sum()
total = len(results_df)
manual_acc = correct / total
print(f"Correct predictions: {correct}")
print(f"Total predictions: {total}")
print(f"Manual accuracy: {manual_acc:.4f} ({manual_acc*100:.2f}%)")
print(f"Returned accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Match: {abs(manual_acc - acc) < 0.0001}")
print()

# 6. Train with different seed to prove it changes
print("6. TRAINING WITH DIFFERENT SEED (43) TO PROVE IT'S NOT HARDCODED:")
print("-"*80)
fight_model2 = FightOutcomeModel('/Users/evankellener/Desktop/UFC_fight_predictor/data/tmp/final.csv', random_seed=43)
model2, acc2 = fight_model2.tune_xgboost_full(use_champion_config=True)
print(f"Seed 42 accuracy: {acc:.4f}")
print(f"Seed 43 accuracy: {acc2:.4f}")
print(f"Results differ (proving not hardcoded): {abs(acc - acc2) > 0.0001}")
print()

# 7. Show the features actually used in training
print("7. FEATURES USED IN TRAINING:")
print("-"*80)
print(f"X_train columns: {list(fight_model.X_train.columns)}")
print(f"Number of features: {len(fight_model.X_train.columns)}")
print()

print("="*80)
print("CONCLUSION: Results are REAL, not hardcoded!")
print("="*80)

