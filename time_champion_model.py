"""
Time the champion XGBoost model training and evaluation.
"""
import time
import sys
sys.path.insert(0, '/Users/evankellener/Desktop/UFC_fight_predictor/src')

from ensemble_model_best import FightOutcomeModel

print("="*80)
print("TIMING CHAMPION XGBOOST MODEL")
print("="*80)

# Initialize model
print("\n[1/5] Initializing FightOutcomeModel...")
start_init = time.time()
fight_model = FightOutcomeModel('/Users/evankellener/Desktop/UFC_fight_predictor/data/tmp/final.csv', random_seed=42)
init_time = time.time() - start_init
print(f"✓ Initialization complete: {init_time:.2f} seconds")

# Train champion model
print("\n[2/5] Training XGBoost with champion config...")
start_train = time.time()
model, acc = fight_model.tune_xgboost_full(use_champion_config=True)
train_time = time.time() - start_train
print(f"✓ Training complete: {train_time:.2f} seconds")
print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# Generate odds table
print("\n[3/5] Generating odds table...")
start_odds = time.time()
odd_df = fight_model.generate_odds_table()
odds_time = time.time() - start_odds
print(f"✓ Odds table generated: {odds_time:.2f} seconds")

# Save CSV
print("\n[4/5] Saving odds table to CSV...")
start_save = time.time()
output_path = '/Users/evankellener/Desktop/UFC_fight_predictor/data/tmp/odds_table_xgb_champion.csv'
odd_df.to_csv(output_path, index=False)
save_time = time.time() - start_save
print(f"✓ CSV saved: {save_time:.2f} seconds")

# Calculate ROI
print("\n[5/5] Calculating ROI...")
start_roi = time.time()
roi_df = fight_model.calculate_roi(
    odds_table_path='/Users/evankellener/Desktop/UFC_fight_predictor/data/tmp/odds_table_xgb_champion.csv',
    vegas_data_path='final_with_odds_clamped.csv'
)
roi_time = time.time() - start_roi
print(f"✓ ROI calculated: {roi_time:.2f} seconds")

# Total time
total_time = init_time + train_time + odds_time + save_time + roi_time

print("\n" + "="*80)
print("TIMING SUMMARY")
print("="*80)
print(f"[1] Model Initialization:     {init_time:6.2f} seconds")
print(f"[2] XGBoost Training:         {train_time:6.2f} seconds  ⭐")
print(f"[3] Odds Table Generation:    {odds_time:6.2f} seconds")
print(f"[4] CSV Save:                 {save_time:6.2f} seconds")
print(f"[5] ROI Calculation:          {roi_time:6.2f} seconds")
print("─"*80)
print(f"TOTAL TIME:                   {total_time:6.2f} seconds")
print("="*80)

print(f"\n✓ Complete! Model trained with {acc*100:.2f}% accuracy in {train_time:.1f}s")

