"""
Evaluate Clean Model Performance
Proper train/test split to measure expected performance on unseen data
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, log_loss
from datetime import timedelta

print("="*80)
print("EVALUATING CLEAN MODEL")
print("="*80)
print()

# Load data
df = pd.read_csv('data/tmp/final.csv', low_memory=False)
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

print(f"Original data: {len(df)} rows")

# Apply filters (same as production)
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

# Prepare data
df_clean = df.dropna(subset=['win_numeric'])
df_clean = df_clean.dropna(subset=features)

print(f"After cleaning: {len(df_clean)} rows")
print(f"Date range: {df_clean['DATE'].min().date()} to {df_clean['DATE'].max().date()}")
print()

# Train/test split (1 year holdout)
cutoff_date = df_clean['DATE'].max() - timedelta(days=365)
train_df = df_clean[df_clean['DATE'] < cutoff_date]
test_df = df_clean[df_clean['DATE'] >= cutoff_date]

print("="*80)
print("DATA SPLIT")
print("="*80)
print(f"Train: {len(train_df)} rows ({train_df['DATE'].min().date()} to {train_df['DATE'].max().date()})")
print(f"Test:  {len(test_df)} rows ({test_df['DATE'].min().date()} to {test_df['DATE'].max().date()})")
print()

# Check balance
train_wins = (train_df['win_numeric'] == 1).sum()
train_losses = (train_df['win_numeric'] == 0).sum()
test_wins = (test_df['win_numeric'] == 1).sum()
test_losses = (test_df['win_numeric'] == 0).sum()

print("Data Balance:")
print(f"  Train: {train_wins} wins, {train_losses} losses ({train_wins/(train_wins+train_losses)*100:.2f}% win rate)")
print(f"  Test:  {test_wins} wins, {test_losses} losses ({test_wins/(test_wins+test_losses)*100:.2f}% win rate)")
print()

# Prepare features
X_train = train_df[features].copy()
y_train = train_df['win_numeric'].copy()
X_test = test_df[features].copy()
y_test = test_df['win_numeric'].copy()

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
print("TRAINING")
print("="*80)
print("Training XGBoost with champion hyperparameters...")
print()

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, verbose=False)

# Make predictions
y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
train_acc = accuracy_score(y_train, y_train_pred)
train_ll = log_loss(y_train, y_train_proba)
test_acc = accuracy_score(y_test, y_test_pred)
test_ll = log_loss(y_test, y_test_proba)

print("="*80)
print("RESULTS - CLEAN MODEL (Version B, NO rolling_ema)")
print("="*80)
print()

print("Training Performance:")
print(f"  Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Log Loss:  {train_ll:.4f}")
print()

print("Test Performance (1-year holdout):")
print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Log Loss:  {test_ll:.4f}")
print()

# Calculate 95% confidence interval for test accuracy
from scipy import stats
n = len(y_test)
ci = 1.96 * np.sqrt((test_acc * (1 - test_acc)) / n)
print(f"  95% CI:    [{(test_acc - ci)*100:.2f}% - {(test_acc + ci)*100:.2f}%]")
print()

print("="*80)
print("COMPARISON TO PREVIOUS RESULTS")
print("="*80)
print()

print("Previous Model (Version A + rolling_ema):")
print("  Test Accuracy: 63.54%")
print("  Test Log Loss: 0.6258")
print()

print("Previous Model (Version B + rolling_ema):")
print("  Test Accuracy: 70.19%")
print("  Test Log Loss: 0.5641")
print()

print("Current Model (Version B, NO rolling_ema):")
print(f"  Test Accuracy: {test_acc*100:.2f}%")
print(f"  Test Log Loss: {test_ll:.4f}")
print()

acc_diff_vs_ema = (test_acc - 0.7019) * 100
ll_diff_vs_ema = test_ll - 0.5641

print("Difference vs Version B + rolling_ema:")
print(f"  Accuracy:  {acc_diff_vs_ema:+.2f}%")
print(f"  Log Loss:  {ll_diff_vs_ema:+.4f}")
print()

if abs(acc_diff_vs_ema) < 3.0:
    print("✅ Clean model performance is comparable (within 3%)")
    print("   The suspect rolling_ema feature was NOT providing real value")
elif acc_diff_vs_ema < -3.0:
    print("⚠️  Clean model is worse by >3%")
    print("   rolling_ema may have had some real signal (investigate further)")
else:
    print("🎉 Clean model is BETTER!")
    print("   Removing rolling_ema actually improved performance")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print(f"Expected Performance on Future Fights:")
print(f"  • Accuracy: {test_acc*100:.1f}% (±{ci*100:.1f}%)")
print(f"  • Log Loss: {test_ll:.4f}")
print()
print("This is your honest estimate for production betting.")
print("="*80)

