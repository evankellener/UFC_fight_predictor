"""
Get a more honest estimate of future performance using nested time series validation
"""
import sys
sys.path.insert(0, '/Users/evankellener/Desktop/UFC_fight_predictor/src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Load data
df = pd.read_csv('/Users/evankellener/Desktop/UFC_fight_predictor/data/tmp/final.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

# Calculate rolling_ema
df['win_numeric'] = pd.to_numeric(df['win'], errors='coerce')
rolling_ema_full = df['win_numeric'].ewm(span=200, min_periods=20).mean()
df['precomp_rolling_ema'] = rolling_ema_full.shift(1)
df = df.dropna(subset=['precomp_rolling_ema'])

# Champion features + rolling_ema
features = [
    "precomp_elo_diff", "precomp_strike_elo_diff", "precomp_grapple_elo_diff",
    "precomp_legacc_perc5", "opp_precomp_sigstr_pm5", "opp_precomp_grapple_strike_mix",
    "opp_precomp_clinchacc_perc", "opp_age_ratio_difference", "opp_precomp_elo",
    "age_ratio_difference", "precomp_distacc_perc", "opp_precomp_winsum",
    "precomp_tdavg3", "opp_precomp_legacc_perc3", "opp_precomp_str_eff_diff3",
    "precomp_winsum", "opp_precomp_sapm3", "precomp_groundacc_perc",
    "opp_precomp_ctrl_per_min", "opp_REACH", "precomp_winsum5",
    "opp_precomp_strdef5", "precomp_ctrl_per_min", "opp_precomp_tdavg5",
    "opp_precomp_headacc_perc5", "precomp_elo_change_5", "opp_precomp_winsum3",
    "opp_precomp_groundacc_perc5", "precomp_rolling_ema"
]

# Filter for valid data
df_filtered = df.dropna(subset=['win'] + features)

# Walk-forward validation: simulate predicting each of the last 3 years
most_recent_date = df_filtered['DATE'].max()
results = []

for years_back in [3, 2, 1]:
    test_end = most_recent_date - timedelta(days=365 * (years_back - 1))
    test_start = test_end - timedelta(days=365)
    
    train_df = df_filtered[df_filtered['DATE'] < test_start]
    test_df = df_filtered[(df_filtered['DATE'] >= test_start) & (df_filtered['DATE'] < test_end)]
    
    if len(test_df) < 50:
        continue
    
    X_train = train_df[features]
    y_train = train_df['win']
    X_test = test_df[features]
    y_test = test_df['win']
    
    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()
    
    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Train XGBoost with champion config
    model = XGBClassifier(
        max_depth=4, learning_rate=0.15, n_estimators=250,
        min_child_weight=3, subsample=0.9, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.1, reg_lambda=0.5,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    results.append({
        'test_period': f'{test_start.strftime("%Y-%m")} to {test_end.strftime("%Y-%m")}',
        'train_size': len(train_df),
        'test_size': len(test_df),
        'accuracy': acc,
        'log_loss': ll
    })
    
    print(f"\nTest Period: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")
    print(f"  Train size: {len(train_df)} | Test size: {len(test_df)}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Log Loss: {ll:.4f}")

print("\n" + "="*80)
print("WALK-FORWARD VALIDATION SUMMARY")
print("="*80)

if results:
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_ll = np.mean([r['log_loss'] for r in results])
    std_acc = np.std([r['accuracy'] for r in results])
    
    print(f"Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%) ± {std_acc*100:.2f}%")
    print(f"Average Log Loss: {avg_ll:.4f}")
    print()
    print("📊 INTERPRETATION:")
    print(f"   Your model's true performance on future fights is likely around {avg_acc*100:.1f}%")
    print(f"   This is {'BETTER' if avg_acc > 0.682 else 'SIMILAR'} to the baseline of 68.2%")
    print()
    print("   The variation across test periods shows how stable the model is.")
    print(f"   Standard deviation of {std_acc*100:.2f}% means expect ±{std_acc*100*1.96:.1f}% variation.")
print("="*80)

