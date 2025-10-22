#!/usr/bin/env python3
"""
XGBoost Greedy Forward Feature Selection
Fast greedy search to establish baseline before GA
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from ensemble_model_best import FightOutcomeModel
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import time

class XGBoostModel(FightOutcomeModel):
    """Model with XGBoost evaluation."""
    
    def _calculate_xgboost(self, features, params=None):
        """
        Evaluate features with XGBoost.
        Train/test split is already handled by FightOutcomeModel initialization.
        """
        # Default params optimized for speed and performance
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.random_seed,
                'n_jobs': -1
            }
        
        # Prepare data
        sub_train = self.train_df.copy()
        sub_test = self.test_df.copy()
        
        imp = SimpleImputer(strategy='median')
        sub_train[features] = imp.fit_transform(sub_train[features])
        sub_test[features] = imp.transform(sub_test[features])
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(sub_train[features])
        X_test_scaled = scaler.transform(sub_test[features])
        
        y_train = sub_train['win']
        y_test = sub_test['win']
        
        # Build XGBoost
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=20,
            **params
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)
        
        return {'accuracy': accuracy, 'log_loss': ll, 'combined': accuracy - ll}


# Main Greedy Search
print("🌲 XGBoost Greedy Forward Feature Selection")
print("=" * 80)
print("Strategy: Add one best feature at a time")
print("Train/Test: Training on 2009-2024, Testing on last year (2024-2025)")
print("=" * 80)
print()

# Initialize
model = XGBoostModel("data/tmp/final.csv", random_seed=42)

# Setup
base_features = ['precomp_elo_diff', 'precomp_strike_elo_diff', 'precomp_grapple_elo_diff']
available_cols = set(model.train_df.columns)
all_features = set(model.main_stats_cols)
valid_features = [f for f in all_features if f in available_cols]
candidate_features = [f for f in valid_features if f not in base_features]

print(f"📊 Configuration:")
print(f"   Base features: {len(base_features)}")
print(f"   Candidate features: {len(candidate_features)}")
print(f"   Max features to add: 25")
print()

# Evaluate baseline
print("📍 Baseline (3 base features):")
baseline_metrics = model._calculate_xgboost(base_features)
print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")
print(f"   Log Loss: {baseline_metrics['log_loss']:.4f}")
print(f"   Combined: {baseline_metrics['combined']:.6f}")
print()

# Greedy search
selected_features = base_features.copy()
remaining_features = candidate_features.copy()
max_additions = 25
best_combined = baseline_metrics['combined']

start_time = time.time()

print("🔍 Starting Greedy Search...")
print("=" * 80)

for iteration in range(1, max_additions + 1):
    iter_start = time.time()
    
    print(f"\n🔄 Iteration {iteration}/{max_additions}")
    
    if not remaining_features:
        print("   ⚠️  No more features to add!")
        break
    
    best_feature = None
    best_score = best_combined
    
    # Try adding each remaining feature
    tested = 0
    for feature in remaining_features:
        test_features = selected_features + [feature]
        
        try:
            metrics = model._calculate_xgboost(test_features)
            combined = metrics['combined']
            
            if combined > best_score:
                best_score = combined
                best_feature = feature
                best_metrics = metrics
            
            tested += 1
            
        except Exception as e:
            continue
    
    iter_time = time.time() - iter_start
    
    # If we found improvement, add the feature
    if best_feature:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_combined = best_score
        
        improvement = (best_score - baseline_metrics['combined']) / baseline_metrics['combined'] * 100
        
        print(f"   ✅ Added: {best_feature}")
        print(f"   📈 Combined: {best_score:.6f} (+{improvement:.2f}% vs baseline)")
        print(f"   📊 Accuracy: {best_metrics['accuracy']:.4f} | Log Loss: {best_metrics['log_loss']:.4f}")
        print(f"   ⏱️  Time: {iter_time:.1f}s | Tested: {tested} features")
        print(f"   📋 Total features: {len(selected_features)}")
    else:
        print(f"   ⛔ No improvement found (tested {tested} features)")
        print(f"   ⏱️  Time: {iter_time:.1f}s")
        break

total_time = time.time() - start_time

# Final results
print(f"\n{'=' * 80}")
print("🏆 Greedy Search Complete!")
print(f"{'=' * 80}")

final_metrics = model._calculate_xgboost(selected_features)

print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
print(f"\n🎯 FINAL RESULTS:")
print(f"   Combined: {final_metrics['combined']:.6f}")
print(f"   Accuracy: {final_metrics['accuracy']:.4f}")
print(f"   Log Loss: {final_metrics['log_loss']:.4f}")
print(f"   Total features: {len(selected_features)}")
print(f"   Improvement: +{((final_metrics['combined'] - baseline_metrics['combined'])/baseline_metrics['combined']*100):.2f}% vs baseline")

print(f"\n📋 SELECTED FEATURES ({len(selected_features)}):")
print("\n   Base Features:")
for feat in base_features:
    print(f"      📌 {feat}")

added_features = [f for f in selected_features if f not in base_features]
if added_features:
    print(f"\n   Added Features ({len(added_features)}):")
    for feat in added_features:
        print(f"      🆕 {feat}")

# Save results
results = {
    'method': 'XGBoost Greedy Forward',
    'features': selected_features,
    'metrics': final_metrics,
    'baseline_metrics': baseline_metrics,
    'time_minutes': total_time / 60
}

import json
timestamp = int(time.time())
output_file = f"xgboost_greedy_results_{timestamp}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")
print()

