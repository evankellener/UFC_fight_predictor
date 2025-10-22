#!/usr/bin/env python3
"""
Quick test of greedy forward search with a small feature subset.
Run this to test convergence and accuracy calculation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_model_best import FightOutcomeModel

# Quick test with small feature subset
model = FightOutcomeModel("data/final.csv")

# Get available features from the model
print("🔍 Checking available features...")
available_features = set(model.main_stats_cols)
print(f"Total available features: {len(available_features)}")

# Use a small subset of features that actually exist in the dataset
# Let's use some basic features that are likely to exist
small_feature_set = [
    'precomp_sigstr_pm', 'opp_precomp_sigstr_pm',
    'precomp_tdavg', 'opp_precomp_tdavg', 
    'precomp_sapm', 'opp_precomp_sapm',
    'precomp_subavg', 'opp_precomp_subavg',
    'precomp_tddef', 'opp_precomp_tddef',
    'precomp_sigstr_perc', 'opp_precomp_sigstr_perc',
    'precomp_strdef', 'opp_precomp_strdef',
    'precomp_tdacc_perc', 'opp_precomp_tdacc_perc',
    'precomp_totalacc_perc', 'opp_precomp_totalacc_perc',
    'precomp_headacc_perc', 'opp_precomp_headacc_perc',
    'precomp_bodyacc_perc', 'opp_precomp_bodyacc_perc',
    'precomp_legacc_perc', 'opp_precomp_legacc_perc',
    'precomp_distacc_perc', 'opp_precomp_distacc_perc',
    'precomp_clinchacc_perc', 'opp_precomp_clinchacc_perc',
    'precomp_groundacc_perc', 'opp_precomp_groundacc_perc',
    'precomp_totalstr_pm', 'opp_precomp_totalstr_pm',
    'precomp_grapple_strike_mix', 'opp_precomp_grapple_strike_mix',
    'precomp_finish_rate', 'opp_precomp_finish_rate',
    'precomp_ctrl_per_min', 'opp_precomp_ctrl_per_min',
    'precomp_winsum', 'opp_precomp_winsum',
    'precomp_losssum', 'opp_precomp_losssum'
]

# Filter to only include features that actually exist
small_feature_set = [f for f in small_feature_set if f in available_features]
print(f"Filtered to {len(small_feature_set)} features that exist in dataset")

# If we don't have enough features, add some more from the available set
if len(small_feature_set) < 20:
    print("Adding more features from available set...")
    additional_features = [f for f in available_features if f not in small_feature_set and f not in model.importance_columns]
    small_feature_set.extend(additional_features[:20])
    small_feature_set = small_feature_set[:40]  # Limit to 40 features
    print(f"Final feature set: {len(small_feature_set)} features")

print("🧪 Quick Greedy Forward Search Test")
print("=" * 50)
print(f"Using {len(small_feature_set)} features for faster testing")

# Run the search
results = model.greedy_forward_search(
    initial_features=None,
    convergence_threshold=0.001,
    max_iterations=5,  # Just 5 iterations for quick test
    metric='log_loss',
    min_improvement=0.0001,
    test_feature_subset=small_feature_set
)

print(f"\n✅ Test complete!")
print(f"Selected {len(results['best_features'])} features")
print(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
print(f"Final log loss: {results['final_metrics']['log_loss']:.4f}")
