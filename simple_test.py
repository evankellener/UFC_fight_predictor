#!/usr/bin/env python3
"""
Super simple test - just use the first 5 available features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_model_best import FightOutcomeModel

# Initialize model with random seed 42 (for reproducibility)
# MUST use same file as notebook: data/tmp/final.csv
print("🚀 Simple Greedy Test")
model = FightOutcomeModel("data/tmp/final.csv", random_seed=42)

# Get all available test features
available_features = set(model.main_stats_cols)
importance_features = set(model.importance_columns)
test_features = [f for f in available_features if f not in importance_features]

print(f"🎯 Running FULL greedy forward search with COMBINED metric")
print(f"   - Available test features: {len(test_features)}")
print(f"   - Optimizing for: Combined Score (accuracy - log_loss)")
print(f"   - This will test ALL available features until convergence")
print()

# Run FULL greedy search with combined metric
# Model already initialized with random_seed=42, so no need to pass it again
results = model.greedy_forward_search(
    test_feature_subset=None,  # Use ALL available features
    max_iterations=50,  # Allow up to 50 iterations
    convergence_threshold=0.001,
    min_improvement=0.0001,
    metric='combined'  # Use combined score (accuracy - log_loss)
)

print(f"✅ Done! Selected {len(results['best_features'])} features")
print(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
print(f"Final log loss: {results['final_metrics']['log_loss']:.4f}")
