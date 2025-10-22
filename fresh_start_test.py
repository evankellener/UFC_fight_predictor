#!/usr/bin/env python3
"""
Fresh start test - build feature set from scratch starting with just 3 ELO features.
This lets the algorithm discover the optimal feature combination without bias.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_model_best import FightOutcomeModel

# Initialize model with random seed 42 (for reproducibility)
print("🚀 Fresh Start Greedy Search")
print("=" * 80)
print("Starting with ONLY 3 ELO features and building from scratch")
print("This may discover a completely different (and better) feature combination!")
print("=" * 80)
print()

model = FightOutcomeModel("data/tmp/final.csv", random_seed=42)

# Start with ONLY the 3 basic ELO features
initial_features = [
    'precomp_elo_diff',
    'precomp_strike_elo_diff', 
    'precomp_grapple_elo_diff'
]

print(f"🎯 Configuration:")
print(f"   - Starting features: {len(initial_features)} (just ELO diffs)")
print(f"   - Patience: 3 iterations")
print(f"   - Max iterations: 100 (allow deep exploration)")
print(f"   - Metric: Combined Score (accuracy - log_loss)")
print()

# Run greedy search starting from scratch
results = model.greedy_forward_search(
    initial_features=initial_features,  # Start with just 3 ELO features
    test_feature_subset=None,  # Use ALL available features
    max_iterations=100,  # Allow lots of exploration
    metric='combined'  # Use combined score
)

print()
print("=" * 80)
print("🎉 FRESH START SEARCH COMPLETE!")
print("=" * 80)
print(f"✅ Final feature count: {len(results['best_features'])} features")
print(f"📈 Final accuracy: {results['final_metrics']['accuracy']:.4f}")
print(f"📉 Final log loss: {results['final_metrics']['log_loss']:.4f}")
final_combined = results['final_metrics']['accuracy'] - results['final_metrics']['log_loss']
print(f"🎯 Final combined: {final_combined:.4f}")
print(f"🏆 Best combined ever: {results['best_metric_ever']:.4f}")
print(f"📊 Iterations completed: {results['total_iterations']}")
print(f"🛑 Convergence: {results['convergence_reason']}")
print()

print("📋 FINAL FEATURE SET:")
for idx, feat in enumerate(results['best_features'], 1):
    marker = "📌" if feat in initial_features else "🆕"
    print(f"   {marker} {idx:2d}. {feat}")

