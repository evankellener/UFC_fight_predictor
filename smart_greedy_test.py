#!/usr/bin/env python3
"""
Smart test script that automatically discovers available features
and runs greedy forward search with a safe subset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_model_best import FightOutcomeModel

def main():
    print("🧠 Smart Greedy Forward Search Test")
    print("=" * 50)
    
    # Initialize model
    model = FightOutcomeModel("data/final.csv")
    
    print(f"✅ Model initialized!")
    print(f"   - Training samples: {len(model.train_df)}")
    print(f"   - Test samples: {len(model.test_df)}")
    print(f"   - Available features: {len(model.main_stats_cols)}")
    print(f"   - Current importance features: {len(model.importance_columns)}")
    print()
    
    # Get available features
    available_features = set(model.main_stats_cols)
    importance_features = set(model.importance_columns)
    
    # Find features that are available but not in importance_columns
    test_features = [f for f in available_features if f not in importance_features]
    
    print(f"🔍 Available test features: {len(test_features)}")
    print(f"📊 Current importance features: {len(importance_features)}")
    
    # Create a safe subset (first 20 features that exist)
    safe_subset = test_features[:20]
    print(f"🎯 Using safe subset of {len(safe_subset)} features")
    print(f"   Features: {safe_subset[:5]}... (showing first 5)")
    print()
    
    # Run greedy forward search with safe subset
    print("🚀 Running greedy forward search...")
    results = model.greedy_forward_search(
        initial_features=None,  # Start with importance_columns
        convergence_threshold=0.001,
        max_iterations=5,  # Limit iterations for testing
        metric='log_loss',
        min_improvement=0.0001,
        test_feature_subset=safe_subset
    )
    
    # Display results
    print("\n📊 FINAL RESULTS:")
    print("=" * 50)
    print(f"✅ Selected features: {len(results['best_features'])}")
    print(f"📉 Remaining features: {len(results['test_features'])}")
    print(f"🔄 Iterations completed: {results['total_iterations']}")
    print(f"🎯 Convergence reason: {results['convergence_reason']}")
    print(f"📈 Final accuracy: {results['final_metrics']['accuracy']:.4f}")
    print(f"📉 Final log loss: {results['final_metrics']['log_loss']:.4f}")
    
    print(f"\n🎯 Selected features:")
    for i, feature in enumerate(results['best_features'], 1):
        print(f"   {i:2d}. {feature}")
    
    if results['iteration_history']:
        print(f"\n📈 Iteration History:")
        print("Iter | Feature Added           | Accuracy | Log Loss | Improvement")
        print("-" * 70)
        for data in results['iteration_history']:
            print(f"{data['iteration']:4d} | {data['feature_added']:22s} | {data['accuracy']:8.4f} | {data['log_loss']:8.4f} | {data['improvement']:11.6f}")
    
    print("\n✅ Smart test complete!")
    print("💡 This shows the algorithm working with features that actually exist in your dataset.")

if __name__ == "__main__":
    main()
