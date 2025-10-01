#!/usr/bin/env python3
"""
Test script to demonstrate the fixed find_best_feature_to_add method
and validate that feature addition produces consistent results.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_model_best import FightOutcomeModel

def test_feature_addition():
    """
    Test the find_best_feature_to_add method and validate results.
    """
    print("🚀 Testing Feature Addition Consistency")
    print("="*60)
    
    # Initialize the model
    print("📊 Initializing FightOutcomeModel...")
    model = FightOutcomeModel('../data/final.csv')
    
    # Show current importance columns
    print(f"\n📋 Current importance_columns count: {len(model.importance_columns)}")
    print("Current importance columns:", model.importance_columns)
    
    # Find the best feature to add (shows both log loss and accuracy rankings)
    print(f"\n🔍 Finding best feature to add...")
    best_feature, results_df = model.find_best_feature_to_add()
    
    if best_feature is not None:
        print(f"\n✅ Best feature found: {best_feature}")
        
        # Test the new comparison method
        print(f"\n📊 Running detailed feature comparison...")
        comparison_results = model.compare_top_features()
        
        # Validate the feature addition
        print(f"\n🧪 Validating feature addition...")
        validation_results = model.validate_feature_addition(best_feature)
        
        # Show the validation results
        print(f"\n📈 Validation Summary:")
        print(f"Base Accuracy: {validation_results['base_accuracy']:.4f}")
        print(f"Extended Accuracy: {validation_results['extended_accuracy']:.4f}")
        print(f"Base Log Loss: {validation_results['base_log_loss']:.4f}")
        print(f"Extended Log Loss: {validation_results['extended_log_loss']:.4f}")
        print(f"Accuracy Improvement: {validation_results['accuracy_improvement']:.4f}")
        print(f"Log Loss Improvement: {validation_results['log_loss_improvement']:.4f}")
        
        return best_feature, validation_results, comparison_results
    else:
        print("❌ No valid features found to add")
        return None, None, None

if __name__ == "__main__":
    try:
        best_feature, validation_results, comparison_results = test_feature_addition()
        
        if best_feature:
            print(f"\n🎉 Test completed successfully!")
            print(f"Best feature to add: {best_feature}")
            
            if validation_results:
                if validation_results['accuracy_improvement'] > 0 or validation_results['log_loss_improvement'] > 0:
                    print("✅ Feature addition shows improvement!")
                else:
                    print("⚠️  Feature addition shows minimal improvement")
            
            if comparison_results:
                print(f"\n💡 Additional insights:")
                print(f"Best by log loss: {comparison_results['best_log_loss_feature']}")
                print(f"Best by accuracy: {comparison_results['best_accuracy_feature']}")
                
                if comparison_results['best_log_loss_feature'] == comparison_results['best_accuracy_feature']:
                    print("🎉 Same feature is best for both metrics!")
                else:
                    print("🤔 Different features are best for different metrics")
        else:
            print("\n❌ Test failed - no features found")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
