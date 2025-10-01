#!/usr/bin/env python3
"""
Advanced Feature Selection Example for UFC Fight Predictor

This script demonstrates how to use the new advanced feature selection methods
added to the ensemble_model_best.py file.

Usage:
    python advanced_feature_selection_example.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_model_best import FightOutcomeModel

def main():
    """
    Demonstrate advanced feature selection methods for UFC fight prediction.
    """
    print("🥊 UFC Fight Predictor - Advanced Feature Selection Demo")
    print("="*60)
    
    # Initialize the model with your data
    # Make sure to update this path to point to your actual data file
    data_path = "data/final_with_odds.csv"  # Update this path as needed
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path variable to point to your data file.")
        return
    
    try:
        # Initialize the model
        print("📊 Loading data and initializing model...")
        model = FightOutcomeModel(data_path)
        print(f"✅ Model initialized successfully!")
        print(f"   Training samples: {len(model.X_train)}")
        print(f"   Test samples: {len(model.X_test)}")
        print(f"   Features: {len(model.X_train.columns)}")
        
        # Run advanced feature selection methods
        print("\n🔬 Running Advanced Feature Selection Methods...")
        print("-" * 50)
        
        # Method 1: Individual advanced methods
        print("\n1️⃣ Individual Advanced Methods")
        results = model.advanced_feature_selection_methods(
            n_features_to_select=20,  # Select top 20 features
            cv_folds=5               # Use 5-fold cross-validation
        )
        
        # Method 2: Ensemble feature selection
        print("\n2️⃣ Ensemble Feature Selection")
        ensemble_results = model.advanced_ensemble_feature_selection(
            n_features_to_select=20,  # Select top 20 features
            cv_folds=5               # Use 5-fold cross-validation
        )
        
        # Display summary
        print("\n📈 SUMMARY")
        print("="*50)
        print("Advanced feature selection completed successfully!")
        print(f"Best individual method: {min(results.keys(), key=lambda k: results[k]['mean_score'])}")
        print(f"Ensemble selection improvement: {ensemble_results['improvement']:.4f}")
        
        # Save results
        print("\n💾 Saving results...")
        
        # Save individual method results
        results_df = pd.DataFrame([
            {
                'Method': method,
                'Log_Loss': result['mean_score'],
                'Std_Log_Loss': result['std_score'],
                'Time_Seconds': result['time'],
                'N_Features': result.get('n_features', 'N/A')
            }
            for method, result in results.items()
            if method != 'PCA'  # PCA doesn't have traditional features
        ])
        
        results_df.to_csv('advanced_feature_selection_results.csv', index=False)
        print("✅ Results saved to: advanced_feature_selection_results.csv")
        
        # Save ensemble features
        ensemble_features_df = pd.DataFrame({
            'Feature': ensemble_results['ensemble_features'],
            'Votes': [ensemble_results['feature_votes'].get(f, 0) for f in ensemble_results['ensemble_features']]
        })
        ensemble_features_df.to_csv('ensemble_selected_features.csv', index=False)
        print("✅ Ensemble features saved to: ensemble_selected_features.csv")
        
        print("\n🎉 Advanced feature selection demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during feature selection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
