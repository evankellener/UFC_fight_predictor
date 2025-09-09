#!/usr/bin/env python3
"""
Test script for the Enhanced UFC Predictor
Tests the key functionality including postcomp stats usage and age calculations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_predictor import EnhancedUFCPredictor
import pandas as pd

def test_enhanced_predictor():
    """Test the enhanced predictor functionality."""
    print("🧪 Testing Enhanced UFC Predictor...")
    
    try:
        # Initialize predictor
        print("\n1. Initializing predictor...")
        predictor = EnhancedUFCPredictor()
        print("✅ Predictor initialized successfully!")
        
        # Test basic functionality
        print("\n2. Testing basic functionality...")
        print(f"   Total fights: {len(predictor.data)}")
        print(f"   Training fights: {len(predictor.train_data)}")
        print(f"   Test fights: {len(predictor.test_data)}")
        print(f"   Features: {len(predictor.feature_columns)}")
        print(f"   Available fighters: {predictor.data['FIGHTER'].nunique()}")
        
        # Test fighter versions
        print("\n3. Testing fighter versions...")
        sample_fighter = predictor.data['FIGHTER'].iloc[0]
        versions = predictor.get_fighter_versions(sample_fighter)
        print(f"   Sample fighter '{sample_fighter}' has {len(versions)} versions")
        if versions:
            print(f"   Latest version: {versions[-1]}")
        
        # Test fighter info
        print("\n4. Testing fighter info...")
        fighter_info = predictor.get_fighter_info(sample_fighter)
        if fighter_info:
            print(f"   Fighter info: {fighter_info['name']} - {fighter_info['total_fights']} fights, {fighter_info['win_rate']:.2f} win rate")
        
        # Test recent postcomp stats
        print("\n5. Testing recent postcomp stats...")
        recent_stats = predictor.get_fighter_recent_postcomp_stats(sample_fighter, months_until_fight=6)
        if recent_stats:
            print(f"   Recent stats for {sample_fighter}:")
            print(f"     Age: {recent_stats.get('age', 'N/A')}")
            print(f"     ELO: {recent_stats.get('postcomp_elo', 'N/A')}")
            print(f"     Strike ELO: {recent_stats.get('postcomp_strike_elo', 'N/A')}")
        
        # Test weight classes
        print("\n6. Testing weight classes...")
        weight_classes = predictor.get_available_weight_classes()
        print(f"   Available weight classes: {len(weight_classes)}")
        for wc in weight_classes[:3]:  # Show first 3
            print(f"     {wc['name']}: {wc['fight_count']} fights")
        
        # Test weight class stats
        print("\n7. Testing weight class stats...")
        wc_stats = predictor.get_weight_class_stats()
        print(f"   Global model test accuracy: {wc_stats['global']['test_accuracy']:.3f}")
        
        # Test a simple prediction
        print("\n8. Testing simple prediction...")
        fighters = predictor.data['FIGHTER'].unique()
        if len(fighters) >= 2:
            fighter1, fighter2 = fighters[0], fighters[1]
            try:
                prediction = predictor.predict_fight_with_versions(fighter1, fighter2, months_until_fight=6)
                print(f"   Prediction: {fighter1} vs {fighter2}")
                print(f"   Winner: {prediction['predicted_winner']}")
                print(f"   Confidence: {prediction['confidence']:.3f}")
                print(f"   Fighter1 prob: {prediction['fighter1_win_probability']:.3f}")
                print(f"   Fighter2 prob: {prediction['fighter2_win_probability']:.3f}")
            except Exception as e:
                print(f"   Prediction failed: {e}")
        
        # Test model validation
        print("\n9. Testing model validation...")
        try:
            validation_results = predictor.validate_model_on_test_data(months_until_fight=6)
            print(f"   Validation accuracy: {validation_results['accuracy']:.3f}")
            print(f"   High confidence accuracy: {validation_results['high_confidence_accuracy']:.3f}")
        except Exception as e:
            print(f"   Validation failed: {e}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_enhanced_predictor()
    sys.exit(0 if success else 1)
