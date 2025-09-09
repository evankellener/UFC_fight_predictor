#!/usr/bin/env python3
"""
UFC Fight Predictor Demo

This script demonstrates how to use the UFC Fight Predictor to make predictions.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ufc_fight_predictor_app import UFCFightPredictor

def main():
    print("🥊 UFC Fight Predictor Demo")
    print("=" * 50)
    
    # Initialize predictor
    print("Initializing predictor...")
    try:
        predictor = UFCFightPredictor()
        print("✅ Predictor initialized successfully!")
        print(f"📊 Model accuracy: {predictor.test_accuracy:.3f}")
        print(f"👥 Available fighters: {len(predictor.get_available_fighters())}")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return 1
    
    # Example 1: Nassourdine Imavov vs Caio Borralho
    print("\n" + "=" * 50)
    print("🎯 EXAMPLE 1: Nassourdine Imavov vs Caio Borralho")
    print("=" * 50)
    
    fighter1 = "Nassourdine Imavov"
    fighter2 = "Caio Borralho"
    weight_class = 10  # Middleweight (185 lbs)
    fight_date = datetime.now() + timedelta(days=30)  # 1 month from now
    
    try:
        result = predictor.predict_fight(fighter1, fighter2, weight_class, fight_date)
        
        print(f"\n🥊 FIGHT PREDICTION:")
        print(f"Fight: {result['fighter1']} vs {result['fighter2']}")
        print(f"Weight Class: {result['weight_class_name']}")
        print(f"Fight Date: {result['fight_date']}")
        print(f"Predicted Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"American Odds: {result['american_odds']}")
        
        print(f"\n📊 FIGHTER COMPARISON:")
        print(f"{result['fighter1']}:")
        for stat, value in result['fighter1_stats'].items():
            if isinstance(value, float):
                print(f"  {stat}: {value:.2f}")
            else:
                print(f"  {stat}: {value}")
        
        print(f"\n{result['fighter2']}:")
        for stat, value in result['fighter2_stats'].items():
            if isinstance(value, float):
                print(f"  {stat}: {value:.2f}")
            else:
                print(f"  {stat}: {value}")
                
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
    
    # Example 2: Jon Jones vs Stipe Miocic (if available)
    print("\n" + "=" * 50)
    print("🎯 EXAMPLE 2: Jon Jones vs Stipe Miocic")
    print("=" * 50)
    
    fighter1 = "Jon Jones"
    fighter2 = "Stipe Miocic"
    weight_class = 12  # Heavyweight (265 lbs)
    fight_date = datetime.now() + timedelta(days=60)  # 2 months from now
    
    try:
        result = predictor.predict_fight(fighter1, fighter2, weight_class, fight_date)
        
        print(f"\n🥊 FIGHT PREDICTION:")
        print(f"Fight: {result['fighter1']} vs {result['fighter2']}")
        print(f"Weight Class: {result['weight_class_name']}")
        print(f"Fight Date: {result['fight_date']}")
        print(f"Predicted Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"American Odds: {result['american_odds']}")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        print("This might be because one or both fighters are not in the dataset.")
    
    # Show model validation
    print("\n" + "=" * 50)
    print("🔍 MODEL VALIDATION")
    print("=" * 50)
    
    try:
        validation_results = predictor.validate_model_on_test_data()
        
        print(f"\n📈 VALIDATION RESULTS:")
        print(f"Total test fights: {validation_results['total_fights']}")
        print(f"Correct predictions: {validation_results['correct_predictions']}")
        print(f"Overall accuracy: {validation_results['accuracy']:.3f}")
        print(f"High confidence fights: {validation_results['high_confidence_fights']}")
        print(f"High confidence accuracy: {validation_results['high_confidence_accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
