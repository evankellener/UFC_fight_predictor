#!/usr/bin/env python3
"""
UFC Fight Predictor Command Line Interface

A simple command-line interface for predicting UFC fight outcomes.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to the path to import our predictor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ufc_fight_predictor_app import UFCFightPredictor

def main():
    parser = argparse.ArgumentParser(description='UFC Fight Predictor CLI')
    parser.add_argument('--fighter1', '-f1', help='Name of the first fighter')
    parser.add_argument('--fighter2', '-f2', help='Name of the second fighter')
    parser.add_argument('--weight-class', '-w', type=int, help='Weight class (1-12)')
    parser.add_argument('--fight-date', '-d', help='Fight date (YYYY-MM-DD). Defaults to 1 month from now.')
    parser.add_argument('--validate', action='store_true', help='Run model validation on test data')
    parser.add_argument('--list-fighters', action='store_true', help='List available fighters')
    parser.add_argument('--list-weight-classes', action='store_true', help='List available weight classes')
    parser.add_argument('--fighter-info', help='Get detailed info about a specific fighter')
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("🥊 Initializing UFC Fight Predictor...")
    try:
        predictor = UFCFightPredictor()
        print("✅ Predictor initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return 1
    
    # Handle list commands
    if args.list_fighters:
        print("\n👥 Available Fighters:")
        fighters = predictor.get_available_fighters()
        for i, fighter in enumerate(fighters, 1):
            print(f"{i:3d}. {fighter}")
        return 0
    
    if args.list_weight_classes:
        print("\n⚖️  Available Weight Classes:")
        weight_classes = predictor.get_available_weight_classes()
        for wc in weight_classes:
            print(f"{wc['value']:2d}. {wc['display_name']} ({wc['fight_count']} fights)")
        return 0
    
    if args.fighter_info:
        print(f"\n👤 Fighter Info: {args.fighter_info}")
        versions = predictor.get_fighter_versions(args.fighter_info)
        if not versions:
            print("❌ Fighter not found or no data available.")
            return 1
        
        print(f"Total fights: {len(versions)}")
        print("\nRecent fights:")
        for i, version in enumerate(versions[-5:], 1):  # Show last 5 fights
            print(f"{i}. {version['fight_date']} vs {version['opponent']} - {version['result']}")
            print(f"   ELO: {version['postcomp_elo']:.0f}, Age: {version['age']:.1f}")
        return 0
    
    # Handle validation
    if args.validate:
        print("\n🔍 Running model validation...")
        try:
            validation_results = predictor.validate_model_on_test_data()
            print(f"\n📈 Validation Results:")
            print(f"Total test fights: {validation_results['total_fights']}")
            print(f"Correct predictions: {validation_results['correct_predictions']}")
            print(f"Overall accuracy: {validation_results['accuracy']:.3f}")
            print(f"High confidence fights: {validation_results['high_confidence_fights']}")
            print(f"High confidence accuracy: {validation_results['high_confidence_accuracy']:.3f}")
            
            print(f"\n📊 Sample Results:")
            for result in validation_results['sample_results'][:5]:
                print(f"{result['fight_date']}: {result['fighter1']} vs {result['fighter2']}")
                print(f"  Actual: {result['actual_winner']}, Predicted: {result['predicted_winner']}")
                print(f"  Correct: {'✅' if result['correct'] else '❌'}, Confidence: {result['confidence']:.3f}")
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return 1
        return 0
    
    # Handle prediction
    if not all([args.fighter1, args.fighter2, args.weight_class]):
        print("❌ Missing required arguments for prediction.")
        print("Use --help for usage information.")
        return 1
    
    # Parse fight date
    if args.fight_date:
        try:
            fight_date = datetime.strptime(args.fight_date, '%Y-%m-%d')
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD.")
            return 1
    else:
        fight_date = datetime.now() + timedelta(days=30)  # Default to 1 month from now
    
    # Make prediction
    print(f"\n🥊 Making prediction...")
    print(f"Fight: {args.fighter1} vs {args.fighter2}")
    print(f"Weight Class: {predictor.weight_class_names.get(args.weight_class, f'Weight Class {args.weight_class}')}")
    print(f"Fight Date: {fight_date.strftime('%Y-%m-%d')}")
    
    try:
        result = predictor.predict_fight(args.fighter1, args.fighter2, args.weight_class, fight_date)
        
        print(f"\n🏆 PREDICTION RESULT:")
        print(f"Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"American Odds: {result['american_odds']}")
        
        print(f"\n📊 FIGHTER COMPARISON:")
        print(f"\n{result['fighter1']}:")
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
        
        # Create inference example CSV
        create_inference_example(result)
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return 1
    
    return 0

def create_inference_example(result):
    """Create an inference example CSV file."""
    try:
        import pandas as pd
        
        # Create a DataFrame with the prediction data
        data = {
            'fighter': [result['fighter1'], result['fighter2']],
            'age': [result['fighter1_stats']['age'], result['fighter2_stats']['age']],
            'height': [result['fighter1_stats']['height'], result['fighter2_stats']['height']],
            'weight': [result['fighter1_stats']['weight'], result['fighter2_stats']['weight']],
            'reach': [result['fighter1_stats']['reach'], result['fighter2_stats']['reach']],
            'elo': [result['fighter1_stats']['elo'], result['fighter2_stats']['elo']],
            'striking_elo': [result['fighter1_stats']['striking_elo'], result['fighter2_stats']['striking_elo']],
            'grapple_elo': [result['fighter1_stats']['grapple_elo'], result['fighter2_stats']['grapple_elo']],
            'last_fight_date': [result['fighter1_stats']['last_fight_date'], result['fighter2_stats']['last_fight_date']],
            'last_fight_result': [result['fighter1_stats']['last_fight_result'], result['fighter2_stats']['last_fight_result']],
            'last_fight_opponent': [result['fighter1_stats']['last_fight_opponent'], result['fighter2_stats']['last_fight_opponent']],
            'win_probability': [result['fighter1_win_probability'], result['fighter2_win_probability']]
        }
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        filename = f"inference_example_{result['fighter1'].replace(' ', '_')}_vs_{result['fighter2'].replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n💾 Inference example saved to: {filename}")
        
    except Exception as e:
        print(f"⚠️  Could not create inference example CSV: {e}")

if __name__ == '__main__':
    sys.exit(main())
