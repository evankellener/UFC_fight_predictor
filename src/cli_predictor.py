#!/usr/bin/env python3
"""
UFC Fight Predictor CLI Tool

Usage:
    python cli_predictor.py "Fighter 1" "Fighter 2"
    
Example:
    python cli_predictor.py "Israel Adesanya" "Sean Strickland"
"""

import sys
import argparse
from simple_predictor_fixed import SimpleUFCPredictorFixed

def main():
    parser = argparse.ArgumentParser(
        description='UFC Fight Predictor - AI-powered fight outcome predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Israel Adesanya" "Sean Strickland"
  %(prog)s "Jon Jones" "Ciryl Gane"
  %(prog)s --info "Khabib Nurmagomedov"
        """
    )
    
    parser.add_argument(
        'fighter1',
        help='Name of the first fighter'
    )
    
    parser.add_argument(
        'fighter2',
        help='Name of the second fighter'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show detailed fighter information instead of prediction'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output including feature values'
    )
    
    args = parser.parse_args()
    
    try:
        print("🥊 UFC Fight Predictor CLI")
        print("=" * 50)
        
        # Initialize predictor
        print("Loading prediction model...")
        predictor = SimpleUFCPredictorFixed()
        print("✅ Model loaded successfully!\n")
        
        if args.info:
            # Show fighter information
            show_fighter_info(predictor, args.fighter1)
            print()
            show_fighter_info(predictor, args.fighter2)
        else:
            # Make prediction
            result = predictor.predict_fight(args.fighter1, args.fighter2)
            display_prediction(result, args.verbose)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def show_fighter_info(predictor, fighter_name):
    """Display detailed information about a fighter."""
    info = predictor.get_fighter_info(fighter_name)
    
    if info is None:
        print(f"❌ Fighter '{fighter_name}' not found in database")
        return
    
    print(f"👤 {fighter_name}")
    print(f"   📊 Career: {info['wins']}W - {info['losses']}L ({info['win_rate']:.1%})")
    print(f"   🏆 Current ELO: {info['current_elo']}")
    print(f"   👊 Striking ELO: {info['current_striking_elo']}")
    print(f"   📏 Physical: {info['age']} years, {info['height']}\", {info['weight']} lbs, {info['reach']}\" reach")
    print(f"   📅 Last Fight: {info['last_fight_date']} - {info['last_fight_result']}")

def display_prediction(result, verbose=False):
    """Display the prediction result."""
    print("🎯 PREDICTION RESULT")
    print("=" * 50)
    
    # Winner
    print(f"🏆 Predicted Winner: {result['predicted_winner']}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    print(f"💰 Model Odds: {result['american_odds']}")
    
    # Probabilities
    print(f"\n📈 Win Probabilities:")
    print(f"   {result['fighter1']}: {result['fighter1_win_probability']:.1%}")
    print(f"   {result['fighter2']}: {result['fighter2_win_probability']:.1%}")
    
    # Fighter comparison
    print(f"\n📊 FIGHTER COMPARISON")
    print("-" * 30)
    
    print(f"{result['fighter1']}:")
    stats1 = result['fighter1_stats']
    print(f"   ELO: {stats1['elo']} | Striking ELO: {stats1['striking_elo']}")
    print(f"   Age: {stats1['age']} | Height: {stats1['height']}\" | Weight: {stats1['weight']} lbs | Reach: {stats1['reach']}\"")
    
    print(f"\n{result['fighter2']}:")
    stats2 = result['fighter2_stats']
    print(f"   ELO: {stats2['elo']} | Striking ELO: {stats2['striking_elo']}")
    print(f"   Age: {stats2['age']} | Height: {stats2['height']}\" | Weight: {stats2['weight']} lbs | Reach: {stats2['reach']}\"")
    
    # Key factors
    print(f"\n🔑 KEY FACTORS INFLUENCING PREDICTION:")
    print("-" * 40)
    for i, factor in enumerate(result['key_factors'], 1):
        print(f"   {i}. {factor}")
    
    if verbose:
        print(f"\n🔍 VERBOSE FEATURE ANALYSIS:")
        print("-" * 40)
        # You could add more detailed feature analysis here
        print("   (Use the web interface for detailed SHAP analysis)")
    
    # Confidence interpretation
    confidence = result['confidence']
    if confidence >= 0.8:
        confidence_level = "Very High"
    elif confidence >= 0.6:
        confidence_level = "High"
    elif confidence >= 0.5:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"
    
    print(f"\n💡 CONFIDENCE INTERPRETATION:")
    print(f"   Level: {confidence_level}")
    if confidence >= 0.7:
        print("   💪 This prediction has strong statistical support")
    elif confidence >= 0.55:
        print("   ⚖️  This is a close fight with moderate confidence")
    else:
        print("   ⚠️  This prediction has low confidence - consider other factors")

if __name__ == "__main__":
    main()
