#!/usr/bin/env python3
"""
Test script for the UFC Fight Predictor

This script tests the basic functionality of the prediction system.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_predictor_fixed import SimpleUFCPredictorFixed

def test_prediction():
    """Test the prediction functionality."""
    print("🧪 Testing UFC Fight Predictor")
    print("=" * 50)
    
    try:
        # Initialize predictor
        print("1. Initializing predictor...")
        predictor = SimpleUFCPredictorFixed()
        print("   ✅ Predictor initialized successfully")
        
        # Test data loading
        print(f"\n2. Data loaded:")
        print(f"   Total fights: {len(predictor.data)}")
        print(f"   Training fights: {len(predictor.train_data)}")
        print(f"   Test fights: {len(predictor.test_data)}")
        print(f"   Features used: {len(predictor.feature_columns)}")
        
        # Test model training
        print(f"\n3. Model performance:")
        if hasattr(predictor, 'model') and predictor.model is not None:
            X_test = predictor.test_data[predictor.feature_columns].copy()
            y_test = predictor.test_data['win'].copy()
            
            X_test_imputed = predictor.imputer.transform(X_test)
            X_test_scaled = predictor.scaler.transform(X_test_imputed)
            
            test_acc = predictor.model.score(X_test_scaled, y_test)
            print(f"   Test accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        else:
            print("   ❌ Model not available")
            return False
        
        # Test fighter lookup
        print(f"\n4. Testing fighter lookup...")
        available_fighters = predictor.data['FIGHTER'].unique()
        print(f"   Available fighters: {len(available_fighters)}")
        
        # Find some test fighters
        test_fighters = []
        for fighter in available_fighters[:10]:  # Check first 10 fighters
            fighter_data = predictor.data[predictor.data['FIGHTER'] == fighter]
            if len(fighter_data) > 0:
                test_fighters.append(fighter)
                if len(test_fighters) >= 2:
                    break
        
        if len(test_fighters) < 2:
            print("   ❌ Not enough fighters found for testing")
            return False
        
        fighter1, fighter2 = test_fighters[0], test_fighters[1]
        print(f"   Test fighters: {fighter1} vs {fighter2}")
        
        # Test prediction
        print(f"\n5. Testing prediction...")
        result = predictor.predict_fight(fighter1, fighter2)
        
        print(f"   ✅ Prediction successful!")
        print(f"   Predicted winner: {result['predicted_winner']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Model odds: {result['american_odds']}")
        
        # Test fighter info
        print(f"\n6. Testing fighter info...")
        info1 = predictor.get_fighter_info(fighter1)
        info2 = predictor.get_fighter_info(fighter2)
        
        if info1 and info2:
            print(f"   ✅ Fighter info retrieved successfully")
            print(f"   {fighter1}: {info1['wins']}W-{info1['losses']}L, ELO: {info1['current_elo']}")
            print(f"   {fighter2}: {info2['wins']}W-{info2['losses']}L, ELO: {info2['current_elo']}")
        else:
            print("   ❌ Failed to get fighter info")
            return False
        
        print(f"\n🎉 All tests passed! The prediction system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_fighters():
    """Test with specific well-known fighters."""
    print(f"\n🔍 Testing with specific fighters...")
    
    try:
        predictor = SimpleUFCPredictorFixed()
        
        # Test with some well-known fighters (if they exist in the data)
        test_cases = [
            ("Israel Adesanya", "Sean Strickland"),
            ("Jon Jones", "Ciryl Gane"),
            ("Khabib Nurmagomedov", "Conor McGregor"),
            ("Anderson Silva", "Chris Weidman")
        ]
        
        for fighter1, fighter2 in test_cases:
            try:
                print(f"\nTesting: {fighter1} vs {fighter2}")
                result = predictor.predict_fight(fighter1, fighter2)
                
                print(f"   Winner: {result['predicted_winner']}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Odds: {result['american_odds']}")
                
            except ValueError as e:
                print(f"   ❌ {e}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
    except Exception as e:
        print(f"Error in specific fighter test: {e}")

if __name__ == "__main__":
    print("Starting UFC Fight Predictor tests...\n")
    
    # Run basic tests
    success = test_prediction()
    
    if success:
        # Run specific fighter tests
        test_specific_fighters()
        
        print(f"\n✅ Testing completed successfully!")
        print(f"🎯 Your UFC Fight Predictor is ready to use!")
        print(f"\nNext steps:")
        print(f"1. Run the web interface: python app.py")
        print(f"2. Use the CLI tool: python cli_predictor.py 'Fighter 1' 'Fighter 2'")
        print(f"3. Test with real upcoming UFC fights!")
    else:
        print(f"\n❌ Testing failed. Please check the error messages above.")
        sys.exit(1)
