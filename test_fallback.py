#!/usr/bin/env python3
"""
Test script to verify the fallback model works
"""

import sys
import os

# Add the app directory to the path
sys.path.append('app')

def test_fallback_model():
    """Test the fallback model initialization"""
    try:
        # Import the model module directly
        from model import UFCFightPredictor
        import model as model_module
        
        # Mock the FightOutcomeModel to be None to force fallback
        original_import = model_module.FightOutcomeModel
        model_module.FightOutcomeModel = None
        
        print("Testing fallback model...")
        predictor = UFCFightPredictor()
        
        print(f"✅ Fallback model initialized successfully!")
        print(f"✅ Model accuracy: {predictor.accuracy:.3f}")
        print(f"✅ Features: {len(predictor.full_features)}")
        
        # Test a simple prediction
        fighters = predictor.get_available_fighters()
        if len(fighters) >= 2:
            fighter1, fighter2 = fighters[0], fighters[1]
            print(f"✅ Testing prediction: {fighter1} vs {fighter2}")
            result = predictor.predict_fight(fighter1, fighter2)
            print(f"✅ Prediction successful: {result['predicted_winner']} wins ({result['confidence']:.1%})")
        
        # Restore original import
        model_module.FightOutcomeModel = original_import
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fallback_model()
    if success:
        print("\n🎉 All tests passed! The fallback model is ready for deployment.")
    else:
        print("\n💥 Tests failed. Check the errors above.")
