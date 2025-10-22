"""
PRODUCTION EVENT PREDICTOR
Bet on every fight in an event, track ROI over time

Usage:
    python predict_event_production.py <event_csv>
    
Example:
    python predict_event_production.py upcoming_events/ufc_308.csv
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, 'src')
from ensemble_model_best import FightOutcomeModel

class EventPredictor:
    def __init__(self):
        print("="*80)
        print("UFC FIGHT PREDICTOR - PRODUCTION")
        print("="*80)
        print()
        
        # Load model
        print("Loading model...")
        self.model_obj = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
        
        # Load champion features
        with open('xgboost_ga_results_1760303427.json', 'r') as f:
            champion = json.load(f)
            self.model_obj.importance_columns = champion['features']
        
        # Train on ALL data (production mode)
        print("Training on all available data...")
        self.model, self.acc = self.model_obj.tune_xgboost_full(
            use_champion_config=True, 
            use_rolling_ema=False
        )
        print(f"✓ Model trained - Accuracy: {self.acc*100:.2f}%")
        print()
        
        # Create tracking directory
        os.makedirs('bet_tracking', exist_ok=True)
        self.tracking_file = 'bet_tracking/bet_history.csv'
    
    def predict_event(self, event_csv):
        """Predict all fights in an event"""
        print("="*80)
        print("PREDICTING EVENT")
        print("="*80)
        
        # Load event
        event_df = pd.read_csv(event_csv)
        print(f"Event: {event_df['EVENT'].iloc[0]}")
        print(f"Date: {event_df['DATE'].iloc[0]}")
        print(f"Total fights: {len(event_df) // 2}")  # 2 rows per fight
        print()
        
        # Get predictions
        predictions = []
        
        # Process each fight (every 2 rows)
        for i in range(0, len(event_df), 2):
            fighter1 = event_df.iloc[i]
            fighter2 = event_df.iloc[i+1]
            
            # TODO: Extract features and make prediction
            # This requires the same feature engineering as training
            # For now, placeholder
            
            predictions.append({
                'fighter1': fighter1['FIGHTER'],
                'fighter2': fighter2['FIGHTER'],
                'fighter1_prob': 0.5,  # Placeholder
                'fighter2_prob': 0.5,  # Placeholder
                'pick': fighter1['FIGHTER'],  # Placeholder
                'odds': fighter1.get('avg_odds_calculated', None)
            })
        
        return predictions
    
    def display_picks(self, predictions):
        """Display betting recommendations"""
        print("="*80)
        print("BETTING RECOMMENDATIONS")
        print("="*80)
        print()
        
        for i, pred in enumerate(predictions, 1):
            print(f"Fight {i}: {pred['fighter1']} vs {pred['fighter2']}")
            print(f"  Pick: {pred['pick']} ({pred['fighter1_prob']*100:.1f}%)")
            if pred['odds']:
                print(f"  Odds: {pred['odds']:+.0f}")
                # Calculate expected value
                if pred['odds'] > 0:
                    payout = pred['odds'] / 100
                else:
                    payout = 100 / abs(pred['odds'])
                ev = pred['fighter1_prob'] * payout - (1 - pred['fighter1_prob'])
                print(f"  Expected Value: {ev*100:+.2f}%")
            print()
        
        print("="*80)
        print(f"Total recommended bets: {len(predictions)}")
        print("="*80)
    
    def track_results(self, predictions, results):
        """Track betting results over time"""
        # TODO: Implement result tracking
        pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_event_production.py <event_csv>")
        print()
        print("Example:")
        print("  python predict_event_production.py upcoming_events/ufc_308.csv")
        sys.exit(1)
    
    event_csv = sys.argv[1]
    
    if not os.path.exists(event_csv):
        print(f"Error: File not found: {event_csv}")
        sys.exit(1)
    
    # Initialize predictor
    predictor = EventPredictor()
    
    # Predict event
    predictions = predictor.predict_event(event_csv)
    
    # Display picks
    predictor.display_picks(predictions)


if __name__ == '__main__':
    main()

