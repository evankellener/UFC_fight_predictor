# -*- coding: utf-8 -*-
"""
UFC Fight Predictor Flask App
A web interface for predicting UFC fight outcomes using machine learning.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import warnings
import json
warnings.filterwarnings('ignore')

# Add the current directory to the path to import our model module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import UFCFightPredictor

app = Flask(__name__)

# Initialize the predictor
predictor = None

def initialize_predictor():
    """Initialize the UFC Fight Predictor model"""
    global predictor
    try:
        print("Initializing UFC Fight Predictor...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        
        # Check if data files exist
        data_paths = [
            '../data/final.csv',
            'data/final.csv',
            '../data/tmp/final_min_fight1.csv',
            'data/tmp/final_min_fight1.csv'
        ]
        
        found_data = False
        for path in data_paths:
            if os.path.exists(path):
                print(f"Found data file at: {path}")
                found_data = True
                break
        
        if not found_data:
            print("No data file found in any expected location!")
            print("Available files in current directory:")
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.csv'):
                        print(f"  {os.path.join(root, file)}")
            return False
        
        predictor = UFCFightPredictor()
        print("UFC Fight Predictor initialized successfully!")
        return True
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Make sure final.csv is in the correct location")
        return False
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required modules are installed")
        return False
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main page with fighter selection interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle fight prediction requests"""
    global predictor
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        fighter1_name = data.get('fighter1', '').strip()
        fighter2_name = data.get('fighter2', '').strip()
        fight_date = data.get('fight_date', '')
        
        if not fighter1_name or not fighter2_name:
            return jsonify({'error': 'Both fighter names are required'}), 400
        
        if not predictor:
            print("Predictor not initialized, attempting to reinitialize...")
            if not initialize_predictor():
                return jsonify({'error': 'Predictor not initialized and reinitialization failed'}), 500
        
        # Get prediction
        result = predictor.predict_fight(fighter1_name, fighter2_name, fight_date)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/fighters')
def get_fighters():
    """Get list of available fighters"""
    global predictor
    try:
        if not predictor:
            print("Predictor not initialized, attempting to reinitialize...")
            if not initialize_predictor():
                return jsonify({'error': 'Predictor not initialized and reinitialization failed'}), 500
        
        fighters = predictor.get_available_fighters()
        return jsonify({'fighters': fighters})
        
    except Exception as e:
        print(f"Error in get_fighters: {e}")
        return jsonify({'error': f'Failed to get fighters: {str(e)}'}), 500

@app.route('/fighter/<fighter_name>')
def get_fighter_stats(fighter_name):
    """Get detailed stats for a specific fighter"""
    try:
        if not predictor:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        stats = predictor.get_fighter_stats(fighter_name)
        if stats:
            return jsonify(stats)
        else:
            return jsonify({'error': 'Fighter not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Failed to get fighter stats: {str(e)}'}), 500

# Card Builder Routes
@app.route('/card-builder')
def card_builder():
    """Card builder page"""
    return render_template('card_builder.html')

@app.route('/api/cards', methods=['GET'])
def get_cards():
    """Get all saved cards"""
    try:
        # For now, we'll use a simple file-based storage
        # In production, you'd want to use a proper database
        cards_file = 'saved_cards.json'
        if os.path.exists(cards_file):
            with open(cards_file, 'r') as f:
                cards = json.load(f)
        else:
            cards = []
        return jsonify({'cards': cards})
    except Exception as e:
        return jsonify({'error': f'Failed to load cards: {str(e)}'}), 500

@app.route('/api/cards', methods=['POST'])
def save_card():
    """Save a new card"""
    try:
        data = request.get_json()
        card_name = data.get('name', '').strip()
        fights = data.get('fights', [])
        event_date = data.get('event_date', '')
        
        if not card_name:
            return jsonify({'error': 'Card name is required'}), 400
        
        if not fights:
            return jsonify({'error': 'At least one fight is required'}), 400
        
        # Validate fights
        for fight in fights:
            if not fight.get('fighter1') or not fight.get('fighter2'):
                return jsonify({'error': 'Each fight must have both fighters'}), 400
        
        # Create card object
        card = {
            'id': str(datetime.now().timestamp()),
            'name': card_name,
            'event_date': event_date,
            'fights': fights,
            'created_at': datetime.now().isoformat()
        }
        
        # Load existing cards
        cards_file = 'saved_cards.json'
        if os.path.exists(cards_file):
            with open(cards_file, 'r') as f:
                cards = json.load(f)
        else:
            cards = []
        
        # Add new card
        cards.append(card)
        
        # Save back to file
        with open(cards_file, 'w') as f:
            json.dump(cards, f, indent=2)
        
        return jsonify({'success': True, 'card': card})
        
    except Exception as e:
        return jsonify({'error': f'Failed to save card: {str(e)}'}), 500

@app.route('/api/cards/<card_id>/predict', methods=['POST'])
def predict_card():
    """Predict all fights in a card"""
    global predictor
    try:
        if not predictor:
            print("Predictor not initialized, attempting to reinitialize...")
            if not initialize_predictor():
                return jsonify({'error': 'Predictor not initialized and reinitialization failed'}), 500
        
        # Load the card
        cards_file = 'saved_cards.json'
        if not os.path.exists(cards_file):
            return jsonify({'error': 'No cards found'}), 404
        
        with open(cards_file, 'r') as f:
            cards = json.load(f)
        
        card = next((c for c in cards if c['id'] == card_id), None)
        if not card:
            return jsonify({'error': 'Card not found'}), 404
        
        # Predict all fights
        predictions = []
        for i, fight in enumerate(card['fights']):
            try:
                result = predictor.predict_fight(
                    fight['fighter1'], 
                    fight['fighter2'], 
                    card.get('event_date', '')
                )
                
                if result['success']:
                    predictions.append({
                        'fight_number': i + 1,
                        'fighter1': fight['fighter1'],
                        'fighter2': fight['fighter2'],
                        'prediction': result
                    })
                else:
                    predictions.append({
                        'fight_number': i + 1,
                        'fighter1': fight['fighter1'],
                        'fighter2': fight['fighter2'],
                        'error': result['error']
                    })
            except Exception as e:
                predictions.append({
                    'fight_number': i + 1,
                    'fighter1': fight['fighter1'],
                    'fighter2': fight['fighter2'],
                    'error': f'Prediction failed: {str(e)}'
                })
        
        return jsonify({
            'success': True,
            'card_name': card['name'],
            'event_date': card.get('event_date', ''),
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to predict card: {str(e)}'}), 500

@app.route('/api/cards/<card_id>', methods=['DELETE'])
def delete_card():
    """Delete a card"""
    try:
        cards_file = 'saved_cards.json'
        if not os.path.exists(cards_file):
            return jsonify({'error': 'No cards found'}), 404
        
        with open(cards_file, 'r') as f:
            cards = json.load(f)
        
        # Find and remove the card
        original_length = len(cards)
        cards = [c for c in cards if c['id'] != card_id]
        
        if len(cards) == original_length:
            return jsonify({'error': 'Card not found'}), 404
        
        # Save back to file
        with open(cards_file, 'w') as f:
            json.dump(cards, f, indent=2)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete card: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize the predictor when starting the app
    if initialize_predictor():
        port = int(os.environ.get('PORT', 5001))
        debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
    else:
        print("Failed to initialize predictor. Exiting...")
        sys.exit(1)
