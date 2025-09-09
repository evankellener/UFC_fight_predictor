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
            return jsonify({'error': 'Predictor not initialized'}), 500
        
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
    try:
        if not predictor:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        fighters = predictor.get_available_fighters()
        return jsonify({'fighters': fighters})
        
    except Exception as e:
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

if __name__ == '__main__':
    # Initialize the predictor when starting the app
    if initialize_predictor():
        port = int(os.environ.get('PORT', 5001))
        debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
    else:
        print("Failed to initialize predictor. Exiting...")
        sys.exit(1)
