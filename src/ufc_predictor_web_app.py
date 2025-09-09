#!/usr/bin/env python3
"""
UFC Fight Predictor Web App

A Flask web application for predicting UFC fight outcomes.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to the path to import our predictor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ufc_fight_predictor_app import UFCFightPredictor

app = Flask(__name__)

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor on startup."""
    global predictor
    try:
        predictor = UFCFightPredictor()
        print("✅ Predictor initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('ufc_predictor.html')

@app.route('/api/fighters')
def get_fighters():
    """Get list of available fighters."""
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        fighters = predictor.get_available_fighters()
        return jsonify({'fighters': fighters})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weight-classes')
def get_weight_classes():
    """Get list of available weight classes."""
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        weight_classes = predictor.get_available_weight_classes()
        return jsonify({'weight_classes': weight_classes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fighter-versions/<fighter_name>')
def get_fighter_versions(fighter_name):
    """Get available versions for a specific fighter."""
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        versions = predictor.get_fighter_versions(fighter_name)
        return jsonify({'versions': versions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_fight():
    """Predict the outcome of a fight."""
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        
        fighter1 = data.get('fighter1')
        fighter2 = data.get('fighter2')
        weight_class = int(data.get('weight_class'))
        fight_date_str = data.get('fight_date')
        
        if not all([fighter1, fighter2, weight_class, fight_date_str]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Parse fight date
        fight_date = datetime.strptime(fight_date_str, '%Y-%m-%d')
        
        # Make prediction
        result = predictor.predict_fight(fighter1, fighter2, weight_class, fight_date)
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate')
def validate_model():
    """Validate the model on test data."""
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        validation_results = predictor.validate_model_on_test_data()
        return jsonify(validation_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_model_stats():
    """Get model statistics."""
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        stats = {
            'train_accuracy': predictor.train_accuracy,
            'test_accuracy': predictor.test_accuracy,
            'total_fighters': len(predictor.get_available_fighters()),
            'total_weight_classes': len(predictor.get_available_weight_classes()),
            'total_fights': len(predictor.data),
            'training_fights': len(predictor.train_data),
            'test_fights': len(predictor.test_data)
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize predictor on startup
    if initialize_predictor():
        print("🚀 Starting UFC Fight Predictor Web App...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize predictor. Exiting.")
        sys.exit(1)
