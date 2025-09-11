# -*- coding: utf-8 -*-
"""
UFC Fight Predictor Model
Contains the machine learning model and prediction logic for UFC fights.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Add multiple paths to find the ensemble model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))
sys.path.append(os.path.join(current_dir, '..', 'src'))

try:
    from src.ensemble_model_best import FightOutcomeModel
    print("Successfully imported FightOutcomeModel")
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback if the import fails
    FightOutcomeModel = None

class UFCFightPredictor:
    def __init__(self):
        """Initialize the UFC Fight Predictor with data and model"""
        # Use absolute path for production deployment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible locations for the data file
        possible_paths = [
            '../data/tmp/final_min_fight1.csv',
            'data/tmp/final_min_fight1.csv',
            '../data/tmp/final.csv',
            'data/tmp/final.csv',
            '../data/final.csv',
            'data/final.csv',
            '../data/final_with_swapped.csv',
            'data/final_with_swapped.csv',
            os.path.join(current_dir, '..', 'data', 'final_with_swapped.csv'),
            os.path.join(current_dir, 'data', 'final_with_swapped.csv'),
            os.path.join(current_dir, '..', 'data', 'tmp', 'final.csv'),
            os.path.join(current_dir, 'data', 'tmp', 'final.csv'),
            os.path.join(current_dir, '..', 'data', 'final.csv'),
            os.path.join(current_dir, 'data', 'final.csv'),
            os.path.join(os.getcwd(), 'data', 'final_with_swapped.csv'),
            os.path.join(os.getcwd(), '..', 'data', 'final_with_swapped.csv'),
            os.path.join(os.getcwd(), 'data', 'tmp', 'final.csv'),
            os.path.join(os.getcwd(), '..', 'data', 'tmp', 'final.csv'),
            os.path.join(os.getcwd(), 'data', 'final.csv'),
            os.path.join(os.getcwd(), '..', 'data', 'final.csv')
        ]
        
        self.data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.data_path = path
                print(f"Found data file at: {path}")
                break
        
        if self.data_path is None:
            raise FileNotFoundError("Could not find final.csv in any expected location. Tried: " + ", ".join(possible_paths))
        self.target = "win"
        
        # Load the full unfiltered dataset for display purposes
        print("Loading full unfiltered dataset for display...")
        self.full_df = pd.read_csv(self.data_path, low_memory=False)
        self.full_df['DATE'] = pd.to_datetime(self.full_df['DATE'], errors='coerce')
        self.full_df = self.full_df[self.full_df['DATE'] >= '2009-01-01']
        self.full_df = self.full_df[self.full_df['sex'].astype(str) == '2']
        self.full_df['win'] = self.full_df['win'].astype(int)
        print(f"Full dataset loaded: {len(self.full_df)} rows (for display purposes)")
        
        # Initialize the FightOutcomeModel from ensemble_model_best.py
        if FightOutcomeModel is None:
            print("FightOutcomeModel not available, using fallback model...")
            self._initialize_fallback_model()
        else:
            print("Initializing FightOutcomeModel...")
            self.fight_model = FightOutcomeModel(self.data_path)
            
            # Train the tuned logistic regression model
            print("Training tuned logistic regression model...")
            self.model, self.accuracy = self.fight_model.tune_logistic_regression()
            print(f"Logistic Regression Accuracy: {self.accuracy}")
        
        # Get the filtered data and features from the fight model (for training)
        if FightOutcomeModel is not None:
            self.df = self.fight_model.df
            self.X_train = self.fight_model.X_train
            self.y_train = self.fight_model.y_train
            self.X_test = self.fight_model.X_test
            self.y_test = self.fight_model.y_test
            
            # Use the importance columns from the fight model
            self.full_features = self.fight_model.importance_columns
        else:
            # For fallback model, use the full dataset
            self.df = self.full_df
            self.full_features = self._get_fallback_features()
        
        # Map of precomp -> postcomp for carry forward
        self.pre_to_post_map = {
            'precomp_elo': 'postcomp_elo',
            'precomp_tdavg': 'postcomp_tdavg',
            'precomp_tddef': 'postcomp_tddef',
            'precomp_sapm5': 'postcomp_sapm5',
            'precomp_headacc_perc3': 'postcomp_headacc_perc3',
            'precomp_totalacc_perc3': 'postcomp_totalacc_perc3',
            'precomp_elo_change_5': 'postcomp_elo_change_5',
            'REACH': 'REACH',
            'precomp_legacc_perc5': 'postcomp_legacc_perc5',
            'precomp_clinchacc_perc5': 'postcomp_clinchacc_perc5',
            'precomp_winsum3': 'postcomp_winsum3',
            'precomp_sapm': 'postcomp_sapm',
            'precomp_totalacc_perc': 'postcomp_totalacc_perc',
            'precomp_groundacc_perc5': 'postcomp_groundacc_perc5',
            'precomp_losssum5': 'postcomp_losssum5',
            'precomp_strike_elo': 'postcomp_strike_elo',
        }
    
    
    def _truncate_round(self, x):
        """Truncate and round values to 2 decimal places"""
        if pd.isna(x):
            return np.nan
        x = float(x)
        truncated = int(x * 10000) / 10000.0
        return round(truncated, 2)
    
    def _apply_elo_decay(self, x, gap_days, threshold=365, decay=0.978):
        """Apply Elo decay based on time gap"""
        if pd.isna(x):
            return np.nan
        val = float(x)
        if gap_days is not None and gap_days >= threshold:
            val = val * decay
        return self._truncate_round(val)
    
    def _compute_age(self, dob_str, fight_date):
        """Compute age from date of birth and fight date"""
        dob = pd.to_datetime(str(dob_str), errors='coerce')
        if pd.isna(dob) or pd.isna(fight_date):
            return np.nan
        return (fight_date - dob).days / 365.25
    
    def _coerce_numeric(self, df, cols):
        """Convert columns to numeric, handling errors"""
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors='coerce')
            else:
                out[c] = np.nan
        return out
    
    def get_available_fighters(self):
        """Get list of all available fighters"""
        if self.full_df is None:
            return []
        
        fighters = sorted(self.full_df['FIGHTER'].unique().tolist())
        return fighters
    
    def get_fighter_stats(self, fighter_name):
        """Get the latest stats for a specific fighter"""
        if self.full_df is None:
            return None
        
        fighter_data = self.full_df[self.full_df['FIGHTER'] == fighter_name]
        if len(fighter_data) == 0:
            return None
        
        # Get the most recent fight
        latest_fight = fighter_data.sort_values('DATE', ascending=False).iloc[0]
        
        stats = {
            'name': fighter_name,
            'latest_fight_date': latest_fight['DATE'].strftime('%Y-%m-%d') if pd.notna(latest_fight['DATE']) else 'N/A',
            'age': latest_fight.get('age', 'N/A'),
            'height': latest_fight.get('HEIGHT', 'N/A'),
            'weight': latest_fight.get('WEIGHT', 'N/A'),
            'reach': latest_fight.get('REACH', 'N/A'),
            'stance': latest_fight.get('STANCE', 'N/A'),
            'precomp_elo': latest_fight.get('precomp_elo', 'N/A'),
            'postcomp_elo': latest_fight.get('postcomp_elo', 'N/A'),
            'total_fights': len(fighter_data),
            'wins': len(fighter_data[fighter_data['win'] == 1]),
            'losses': len(fighter_data[fighter_data['win'] == 0])
        }
        
        return stats
    
    def predict_fight(self, fighter1_name, fighter2_name, fight_date=None):
        """Predict the outcome of a fight between two fighters using bidirectional prediction"""
        try:
            if self.model is None:
                return {'success': False, 'error': 'Model not trained'}
            
            if fight_date is None:
                fight_date = datetime.now().strftime('%Y-%m-%d')
            
            fight_date = pd.to_datetime(fight_date)
            
            # Get fighter data from full dataset for display
            fighter1_data = self.full_df[self.full_df['FIGHTER'] == fighter1_name]
            fighter2_data = self.full_df[self.full_df['FIGHTER'] == fighter2_name]
            
            if len(fighter1_data) == 0:
                return {'success': False, 'error': f'Fighter "{fighter1_name}" not found in database'}
            
            if len(fighter2_data) == 0:
                return {'success': False, 'error': f'Fighter "{fighter2_name}" not found in database'}
            
            # Get the most recent fight data for each fighter
            fighter1_latest = fighter1_data.sort_values('DATE', ascending=False).iloc[0]
            fighter2_latest = fighter2_data.sort_values('DATE', ascending=False).iloc[0]
            
            # Make two predictions: A vs B and B vs A, then average them
            # Prediction 1: fighter1 vs fighter2
            features_1 = self._create_fight_features(fighter1_latest, fighter2_latest, fight_date)
            prediction_1 = self.model.predict_proba([features_1])[0]
            fighter1_win_prob_1 = prediction_1[1]  # Probability of fighter1 winning in first prediction
            
            # Prediction 2: fighter2 vs fighter1 (swapped)
            features_2 = self._create_fight_features(fighter2_latest, fighter1_latest, fight_date)
            prediction_2 = self.model.predict_proba([features_2])[0]
            fighter2_win_prob_2 = prediction_2[1]  # Probability of fighter2 winning in second prediction
            
            # Average the probabilities (since prediction_2 gives us fighter2's win prob when fighter2 is first)
            fighter1_win_prob = (fighter1_win_prob_1 + (1 - fighter2_win_prob_2)) / 2
            fighter2_win_prob = (fighter2_win_prob_2 + (1 - fighter1_win_prob_1)) / 2
            
            # Normalize to ensure they sum to 1
            total_prob = fighter1_win_prob + fighter2_win_prob
            fighter1_win_prob = fighter1_win_prob / total_prob
            fighter2_win_prob = fighter2_win_prob / total_prob
            
            predicted_winner = fighter1_name if fighter1_win_prob > 0.5 else fighter2_name
            confidence = max(fighter1_win_prob, fighter2_win_prob)
            
            result = {
                'success': True,
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'predicted_winner': predicted_winner,
                'fighter1_win_probability': round(fighter1_win_prob, 3),
                'fighter2_win_probability': round(fighter2_win_prob, 3),
                'confidence': round(confidence, 3),
                'fight_date': fight_date.strftime('%Y-%m-%d') if pd.notna(fight_date) else 'N/A',
                'fighter1_stats': self.get_fighter_stats(fighter1_name),
                'fighter2_stats': self.get_fighter_stats(fighter2_name)
            }
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Prediction failed: {str(e)}'}
    
    def _create_fight_features(self, fighter1_data, fighter2_data, fight_date):
        """Create feature vector for fight prediction using the FightOutcomeModel features"""
        # Calculate ages at fight date
        fighter1_age = self._compute_age(fighter1_data.get('DOB'), fight_date)
        fighter2_age = self._compute_age(fighter2_data.get('OPP_DOB'), fight_date)
        
        # Age ratios
        age_ratio_diff = self._truncate_round(fighter1_age / fighter2_age - 1.0) if fighter2_age > 0 else 0
        opp_age_ratio_diff = self._truncate_round(fighter2_age / fighter1_age - 1.0) if fighter1_age > 0 else 0
        
        # Create feature vector using the importance columns from FightOutcomeModel
        feature_values = {}
        
        # Map fighter data to feature values
        for feature in self.full_features:
            if feature == 'age_ratio_difference':
                feature_values[feature] = age_ratio_diff
            elif feature == 'opp_age_ratio_difference':
                feature_values[feature] = opp_age_ratio_diff
            elif feature == 'age':
                feature_values[feature] = fighter1_age
            elif feature == 'opp_age':
                feature_values[feature] = fighter2_age
            elif feature.startswith('opp_'):
                # For opponent features, get from fighter2_data
                opp_feature = feature[4:]  # Remove 'opp_' prefix
                feature_values[feature] = fighter2_data.get(opp_feature, 0)
            else:
                # For fighter features, get from fighter1_data
                feature_values[feature] = fighter1_data.get(feature, 0)
        
        # Create feature vector in the correct order
        features = []
        for feature in self.full_features:
            value = feature_values.get(feature, 0)
            
            # Convert to numeric and handle various data types
            try:
                # Try to convert to float
                if isinstance(value, str):
                    value = float(value) if value.replace('.', '').replace('-', '').isdigit() else 0
                elif pd.isna(value):
                    value = 0
                else:
                    value = float(value)
                
                # Check for infinite values after conversion
                if np.isinf(value) or np.isnan(value):
                    value = 0
                    
            except (ValueError, TypeError, OverflowError):
                # If conversion fails, use 0
                value = 0
                
            features.append(value)
        
        return np.array(features)
    
    def _initialize_fallback_model(self):
        """Initialize a simple fallback model when FightOutcomeModel is not available"""
        print("Initializing fallback logistic regression model...")
        
        # Simple feature set for fallback
        self.full_features = [
            'precomp_elo', 'opp_precomp_elo', 'age', 'opp_age',
            'precomp_tdavg', 'opp_precomp_tdavg', 'precomp_tddef', 'opp_precomp_tddef',
            'precomp_sapm5', 'opp_precomp_sapm5', 'REACH', 'opp_REACH'
        ]
        
        # Prepare data for training
        train_data = self.full_df[self.full_df['DATE'] < '2024-01-01'].copy()
        test_data = self.full_df[self.full_df['DATE'] >= '2024-01-01'].copy()
        
        # Convert features to numeric and handle missing values
        for feature in self.full_features:
            train_data[feature] = pd.to_numeric(train_data[feature], errors='coerce')
            test_data[feature] = pd.to_numeric(test_data[feature], errors='coerce')
        
        # Fill missing values with median
        for feature in self.full_features:
            median_val = train_data[feature].median()
            train_data[feature] = train_data[feature].fillna(median_val)
            test_data[feature] = test_data[feature].fillna(median_val)
        
        # Prepare training data
        X_train = train_data[self.full_features].values
        y_train = train_data['win'].values
        X_test = test_data[self.full_features].values
        y_test = test_data['win'].values
        
        # Train simple logistic regression
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()
        
        self.accuracy = test_acc
        print(f"Fallback model - Train accuracy: {train_acc:.3f}, Test accuracy: {test_acc:.3f}")
        
        # Store training data for reference
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def _get_fallback_features(self):
        """Get the feature names for the fallback model"""
        return self.full_features
