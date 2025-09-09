import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import sys

class SimpleUFCPredictor:
    """
    A simple interface for making UFC fight predictions using the ensemble model approach.
    """
    
    def __init__(self, data_path='final_with_odds_filtered.csv'):
        """
        Initialize the predictor with the data file.
        
        Args:
            data_path (str): Path to the CSV file with fighter data
        """
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        
        # Load and prepare data
        self._load_data()
        self._prepare_model()
    
    def _load_data(self):
        """Load and prepare the data for predictions."""
        print("Loading fighter data...")
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        self.data['DATE'] = pd.to_datetime(self.data['DATE'], errors='coerce')
        
        # Filter to recent data and male fighters
        self.data = self.data[self.data['DATE'] >= '2009-01-01']
        self.data = self.data[self.data['sex'] == 2]  # Male fighters
        
        # Define the important feature columns (from ensemble_model_best.py)
        self.feature_columns = [
            'age_ratio_difference', 'opp_age_ratio_difference', 'opp_precomp_elo_change_5', 
            'precomp_elo', 'opp_precomp_elo', 'precomp_tdavg', 'opp_precomp_tdavg',
            'opp_precomp_tddef', 'opp_precomp_sapm5', 'precomp_tddef', 'precomp_sapm5',
            'precomp_headacc_perc3', 'opp_precomp_headacc_perc3', 'precomp_totalacc_perc3',
            'precomp_elo_change_5', 'REACH', 'opp_REACH', 'precomp_legacc_perc5',
            'opp_precomp_totalacc_perc3', 'opp_precomp_legacc_perc5', 'opp_precomp_clinchacc_perc5',
            'precomp_clinchacc_perc5', 'precomp_winsum3', 'opp_precomp_winsum3',
            'opp_precomp_sapm', 'precomp_sapm', 'opp_precomp_totalacc_perc', 'precomp_totalacc_perc',
            'precomp_groundacc_perc5', 'opp_precomp_groundacc_perc5', 'precomp_losssum5',
            'opp_precomp_losssum5', 'age', 'opp_age', 'precomp_strike_elo', 'opp_precomp_strike_elo'
        ]
        
        # Filter to only include rows with these features
        available_features = [col for col in self.feature_columns if col in self.data.columns]
        print(f"Available features: {len(available_features)} out of {len(self.feature_columns)}")
        
        # Filter data to only include rows with most features available
        thresh = int(0.7 * len(available_features))
        self.data = self.data.dropna(subset=['win'])
        self.data = self.data[self.data[available_features].isnull().sum(axis=1) < thresh]
        
        print(f"Data loaded: {len(self.data)} fights with {len(available_features)} features")
        
        # Split into train/test by date
        latest = self.data['DATE'].max()
        cutoff = latest - pd.Timedelta(days=365)
        
        self.train_data = self.data[self.data['DATE'] < cutoff]
        self.test_data = self.data[self.data['DATE'] >= cutoff]
        
        print(f"Training data: {len(self.train_data)} fights")
        print(f"Test data: {len(self.test_data)} fights")
    
    def _prepare_model(self):
        """Train a simple logistic regression model for predictions."""
        print("Training prediction model...")
        
        # Prepare training data
        X_train = self.train_data[self.feature_columns].copy()
        y_train = self.train_data['win'].copy()
        
        # Handle missing values
        self.imputer = SimpleImputer(strategy='median')
        X_train_imputed = self.imputer.fit_transform(X_train)
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        # Train model
        self.model = LogisticRegression(
            max_iter=10000,
            C=10,
            class_weight='balanced',
            penalty='l2',
            solver='liblinear',
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        X_test = self.test_data[self.feature_columns].copy()
        y_test = self.test_data['win'].copy()
        
        X_test_imputed = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        print(f"Model trained successfully!")
        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Test accuracy: {test_acc:.3f}")
    
    def predict_fight(self, fighter1_name, fighter2_name):
        """
        Predict the winner of a fight between two fighters.
        
        Args:
            fighter1_name (str): Name of the first fighter
            fighter2_name (str): Name of the second fighter
            
        Returns:
            dict: Prediction results with winner, probability, and details
        """
        print(f"\nPredicting: {fighter1_name} vs {fighter2_name}")
        
        # Find the most recent data for each fighter
        fighter1_data = self.data[self.data['FIGHTER'] == fighter1_name]
        fighter2_data = self.data[self.data['FIGHTER'] == fighter2_name]
        
        if len(fighter1_data) == 0:
            raise ValueError(f"No data found for fighter '{fighter1_name}'")
        if len(fighter2_data) == 0:
            raise ValueError(f"No data found for fighter '{fighter2_name}'")
        
        # Get most recent fight data for each fighter
        fighter1_latest = fighter1_data.sort_values('DATE', ascending=False).iloc[0]
        fighter2_latest = fighter2_data.sort_values('DATE', ascending=False).iloc[0]
        
        # Create feature vector for fighter1 vs fighter2
        features = []
        for col in self.feature_columns:
            if col in fighter1_latest and col in fighter2_latest:
                # For opponent columns, use fighter2's data
                if col.startswith('opp_'):
                    base_col = col.replace('opp_', '')
                    if base_col in fighter1_latest:
                        features.append(fighter1_latest[base_col])
                    else:
                        features.append(0)  # Default value
                else:
                    features.append(fighter1_latest[col])
            else:
                features.append(0)  # Default value for missing features
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        features_imputed = self.imputer.transform(features_array)
        features_scaled = self.scaler.transform(features_imputed)
        
        prob_win = self.model.predict_proba(features_scaled)[0][1]
        
        # Determine winner
        if prob_win > 0.5:
            winner = fighter1_name
            confidence = prob_win
        else:
            winner = fighter2_name
            confidence = 1 - prob_win
        
        # Get fighter stats for context
        fighter1_stats = {
            'elo': fighter1_latest.get('precomp_elo', 'N/A'),
            'striking_elo': fighter1_latest.get('precomp_strike_elo', 'N/A'),
            'age': fighter1_latest.get('age', 'N/A'),
            'height': fighter1_latest.get('HEIGHT', 'N/A'),
            'weight': fighter1_latest.get('WEIGHT', 'N/A'),
            'reach': fighter1_latest.get('REACH', 'N/A'),
            'last_fight_date': fighter1_latest.get('DATE', 'N/A')
        }
        
        fighter2_stats = {
            'elo': fighter2_latest.get('precomp_elo', 'N/A'),
            'striking_elo': fighter2_latest.get('precomp_strike_elo', 'N/A'),
            'age': fighter2_latest.get('age', 'N/A'),
            'height': fighter2_latest.get('HEIGHT', 'N/A'),
            'weight': fighter2_latest.get('WEIGHT', 'N/A'),
            'reach': fighter2_latest.get('REACH', 'N/A'),
            'last_fight_date': fighter2_latest.get('DATE', 'N/A')
        }
        
        # Convert probability to American odds
        def prob_to_american_odds(p):
            if p <= 0 or p >= 1:
                return "N/A"
            if p >= 0.5:
                odds = - (p / (1 - p)) * 100
            else:
                odds = ((1 - p) / p) * 100
            return int(np.sign(odds) * np.round(abs(odds)))
        
        odds = prob_to_american_odds(prob_win)
        
        result = {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'predicted_winner': winner,
            'fighter1_win_probability': prob_win,
            'fighter2_win_probability': 1 - prob_win,
            'confidence': confidence,
            'american_odds': odds,
            'fighter1_stats': fighter1_stats,
            'fighter2_stats': fighter2_stats,
            'key_factors': self._get_key_factors(features, prob_win)
        }
        
        return result
    
    def _get_key_factors(self, features, prob_win):
        """Identify the key factors that influenced this prediction."""
        # Get feature importance from the model
        feature_importance = self.model.coef_[0]
        
        # Create feature importance pairs
        feature_importance_pairs = list(zip(self.feature_columns, feature_importance))
        
        # Sort by absolute importance
        feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 5 most important features
        top_features = feature_importance_pairs[:5]
        
        key_factors = []
        for feature, importance in top_features:
            if feature in self.feature_columns:
                idx = self.feature_columns.index(feature)
                if idx < len(features):
                    value = features[idx]
                    # Create a human-readable description
                    if 'elo' in feature.lower():
                        desc = f"{feature}: {value:.1f}"
                    elif 'age' in feature.lower():
                        desc = f"{feature}: {value:.1f} years"
                    elif 'height' in feature.lower() or 'reach' in feature.lower():
                        desc = f"{feature}: {value:.1f} inches"
                    elif 'weight' in feature.lower():
                        desc = f"{feature}: {value:.1f} lbs"
                    else:
                        desc = f"{feature}: {value:.3f}"
                    key_factors.append(desc)
        
        return key_factors
    
    def get_fighter_info(self, fighter_name):
        """Get detailed information about a specific fighter."""
        fighter_data = self.data[self.data['FIGHTER'] == fighter_name]
        
        if len(fighter_data) == 0:
            return None
        
        # Get most recent data
        latest = fighter_data.sort_values('DATE', ascending=False).iloc[0]
        
        # Get career stats
        career_wins = len(fighter_data[fighter_data['win'] == 1])
        career_losses = len(fighter_data[fighter_data['win'] == 0])
        total_fights = career_wins + career_losses
        
        info = {
            'name': fighter_name,
            'total_fights': total_fights,
            'wins': career_wins,
            'losses': career_losses,
            'win_rate': career_wins / total_fights if total_fights > 0 else 0,
            'current_elo': latest.get('precomp_elo', 'N/A'),
            'current_striking_elo': latest.get('precomp_strike_elo', 'N/A'),
            'age': latest.get('age', 'N/A'),
            'height': latest.get('HEIGHT', 'N/A'),
            'weight': latest.get('WEIGHT', 'N/A'),
            'reach': latest.get('REACH', 'N/A'),
            'last_fight_date': latest.get('DATE', 'N/A'),
            'last_fight_result': 'Win' if latest.get('win') == 1 else 'Loss'
        }
        
        return info

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = SimpleUFCPredictor()
    
    # Example prediction
    try:
        result = predictor.predict_fight("Israel Adesanya", "Sean Strickland")
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Odds: {result['american_odds']}")
        
        print(f"\n📊 FIGHTER COMPARISON:")
        print(f"{result['fighter1']}:")
        for stat, value in result['fighter1_stats'].items():
            print(f"  {stat}: {value}")
        
        print(f"\n{result['fighter2']}:")
        for stat, value in result['fighter2_stats'].items():
            print(f"  {stat}: {value}")
        
        print(f"\n🔑 KEY FACTORS:")
        for factor in result['key_factors']:
            print(f"  • {factor}")
            
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Make sure the fighter names exist in your dataset.")
