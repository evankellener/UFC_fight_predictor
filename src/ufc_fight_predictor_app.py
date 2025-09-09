#!/usr/bin/env python3
"""
UFC Fight Predictor App

This app allows users to predict the outcome of UFC fights by:
1. Selecting two fighters
2. Specifying the weight class
3. Specifying when the fight will take place

The model uses postcomp stats from each fighter's most recent fight to predict upcoming fights,
with proper ELO decay and age calculations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UFCFightPredictor:
    """
    UFC Fight Predictor that uses postcomp stats from previous fights to predict upcoming fights.
    """
    
    def __init__(self, data_path='data/tmp/final.csv'):
        """
        Initialize the UFC Fight Predictor.
        
        Args:
            data_path (str): Path to the CSV file with fighter data
        """
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.fighter_versions = {}  # Store available versions for each fighter
        self.last_fight_dict = {}  # Track last fight dates for ELO decay
        
        # Weight class mapping for display
        self.weight_class_names = {
            1: 'Strawweight (115 lbs)',
            2: 'Women\'s Flyweight (125 lbs)',
            3: 'Women\'s Bantamweight (135 lbs)',
            4: 'Women\'s Featherweight (145 lbs)',
            5: 'Flyweight (125 lbs)',
            6: 'Bantamweight (135 lbs)',
            7: 'Featherweight (145 lbs)',
            8: 'Lightweight (155 lbs)',
            9: 'Welterweight (170 lbs)',
            10: 'Middleweight (185 lbs)',
            11: 'Light Heavyweight (205 lbs)',
            12: 'Heavyweight (265 lbs)'
        }
        
        # Define base feature columns - these are the features the model expects
        self.base_feature_columns = [
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
        
        # Load and prepare data
        self._load_data()
        self._prepare_model()
        self._build_fighter_versions()
    
    def _load_data(self):
        """Load and prepare the data for predictions."""
        print("Loading fighter data...")
        
        # Load the main data
        self.data = pd.read_csv(self.data_path)
        self.data['DATE'] = pd.to_datetime(self.data['DATE'], errors='coerce')
        
        print(f"Raw data loaded: {len(self.data)} total rows")
        
        # Filter to recent data and male fighters
        self.data = self.data[self.data['DATE'] >= '2009-01-01']
        print(f"After date filter: {len(self.data)} rows")
        
        if 'sex' in self.data.columns:
            self.data = self.data[self.data['sex'] == 2]  # Male fighters
            print(f"After sex filter: {len(self.data)} rows")
        
        # Load DOB data if available
        self.dob_data = None
        try:
            dob_path = 'data/tmp/final_data_with_dob.csv'
            if os.path.exists(dob_path):
                self.dob_data = pd.read_csv(dob_path)
                self.dob_data['DOB'] = pd.to_datetime(self.dob_data['DOB'], errors='coerce')
                print(f"DOB data loaded: {len(self.dob_data)} fighters")
                # Merge DOB data
                self.data = self.data.merge(
                    self.dob_data[['FIGHTER', 'DOB']], 
                    on='FIGHTER', 
                    how='left'
                )
        except Exception as e:
            print(f"Warning: Could not load DOB data: {e}")
        
        # Apply time-based ELO decay
        self._apply_elo_decay()
        
        # Check which features are actually available
        available_features = [col for col in self.base_feature_columns if col in self.data.columns]
        missing_features = [col for col in self.base_feature_columns if col not in self.data.columns]
        
        print(f"Available features: {len(available_features)} out of {len(self.base_feature_columns)}")
        if missing_features:
            print(f"Missing features: {missing_features[:5]}...")  # Show first 5 missing
        
        # Use only available features
        self.feature_columns = available_features
        
        # Filter data
        self.data = self.data.dropna(subset=['win'])
        print(f"After win filter: {len(self.data)} rows")
        
        # Only require a few key features to be present
        key_features = ['precomp_elo', 'opp_precomp_elo', 'age', 'opp_age']
        key_features = [f for f in key_features if f in self.data.columns]
        
        if key_features:
            # Require at least 2 key features to be present
            self.data = self.data[self.data[key_features].isnull().sum(axis=1) <= len(key_features) - 2]
            print(f"After key feature filter: {len(self.data)} rows")
        
        print(f"Final data: {len(self.data)} fights with {len(self.feature_columns)} features")
        
        # Split into train/test by date - training on 2009-2024, testing on 2024+
        cutoff_date = pd.Timestamp('2024-01-01')
        
        self.train_data = self.data[self.data['DATE'] < cutoff_date]
        self.test_data = self.data[self.data['DATE'] >= cutoff_date]
        
        print(f"Training data: {len(self.train_data)} fights (2009-2023)")
        print(f"Test data: {len(self.test_data)} fights (2024+)")
        
        # Show some sample data
        print(f"\nSample fighters available: {self.data['FIGHTER'].nunique()}")
        print(f"Sample fighters: {', '.join(self.data['FIGHTER'].unique()[:5])}")
    
    def _apply_elo_decay(self):
        """Apply time-based ELO decay based on days since last fight."""
        print("Applying time-based ELO decay...")
        
        # Sort by date to process chronologically
        self.data = self.data.sort_values('DATE').reset_index(drop=True)
        
        # Track last fight date for each fighter
        last_fight_dates = {}
        
        # ELO columns to apply decay to
        elo_columns = ['precomp_elo', 'precomp_strike_elo', 'precomp_grapple_elo']
        opp_elo_columns = ['opp_precomp_elo', 'opp_precomp_strike_elo', 'opp_precomp_grapple_elo']
        
        # Filter to only include columns that exist in the data
        elo_columns = [col for col in elo_columns if col in self.data.columns]
        opp_elo_columns = [col for col in opp_elo_columns if col in self.data.columns]
        
        print(f"Applying decay to ELO columns: {elo_columns}")
        print(f"Applying decay to opponent ELO columns: {opp_elo_columns}")
        
        for idx, row in self.data.iterrows():
            fighter = row['FIGHTER']
            opponent = row.get('opp_FIGHTER')
            fight_date = row['DATE']
            
            # Apply decay to fighter's ELO if they haven't fought in over 365 days
            if fighter in last_fight_dates:
                days_since_last = (fight_date - last_fight_dates[fighter]).days
                if days_since_last > 365:
                    # Apply decay: 0.978 per year (same as gpt_elo_best_time_finnish)
                    decay_factor = 0.978 ** (days_since_last / 365)
                    
                    for col in elo_columns:
                        if pd.notna(self.data.at[idx, col]):
                            original_elo = self.data.at[idx, col]
                            decayed_elo = original_elo * decay_factor
                            self.data.at[idx, col] = decayed_elo
            
            # Apply decay to opponent's ELO
            if opponent and opponent in last_fight_dates:
                days_since_last = (fight_date - last_fight_dates[opponent]).days
                if days_since_last > 365:
                    decay_factor = 0.978 ** (days_since_last / 365)
                    
                    for col in opp_elo_columns:
                        if pd.notna(self.data.at[idx, col]):
                            original_elo = self.data.at[idx, col]
                            decayed_elo = original_elo * decay_factor
                            self.data.at[idx, col] = decayed_elo
            
            # Update last fight dates
            last_fight_dates[fighter] = fight_date
            if opponent:
                last_fight_dates[opponent] = fight_date
        
        # Store for later use
        self.last_fight_dict = last_fight_dates
        
        print(f"Time-based ELO decay applied successfully!")
        print(f"Processed {len(self.data)} fights for {len(last_fight_dates)} fighters")
    
    def _prepare_model(self):
        """Train the logistic regression model using precomp stats (standard approach)."""
        print("Training logistic regression model using precomp stats...")
        
        # Use all data for training
        train_data = self.train_data.copy()
        test_data = self.test_data.copy()
        
        print(f"Training data: {len(train_data)} fights")
        print(f"Test data: {len(test_data)} fights")
        
        # Get features
        features = self.feature_columns.copy()
        
        # Prepare training data
        X_train = train_data[features].copy()
        y_train = train_data['win'].copy()
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Training target shape: {y_train.shape}")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        
        # Train model with strong regularization
        model = LogisticRegression(
            C=0.1,  # Strong regularization
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on training data
        train_acc = model.score(X_train_scaled, y_train)
        
        # Evaluate on test data
        X_test = test_data[features].copy()
        y_test = test_data['win'].copy()
        
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        test_acc = model.score(X_test_scaled, y_test)
        
        print(f"Model trained successfully!")
        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Test accuracy: {test_acc:.3f}")
        
        # Store the model components
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        
        self.train_accuracy = train_acc
        self.test_accuracy = test_acc
    
    def _create_training_data_from_postcomp(self):
        """Create training data using postcomp stats from previous fights."""
        print("Creating training data from postcomp stats...")
        
        features_list = []
        labels_list = []
        
        # Use training data (2009-2023)
        train_fights = self.train_data.copy()
        train_fights = train_fights.sort_values('DATE')
        
        print(f"Processing {len(train_fights)} training fights...")
        
        for idx, fight in train_fights.iterrows():
            fighter1_name = fight['FIGHTER']
            fighter2_name = fight.get('opp_FIGHTER')
            fight_date = fight['DATE']
            actual_winner = fight['win']  # 1 if fighter1 wins, 0 if fighter2 wins
            
            if not fighter2_name:
                continue
            
            try:
                # Get fighter stats from their previous fight (not the current fight)
                fighter1_stats = self.get_fighter_previous_postcomp_stats(fighter1_name, fight_date)
                fighter2_stats = self.get_fighter_previous_postcomp_stats(fighter2_name, fight_date)
                
                if fighter1_stats is None or fighter2_stats is None:
                    continue
                
                # Create feature vector using postcomp stats
                features = self._create_feature_vector(fighter1_stats, fighter2_stats)
                
                features_list.append(features)
                labels_list.append(actual_winner)
                
            except Exception as e:
                continue
        
        if len(features_list) == 0:
            return np.array([]), np.array([])
        
        return np.array(features_list), np.array(labels_list)
    
    def _create_test_data_from_postcomp(self):
        """Create test data using postcomp stats from previous fights."""
        print("Creating test data from postcomp stats...")
        
        features_list = []
        labels_list = []
        
        # Use test data (2024+)
        test_fights = self.test_data.copy()
        test_fights = test_fights.sort_values('DATE')
        
        print(f"Processing {len(test_fights)} test fights...")
        
        for idx, fight in test_fights.iterrows():
            fighter1_name = fight['FIGHTER']
            fighter2_name = fight.get('opp_FIGHTER')
            fight_date = fight['DATE']
            actual_winner = fight['win']  # 1 if fighter1 wins, 0 if fighter2 wins
            
            if not fighter2_name:
                continue
            
            try:
                # Get fighter stats from their previous fight (not the current fight)
                fighter1_stats = self.get_fighter_previous_postcomp_stats(fighter1_name, fight_date)
                fighter2_stats = self.get_fighter_previous_postcomp_stats(fighter2_name, fight_date)
                
                if fighter1_stats is None or fighter2_stats is None:
                    continue
                
                # Create feature vector using postcomp stats
                features = self._create_feature_vector(fighter1_stats, fighter2_stats)
                
                features_list.append(features)
                labels_list.append(actual_winner)
                
            except Exception as e:
                continue
        
        if len(features_list) == 0:
            return np.array([]), np.array([])
        
        return np.array(features_list), np.array(labels_list)
    
    def _build_fighter_versions(self):
        """Build a dictionary of available versions (post-fight stats) for each fighter."""
        print("Building fighter versions database...")
        
        for fighter_name in self.data['FIGHTER'].unique():
            fighter_data = self.data[self.data['FIGHTER'] == fighter_name].copy()
            
            # Sort by date and get postcomp stats for each fight
            fighter_data = fighter_data.sort_values('DATE')
            
            versions = []
            for idx, row in fighter_data.iterrows():
                # Get all available postcomp stats for this fighter
                postcomp_stats = {}
                for col in self.data.columns:
                    if col.startswith('postcomp_') and pd.notna(row[col]):
                        postcomp_stats[col] = row[col]
                
                version_info = {
                    'fight_date': row['DATE'],
                    'opponent': row.get('opp_FIGHTER', 'Unknown'),
                    'result': 'Win' if row['win'] == 1 else 'Loss',
                    'postcomp_elo': row.get('postcomp_elo', None),
                    'postcomp_strike_elo': row.get('postcomp_strike_elo', None),
                    'postcomp_grapple_elo': row.get('postcomp_grapple_elo', None),
                    'age': row.get('age', None),
                    'weight': row.get('WEIGHT', None),
                    'reach': row.get('REACH', None),
                    'height': row.get('HEIGHT', None),
                    'dob': row.get('DOB', None),
                    # Include all postcomp stats
                    **postcomp_stats
                }
                versions.append(version_info)
            
            self.fighter_versions[fighter_name] = versions
        
        print(f"Built versions for {len(self.fighter_versions)} fighters")
    
    def get_fighter_versions(self, fighter_name):
        """
        Get available versions (post-fight stats) for a specific fighter.
        
        Args:
            fighter_name (str): Name of the fighter
            
        Returns:
            list: List of available versions with fight details
        """
        if fighter_name not in self.fighter_versions:
            return []
        
        versions = self.fighter_versions[fighter_name]
        
        # Format versions for display
        formatted_versions = []
        for i, version in enumerate(versions):
            formatted_version = {
                'version_id': i,
                'fight_date': version['fight_date'].strftime('%Y-%m-%d'),
                'opponent': version['opponent'],
                'result': version['result'],
                'postcomp_elo': version['postcomp_elo'],
                'postcomp_strike_elo': version['postcomp_strike_elo'],
                'postcomp_grapple_elo': version['postcomp_grapple_elo'],
                'age': version['age'],
                'weight': version['weight'],
                'reach': version['reach'],
                'height': version['height']
            }
            formatted_versions.append(formatted_version)
        
        return formatted_versions
    
    def get_fighter_recent_postcomp_stats(self, fighter_name, fight_date):
        """
        Get the most recent postcomp stats for a fighter, adjusted for the fight date.
        For fighters with no previous fights, use default values.
        
        Args:
            fighter_name (str): Name of the fighter
            fight_date (datetime): Date of the upcoming fight
            
        Returns:
            dict: Fighter stats adjusted for the prediction date
        """
        if fighter_name not in self.fighter_versions:
            # Fighter not found, return default stats
            return self._get_default_fighter_stats(fight_date)
        
        versions = self.fighter_versions[fighter_name]
        if not versions:
            # No versions available, return default stats
            return self._get_default_fighter_stats(fight_date)
        
        # Get most recent version that happened before the fight date
        latest = None
        for version in reversed(versions):
            if version['fight_date'] < fight_date:
                latest = version
                break
        
        if latest is None:
            # No previous fight found, return default stats
            return self._get_default_fighter_stats(fight_date)
        
        # Calculate age at fight time
        if 'dob' in latest and pd.notna(latest['dob']):
            try:
                age_at_fight = (fight_date - latest['dob']).days / 365.25
            except:
                age_at_fight = latest.get('age', 30)
        else:
            age_at_fight = latest.get('age', 30)
        
        # Apply ELO decay if more than 365 days since last fight
        days_since_last = (fight_date - latest['fight_date']).days
        decay_factor = 1.0
        if days_since_last > 365:
            decay_factor = 0.978 ** (days_since_last / 365)
        
        # Create stats dictionary with all available postcomp stats
        stats = {
            'age': age_at_fight,
            'height': latest.get('height'),
            'weight': latest.get('weight'),
            'reach': latest.get('reach'),
            'dob': latest.get('dob'),
            'fight_date': latest['fight_date'],
            'last_fight_date': latest['fight_date'],
            'result': latest.get('result', 'Unknown'),
            'last_fight_result': latest.get('result', 'Unknown'),
            'last_fight_opponent': latest.get('opponent', 'Unknown')
        }
        
        # Add all postcomp stats with decay applied
        for key, value in latest.items():
            if key.startswith('postcomp_') and pd.notna(value):
                if 'elo' in key and days_since_last > 365:
                    stats[key] = value * decay_factor
                else:
                    stats[key] = value
        
        # Also add precomp stats as fallbacks if postcomp not available
        for key, value in latest.items():
            if key.startswith('precomp_') and pd.notna(value):
                postcomp_key = key.replace('precomp_', 'postcomp_')
                if postcomp_key not in stats or pd.isna(stats[postcomp_key]):
                    if 'elo' in key and days_since_last > 365:
                        stats[postcomp_key] = value * decay_factor
                    else:
                        stats[postcomp_key] = value
        
        # Ensure we have the key stats that the model needs
        if 'postcomp_elo' not in stats or pd.isna(stats['postcomp_elo']):
            stats['postcomp_elo'] = 1500.0
        if 'postcomp_strike_elo' not in stats or pd.isna(stats['postcomp_strike_elo']):
            stats['postcomp_strike_elo'] = 1500.0
        if 'postcomp_grapple_elo' not in stats or pd.isna(stats['postcomp_grapple_elo']):
            stats['postcomp_grapple_elo'] = 1500.0
        
        return stats
    
    def get_fighter_previous_postcomp_stats(self, fighter_name, fight_date):
        """
        Get postcomp stats from the fighter's previous fight (not the current fight).
        
        Args:
            fighter_name (str): Name of the fighter
            fight_date (datetime): Date of the current fight
            
        Returns:
            dict: Fighter stats from their previous fight
        """
        if fighter_name not in self.fighter_versions:
            # Fighter not found, return default stats
            return self._get_default_fighter_stats(fight_date)
        
        versions = self.fighter_versions[fighter_name]
        if not versions:
            # No versions available, return default stats
            return self._get_default_fighter_stats(fight_date)
        
        # Find the version that comes before the current fight date
        previous_version = None
        for version in reversed(versions):
            if version['fight_date'] < fight_date:
                previous_version = version
                break
        
        if previous_version is None:
            # No previous fight found, return default stats
            return self._get_default_fighter_stats(fight_date)
        
        # Calculate age at fight time
        if 'dob' in previous_version and pd.notna(previous_version['dob']):
            try:
                age_at_fight = (fight_date - previous_version['dob']).days / 365.25
            except:
                age_at_fight = previous_version.get('age', 30)
        else:
            age_at_fight = previous_version.get('age', 30)
        
        # Apply ELO decay if more than 365 days since last fight
        days_since_last = (fight_date - previous_version['fight_date']).days
        decay_factor = 1.0
        if days_since_last > 365:
            decay_factor = 0.978 ** (days_since_last / 365)
        
        # Create stats dictionary with all available postcomp stats
        stats = {
            'age': age_at_fight,
            'height': previous_version.get('height'),
            'weight': previous_version.get('weight'),
            'reach': previous_version.get('reach'),
            'dob': previous_version.get('dob'),
            'fight_date': previous_version['fight_date'],
            'last_fight_date': previous_version['fight_date'],
            'result': previous_version.get('result', 'Unknown'),
            'last_fight_result': previous_version.get('result', 'Unknown'),
            'last_fight_opponent': previous_version.get('opponent', 'Unknown')
        }
        
        # Add all postcomp stats with decay applied
        for key, value in previous_version.items():
            if key.startswith('postcomp_') and pd.notna(value):
                if 'elo' in key and days_since_last > 365:
                    stats[key] = value * decay_factor
                else:
                    stats[key] = value
        
        # Also add precomp stats as fallbacks if postcomp not available
        for key, value in previous_version.items():
            if key.startswith('precomp_') and pd.notna(value):
                postcomp_key = key.replace('precomp_', 'postcomp_')
                if postcomp_key not in stats or pd.isna(stats[postcomp_key]):
                    if 'elo' in key and days_since_last > 365:
                        stats[postcomp_key] = value * decay_factor
                    else:
                        stats[postcomp_key] = value
        
        # Ensure we have the key stats that the model needs
        if 'postcomp_elo' not in stats or pd.isna(stats['postcomp_elo']):
            stats['postcomp_elo'] = 1500.0
        if 'postcomp_strike_elo' not in stats or pd.isna(stats['postcomp_strike_elo']):
            stats['postcomp_strike_elo'] = 1500.0
        if 'postcomp_grapple_elo' not in stats or pd.isna(stats['postcomp_grapple_elo']):
            stats['postcomp_grapple_elo'] = 1500.0
        
        return stats
    
    def _get_default_fighter_stats(self, fight_date):
        """Get default fighter stats for fighters with no previous fights."""
        return {
            'age': 30.0,
            'height': 70.0,
            'weight': 170.0,
            'reach': 70.0,
            'dob': None,
            'fight_date': fight_date,
            'last_fight_date': fight_date,
            'result': 'Unknown',
            'last_fight_result': 'Unknown',
            'last_fight_opponent': 'Unknown',
            'postcomp_elo': 1500.0,
            'postcomp_strike_elo': 1500.0,
            'postcomp_grapple_elo': 1500.0
        }
    
    def _create_feature_vector(self, fighter1_stats, fighter2_stats):
        """
        Create a feature vector for fighter1 vs fighter2.
        This maps precomp_ features to postcomp_ stats from previous fights.
        """
        features = {}
        
        # Helper function to get postcomp stat with fallback
        def get_postcomp_stat(stats, stat_name, default=0.0):
            """Get postcomp stat, falling back to precomp if available, then default."""
            # Try postcomp first
            postcomp_key = f'postcomp_{stat_name}'
            if postcomp_key in stats and pd.notna(stats[postcomp_key]):
                value = stats[postcomp_key]
                return float(value) if value is not None else default
            
            # Try precomp as fallback
            precomp_key = f'precomp_{stat_name}'
            if precomp_key in stats and pd.notna(stats[precomp_key]):
                value = stats[precomp_key]
                return float(value) if value is not None else default
            
            return default
        
        # Map each feature column to the appropriate stats
        for col in self.feature_columns:
            if col.startswith('opp_'):
                # For opponent columns, use fighter2's stats
                base_col = col.replace('opp_', '')
                
                if base_col == 'precomp_elo':
                    features[col] = fighter2_stats.get('postcomp_elo', 1500.0) or 1500.0
                elif base_col == 'precomp_strike_elo':
                    features[col] = fighter2_stats.get('postcomp_strike_elo', 1500.0) or 1500.0
                elif base_col == 'precomp_grapple_elo':
                    features[col] = fighter2_stats.get('postcomp_grapple_elo', 1500.0) or 1500.0
                elif base_col == 'age':
                    features[col] = float(fighter2_stats.get('age', 30.0)) or 30.0
                elif base_col == 'REACH':
                    features[col] = float(fighter2_stats.get('reach', 70.0)) or 70.0
                elif base_col == 'WEIGHT':
                    features[col] = float(fighter2_stats.get('weight', 170.0)) or 170.0
                elif base_col == 'HEIGHT':
                    features[col] = float(fighter2_stats.get('height', 70.0)) or 70.0
                elif base_col == 'precomp_elo_change_5':
                    features[col] = get_postcomp_stat(fighter2_stats, 'elo_change_5', 0.0)
                elif base_col == 'precomp_tdavg':
                    features[col] = get_postcomp_stat(fighter2_stats, 'tdavg', 1.5)
                elif base_col == 'precomp_tddef':
                    features[col] = get_postcomp_stat(fighter2_stats, 'tddef', 0.5)
                elif base_col == 'precomp_sapm5':
                    features[col] = get_postcomp_stat(fighter2_stats, 'sapm5', 1.2)
                elif base_col == 'precomp_sapm':
                    features[col] = get_postcomp_stat(fighter2_stats, 'sapm', 1.2)
                elif base_col == 'precomp_totalacc_perc3':
                    features[col] = get_postcomp_stat(fighter2_stats, 'totalacc_perc3', 0.45)
                elif base_col == 'precomp_totalacc_perc':
                    features[col] = get_postcomp_stat(fighter2_stats, 'totalacc_perc', 0.45)
                elif base_col == 'precomp_legacc_perc5':
                    features[col] = get_postcomp_stat(fighter2_stats, 'legacc_perc5', 0.3)
                elif base_col == 'precomp_clinchacc_perc5':
                    features[col] = get_postcomp_stat(fighter2_stats, 'clinchacc_perc5', 0.35)
                elif base_col == 'precomp_winsum3':
                    features[col] = get_postcomp_stat(fighter2_stats, 'winsum3', 2.0)
                elif base_col == 'precomp_losssum5':
                    features[col] = get_postcomp_stat(fighter2_stats, 'losssum5', 1.0)
                elif base_col == 'precomp_headacc_perc3':
                    features[col] = get_postcomp_stat(fighter2_stats, 'headacc_perc3', 0.4)
                elif base_col == 'precomp_groundacc_perc5':
                    features[col] = get_postcomp_stat(fighter2_stats, 'groundacc_perc5', 0.25)
                else:
                    # For other stats, try to get from fighter2_stats or use default
                    value = get_postcomp_stat(fighter2_stats, base_col.replace('precomp_', ''), 0.0)
                    features[col] = float(value) if value is not None else 0.0
            else:
                # Regular columns use fighter1's stats
                if col == 'precomp_elo':
                    features[col] = fighter1_stats.get('postcomp_elo', 1500.0) or 1500.0
                elif col == 'precomp_strike_elo':
                    features[col] = fighter1_stats.get('postcomp_strike_elo', 1500.0) or 1500.0
                elif col == 'precomp_grapple_elo':
                    features[col] = fighter1_stats.get('postcomp_grapple_elo', 1500.0) or 1500.0
                elif col == 'age':
                    features[col] = float(fighter1_stats.get('age', 30.0)) or 30.0
                elif col == 'REACH':
                    features[col] = float(fighter1_stats.get('reach', 70.0)) or 70.0
                elif col == 'WEIGHT':
                    features[col] = float(fighter1_stats.get('weight', 170.0)) or 170.0
                elif col == 'HEIGHT':
                    features[col] = float(fighter1_stats.get('height', 70.0)) or 70.0
                elif col == 'precomp_elo_change_5':
                    features[col] = get_postcomp_stat(fighter1_stats, 'elo_change_5', 0.0)
                elif col == 'precomp_tdavg':
                    features[col] = get_postcomp_stat(fighter1_stats, 'tdavg', 1.5)
                elif col == 'precomp_tddef':
                    features[col] = get_postcomp_stat(fighter1_stats, 'tddef', 0.5)
                elif col == 'precomp_sapm5':
                    features[col] = get_postcomp_stat(fighter1_stats, 'sapm5', 1.2)
                elif col == 'precomp_sapm':
                    features[col] = get_postcomp_stat(fighter1_stats, 'sapm', 1.2)
                elif col == 'precomp_totalacc_perc3':
                    features[col] = get_postcomp_stat(fighter1_stats, 'totalacc_perc3', 0.45)
                elif col == 'precomp_totalacc_perc':
                    features[col] = get_postcomp_stat(fighter1_stats, 'totalacc_perc', 0.45)
                elif col == 'precomp_legacc_perc5':
                    features[col] = get_postcomp_stat(fighter1_stats, 'legacc_perc5', 0.3)
                elif col == 'precomp_clinchacc_perc5':
                    features[col] = get_postcomp_stat(fighter1_stats, 'clinchacc_perc5', 0.35)
                elif col == 'precomp_winsum3':
                    features[col] = get_postcomp_stat(fighter1_stats, 'winsum3', 2.0)
                elif col == 'precomp_losssum5':
                    features[col] = get_postcomp_stat(fighter1_stats, 'losssum5', 1.0)
                elif col == 'precomp_headacc_perc3':
                    features[col] = get_postcomp_stat(fighter1_stats, 'headacc_perc3', 0.4)
                elif col == 'precomp_groundacc_perc5':
                    features[col] = get_postcomp_stat(fighter1_stats, 'groundacc_perc5', 0.25)
                else:
                    # For other stats, try to get from fighter1_stats or use default
                    value = get_postcomp_stat(fighter1_stats, col.replace('precomp_', ''), 0.0)
                    features[col] = float(value) if value is not None else 0.0
        
        # Calculate derived features that require both fighters' stats
        if 'age_ratio_difference' in self.feature_columns:
            age1 = float(fighter1_stats.get('age', 30.0))
            age2 = float(fighter2_stats.get('age', 30.0))
            if age1 > 0 and age2 > 0:
                features['age_ratio_difference'] = age1 / age2 - 1
            else:
                features['age_ratio_difference'] = 0.0
        
        if 'opp_age_ratio_difference' in self.feature_columns:
            age1 = float(fighter1_stats.get('age', 30.0))
            age2 = float(fighter2_stats.get('age', 30.0))
            if age1 > 0 and age2 > 0:
                features['opp_age_ratio_difference'] = age2 / age1 - 1
            else:
                features['opp_age_ratio_difference'] = 0.0
        
        # Convert to list in the correct order and ensure all values are numeric
        feature_list = []
        for col in self.feature_columns:
            value = features.get(col, 0.0)
            # Ensure the value is numeric
            try:
                feature_list.append(float(value))
            except (ValueError, TypeError):
                print(f"Warning: Non-numeric value '{value}' for feature '{col}', using 0.0")
                feature_list.append(0.0)
        
        return feature_list
    
    def predict_fight(self, fighter1_name, fighter2_name, weight_class, fight_date):
        """
        Predict the winner of a fight between two fighters.
        
        Args:
            fighter1_name (str): Name of the first fighter
            fighter2_name (str): Name of the second fighter
            weight_class (int): Weight class of the fight
            fight_date (datetime): Date of the fight
            
        Returns:
            dict: Prediction results with winner, probability, and details
        """
        print(f"\nPredicting: {fighter1_name} vs {fighter2_name}")
        print(f"Weight class: {self.weight_class_names.get(weight_class, f'Weight Class {weight_class}')}")
        print(f"Fight date: {fight_date.strftime('%Y-%m-%d')}")
        
        # Get fighter data
        fighter1_data = self.data[self.data['FIGHTER'] == fighter1_name]
        fighter2_data = self.data[self.data['FIGHTER'] == fighter2_name]
        
        if len(fighter1_data) == 0:
            raise ValueError(f"No data found for fighter '{fighter1_name}'")
        if len(fighter2_data) == 0:
            raise ValueError(f"No data found for fighter '{fighter2_name}'")
        
        # Get fighter stats from their most recent fights
        fighter1_stats = self.get_fighter_recent_postcomp_stats(fighter1_name, fight_date)
        fighter2_stats = self.get_fighter_recent_postcomp_stats(fighter2_name, fight_date)
        
        if fighter1_stats is None or fighter2_stats is None:
            raise ValueError(f"Could not get stats for one or both fighters")
        
        print(f"Using {fighter1_name} stats from {fighter1_stats['fight_date']} vs {fighter2_name} stats from {fighter2_stats['fight_date']}")
        print(f"Fighter1 age at fight: {fighter1_stats['age']:.1f}")
        print(f"Fighter2 age at fight: {fighter2_stats['age']:.1f}")
        
        # Create feature vector
        features = self._create_feature_vector(fighter1_stats, fighter2_stats)
        
        # Convert to numpy array
        features = np.array(features, dtype=float)
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Apply imputation
        features_imputed = self.imputer.transform(features)
        
        # Apply scaling
        features_scaled = self.scaler.transform(features_imputed)
        
        # Apply clipping
        features_clipped = np.clip(features_scaled, -10, 10)
        
        # Make prediction
        raw_probs = self.model.predict_proba(features_clipped)[0]
        prob_fighter1 = raw_probs[1]  # Class 1 = fighter1 wins
        
        # Clip probabilities to prevent extreme values
        prob_fighter1 = np.clip(prob_fighter1, 0.15, 0.85)
        prob_fighter2 = 1 - prob_fighter1
        
        print(f"Probability: {fighter1_name} {prob_fighter1:.4f} vs {fighter2_name} {prob_fighter2:.4f}")
        
        # Determine winner
        if prob_fighter1 > prob_fighter2:
            winner = fighter1_name
            confidence = prob_fighter1
        else:
            winner = fighter2_name
            confidence = prob_fighter2
        
        # Convert probability to American odds
        def prob_to_american_odds(p):
            if p <= 0 or p >= 1:
                return "N/A"
            if p >= 0.5:
                odds = - (p / (1 - p)) * 100
            else:
                odds = ((1 - p) / p) * 100
            return int(np.sign(odds) * np.round(abs(odds)))
        
        odds = prob_to_american_odds(prob_fighter1)
        
        result = {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'weight_class': weight_class,
            'weight_class_name': self.weight_class_names.get(weight_class, f'Weight Class {weight_class}'),
            'fight_date': fight_date.strftime('%Y-%m-%d'),
            'predicted_winner': winner,
            'fighter1_win_probability': prob_fighter1,
            'fighter2_win_probability': prob_fighter2,
            'confidence': confidence,
            'american_odds': odds,
            'fighter1_stats': {
                'elo': fighter1_stats['postcomp_elo'],
                'striking_elo': fighter1_stats['postcomp_strike_elo'],
                'grapple_elo': fighter1_stats['postcomp_grapple_elo'],
                'age': fighter1_stats['age'],
                'height': fighter1_stats['height'],
                'weight': fighter1_stats['weight'],
                'reach': fighter1_stats['reach'],
                'last_fight_date': fighter1_stats['fight_date'].strftime('%Y-%m-%d'),
                'last_fight_result': fighter1_stats.get('result', 'Unknown'),
                'last_fight_opponent': fighter1_stats.get('opponent', 'Unknown')
            },
            'fighter2_stats': {
                'elo': fighter2_stats['postcomp_elo'],
                'striking_elo': fighter2_stats['postcomp_strike_elo'],
                'grapple_elo': fighter2_stats['postcomp_grapple_elo'],
                'age': fighter2_stats['age'],
                'height': fighter2_stats['height'],
                'weight': fighter2_stats['weight'],
                'reach': fighter2_stats['reach'],
                'last_fight_date': fighter2_stats['fight_date'].strftime('%Y-%m-%d'),
                'last_fight_result': fighter2_stats.get('result', 'Unknown'),
                'last_fight_opponent': fighter2_stats.get('opponent', 'Unknown')
            }
        }
        
        return result
    
    def get_available_fighters(self):
        """Get list of available fighters."""
        return sorted(self.data['FIGHTER'].unique())
    
    def get_available_weight_classes(self):
        """Get list of available weight classes."""
        weight_classes = []
        if 'weight_of_fight' in self.data.columns:
            available_weight_classes = sorted(self.data['weight_of_fight'].dropna().unique())
            for weight_class in available_weight_classes:
                weight_class_info = {
                    'value': weight_class,
                    'name': self.weight_class_names.get(weight_class, f'Weight Class {weight_class}'),
                    'display_name': self.weight_class_names.get(weight_class, f'Weight Class {weight_class}'),
                    'fight_count': len(self.data[self.data['weight_of_fight'] == weight_class])
                }
                weight_classes.append(weight_class_info)
        return weight_classes
    
    def validate_model_on_test_data(self):
        """
        Validate the model by running predictions on test data.
        This mimics real-world prediction by using postcomp stats from previous fights
        to predict current fights, with proper ELO decay and age adjustments.
        
        Returns:
            dict: Validation results with accuracy metrics and detailed results
        """
        print(f"\nValidating model on test data...")
        print("This validation mimics real-world prediction using postcomp data from previous fights.")
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        # Get test data fights
        test_fights = self.test_data.copy()
        
        # Sort by date to ensure we process chronologically
        test_fights = test_fights.sort_values('DATE')
        
        print(f"Validating on {len(test_fights)} test fights...")
        
        for idx, fight in test_fights.iterrows():
            fighter1_name = fight['FIGHTER']
            fighter2_name = fight.get('opp_FIGHTER')
            actual_winner = fighter1_name if fight['win'] == 1 else fighter2_name
            actual_winner_is_fighter1 = fight['win'] == 1
            fight_date = fight['DATE']
            
            if not fighter2_name:
                continue
            
            try:
                # Get fighter stats from their most recent fight before this fight date
                fighter1_stats = self.get_fighter_recent_postcomp_stats(fighter1_name, fight_date)
                fighter2_stats = self.get_fighter_recent_postcomp_stats(fighter2_name, fight_date)
                
                if fighter1_stats is None or fighter2_stats is None:
                    continue
                
                # Create feature vector
                features = self._create_feature_vector(fighter1_stats, fighter2_stats)
                
                # Convert to numpy array
                features = np.array(features, dtype=float)
                
                # Reshape for prediction
                features = features.reshape(1, -1)
                
                # Apply imputation
                features_imputed = self.imputer.transform(features)
                
                # Apply scaling
                features_scaled = self.scaler.transform(features_imputed)
                
                # Apply clipping
                features_clipped = np.clip(features_scaled, -10, 10)
                
                # Make prediction
                raw_probs = self.model.predict_proba(features_clipped)[0]
                prob_fighter1 = raw_probs[1]
                
                # Clip probabilities
                prob_fighter1 = np.clip(prob_fighter1, 0.15, 0.85)
                
                # Determine predicted winner
                predicted_winner_is_fighter1 = prob_fighter1 > 0.5
                predicted_winner = fighter1_name if predicted_winner_is_fighter1 else fighter2_name
                
                # Check if prediction was correct
                correct = (predicted_winner_is_fighter1 == actual_winner_is_fighter1)
                if correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Store result
                result = {
                    'fight_date': fight['DATE'].strftime('%Y-%m-%d'),
                    'fighter1': fighter1_name,
                    'fighter2': fighter2_name,
                    'actual_winner': actual_winner,
                    'predicted_winner': predicted_winner,
                    'fighter1_win_probability': prob_fighter1,
                    'fighter2_win_probability': 1 - prob_fighter1,
                    'correct': correct,
                    'confidence': max(prob_fighter1, 1 - prob_fighter1)
                }
                results.append(result)
                
                # Progress indicator
                if total_predictions % 50 == 0:
                    current_acc = correct_predictions / total_predictions
                    print(f"  Processed {total_predictions} fights, current accuracy: {current_acc:.3f}")
                
            except Exception as e:
                print(f"  Error predicting fight {fighter1_name} vs {fighter2_name}: {e}")
                continue
        
        # Calculate final metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate additional metrics
        high_confidence_results = [r for r in results if r['confidence'] >= 0.7]
        high_confidence_accuracy = 0
        if high_confidence_results:
            high_confidence_correct = sum(1 for r in high_confidence_results if r['correct'])
            high_confidence_accuracy = high_confidence_correct / len(high_confidence_results)
        
        validation_results = {
            'total_fights': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'high_confidence_fights': len(high_confidence_results),
            'high_confidence_accuracy': high_confidence_accuracy,
            'sample_results': results[:10],  # First 10 results for inspection
            'validation_method': 'postcomp_stats_from_previous_fights'
        }
        
        print(f"\nValidation complete!")
        print(f"Total fights: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Overall accuracy: {accuracy:.3f}")
        print(f"High confidence fights: {len(high_confidence_results)}")
        print(f"High confidence accuracy: {high_confidence_accuracy:.3f}")
        
        return validation_results


def main():
    """Main function to demonstrate the UFC Fight Predictor."""
    print("🥊 UFC Fight Predictor App")
    print("=" * 50)
    
    # Initialize predictor
    try:
        predictor = UFCFightPredictor()
        print(f"✅ Predictor initialized successfully!")
        print(f"📊 Model accuracy: {predictor.test_accuracy:.3f}")
        print(f"👥 Available fighters: {len(predictor.get_available_fighters())}")
        print(f"⚖️  Available weight classes: {len(predictor.get_available_weight_classes())}")
        
        # Example prediction: Nassourdine Imavov vs Caio Borralho
        print("\n" + "=" * 50)
        print("🎯 EXAMPLE PREDICTION")
        print("=" * 50)
        
        fighter1 = "Nassourdine Imavov"
        fighter2 = "Caio Borralho"
        weight_class = 10  # Middleweight (185 lbs)
        fight_date = datetime.now() + timedelta(days=30)  # 1 month from now
        
        try:
            result = predictor.predict_fight(fighter1, fighter2, weight_class, fight_date)
            
            print(f"\n🥊 FIGHT PREDICTION:")
            print(f"Fight: {result['fighter1']} vs {result['fighter2']}")
            print(f"Weight Class: {result['weight_class_name']}")
            print(f"Fight Date: {result['fight_date']}")
            print(f"Predicted Winner: {result['predicted_winner']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"American Odds: {result['american_odds']}")
            
            print(f"\n📊 FIGHTER COMPARISON:")
            print(f"{result['fighter1']}:")
            for stat, value in result['fighter1_stats'].items():
                print(f"  {stat}: {value}")
            
            print(f"\n{result['fighter2']}:")
            for stat, value in result['fighter2_stats'].items():
                print(f"  {stat}: {value}")
                
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            print("Make sure the fighter names exist in your dataset.")
        
        # Validate model on test data
        print("\n" + "=" * 50)
        print("🔍 MODEL VALIDATION")
        print("=" * 50)
        
        validation_results = predictor.validate_model_on_test_data()
        
        print(f"\n📈 VALIDATION RESULTS:")
        print(f"Total test fights: {validation_results['total_fights']}")
        print(f"Correct predictions: {validation_results['correct_predictions']}")
        print(f"Overall accuracy: {validation_results['accuracy']:.3f}")
        print(f"High confidence fights: {validation_results['high_confidence_fights']}")
        print(f"High confidence accuracy: {validation_results['high_confidence_accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
