# Modify the model.py file to include a hyperparameter tuning method using Optuna and MLflow
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import scaler
import random
import mlflow
import mlflow.tensorflow
import optuna
import shap

class DNNFightPredictor:
    def __init__(self, file_path, test_size=0.2, layer_sizes=[64, 32], dropout_rate=0.2, l2_reg=0.01, epochs=50, batch_size=32, patience=5, learning_rate=0.05, temperature=2.0):
        self.file_path = file_path
        self.test_size = test_size
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.temperature = temperature  # Temperature for sigmoid scaling

        seed_value = 21
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        self.data = pd.read_csv(self.file_path)

        self.X, self.y = self.load_and_preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def load_and_preprocess_data(self):
        # Drop rows with missing values
        self.data = self.data.dropna()
        
        # Convert DATE to datetime
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])
        
        # Filter data from 2010-01-01 onwards
        filtered_data = self.data[self.data['DATE'] >= '2009-01-01']
        
        # Store the filtered data for splitting by date later
        self.filtered_data = filtered_data
        
        # Store column names for later use
        self.elo_columns = [
            'age', 'HEIGHT', 'WEIGHT', 'REACH', 'weightindex',
            'precomp_sigstr_pm', 'precomp_tdavg', 'precomp_sapm', 'precomp_subavg',
            'precomp_tddef', 'precomp_sigstr_perc', 'precomp_strdef', 'precomp_tdacc_perc',
            'precomp_totalacc_perc', 'precomp_headacc_perc', 'precomp_bodyacc_perc', 'precomp_legacc_perc',
            'precomp_distacc_perc','precomp_clinchacc_perc','precomp_groundacc_perc',
            'precomp_winsum', 'precomp_losssum','precomp_elo',
            'precomp_sigstr_pm5', 'precomp_tdavg5', 'precomp_sapm5', 'precomp_subavg5',
            'precomp_tddef5', 'precomp_sigstr_perc5', 'precomp_strdef5', 'precomp_tdacc_perc5',
            'precomp_totalacc_perc5', 'precomp_headacc_perc5', 'precomp_bodyacc_perc5', 'precomp_legacc_perc5',
            'precomp_distacc_perc5','precomp_clinchacc_perc5','precomp_groundacc_perc5',
            'precomp_winsum5', 'precomp_losssum5','precomp_elo_change_5',
            'precomp_sigstr_pm3', 'precomp_tdavg3', 'precomp_sapm3', 'precomp_subavg3',
            'precomp_tddef3', 'precomp_sigstr_perc3', 'precomp_strdef3', 'precomp_tdacc_perc3',
            'precomp_totalacc_perc3', 'precomp_headacc_perc3', 'precomp_bodyacc_perc3', 'precomp_legacc_perc3',
            'precomp_distacc_perc3','precomp_clinchacc_perc3','precomp_groundacc_perc3',
            'precomp_winsum3', 'precomp_losssum3','precomp_elo_change_3',
            'opp_age', 'opp_HEIGHT', 'opp_WEIGHT', 'opp_REACH','opp_weightindex', 'opp_weight_of_fight',
            'opp_precomp_sigstr_pm', 'opp_precomp_tdavg', 'opp_precomp_sapm', 'opp_precomp_subavg',
            'opp_precomp_tddef', 'opp_precomp_sigstr_perc', 'opp_precomp_strdef', 'opp_precomp_tdacc_perc',
            'opp_precomp_totalacc_perc', 'opp_precomp_headacc_perc','opp_precomp_bodyacc_perc','opp_precomp_legacc_perc',
            'opp_precomp_distacc_perc','opp_precomp_clinchacc_perc','opp_precomp_groundacc_perc',
            'opp_precomp_winsum', 'opp_precomp_losssum', 'opp_precomp_elo',
            'opp_precomp_sigstr_pm5', 'opp_precomp_tdavg5', 'opp_precomp_sapm5', 'opp_precomp_subavg5',
            'opp_precomp_tddef5', 'opp_precomp_sigstr_perc5', 'opp_precomp_strdef5', 'opp_precomp_tdacc_perc5',
            'opp_precomp_totalacc_perc5', 'opp_precomp_headacc_perc5','opp_precomp_bodyacc_perc5','opp_precomp_legacc_perc5',
            'opp_precomp_distacc_perc5','opp_precomp_clinchacc_perc5','opp_precomp_groundacc_perc5',
            'opp_precomp_winsum5', 'opp_precomp_losssum5','opp_precomp_elo_change_5',
            'opp_precomp_sigstr_pm3', 'opp_precomp_tdavg3', 'opp_precomp_sapm3', 'opp_precomp_subavg3',
            'opp_precomp_tddef3', 'opp_precomp_sigstr_perc3', 'opp_precomp_strdef3', 'opp_precomp_tdacc_perc3',
            'opp_precomp_totalacc_perc3', 'opp_precomp_headacc_perc3','opp_precomp_bodyacc_perc3','opp_precomp_legacc_perc3',
            'opp_precomp_distacc_perc3','opp_precomp_clinchacc_perc3','opp_precomp_groundacc_perc3',
            'opp_precomp_winsum3', 'opp_precomp_losssum3','opp_precomp_elo_change_3'
        ]
        
        elo_columns = ['precomp_elo','precomp_elo_change_3', 'precomp_elo_change_5', 'opp_precomp_elo','opp_precomp_elo_change_3', 'opp_precomp_elo_change_5',]

        # Make sure all columns in elo_columns exist, if not create them with default value 0
        for col in self.elo_columns:
            if col not in filtered_data.columns:
                filtered_data[col] = 0
                
        # Convert all columns to numeric to avoid type issues
        for col in self.elo_columns:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
            
        # Fill NaN values with 0
        filtered_data[self.elo_columns] = filtered_data[self.elo_columns].fillna(0)

        # Extract features from filtered data
        X = filtered_data[self.elo_columns]
        
        # Convert result column to numeric and ensure it's binary (0 or 1)
        # This is crucial to fix the "Invalid dtype: object" error in TensorFlow
        filtered_data['result'] = pd.to_numeric(filtered_data['result'], errors='coerce').fillna(0).astype(int)
        y = filtered_data['result']
        
        # Save feature names to a file for later use using absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        feature_names_path = os.path.join(project_root, 'saved_models', 'feature_names.txt')
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(feature_names_path), exist_ok=True)
        
        with open(feature_names_path, 'w') as f:
            for col in self.elo_columns:
                f.write(f"{col}\n")
                
        return X, y
    
    def split_data(self):
        """
        Split the data based on date, using all data from 2009-01-01 up to the most recent date minus one year
        for training, and the latest year of data for testing.
        """
        # Get the most recent date in the dataset
        most_recent_date = self.filtered_data['DATE'].max()
        
        # Calculate the cutoff date (one year before the most recent date)
        one_year = pd.DateOffset(years=1)
        cutoff_date = most_recent_date - one_year
        
        print(f"Most recent date in dataset: {most_recent_date}")
        print(f"Training cutoff date: {cutoff_date}")
        print(f"Using data from 2009-01-01 to {cutoff_date} for training")
        print(f"Using data from {cutoff_date} to {most_recent_date} for testing")
        
        # Split data by date
        train_data = self.filtered_data[self.filtered_data['DATE'] <= cutoff_date]
        test_data = self.filtered_data[self.filtered_data['DATE'] > cutoff_date]
        
        # Check that we have enough data in both sets
        print(f"Training set size: {len(train_data)}")
        print(f"Testing set size: {len(test_data)}")
        
        # Extract features and targets using the stored column names
        train_X = train_data[self.elo_columns]
        train_y = train_data['result']


        
        test_X = test_data[self.elo_columns]
        test_y = test_data['result']
        
        # Standardize features based on training data
        self.scaler = StandardScaler()
        train_X_scaled = self.scaler.fit_transform(train_X)
        test_X_scaled = self.scaler.transform(test_X)
        
        # Save the scaler for later use in predictions
        from joblib import dump
        import os
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  'saved_models', 'feature_scaler.joblib')
        dump(self.scaler, scaler_path)
        print(f"Saved feature scaler to {scaler_path}")
        
        return train_X_scaled, test_X_scaled, train_y, test_y
    
    