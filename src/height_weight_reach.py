import pandas as pd
import numpy as np
import sqlite3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class hwr_process:
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name
        self.connection = sqlite3.connect(db_path)
        self.models = {}

    def _convert_height_to_inches(self, height):
        if height == "--" or pd.isna(height):
            return None
        height = str(height).split(',')[0].replace('"', '').replace("\\", "")
        if ' ' in height:
            feet, inches = map(float, height.replace("'", "").split())
        else:
            feet, inches = float(height.replace("'", "")), 0
        return feet * 12 + inches

    def _convert_weight_to_lbs(self, weight):
        print(f"Processing: {weight} ({type(weight)})")  # Debugging
        if pd.isna(weight) or weight == "--":
            return None
        try:
            return float(str(weight).replace(' lbs.', '').replace('lbs.', '').strip())
        except ValueError:
            return None
            
    def preprocess_data(self):
        data = pd.read_sql_query(f"SELECT * FROM {self.table_name}", self.connection)
        data['HEIGHT'] = data['HEIGHT'].apply(self._convert_height_to_inches)
        data['WEIGHT'] = data['WEIGHT'].apply(self._convert_weight_to_lbs)
        data['REACH'] = data['REACH'].astype(str).str.replace('"', '').replace('--', None).astype(float)
        return data.dropna(subset=['HEIGHT', 'WEIGHT', 'REACH'], how='all')

    def train_model(self, feature_columns, target_column):
        data = self.preprocess_data()
        train_data = data.dropna(subset=feature_columns + [target_column])
        X = train_data[feature_columns].to_numpy()
        y = train_data[target_column].to_numpy()

        # Add bias term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Calculate weights using the normal equation
        weights = np.linalg.pinv(X.T @ X) @ X.T @ y

        self.models[target_column] = weights

        # Calculate predictions for evaluation
        predictions = X @ weights
        mse = np.mean((y - predictions) ** 2)
        r_squared = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        return weights, mse, r_squared

    def predict_missing_values(self, feature_columns, target_column):
        if target_column not in self.models:
            raise ValueError(f"Model for {target_column} has not been trained.")

        data = self.preprocess_data()
        missing_data = data[data[target_column].isna() & data[feature_columns].notna().all(axis=1)]
        X_missing = missing_data[feature_columns].to_numpy()

        # Add bias term to X_missing
        X_missing = np.c_[np.ones(X_missing.shape[0]), X_missing]
        predictions = X_missing @ self.models[target_column]
        data.loc[missing_data.index, target_column] = np.round(predictions, 1)

        return data

    def save_model(self, target_column, file_path):
        if target_column not in self.models:
            raise ValueError(f"Model for {target_column} has not been trained.")
        with open(file_path, 'wb') as file:
            pickle.dump(self.models[target_column], file)

    def load_model(self, target_column, file_path):
        with open(file_path, 'rb') as file:
            self.models[target_column] = pickle.load(file)

    def close_connection(self):
        self.connection.close()
