import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    """
    Load and prepare the filtered dataset for model training.
    """
    # Define file path
    data_path = os.path.join('data', 'tmp', 'interleaved_with_enhanced_elo_filtered.csv')
    
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Define features to include in the model
    elo_columns = [
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
        'weight_avg3',
        'opp_precomp_sigstr_pm5', 'opp_precomp_tdavg5', 'opp_precomp_sapm5', 'opp_precomp_subavg5',
        'opp_precomp_tddef5', 'opp_precomp_sigstr_perc5', 'opp_precomp_strdef5', 'opp_precomp_tdacc_perc5',
        'opp_precomp_totalacc_perc5', 'opp_precomp_headacc_perc5','opp_precomp_bodyacc_perc5','opp_precomp_legacc_perc5',
        'opp_precomp_distacc_perc5','opp_precomp_clinchacc_perc5','opp_precomp_groundacc_perc5',
        'opp_precomp_winsum5', 'opp_precomp_losssum5','opp_precomp_elo_change_5',
        'opp_age', 'opp_HEIGHT', 'opp_WEIGHT', 'opp_REACH','opp_weightindex',
        'opp_precomp_sigstr_pm', 'opp_precomp_tdavg', 'opp_precomp_sapm', 'opp_precomp_subavg',
        'opp_precomp_tddef', 'opp_precomp_sigstr_perc', 'opp_precomp_strdef', 'opp_precomp_tdacc_perc',
        'opp_precomp_totalacc_perc', 'opp_precomp_headacc_perc','opp_precomp_bodyacc_perc','opp_precomp_legacc_perc',
        'opp_precomp_distacc_perc','opp_precomp_clinchacc_perc','opp_precomp_groundacc_perc',
        'opp_precomp_winsum', 'opp_precomp_losssum', 'opp_precomp_elo',
        'opp_precomp_sigstr_pm3', 'opp_precomp_tdavg3', 'opp_precomp_sapm3', 'opp_precomp_subavg3',
        'opp_precomp_tddef3', 'opp_precomp_sigstr_perc3', 'opp_precomp_strdef3', 'opp_precomp_tdacc_perc3',
        'opp_precomp_totalacc_perc3', 'opp_precomp_headacc_perc3','opp_precomp_bodyacc_perc3','opp_precomp_legacc_perc3',
        'opp_precomp_distacc_perc3','opp_precomp_clinchacc_perc3','opp_precomp_groundacc_perc3',
        'opp_precomp_winsum3', 'opp_precomp_losssum3','opp_precomp_elo_change_3','opp_weight_avg3'
    ]
    
    # Check which columns actually exist in the dataset
    available_cols = [col for col in elo_columns if col in data.columns]
    missing_cols = [col for col in elo_columns if col not in data.columns]
    
    if missing_cols:
        print(f"Warning: The following columns are missing from the dataset: {missing_cols}")
    
    # Prepare features and target
    X = data[available_cols]
    y = data['result']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, available_cols

def build_model(input_shape):
    """
    Build a neural network model for fight prediction.
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    """
    Train the fight prediction model.
    """
    # Load and prepare data
    X_train, X_test, y_train, y_test, available_cols = load_and_prepare_data()
    
    # Build model
    model = build_model(X_train.shape[1])
    
    # Set up callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
    # Save model
    model.save('saved_models/enhanced_elo_model.h5')
    print("Model saved to saved_models/enhanced_elo_model.h5")
    
    # Save feature names
    with open('saved_models/feature_names.txt', 'w') as f:
        for feature in available_cols:
            f.write(f"{feature}\n")
    
    return model, history

if __name__ == "__main__":
    train()