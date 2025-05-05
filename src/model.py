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

        # Use absolute paths for MLflow tracking
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        tracking_uri = os.path.join(project_root, "saved_models")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("UFC_Fight_Prediction")

        self.X, self.y = self.load_and_preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.model = self.build_dnn_model()

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
        '''
        self.main_stats_cols = [
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
        '''

        self.elo_columns = ['precomp_elo','precomp_elo_change_3', 'precomp_elo_change_5', 'opp_precomp_elo','opp_precomp_elo_change_3', 'opp_precomp_elo_change_5',]

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
        #save the test data to a file
        test_X.to_csv('../data/tmp/test_data.csv', index=False)
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

    def build_dnn_model(self):
        dnn_model = models.Sequential()
        dnn_model.add(layers.Input(shape=(self.X_train.shape[1],)))

        for size in self.layer_sizes:
            dnn_model.add(layers.Dense(size, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg)))
            dnn_model.add(layers.Dropout(self.dropout_rate))

        # Using a modified activation for better probability separation
        # We'll use a custom layer that applies temperature scaling
        dnn_model.add(layers.Dense(1, activation=None))  # No activation yet
        # Apply temperature scaling to get more separated probabilities
        # Temperature < 1 makes probabilities more extreme
        # Temperature > 1 makes probabilities more moderate
        dnn_model.add(layers.Lambda(lambda x: tf.sigmoid(x / self.temperature)))

        dnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                          loss='binary_crossentropy', metrics=['accuracy'])

        return dnn_model

    def hyperparameter_tuning(self, n_trials=50):
        def objective(trial):
            # Define the hyperparameter search space with updated Optuna syntax
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)
            layer_sizes = [trial.suggest_int(f"layer_size_{i}", 16, 128, step=16) for i in range(4)]

            try:
                with mlflow.start_run():
                    mlflow.log_param("learning_rate", learning_rate)
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("dropout_rate", dropout_rate)
                    mlflow.log_param("l2_reg", l2_reg)
                    mlflow.log_param("layer_sizes", layer_sizes)

                    # Build and train model
                    model = models.Sequential()
                    model.add(layers.Input(shape=(self.X_train.shape[1],)))
                    for size in layer_sizes:
                        model.add(layers.Dense(size, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
                        model.add(layers.Dropout(dropout_rate))
                    model.add(layers.Dense(1, activation='sigmoid'))

                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                  loss='binary_crossentropy', metrics=['accuracy'])

                    # Create a true validation set for model development
                    # Instead of using test data for validation during training
                    # This avoids data leakage between tuning and final evaluation
                    from sklearn.model_selection import train_test_split
                    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
                        self.X_train, self.y_train, test_size=0.2, random_state=42
                    )
                    
                    early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

                    # Use the validation split of training data instead of test data
                    # This prevents information leakage during model selection
                    history = model.fit(X_train_subset, y_train_subset, 
                                        validation_data=(X_val, y_val),
                                        epochs=self.epochs, batch_size=batch_size,
                                        callbacks=[early_stop], verbose=0)

                    # Evaluate on validation data, not test data
                    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

                    mlflow.log_metric("validation_loss", val_loss)
                    mlflow.log_metric("validation_accuracy", val_accuracy)
                    
                    # Store the model and metrics for this trial
                    trial.set_user_attr("model", model)
                    trial.set_user_attr("val_accuracy", val_accuracy)
                    trial.set_user_attr("val_loss", val_loss)
                    
                    # Also evaluate the model on the test set to check for potential
                    # generalization issues during the optimization process
                    test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
                    mlflow.log_metric("test_loss", test_loss)
                    mlflow.log_metric("test_accuracy", test_accuracy)
                    trial.set_user_attr("test_accuracy", test_accuracy)
                    trial.set_user_attr("test_loss", test_loss)
                    
                    # Optionally, log the generalization gap to help identify models that overfit
                    gen_gap = val_accuracy - test_accuracy
                    mlflow.log_metric("generalization_gap", gen_gap)
                    trial.set_user_attr("generalization_gap", gen_gap)

                    # We still return validation accuracy as the optimization metric
                    # But now we're tracking test performance to be aware of overfitting
                    return val_accuracy  # Maximizing accuracy
            except Exception as e:
                print(f"Trial failed with error: {str(e)}")
                # Return a very low score to mark this trial as failed
                return 0.0

        # Run the hyperparameter tuning with error handling
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            print("Best Hyperparameters Found:", best_params)
            
            # Save best parameters to a file for future reference
            import json
            params_path = os.path.join(project_root, 'saved_models', 'best_params.json')
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
                
            # Check all trials to find the one with the best generalization performance
            # First get the standard best trial based on validation accuracy
            best_val_trial = study.best_trial
            
            # Track the test performance but don't use it for model selection
            # This avoids data leakage from the test set to the model selection process
            best_test_acc = 0
            best_test_trial = None
            
            for trial in study.trials:
                if "test_accuracy" in trial.user_attrs and trial.user_attrs["test_accuracy"] > best_test_acc:
                    best_test_acc = trial.user_attrs["test_accuracy"]
                    best_test_trial = trial
            
            # Always use the model with the best validation accuracy for proper generalization
            print(f"Using model with best validation accuracy: {best_val_trial.user_attrs.get('val_accuracy', 0):.4f}")
            print(f"Test accuracy of this model: {best_val_trial.user_attrs.get('test_accuracy', 0):.4f}")
            best_trial = best_val_trial
            
            # Track but don't use the best test accuracy model (just for informational purposes)
            if best_test_trial:
                print(f"Note: Best test accuracy was: {best_test_acc:.4f} (not used for model selection)")
                print(f"Validation accuracy of that model: {best_test_trial.user_attrs.get('val_accuracy', 0):.4f}")
                
            if "model" in best_trial.user_attrs:
                best_model = best_trial.user_attrs["model"]
                
                # Save the best model
                model_path = os.path.join(project_root, 'saved_models', 'best_model.h5')
                best_model.save(model_path)
                print(f"Best model saved to {model_path}")
                
                # Store the best model for use in evaluate_generalization
                self.best_model = best_model
                print(f"Best model metrics:")
                print(f"- Validation accuracy: {best_trial.user_attrs.get('val_accuracy', 0):.4f}")
                print(f"- Test accuracy: {best_trial.user_attrs.get('test_accuracy', 0):.4f}")
                print(f"- Generalization gap: {best_trial.user_attrs.get('generalization_gap', 0):.4f}")
                
                # Log the best model in MLflow
                with mlflow.start_run():
                    mlflow.log_params(best_params)
                    mlflow.log_metric("validation_accuracy", best_trial.user_attrs.get("val_accuracy", 0))
                    mlflow.log_metric("validation_loss", best_trial.user_attrs.get("val_loss", 0))
                    mlflow.log_metric("test_accuracy", best_trial.user_attrs.get("test_accuracy", 0))
                    mlflow.log_metric("test_loss", best_trial.user_attrs.get("test_loss", 0))
                    mlflow.log_metric("generalization_gap", best_trial.user_attrs.get("generalization_gap", 0))
                    mlflow.tensorflow.log_model(best_model, "best_model")
                    print(f"Best model logged in MLflow with run ID: {mlflow.active_run().info.run_id}")
            else:
                print("Warning: Best model not found in trial user attributes")
                
            return best_params
        except Exception as e:
            print(f"Hyperparameter tuning failed: {str(e)}")
            # Return some default parameters if tuning fails
            default_params = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "dropout_rate": 0.2,
                "l2_reg": 0.001,
                "layer_size_0": 64,
                "layer_size_1": 32,
                "layer_size_2": 16,
                "layer_size_3": 8
            }
            return default_params

    def mlflow_run(self):
        with mlflow.start_run():
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("layer_sizes", self.layer_sizes)
            mlflow.log_param("dropout_rate", self.dropout_rate)
            mlflow.log_param("l2_reg", self.l2_reg)

            early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
            history = self.model.fit(self.X_train, self.y_train,
                                     validation_data=(self.X_test, self.y_test),
                                     epochs=self.epochs, batch_size=self.batch_size,
                                     callbacks=[early_stop], verbose=1)

            val_loss, val_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            mlflow.log_metric("validation_loss", val_loss)
            mlflow.log_metric("validation_accuracy", val_accuracy)

            model_path = "../saved_models/model.h5"
            self.model.save(model_path)
            mlflow.tensorflow.log_model(self.model, "model")

            print(f"Run ID: {mlflow.active_run().info.run_id}")
    def train_with_best_params(self, n_trials=50):
        """
        Finds the best hyperparameters using Optuna and retrains the model with them.
        """
        best_params = self.hyperparameter_tuning(n_trials=n_trials)

        # Update class attributes with the best parameters
        self.learning_rate = best_params["learning_rate"]
        self.batch_size = best_params["batch_size"]
        self.dropout_rate = best_params["dropout_rate"]
        self.l2_reg = best_params["l2_reg"]
        self.layer_sizes = [best_params[f"layer_size_{i}"] for i in range(4)]

        # Rebuild the model with the new best parameters
        self.model = self.build_dnn_model()

        # Retrain the model
        early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        self.model.fit(self.X_train, self.y_train,
                       validation_data=(self.X_test, self.y_test),
                       epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[early_stop], verbose=1)

        print("Retrained model with best hyperparameters.")
        
    def train_with_temperature(self, temperature=2.0, save_path=None):
        """
        Retrain the model with a new temperature scaling parameter.
        Higher temperature (>1) makes predictions more moderate and spread out.
        Lower temperature (<1) makes predictions more extreme (closer to 0 or 1).
        
        Args:
            temperature (float): The temperature scaling parameter. Default: 2.0
            save_path (str): Path to save the retrained model. If None, uses default path.
        """
        # Update the temperature
        self.temperature = temperature
        print(f"Setting temperature scaling to {temperature}")
        
        # Rebuild the model with the new temperature
        self.model = self.build_dnn_model()
        
        # Retrain the model
        early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        history = self.model.fit(self.X_train, self.y_train,
                   validation_data=(self.X_test, self.y_test),
                   epochs=self.epochs, batch_size=self.batch_size,
                   callbacks=[early_stop], verbose=1)
        
        # Save the retrained model
        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            save_path = os.path.join(project_root, 'saved_models', f'model_temp_{temperature}.h5')
        
        self.model.save(save_path)
        print(f"Model with temperature={temperature} saved to {save_path}")
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test loss: {loss:.4f}")
        
        # Make a few predictions to show the effect of temperature
        print("\nSample predictions with new temperature:")
        sample_size = min(5, len(self.X_test))
        sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        
        for idx in sample_indices:
            x = self.X_test[idx:idx+1]
            true_y = self.y_test.iloc[idx]
            pred_prob = self.model.predict(x, verbose=0)[0][0]
            pred_class = 1 if pred_prob > 0.5 else 0
            correct = pred_class == true_y
            
            print(f"Sample {idx}: Prediction={pred_prob:.4f}, True={true_y}, {'✓' if correct else '✗'}")
        
        return history

    def evaluate_generalization_with_postcomp_stats(self):
        """
        I want a dataset that contains all matchups in the data for the last year but it contains the postcomp values for every fighter's most recent fight.

        And then evaluate the generalization performance of the model on this dataset.

        For example:

        Today is 2025-05-02

        The earlist fight in the dataset going back one year is 2024-05-04

        AlexandrePantoja vs. SteveErceg happened on 2024-05-04. I would want to include this fight in the dataset but I want to use the postcomp values from their most recent fights.

        AlexandrePantoja's most recent fight was on 2023-12-16 against Brandon Royval. I want to use the postcomp values where for example I would replace the 'precomp_sigstr_pm' from AlexandrePantoja vs Steve Erceg with 'postcomp_sigstr_pm' AlexandrePantoja vs Brandon Royval and so on.
        Steve Erceg's most fight was on 2024-03-02 against Matt Schnell. I want to use the postcomp values where for example I would replace the 'precomp_sigstr_pm' from Steve Erceg vs AlexandrePantoja with 'postcomp_sigstr_pm' Steve Erceg vs Matt Schnell and so on.

        I want to apply this to all fights in the dataset that happened in the last year.
        """
        # Get today's date
        today = pd.to_datetime("today")
        # Get the date one year ago
        one_year_ago = today - pd.DateOffset(years=1)
        # Filter the dataset to include only fights that happened in the last year
        recent_fights = self.data[self.data['DATE'] >= one_year_ago]
        #postcomp stats
        postcomp_cols = [
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
            'precomp_winsum3', 'precomp_losssum3','precomp_elo_change_3']
        # for every fight in the dataset I want to get the most postcomp values for every fighter's most recent fight
        for index, row in recent_fights.iterrows():
            fighter1 = row['FIGHTER']
            fighter2 = row['opp_FIGHTER']
            date = row['DATE']
            # Get the most recent fight for fighter1
            fighter1_recent_fight = self.data[(self.data['FIGHTER'] == fighter1) | (self.data['opp_FIGHTER'] == fighter1)]
            fighter1_recent_fight = fighter1_recent_fight[fighter1_recent_fight['DATE'] < date]
            if len(fighter1_recent_fight) > 0:
                fighter1_recent_fight = fighter1_recent_fight.iloc[-1]
                for col in postcomp_cols:
                    self.data.at[index, col] = fighter1_recent_fight[col.replace('postcomp', 'precomp')]
            # Get the most recent fight for fighter2
            fighter2_recent_fight = self.data[(self.data['FIGHTER'] == fighter2) | (self.data['opp_FIGHTER'] == fighter2)]
            fighter2_recent_fight = fighter2_recent_fight[fighter2_recent_fight['DATE'] < date]
            if len(fighter2_recent_fight) > 0:
                fighter2_recent_fight = fighter2_recent_fight.iloc[-1]
                for col in postcomp_cols:
                    self.data.at[index, col] = fighter2_recent_fight[col.replace('precomp', 'postcomp')]

            # Save the modified row back to the dataset
            self.data.at[index, 'DATE'] = date
        # Now we can use this modified dataset to evaluate the model
        # Filter the dataset to include only fights that happened in the last year
        recent_fights = self.data[self.data['DATE'] >= one_year_ago]
        # Extract features and targets
        X_recent = recent_fights[self.elo_columns]
        y_recent = recent_fights['result']
        # Standardize features based on training data
        X_recent_scaled = self.scaler.transform(X_recent)
        # Evaluate the model on this new dataset
        loss, accuracy = self.model.evaluate(X_recent_scaled, y_recent, verbose=0)
        print(f"Generalization accuracy on recent fights: {accuracy:.4f}")
        # Get predictions for the test set
        y_pred_proba = self.model.predict(X_recent_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        # Calculate classification metrics
        from sklearn.metrics import (
            classification_report, precision_score, recall_score,
            f1_score, roc_auc_score, roc_curve, confusion_matrix
        )
        import matplotlib.pyplot as plt
        # Calculate metrics
        precision = precision_score(y_recent, y_pred)
        recall = recall_score(y_recent, y_pred)
        f1 = f1_score(y_recent, y_pred)
        auc_roc = roc_auc_score(y_recent, y_pred_proba)
        conf_matrix = confusion_matrix(y_recent, y_pred)
        report = classification_report(y_recent, y_pred)
        # Print comprehensive results
        print("\n========== Generalization Performance on Recent Fights ==========")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test loss: {loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")


        print(f"AUC-ROC: {auc_roc:.4f}")
        print("\nConfusion Matrix:")
        print(f"TN: {conf_matrix[0, 0]} | FP: {conf_matrix[0, 1]}")
        print(f"FN: {conf_matrix[1, 0]} | TP: {conf_matrix[1, 1]}")
        # Calculate positive and negative predictive values
        ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
        npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0]) if (conf_matrix[0, 0] + conf_matrix[1, 0]) > 0 else 0
        print(f"Positive Predictive Value: {ppv:.4f}")
        print(f"Negative Predictive Value: {npv:.4f}")
        print("\nDetailed Classification Report:")
        print(report)



    def evaluate_generalization(self, use_best_model=True):
        """
        Evaluate the model's generalization performance on chronologically separated test data.
        This tests how well the model performs on future fights that happened after its training period.
        Calculates accuracy, precision, recall, F1-score, and AUC-ROC.
        
        Args:
            use_best_model (bool): If True, uses the best model from hyperparameter tuning
                                  if available, otherwise falls back to the default model.
        """
        # Determine which model to use for evaluation
        evaluation_model = None
        model_source = ""
        
        if use_best_model and hasattr(self, 'best_model') and self.best_model is not None:
            # Use the best model from hyperparameter tuning
            evaluation_model = self.best_model
            model_source = "best model from hyperparameter tuning"
        elif use_best_model:
            # Try to load the saved best model if it exists
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            best_model_path = os.path.join(project_root, 'saved_models', 'best_model.h5')
            
            if os.path.exists(best_model_path):
                try:
                    evaluation_model = tf.keras.models.load_model(best_model_path)
                    model_source = f"saved best model loaded from {best_model_path}"
                except Exception as e:
                    print(f"Failed to load best model: {str(e)}")
                    # Fall back to the default model
                    evaluation_model = self.model
                    model_source = "default model (best model failed to load)"
            else:
                # Use the default model if no best model is found
                evaluation_model = self.model
                model_source = "default model (no best model found)"
        else:
            # Explicitly use the default model
            evaluation_model = self.model
            model_source = "default model (as requested)"
        
        # Make sure a model is available for evaluation
        if evaluation_model is None:
            raise ValueError("No model available for evaluation. Please train a model first.")
        
        print(f"\nUsing {model_source} for generalization evaluation")
            
        # Evaluate on the test set (which contains the latest year of data)
        # NOTE: Use evaluation_model instead of self.model to ensure we're testing the right model
        loss, accuracy = evaluation_model.evaluate(self.X_test, self.y_test, verbose=1)
        # save self.X_test to a csv file in '../data/tmp'
        # puth the feature neames from saved_models/feature_names.txt and put them into a list called feature_names
        feature_names = []
        with open('../saved_models/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        #convers self.X_test to a dataframe
        self.X_test = pd.DataFrame(self.X_test, columns=feature_names)
        self.X_test.to_csv('../data/tmp/X_test.csv', index=False)
        
        # Get predictions for the test set
        # NOTE: Use evaluation_model instead of self.model to ensure we're testing the right model
        y_pred_proba = evaluation_model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate classification metrics
        from sklearn.metrics import (
            classification_report, precision_score, recall_score, 
            f1_score, roc_auc_score, roc_curve, confusion_matrix
        )
        import matplotlib.pyplot as plt
        
        # Calculate metrics
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        # Print comprehensive results
        print("\n========== Generalization Performance ==========")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test loss: {loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"TN: {conf_matrix[0, 0]} | FP: {conf_matrix[0, 1]}")
        print(f"FN: {conf_matrix[1, 0]} | TP: {conf_matrix[1, 1]}")
        
        # Calculate positive and negative predictive values
        ppv = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
        npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0]) if (conf_matrix[0, 0] + conf_matrix[1, 0]) > 0 else 0
        print(f"Positive Predictive Value: {ppv:.4f}")
        print(f"Negative Predictive Value: {npv:.4f}")
        
        print("\nDetailed Classification Report:")
        print(report)
        
        # Calculate accuracy by month to see if there's a time-based decay in performance
        test_dates = self.filtered_data[self.filtered_data['DATE'] > (self.filtered_data['DATE'].max() - pd.DateOffset(years=1))]['DATE']
        
        if len(test_dates) > 0:
            # Group by month and calculate accuracy
            test_data_with_preds = pd.DataFrame({
                'date': test_dates.reset_index(drop=True),
                'actual': self.y_test.reset_index(drop=True),
                'predicted': y_pred,
                'probability': y_pred_proba.flatten()
            })
            
            test_data_with_preds['year_month'] = test_data_with_preds['date'].dt.to_period('M')
            
            # Calculate monthly metrics
            monthly_metrics = test_data_with_preds.groupby('year_month').apply(
                lambda x: pd.Series({
                    'accuracy': accuracy_score(x['actual'], x['predicted']),
                    'precision': precision_score(x['actual'], x['predicted'], zero_division=0),
                    'recall': recall_score(x['actual'], x['predicted'], zero_division=0),
                    'f1': f1_score(x['actual'], x['predicted'], zero_division=0),
                    'auc': roc_auc_score(x['actual'], x['probability']) if len(set(x['actual'])) > 1 else np.nan,
                    'count': len(x)
                })
            )
            
            print("\nMetrics by Month:")
            print(monthly_metrics.to_string())
            
            # Plot ROC curve
            try:
                plt.figure(figsize=(10, 8))
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                
                # Save plot to project directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                plot_path = os.path.join(project_root, 'roc_curve.png')
                plt.savefig(plot_path)
                print(f"ROC curve saved to {plot_path}")
            except Exception as e:
                print(f"Could not generate ROC curve plot: {e}")
        
        # Return comprehensive metrics as a dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
        
        return metrics

    def predict_with_postcomp_data(self, fighter1, fighter2, weight1=None, weight2=None):
        """
        Predicts the winner of a fight between two UFC fighters using post-computation values
        from their most recent fights.
        
        Args:
            fighter1 (str): Name of the first fighter
            fighter2 (str): Name of the second fighter
            weight1 (float, optional): Weight of first fighter, to override database value
            weight2 (float, optional): Weight of second fighter, to override database value
            
        Returns:
            tuple: (predicted winner name (str), winning probability (float))
        """
        import os
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        from datetime import datetime, date
        import pandas as pd
        import numpy as np
        
        # Define the post-computation columns to use
        postcomp_cols = [
            'age', 'HEIGHT', 'WEIGHT', 'REACH', 'weightindex',
            'postcomp_sigstr_pm', 'postcomp_tdavg', 'postcomp_sapm', 'postcomp_subavg',
            'postcomp_tddef', 'postcomp_sigstr_perc', 'postcomp_strdef', 'postcomp_tdacc_perc',
            'postcomp_totalacc_perc', 'postcomp_headacc_perc', 'postcomp_bodyacc_perc', 'postcomp_legacc_perc',
            'postcomp_distacc_perc','postcomp_clinchacc_perc','postcomp_groundacc_perc',
            'postcomp_winsum', 'postcomp_losssum','postcomp_elo',
            'postcomp_sigstr_pm5', 'postcomp_tdavg5', 'postcomp_sapm5', 'postcomp_subavg5',
            'postcomp_tddef5', 'postcomp_sigstr_perc5', 'postcomp_strdef5', 'postcomp_tdacc_perc5',
            'postcomp_totalacc_perc5', 'postcomp_headacc_perc5', 'postcomp_bodyacc_perc5', 'postcomp_legacc_perc5',
            'postcomp_distacc_perc5','postcomp_clinchacc_perc5','postcomp_groundacc_perc5',
            'postcomp_winsum5', 'postcomp_losssum5','postcomp_elo_change_5',
            'postcomp_sigstr_pm3', 'postcomp_tdavg3', 'postcomp_sapm3', 'postcomp_subavg3',
            'postcomp_tddef3', 'postcomp_sigstr_perc3', 'postcomp_strdef3', 'postcomp_tdacc_perc3',
            'postcomp_totalacc_perc3', 'postcomp_headacc_perc3', 'postcomp_bodyacc_perc3', 'postcomp_legacc_perc3',
            'postcomp_distacc_perc3','postcomp_clinchacc_perc3','postcomp_groundacc_perc3',
            'postcomp_winsum3', 'postcomp_losssum3','postcomp_elo_change_3',
        ]
        
        opp_postcomp_cols = [
            'opp_age', 'opp_HEIGHT', 'opp_WEIGHT', 'opp_REACH', 'opp_weightindex', 'opp_weight_of_fight',
            'opp_postcomp_sigstr_pm', 'opp_postcomp_tdavg', 'opp_postcomp_sapm', 'opp_postcomp_subavg',
            'opp_postcomp_tddef', 'opp_postcomp_sigstr_perc', 'opp_postcomp_strdef', 'opp_postcomp_tdacc_perc',
            'opp_postcomp_totalacc_perc', 'opp_postcomp_headacc_perc', 'opp_postcomp_bodyacc_perc', 'opp_postcomp_legacc_perc',
            'opp_postcomp_distacc_perc', 'opp_postcomp_clinchacc_perc', 'opp_postcomp_groundacc_perc',
            'opp_postcomp_winsum', 'opp_postcomp_losssum', 'opp_postcomp_elo',
            'opp_postcomp_sigstr_pm5', 'opp_postcomp_tdavg5', 'opp_postcomp_sapm5', 'opp_postcomp_subavg5',
            'opp_postcomp_tddef5', 'opp_postcomp_sigstr_perc5', 'opp_postcomp_strdef5', 'opp_postcomp_tdacc_perc5',
            'opp_postcomp_totalacc_perc5', 'opp_postcomp_headacc_perc5', 'opp_postcomp_bodyacc_perc5', 'opp_postcomp_legacc_perc5',
            'opp_postcomp_distacc_perc5', 'opp_postcomp_clinchacc_perc5', 'opp_postcomp_groundacc_perc5',
            'opp_postcomp_winsum5', 'opp_postcomp_losssum5', 'opp_postcomp_elo_change_5',
            'opp_postcomp_sigstr_pm3', 'opp_postcomp_tdavg3', 'opp_postcomp_sapm3', 'opp_postcomp_subavg3',
            'opp_postcomp_tddef3', 'opp_postcomp_sigstr_perc3', 'opp_postcomp_strdef3', 'opp_postcomp_tdacc_perc3',
            'opp_postcomp_totalacc_perc3', 'opp_postcomp_headacc_perc3', 'opp_postcomp_bodyacc_perc3', 'opp_postcomp_legacc_perc3',
            'opp_postcomp_distacc_perc3', 'opp_postcomp_clinchacc_perc3', 'opp_postcomp_groundacc_perc3',
            'opp_postcomp_winsum3', 'opp_postcomp_losssum3', 'opp_postcomp_elo_change_3'
        ]
        
        # Create a mapping from postcomp to precomp columns
        postcomp_to_precomp = {}
        for col in postcomp_cols:
            if col.startswith('postcomp_'):
                precomp_col = col.replace('postcomp_', 'precomp_')
                postcomp_to_precomp[col] = precomp_col
        
        for col in opp_postcomp_cols:
            if col.startswith('opp_postcomp_'):
                precomp_col = col.replace('opp_postcomp_', 'opp_precomp_')
                postcomp_to_precomp[col] = precomp_col
        
        # 1. Find most recent fight data for each fighter
        fighter1_data = self.data[self.data['FIGHTER'] == fighter1]
        print(f"Fighter1 data: {fighter1_data.shape}")
        fighter2_data = self.data[self.data['FIGHTER'] == fighter2]
        print(f"Fighter2 data: {fighter2_data.shape}")
        
        # Error handling if fighters are not found
        if len(fighter1_data) == 0:
            raise ValueError(f"No data found for fighter '{fighter1}'")
            
        if len(fighter2_data) == 0:
            raise ValueError(f"No data found for fighter '{fighter2}'")
        
        # Sort by date in descending order and get the most recent fight
        fighter1_data = fighter1_data.sort_values('DATE', ascending=False).iloc[0].copy()
        fighter2_data = fighter2_data.sort_values('DATE', ascending=False).iloc[0].copy()
        
        print(f"Most recent fight data found for both fighters")
        
        # Update WEIGHT if provided
        if weight1 is not None:
            fighter1_data['WEIGHT'] = float(weight1)
            print(f"Using overridden weight for {fighter1}: {weight1}")
        
        if weight2 is not None:
            fighter2_data['WEIGHT'] = float(weight2)
            print(f"Using overridden weight for {fighter2}: {weight2}")
        
        # Update AGE to current age
        try:
            if 'DOB' in fighter1_data and not pd.isna(fighter1_data['DOB']):
                dob = pd.to_datetime(fighter1_data['DOB'])
                today = date.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                fighter1_data['age'] = age
                print(f"Calculated current age for {fighter1}: {age}")
            
            if 'DOB' in fighter2_data and not pd.isna(fighter2_data['DOB']):
                dob = pd.to_datetime(fighter2_data['DOB'])
                today = date.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                fighter2_data['age'] = age
                print(f"Calculated current age for {fighter2}: {age}")
        except Exception as e:
            print(f"Error updating ages: {e}")
        
        # 2. Extract postcomp features for each fighter
        fighter1_features = {}
        fighter2_features = {}
        
        # Regular (non-opponent) features
        for col in postcomp_cols:
            if col in fighter1_data:
                fighter1_features[col] = fighter1_data[col]
            else:
                fighter1_features[col] = 0
                
            if col in fighter2_data:
                fighter2_features[col] = fighter2_data[col]
            else:
                fighter2_features[col] = 0
                
        # 3. Create the feature dataframes with the correct mapping to precomp values
        # for model compatibility
        fighter1_model_features = {}
        fighter2_model_features = {}
        
        # Map postcomp values to precomp column names for the model
        for col, value in fighter1_features.items():
            if col in postcomp_to_precomp:
                # If it's a postcomp column, map to precomp column name
                precomp_col = postcomp_to_precomp[col]
                fighter1_model_features[precomp_col] = value
            else:
                # If it's a non-comp column (like age, HEIGHT), keep as is
                fighter1_model_features[col] = value
                
        for col, value in fighter2_features.items():
            if col in postcomp_to_precomp:
                # If it's a postcomp column, map to precomp column name
                precomp_col = postcomp_to_precomp[col]
                fighter2_model_features[precomp_col] = value
            else:
                # If it's a non-comp column (like age, HEIGHT), keep as is
                fighter2_model_features[col] = value
        
        # Add opponent features (fighter2's features as fighter1's opponent features)
        # And vice versa
        fight1_features = fighter1_model_features.copy()
        fight2_features = fighter2_model_features.copy()
        
        # Add opponent features for fight 1 (fighter1 vs fighter2)
        for col, value in fighter2_model_features.items():
            if not col.startswith('opp_'):
                # Create the opponent feature name by adding 'opp_' prefix
                # or replacing 'precomp_' with 'opp_precomp_'
                if col.startswith('precomp_'):
                    opp_col = col.replace('precomp_', 'opp_precomp_')
                else:
                    opp_col = f'opp_{col}'
                fight1_features[opp_col] = value
        
        # Add opponent features for fight 2 (fighter2 vs fighter1)
        for col, value in fighter1_model_features.items():
            if not col.startswith('opp_'):
                # Create the opponent feature name by adding 'opp_' prefix
                # or replacing 'precomp_' with 'opp_precomp_'
                if col.startswith('precomp_'):
                    opp_col = col.replace('precomp_', 'opp_precomp_')
                else:
                    opp_col = f'opp_{col}'
                fight2_features[opp_col] = value
        
        # Convert to dataframes
        fight1_df = pd.DataFrame([fight1_features])
        fight2_df = pd.DataFrame([fight2_features])
        
        # Check if feature_names.txt exists, if not create it using elo_columns
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        feature_file_path = os.path.join(project_root, 'saved_models', 'feature_names.txt')
        
        if not os.path.exists(feature_file_path):
            print(f"Feature names file not found, creating from current columns")
            os.makedirs(os.path.dirname(feature_file_path), exist_ok=True)
            with open(feature_file_path, 'w') as f:
                for col in self.elo_columns:
                    f.write(f"{col}\n")
            all_features = self.elo_columns
        else:
            # Load feature list from file
            with open(feature_file_path, 'r') as f:
                all_features = [line.strip() for line in f.readlines()]
        
        # Ensure we have all required features for the model
        for col in all_features:
            if col not in fight1_df.columns:
                fight1_df[col] = 0
            if col not in fight2_df.columns:
                fight2_df[col] = 0
                
        # Order the columns properly to match the training data format
        fight1_df = fight1_df[all_features]
        fight2_df = fight2_df[all_features]
        
        # 4. Combine both fights into a single dataframe
        fights_df = pd.concat([fight1_df, fight2_df])
        
        print("Feature data prepared with postcomp values for model prediction")
        
        # 5. Prepare input for the model using the same scaler as during training
        try:
            from joblib import load
            
            scaler_path = os.path.join(project_root, 'saved_models', 'feature_scaler.joblib')
            
            if os.path.exists(scaler_path):
                print(f"Loading saved scaler from {scaler_path}")
                scaler = load(scaler_path)
            else:
                print("Saved scaler not found, fitting a new one (this may cause prediction issues)")
                scaler = StandardScaler()
                # Use a subset of training data to fit the scaler
                if hasattr(self, 'X_train') and self.X_train is not None:
                    X_train_subset = self.X_train[:100]
                    scaler.fit(X_train_subset)
                else:
                    # If no training data available, fit on the fight data
                    print("WARNING: No training data available, scaling with current fight data only")
                    scaler.fit(fights_df)
            
            fights_scaled = scaler.transform(fights_df)
        except Exception as e:
            print(f"Error loading/using scaler: {str(e)}, using basic standardization")
            # Fallback to basic scaling if there's an error
            scaler = StandardScaler()
            scaler.fit(fights_df)
            fights_scaled = scaler.transform(fights_df)
        
        # 6. Load the model
        try:
            model_path = os.path.join(project_root, 'saved_models', 'best_model.h5')
            print(f"Loading best model from {model_path}...")
            prediction_model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Failed to load best model: {str(e)}")
            model_path = os.path.join(project_root, 'saved_models', 'model.h5')
            print(f"Loading fallback model from {model_path}...")
            prediction_model = tf.keras.models.load_model(model_path)
        
        # 7. Make prediction
        predictions = prediction_model.predict(fights_scaled, verbose=0)
        
        # Print raw predictions for debugging
        print(f"Raw predictions: {predictions}")
        
        # 8. Interpret the results
        # For first fight: 1 means fighter1 wins, 0 means fighter2 wins
        # For second fight: 1 means fighter2 wins, 0 means fighter1 wins
        prob_fighter1_wins = float(predictions[0][0])
        prob_fighter2_wins = float(predictions[1][0])
        
        print(f"Prediction for {fighter1}: {prob_fighter1_wins:.4f}")
        print(f"Prediction for {fighter2}: {prob_fighter2_wins:.4f}")
        
        # Apply normalization to the probabilities to make them more separated and sum closer to 1
        # Calculate a more balanced probability using both predictions
        # We convert both to a probability for fighter1 (inverting the second prediction)
        fighter1_prob_from_direct = prob_fighter1_wins
        fighter1_prob_from_inverse = 1 - prob_fighter2_wins
        
        # Calculate the average probability
        raw_prob_fighter1 = (fighter1_prob_from_direct + fighter1_prob_from_inverse) / 2
        raw_prob_fighter2 = 1 - raw_prob_fighter1
        
        # Apply softmax to increase separation
        def softmax_with_temp(probs, temperature=0.5):
            # Lower temperature creates more separation
            probs = np.array(probs) / temperature
            exp_probs = np.exp(probs - np.max(probs))  # Subtract max for numerical stability
            return exp_probs / np.sum(exp_probs)
        
        # Apply softmax to get more separated probabilities
        normalized_probs = softmax_with_temp([raw_prob_fighter1, raw_prob_fighter2])
        
        # Extract the normalized probabilities
        normalized_prob_fighter1 = normalized_probs[0]
        normalized_prob_fighter2 = normalized_probs[1]
        
        print(f"Normalized probabilities:")
        print(f"{fighter1}: {normalized_prob_fighter1:.4f}")
        print(f"{fighter2}: {normalized_prob_fighter2:.4f}")
        print(f"Sum: {normalized_prob_fighter1 + normalized_prob_fighter2:.4f}")
        
        # Determine winner based on normalized probabilities
        if normalized_prob_fighter1 > normalized_prob_fighter2:
            winner = fighter1
            win_probability = normalized_prob_fighter1
        else:
            winner = fighter2
            win_probability = normalized_prob_fighter2
        
        print(f"\nPrediction: {winner} wins with {win_probability:.2%} probability")
        return winner, win_probability
        
    def shap_analysis(self, use_best_model=True):
        """
        Compute SHAP values for feature importance analysis on the model.
        Prints and stores the top 20 most important features in reverse order (least to most important).
        
        Args:
            use_best_model (bool): If True, uses the best model from hyperparameter tuning
                                  if available, otherwise falls back to the default model.
        """
        # Determine which model to use for analysis
        analysis_model = None
        model_source = ""
        
        if use_best_model and hasattr(self, 'best_model') and self.best_model is not None:
            # Use the best model from hyperparameter tuning
            analysis_model = self.best_model
            model_source = "best model from hyperparameter tuning"
        elif use_best_model:
            # Try to load the saved best model if it exists
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            best_model_path = os.path.join(project_root, 'saved_models', 'best_model.h5')
            
            if os.path.exists(best_model_path):
                try:
                    analysis_model = tf.keras.models.load_model(best_model_path)
                    model_source = f"saved best model loaded from {best_model_path}"
                except Exception as e:
                    print(f"Failed to load best model: {str(e)}")
                    # Fall back to the default model
                    analysis_model = self.model
                    model_source = "default model (best model failed to load)"
            else:
                # Use the default model if no best model is found
                analysis_model = self.model
                model_source = "default model (no best model found)"
        else:
            # Explicitly use the default model
            analysis_model = self.model
            model_source = "default model (as requested)"
        
        # Make sure a model is available for analysis
        if analysis_model is None:
            raise ValueError("No model available for analysis. Please train a model first.")
        
        print(f"\nUsing {model_source} for SHAP feature importance analysis")

        # Use a subset of test data for SHAP analysis to speed up computation
        background = self.X_train[:100]  # Select a small sample for SHAP background
        try:
            explainer = shap.Explainer(analysis_model, background)
            
            # Compute SHAP values for test data
            shap_values = explainer(self.X_test[:100])  # Analyze a subset for efficiency
            
            # Compute mean absolute SHAP values across all test samples
            mean_shap_values = np.abs(shap_values.values).mean(axis=0)
            
            # Get the features from elo_columns
            feature_importance = list(zip(self.elo_columns, mean_shap_values))
            feature_importance.sort(key=lambda x: x[1])  # Sort by SHAP value (ascending)
            
            # Get the top 20 features
            top_20_features = feature_importance[-20:]  # Select most important 20
            
            # Print the top 20 features in reverse order
            print("\nTop 20 Most Important Features (Least to Most Important):")
            for feature, importance in top_20_features:
                print(f"{feature}: {importance:.5f}")

            # Store the feature names ranked least to most important
            self.top_20_features = [feature for feature, _ in top_20_features]
            
            # Save top features to file
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                top_features_path = os.path.join(project_root, 'saved_models', 'top_features.txt')
                
                with open(top_features_path, 'w') as f:
                    for feature, importance in reversed(top_20_features):  # Save in most to least important order
                        f.write(f"{feature}: {importance:.6f}\n")
                
                print(f"Top features saved to {top_features_path}")
            except Exception as e:
                print(f"Failed to save top features: {str(e)}")

            return self.top_20_features
            
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
            print("This may be due to incompatible SHAP versions or model structure.")
            print("Try updating SHAP or using a simpler model architecture.")
            return []