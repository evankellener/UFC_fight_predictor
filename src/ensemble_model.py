import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.models import load_model
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, log_loss

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False


class FightOutcomeModel:
    def __init__(self, file_path, scaler_path='../saved_models/feature_scaler.joblib'):
        self.elo_columns = [
            'precomp_elo', 'precomp_elo_prev', 'precomp_elo_change_3', 'precomp_elo_change_5',
            'opp_precomp_elo', 'opp_precomp_elo_prev', 'opp_precomp_elo_change_3', 'opp_precomp_elo_change_5'
        ]
        self.main_stats_cols = [
            'age', 'HEIGHT', 'WEIGHT', 'REACH', 'weightindex',
            'precomp_sigstr_pm', 'precomp_tdavg', 'precomp_sapm', 'precomp_subavg',
            'precomp_tddef', 'precomp_sigstr_perc', 'precomp_strdef', 'precomp_tdacc_perc',
            'precomp_totalacc_perc', 'precomp_headacc_perc', 'precomp_bodyacc_perc', 'precomp_legacc_perc',
            'precomp_distacc_perc','precomp_clinchacc_perc','precomp_groundacc_perc',
            'precomp_winsum', 'precomp_losssum','precomp_elo','precomp_elo_prev',
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
            'opp_precomp_winsum', 'opp_precomp_losssum', 'opp_precomp_elo','opp_precomp_elo_prev',
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
        self.importance_columns = [
            'precomp_elo', 'precomp_elo_prev', 'precomp_elo_change_3', 'precomp_elo_change_5',
            'opp_precomp_elo', 'opp_precomp_elo_prev', 'opp_precomp_elo_change_3', 'opp_precomp_elo_change_5',
            'WEIGHT', 'REACH', 'age', 'opp_precomp_sapm5',
            'opp_WEIGHT', 'opp_REACH', 'opp_age'
        ]
        self.df = pd.read_csv(file_path)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.scaler = load(scaler_path)
        self._prepare_data()

    def _prepare_data(self):
        latest_date = self.df['DATE'].max()
        test_start_date = latest_date - timedelta(days=365)
        
        self.train_df = self.df[self.df['DATE'] < test_start_date].dropna(subset=self.elo_columns + ['win'])
        self.test_df = self.df[self.df['DATE'] >= test_start_date].dropna(subset=self.elo_columns + ['win'])

        self.X_train = self.train_df[self.elo_columns]
        self.y_train = self.train_df['win']
        self.X_test = self.test_df[self.elo_columns]
        self.y_test = self.test_df['win']
        #print sze of the datasets
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")


    def build_mlp(self):
        model = load_model('../saved_models/best_model.h5')

        # Scale the features
        X_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Predict and evaluate
        preds = (model.predict(X_test_scaled) > 0.5).astype("int32").flatten()
        acc = accuracy_score(self.y_test, preds)
        return model, acc

    def tune_logistic_regression(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=10000, random_state=42))
        ])
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'saga'],
            'clf__class_weight': [None, 'balanced']
        }
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        log_probs = grid.best_estimator_.predict_log_proba(self.X_test)
        print("log_probs", log_probs)
        log_probs_class_1 = log_probs[:, 1]
        probs = grid.best_estimator_.predict_proba(self.X_test)[:, 1]
        loss = log_loss(self.y_test, probs)
        print("loss", loss)

        print("log_probs_class_1", log_probs_class_1)
        acc = accuracy_score(self.y_test, grid.predict(self.X_test))
        return grid.best_estimator_, acc

    def tune_svm(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True))
        ])
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['rbf', 'linear'],
            'clf__gamma': ['scale', 'auto'],
            'clf__class_weight': [None, 'balanced']
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_test, grid.predict(self.X_test))
        return grid.best_estimator_, acc
    
    def build_naive_bayes(self):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_test, model.predict(self.X_test))
        return model, acc

    def tune_xgboost(self):
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_test, grid.predict(self.X_test))
        return grid.best_estimator_, acc
    
    def train_ensemble(self):
        log_model, _ = self.tune_logistic_regression()
        #svm_model, _ = self.tune_svm()
        xgb_model, _ = self.tune_xgboost()
        mlp, _ = self.build_mlp()
        #nb, _ = self.build_naive_bayes()


        ensemble = VotingClassifier(
            estimators=[
                ('logreg', log_model),
                #('svm', svm_model),
                ('xgb', xgb_model),
                ('mlp', mlp)
                #('nb', nb)
            ],
            voting='soft'
        )
        ensemble.fit(self.X_train, self.y_train)
        preds = ensemble.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        return ensemble, acc
    
    import numpy as np

    def custom_soft_voting_ensemble(self):
        # Fit sklearn models
        log_model, _ = self.tune_logistic_regression()
        print("logistic log probabilities ", log_model)
        xgb_model, _ = self.tune_xgboost()
        print("xgboost log_model", xgb_model)
        mlp_model = load_model('../saved_models/best_model.h5')
        print("mlp_model log probabilities", mlp_model)
        # Scale features for MLP
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Predict probabilities
        log_probs = log_model.predict_proba(self.X_test)[:, 1]
        print("log_probs", log_probs)
        xgb_probs = xgb_model.predict_proba(self.X_test)[:, 1]
        mlp_probs = mlp_model.predict(X_test_scaled).flatten()

        # Average probabilities
        avg_probs = (log_probs + xgb_probs + mlp_probs) / 3.0
        final_preds = (avg_probs > 0.5).astype(int)
        #print out log loss
        print("log loss", log_loss(self.y_test, avg_probs))

        acc = accuracy_score(self.y_test, final_preds)
        return final_preds, acc
    
    def custom_hard_voting_ensemble(self):
        # Fit sklearn models
        log_model, _ = self.tune_logistic_regression()
        xgb_model, _ = self.tune_xgboost()
        mlp_model = load_model('../saved_models/best_model.h5')
        
        # Scale features for MLP
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Predict classes
        log_preds = log_model.predict(self.X_test)
        xgb_preds = xgb_model.predict(self.X_test)
        mlp_preds = (mlp_model.predict(X_test_scaled) > 0.5).astype("int32").flatten()

        # Majority voting
        preds_matrix = np.array([log_preds, xgb_preds, mlp_preds])
        final_preds = np.array([np.bincount(row).argmax() for row in preds_matrix.T])

        acc = accuracy_score(self.y_test, final_preds)
        return final_preds, acc
    
    def basic_elo_prediction(self):
        self.df['elo_prediction'] = np.where(self.df['precomp_elo'] > self.df['opp_precomp_elo'], 1, 0)
        self.df['elo_prediction'] = np.where(self.df['precomp_elo'] == self.df['opp_precomp_elo'], 0.5, self.df['elo_prediction'])
        self.df.dropna(subset=['elo_prediction'], inplace=True)
        acc = accuracy_score(self.df['win'], self.df['elo_prediction'])
        return acc
    
    def basic_elo_pred(self):
        correct = 0
        total = 0

        for _, row in self.test_df.iterrows():
            fighter_elo = row['precomp_elo']
            opponent_elo = row['opp_precomp_elo']
            win = row['win']

            if fighter_elo > opponent_elo and win == 1:
                correct += 1
            elif fighter_elo < opponent_elo and win == 0:
                correct += 1
            # if equal or wrong prediction, don't count as correct
            total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def print_top_n_fighters_by_elo(self, n=10):
        if 'FIGHTER' not in self.df.columns:
            print("Column 'name' not found in dataset.")
            return

        # Sort by date to get the latest Elo per fighter
        latest_elos = self.df.sort_values('DATE').groupby('FIGHTER').tail(1)

        # Drop NaNs just in case
        latest_elos = latest_elos.dropna(subset=['precomp_elo'])

        # Sort descending by Elo and print top n
        top_fighters = latest_elos.sort_values(by='precomp_elo', ascending=False).head(n)

        print(f"\nTop {n} Fighters by Elo Rating:")
        print(top_fighters[['FIGHTER', 'precomp_elo']])
    def elo_log_loss(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x / 200))  # 400 is standard in Elo logistic scaling

        # Calculate the probability that the fighter wins using Elo difference
        elo_prev = self.test_df['precomp_elo'] - self.test_df['opp_precomp_elo']
        probs = sigmoid(elo_prev)

        # Ground truth
        y_true = self.test_df['win']

        # Compute log loss
        loss = log_loss(y_true, probs)

        # Also print average predicted confidence
        avg_conf = np.mean(np.maximum(probs, 1 - probs))
        print(f"Elo Log Loss: {loss:.4f}")
        print(f"Avg Confidence: {avg_conf:.4f}")
        return loss
