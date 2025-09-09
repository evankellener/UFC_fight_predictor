import pandas as pd
from datetime import timedelta, datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.models import load_model
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from statsmodels.stats.proportion import proportion_confint
import shap
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from datetime import datetime

from sklearn.metrics import accuracy_score, log_loss

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

def prob_to_american_odds(p):
    """
    Convert win probability p (0 < p < 1) into American odds.
    - If p >= 0.5 → negative odds: amount you must risk to win 100.
    - If p < 0.5  → positive odds: amount you win on a 100 risk.
    """
    if p <= 0 or p >= 1:
        return np.nan
    if p >= 0.5:
        odds = - (p / (1 - p)) * 100
    else:
        odds = ((1 - p) / p) * 100
    return int(np.sign(odds) * np.round(abs(odds)))


def make_consistent_odds_table(test_df, probs):
    """
    Build a table of consistent moneyline odds per fight.
    Columns: DATE, EVENT, BOUT, FIGHTER, prob_norm, odds
    """
    df = test_df[['DATE', 'EVENT', 'BOUT', 'FIGHTER']].copy()
    df['prob_raw'] = probs
    rows = []
    for bout_id, grp in df.groupby('BOUT'):
        grp = grp.copy()
        if len(grp) == 2:
            p1, p2 = grp['prob_raw'].values
            total = p1 + p2
            grp.loc[grp.index[0], 'prob_norm'] = p1 / total
            grp.loc[grp.index[1], 'prob_norm'] = p2 / total
        else:
            grp['prob_norm'] = grp['prob_raw']
        grp['odds'] = grp['prob_norm'].map(prob_to_american_odds)
        rows.append(grp)
    result = pd.concat(rows, ignore_index=True)
    return result[['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'prob_norm', 'odds']]


class FightOutcomeModel:
    def __init__(self, file_path, scaler_path='../saved_models/feature_scaler.joblib'):
        self.elo_columns = [
            'precomp_elo', 'precomp_elo_prev', 'precomp_elo_change_3', 'precomp_elo_change_5',
            'opp_precomp_elo', 'opp_precomp_elo_prev', 'opp_precomp_elo_change_3', 'opp_precomp_elo_change_5'
        ]
        self.main_stats_cols = [
            'age', 'HEIGHT', 'WEIGHT', 'REACH','weightindex','age_ratio_difference',
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
            'opp_age', 'opp_HEIGHT', 'opp_WEIGHT', 'opp_REACH','opp_weightindex', 'opp_weight_of_fight','opp_age_ratio_difference',
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
            'opp_precomp_winsum3', 'opp_precomp_losssum3','opp_precomp_elo_change_3','precomp_strike_elo', 'opp_precomp_strike_elo',
            'precomp_strike_elo_change_3', 'opp_precomp_strike_elo_change_3','precomp_strike_elo_change_5', 'opp_precomp_strike_elo_change_5'
        ]

        self.importance_columns = [
            'WEIGHT',
            'opp_WEIGHT',
            'precomp_elo_change_3',
            'opp_precomp_elo_change_3',
            'precomp_elo_change_5',
            'opp_precomp_elo_change_5',
            'opp_precomp_strike_elo',
            'precomp_strike_elo',
            'precomp_strike_elo_change_5',
            'opp_precomp_strike_elo_change_5',
            'precomp_strike_elo_change_3',
            'opp_precomp_strike_elo_change_3',
            'opp_REACH',
            'REACH',
            'opp_weight_of_fight',
            'opp_weightindex',
            'weightindex',
            'opp_HEIGHT',
            'HEIGHT',
            'opp_precomp_subavg5',
            'precomp_subavg5',
            'opp_precomp_subavg',
            'precomp_subavg',
            'opp_precomp_subavg3',
            'precomp_subavg3',
            'opp_precomp_tdavg',
            'precomp_tdavg',
            'precomp_tdavg3',
            'opp_precomp_tdavg3',
            'precomp_tdavg5'
        ]

        # best log loss:
        """
        'age_ratio_difference','opp_age_ratio_difference','opp_precomp_elo_change_5', 'precomp_elo','opp_precomp_elo','precomp_tdavg', 'opp_precomp_tdavg','opp_precomp_tddef',
            'opp_precomp_sapm5','precomp_tddef','precomp_sapm5','precomp_headacc_perc3','opp_precomp_headacc_perc3','precomp_totalacc_perc3','precomp_elo_change_5','REACH','opp_REACH',
            'precomp_legacc_perc5','opp_precomp_totalacc_perc3','opp_precomp_legacc_perc5','opp_precomp_clinchacc_perc5','precomp_clinchacc_perc5','precomp_winsum3','opp_precomp_winsum3',
            'opp_precomp_sapm','precomp_sapm','opp_precomp_totalacc_perc','precomp_totalacc_perc','precomp_groundacc_perc5', 'opp_precomp_groundacc_perc5','precomp_losssum5','opp_precomp_losssum5',
            'age','opp_age','precomp_strike_elo', 'opp_precomp_strike_elo'
        """
        self.df = pd.read_csv(file_path)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df = self.df[self.df['DATE'] >= '2009-01-01']
        self.df = self.df[self.df['sex'] == 2]
        self.scaler = load(scaler_path)
        self._prepare_data()
        self.debug_data_split()

    def _prepare_data(self):
        latest = self.df['DATE'].max()
        cutoff = latest - timedelta(days=548)
        valid_cols = [c for c in getattr(self, 'importance_columns', []) if c in self.df.columns]
        self.df = self.df.dropna(subset=['win'])
        thresh = int(0.7 * len(valid_cols))
        self.df = self.df[self.df[valid_cols].isnull().sum(axis=1) < thresh]
        imp = SimpleImputer(strategy='median')
        self.df[valid_cols] = imp.fit_transform(self.df[valid_cols])
        self.train_df = self.df[self.df['DATE'] < cutoff]
        self.test_df  = self.df[self.df['DATE'] >= cutoff]
        self.X_train  = self.train_df[valid_cols]
        self.y_train  = self.train_df['win']
        self.X_test   = self.test_df[valid_cols]
        self.y_test   = self.test_df['win']
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Feature count: {len(valid_cols)}")
        return self.X_train, self.y_train, self.X_test, self.y_test

    def covariance_feature_analysis(self, top_n=20):
        import seaborn as sns

        # Ensure we're working with numeric features
        numeric_features = self.X_train.select_dtypes(include=[np.number]).copy()
        numeric_features['win'] = self.y_train

        # Compute Pearson correlation
        correlation_matrix = numeric_features.corr()

        # Extract correlation with 'win'
        win_corr = correlation_matrix['win'].drop('win')

        # Sort by absolute correlation values
        sorted_corr = win_corr.reindex(win_corr.abs().sort_values(ascending=False).index)

        print("\n🔍 Top Features Most Correlated with 'win':")
        print(sorted_corr.head(top_n))

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_corr.abs().head(top_n), y=sorted_corr.head(top_n).index, palette="viridis")
        plt.title(f"Top {top_n} Features Correlated with 'win'")
        plt.xlabel("Absolute Pearson Correlation")
        plt.ylabel("Feature")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return sorted_corr

    def build_mlp(self):
        model = load_model('../saved_models/best_model.h5')

        # Scale the features
        X_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Predict and evaluate
        preds = (model.predict(X_test_scaled) > 0.5).astype("int32").flatten()
        acc = accuracy_score(self.y_test, preds)
        return model, acc

    from sklearn.impute import SimpleImputer

    def debug_data_split(self):
        print("\n🔍 Data Split Diagnostics:")
        print(f"Train win rate: {self.y_train.mean():.3f}")
        print(f"Test  win rate: {self.y_test.mean():.3f}\n")
        print("Train feature summaries:\n", self.X_train.describe().transpose())
        print("Test  feature summaries:\n", self.X_test.describe().transpose())

    def tune_logistic_regression(self):
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=10000, random_state=42))
        ])
        params = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'saga'],
            'clf__class_weight': [None, 'balanced']
        }
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(pipeline, params, cv=tscv, scoring='accuracy', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        best = grid.best_estimator_
        self.probs = best.predict_proba(self.X_test)[:, 1]
        self.ml_odds = [prob_to_american_odds(p) for p in self.probs]
        ll = log_loss(self.y_test, self.probs)
        preds = best.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print("Best params:", grid.best_params_)
        print(f"Train accuracy: {best.score(self.X_train, self.y_train):.3f}")
        print(f"Log loss: {ll:.3f}")
        print(f"Test accuracy: {acc:.3f}")
        # 95% confidence interval for test accuracy
        count = int(acc * len(self.y_test))
        lower, upper = proportion_confint(count=count, nobs=len(self.y_test), method='wilson')
        print(f"95% CI for test accuracy: {lower:.3f} - {upper:.3f}\n")

        # SHAP explanations
        display_df = self.test_df[['DATE', 'EVENT', 'BOUT', 'FIGHTER']].copy()
        display_df['prob_win'] = np.round(self.probs, 3)
        display_df['odds'] = self.ml_odds
        print(display_df.head(5).to_string(index=False))
        model = best.named_steps['clf']
        imputed = best.named_steps['imputer'].transform(self.X_test)
        scaled  = best.named_steps['scaler'].transform(imputed)
        expl = shap.Explainer(model, scaled)
        sv = expl(scaled)
        shap.summary_plot(sv, scaled, feature_names=self.X_test.columns)
        self.best_model = best
        return best, acc
    
    def generate_odds_table(self):
        if not hasattr(self, 'probs'):
            raise RuntimeError("Run tune_logistic_regression() first.")
        return make_consistent_odds_table(self.test_df, self.probs)
    
    #def calculate_roi(self, odds_data):
    def calculate_roi(self,
                      odds_table_path: str,
                      vegas_data_path: str,
                      vegas_cols: list = None,
                      stake: float = 100) -> pd.DataFrame:
        """
        Compute betting ROI by merging your model's odds with Vegas lines.
        - odds_table_path: path to CSV from generate_odds_table (DATE, EVENT, BOUT, FIGHTER, odds)
        - vegas_data_path: path to your full dataset CSV (must include win + Vegas odds columns)
        - vegas_cols: list of Vegas odds column names, defaults to DraftKings, FanDuel, BetMGM, Bovada
        - stake: amount to risk per fight (default $100)
        Returns a DataFrame of picks with cumulative ROI metrics.
        """
        import pandas as pd

        # 1. Load model odds and Vegas data
        df_model = pd.read_csv(odds_table_path, parse_dates=['DATE'])
        df_vegas = pd.read_csv(vegas_data_path, parse_dates=['DATE'])

        # 2. Drop timezone if present
        try:
            df_vegas['DATE'] = df_vegas['DATE'].dt.tz_convert(None)
        except Exception:
            pass

        # 3. Prepare merge keys
        key_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER']
        # Default sportsbooks
        vegas_cols = vegas_cols or ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bovada_odds']

        # 4. Merge
        df = pd.merge(
            df_model[key_cols + ['odds']],
            df_vegas[key_cols + ['win'] + vegas_cols],
            on=key_cols,
            how='inner'
        )

        # 5. Average the Vegas odds
        df['avg_vegas_odds'] = df[vegas_cols].mean(axis=1, skipna=True)

        # 6. Pick your model's favorite by lowest American odds
        idx = df.groupby('BOUT')['odds'].idxmin()
        picks = df.loc[idx].copy()

        # 7. Calculate profit per fight
        def calc_profit(row):
            ml = row['avg_vegas_odds']
            if row['win'] == 1:
                return stake * (ml / 100) if ml > 0 else stake * (100 / abs(ml))
            return -stake

        picks['stake'] = stake
        picks['profit'] = picks.apply(calc_profit, axis=1)

        # 8. Compute cumulative ROI
        picks = picks.sort_values('DATE')
        picks['cum_profit'] = picks['profit'].cumsum()
        picks['cum_stake'] = picks['stake'].cumsum()
        picks['cum_roi'] = picks['cum_profit'] / picks['cum_stake']

        # 9. Report
        overall = picks['cum_roi'].iloc[-1]
        print(f"Overall ROI: {overall:.4%}")

        return picks


    def tune_svm(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True))
        ])
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['rbf'],
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
        print("Best parameters for XGBoost:")
        print(param_grid)
        print("Fitting XGBoost model...")
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
        svm_model, _ = self.tune_svm()
        print("svm_model log probabilities", svm_model)
        nb_model, _ = self.build_naive_bayes()
        print("nb_model log probabilities", nb_model)
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
        self.df['elo_prediction'] = np.where(self.df['precomp_strike_elo'] > self.df['precomp_strike_elo'], 1, 0)
        self.df['elo_prediction'] = np.where(self.df['precomp_strike_elo'] == self.df['opp_precomp_strike_elo'], 0.5, self.df['elo_prediction'])
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
    
    def find_best_feature_to_add(self, base_features=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, log_loss

        if base_features is None:
            base_features = self.importance_columns

        base_set = set(base_features)
        all_candidates = [f for f in self.main_stats_cols if f not in base_set]
        results = []

        print(f"Evaluating {len(all_candidates)} features not in importance_columns...\n")

        for candidate in all_candidates:
            current_features = base_features + [candidate]

            # Drop missing values specific to current feature set
            sub_train = self.train_df.copy()
            sub_test = self.test_df.copy()

            train_medians = sub_train[current_features].median()
            sub_train[current_features] = sub_train[current_features].fillna(train_medians)
            sub_test[current_features] = sub_test[current_features].fillna(train_medians)

            # Fill missing with median to match _prepare_data logic
            X_train = sub_train[current_features]
            y_train = sub_train['win']
            X_test = sub_test[current_features]
            y_test = sub_test['win']


            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    max_iter=10000,
                    C=1,
                    class_weight='balanced',
                    penalty='l2',
                    solver='liblinear',
                    random_state=42
                ))
            ])

            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                probs = pipeline.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, preds)
                loss = log_loss(y_test, probs)

                results.append({
                    'feature_added': candidate,
                    'accuracy': acc,
                    'log_loss': loss
                })
            except Exception as e:
                print(f"⚠️ Skipping {candidate} due to error: {e}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['log_loss', 'accuracy'], ascending=[True, True])

        print("\n🏆 Top 5 candidates by lowest log loss:")
        print(results_df.head(5))

        best_feature = results_df.iloc[0]['feature_added']
        print(f"\n✅ Best feature to add: {best_feature}")
        return best_feature, results_df

    
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
            return 1 / (1 + np.exp(-x / 170))  # 400 is standard in Elo logistic scaling

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
    
    def print_fighter_elo(self, fighter_name):
        #graph the fighter's elo over time
        #make the graph pretty and mark every time the fighter fought
        #make sure to show the precomp_elo and postcomp_elo
        if 'FIGHTER' not in self.df.columns:
            print("Column 'name' not found in dataset.")
            return
        fighter_data = self.df[self.df['FIGHTER'] == fighter_name]
        if fighter_data.empty:
            print(f"No data found for fighter: {fighter_name}")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(fighter_data['DATE'], fighter_data['precomp_elo'], label='Pre-Fight Elo', color='blue', alpha=0.7)
        plt.plot(fighter_data['DATE'], fighter_data['postcomp_elo'], label='Post-Fight Elo', color='red', alpha=0.7)
        plt.title(f'Elo Ratings Over Time for {fighter_name}')
        plt.xlabel('Date')
        plt.ylabel('Elo Rating')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        return fighter_data[['DATE', 'precomp_elo', 'postcomp_elo']]
    
    def plot_elo_distribution(self):
        # Plot the distribution of Elo ratings
        plt.figure(figsize=(12, 6))
        plt.hist(self.df['precomp_elo'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribution of Fighter Elo Ratings')
        plt.xlabel('Elo Rating')
        plt.ylabel('Frequency')
        plt.grid()
        plt.tight_layout()
        plt.show()
        return self.df['precomp_elo'].describe()
    
    def plot_elo_vs_outcome(self):
        # Plot Elo ratings against fight outcomes
        plt.figure(figsize=(12, 6))
        plt.scatter(self.df['precomp_elo'], self.df['win'], alpha=0.5, color='blue')
        plt.title('Elo Ratings vs Fight Outcomes')
        plt.xlabel('Elo Rating')
        plt.ylabel('Fight Outcome (1 = Win, 0 = Loss)')
        plt.grid()
        plt.tight_layout()
        plt.show()
        return self.df[['precomp_elo', 'win']].describe()
    
    def plot_feature_importance(self, model):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError("Model does not have feature importances or coefficients.")

        # Sort the feature importances
        indices = np.argsort(importances)[::-1]
        features = self.X_train.columns[indices]

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), features, rotation=90)
        plt.xlim([-1, len(importances)])
        plt.tight_layout()
        plt.show()
        return importances[indices]


    def hierarchical_feature_selector(self, total_features=50, n_batches=10, top_per_batch=10, scoring_metric='neg_log_loss'):
        import time
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, log_loss

        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        # Only use numeric columns that survive imputation
        numeric_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_numeric = self.X_train[numeric_features].copy()
        X_test_numeric = self.X_test[numeric_features].copy()

        X_train_imputed = imputer.fit_transform(X_train_numeric)
        X_test_imputed = imputer.transform(X_test_numeric)

        # After imputation, get final valid feature count (some cols may have been dropped)
        valid_feature_count = X_train_imputed.shape[1]
        feature_names_after_imputation = [numeric_features[i] for i in range(valid_feature_count)]

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=feature_names_after_imputation)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=feature_names_after_imputation)

        total_feature_count = len(feature_names_after_imputation)
        batch_size = total_feature_count // n_batches

        base_model = LogisticRegression(
            max_iter=10000,
            C=10,
            class_weight='balanced',
            penalty='l2',
            solver='liblinear',
            random_state=42
        )

        selected_feature_names = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size if i != n_batches - 1 else total_feature_count
            batch_features = feature_names_after_imputation[start_idx:end_idx]
            X_train_batch = X_train_scaled[batch_features].values

            print(f"\n📦 Batch {i+1}: Selecting from {len(batch_features)} features")

            selector = SequentialFeatureSelector(
                base_model,
                n_features_to_select=min(top_per_batch, len(batch_features)),
                direction='forward',
                scoring=scoring_metric,
                cv=3,
                n_jobs=-1
            )

            selector.fit(X_train_batch, self.y_train)
            selected_batch_features = [batch_features[idx] for idx, selected in enumerate(selector.get_support()) if selected]
            selected_feature_names.extend(selected_batch_features)

        selected_feature_names = sorted(list(set(selected_feature_names)))
        print(f"\n✅ Total candidates after batch selection: {len(selected_feature_names)}")

        X_train_reduced = X_train_scaled[selected_feature_names].values
        X_test_reduced = X_test_scaled[selected_feature_names].values

        print("\n🚀 Running final selection on reduced feature set...")
        selector_final = SequentialFeatureSelector(
            base_model,
            n_features_to_select=min(total_features, len(selected_feature_names)),
            direction='forward',
            scoring=scoring_metric,
            cv=5,
            n_jobs=-1
        )

        start_time = time.time()
        selector_final.fit(X_train_reduced, self.y_train)
        selected_mask = selector_final.get_support()
        final_features = [selected_feature_names[idx] for idx, sel in enumerate(selected_mask) if sel]
        duration = time.time() - start_time

        X_train_final = X_train_scaled[final_features].values
        X_test_final = X_test_scaled[final_features].values

        model = base_model.fit(X_train_final, self.y_train)
        preds = model.predict(X_test_final)
        probs = model.predict_proba(X_test_final)[:, 1]

        acc = accuracy_score(self.y_test, preds)
        loss = log_loss(self.y_test, probs)

        print(f"\n🎯 Final Accuracy: {acc:.4f}")
        print(f"📉 Final Log Loss: {loss:.4f}")
        print(f"⏱ Total Time: {duration:.2f} seconds")
        print(f"🏁 Final Selected Features: {final_features}")

        return final_features, acc, loss, duration
