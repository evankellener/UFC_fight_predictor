import pandas as pd
import os
from datetime import timedelta, datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from keras.models import load_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        # Fallback for when neither keras nor tensorflow.keras is available
        load_model = None
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV, SelectFromModel
from sklearn.impute import SimpleImputer
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
try:
    import shap
except ImportError:
    shap = None
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

# Odds filtering: ensure only valid sportsbook odds are used in the pipeline
try:
    from odd_filter import filter_sportsbook_odds
    filter_sportsbook_odds(
        input_csv="final_with_odds.csv",
        output_csv="final_with_odds_filtered.csv"
    )
except Exception as e:
    print(f"[Warning] Odds filtering step failed: {e}")

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
        # --- Odds rounding logic ---
        def fix_model_odds(odds):
            if 0 < odds < 100:
                return 100
            if -100 < odds < 0:
                return -100
            return odds
        grp['odds'] = grp['odds'].apply(fix_model_odds)
        rows.append(grp)
    result = pd.concat(rows, ignore_index=True)
    return result[['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'prob_norm', 'odds']]


class FightOutcomeModel:
    def __init__(self, file_path, scaler_path=None):
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
            'precomp_str_eff_diff', 'precomp_str_eff_diff3', 'precomp_str_eff_diff5',
            'precomp_totalstr_pm', 'precomp_totalstr_pm3', 'precomp_totalstr_pm5',
            'precomp_grapple_strike_mix', 'precomp_grapple_strike_mix3', 'precomp_grapple_strike_mix5',
            'precomp_finish_rate', 'precomp_finish_rate3', 'precomp_finish_rate5',
            'precomp_ctrl_per_min', 'precomp_ctrl_per_min3', 'precomp_ctrl_per_min5',
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
            'opp_precomp_str_eff_diff', 'opp_precomp_str_eff_diff3', 'opp_precomp_str_eff_diff5',
            'opp_precomp_totalstr_pm', 'opp_precomp_totalstr_pm3', 'opp_precomp_totalstr_pm5',
            'opp_precomp_grapple_strike_mix', 'opp_precomp_grapple_strike_mix3', 'opp_precomp_grapple_strike_mix5',
            'opp_precomp_finish_rate', 'opp_precomp_finish_rate3', 'opp_precomp_finish_rate5',
            'opp_precomp_ctrl_per_min', 'opp_precomp_ctrl_per_min3', 'opp_precomp_ctrl_per_min5',
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
            #'age_ratio_difference', 'opp_age_ratio_difference', 'opp_precomp_elo', 'opp_precomp_sigstr_perc5', 'opp_precomp_strdef5', 'opp_precomp_strike_elo', 'opp_precomp_tdavg', 'precomp_elo', 'precomp_strdef5', 'precomp_strike_elo', 'precomp_tdavg',   
            'age_ratio_difference','opp_age_ratio_difference','precomp_elo','opp_precomp_elo', 'precomp_losssum3', 'opp_precomp_losssum3', 'opp_precomp_strike_elo', 'precomp_strike_elo', 'precomp_strdef5', 'opp_precomp_strdef5', 'precomp_sapm5', 'opp_precomp_headacc_perc', 'precomp_headacc_perc', 'opp_precomp_tdavg3', 'precomp_tdavg3', 'precomp_tdavg',
            'weightindex', 'opp_weight_of_fight', 'REACH', 'opp_REACH', 'opp_precomp_tdavg', 'precomp_groundacc_perc3', 'opp_precomp_winsum5', 'opp_precomp_bodyacc_perc', 'opp_precomp_bodyacc_perc5', 'opp_precomp_sigstr_perc', 'precomp_subavg3', 'opp_precomp_headacc_perc3', 'opp_precomp_distacc_perc', 'opp_precomp_totalacc_perc', 'opp_precomp_tddef3', 
            'opp_precomp_sigstr_perc5', 'opp_precomp_subavg3', 'precomp_legacc_perc3', 'opp_age', 'opp_precomp_losssum'
        ]

        # best log loss:
        """
                    'precomp_elo','opp_precomp_elo', 'precomp_elo_change_3', 'opp_precomp_elo_change_3', 'precomp_elo_change_5', 'opp_precomp_elo_change_5', 
            'precomp_tdavg3', 'opp_precomp_tdavg3', 'precomp_tdavg5', 'opp_precomp_tdavg5', 'precomp_tddef3', 'opp_precomp_tddef3', 'precomp_tddef5', 'opp_precomp_tddef5',
            'precomp_totalacc_perc' , 'opp_precomp_totalacc_perc', 'precomp_totalacc_perc3', 'opp_precomp_totalacc_perc3', 'precomp_totalacc_perc5', 'opp_precomp_totalacc_perc5',
            'precomp_strdef', 'opp_precomp_strdef', 'precomp_strdef3', 'opp_precomp_strdef3', 'precomp_strdef5', 'opp_precomp_strdef5',
            'age_ratio_difference', 'opp_age_ratio_difference', 'precomp_strike_elo', 'opp_precomp_strike_elo', 'precomp_strike_elo_change_3', 'opp_precomp_strike_elo_change_3', 'precomp_strike_elo_change_5', 'opp_precomp_strike_elo_change_5',
            'opp_precomp_tdavg', 'precomp_tdavg','opp_precomp_tdacc_perc5', 'precomp_tdacc_perc5', 'REACH', 'opp_REACH', 'precomp_winsum3', 'opp_precomp_winsum3', 'weightindex', 'opp_weightindex', 'weight_of_fight', 'opp_weight_of_fight',
            'precomp_distacc_perc', 'opp_precomp_distacc_perc', 'precomp_tdacc_perc3', 'opp_precomp_tdacc_perc3', 'precomp_legacc_perc3', 'opp_precomp_legacc_perc3', 'precomp_distacc_perc5', 'opp_precomp_headacc_perc3'
        """
        
        self.df = pd.read_csv(file_path, low_memory=False)
        
        # Save the truly unfiltered dataset first (before any preprocessing)
        print("Saving truly unfiltered dataset (before any preprocessing)...")
        truly_unfiltered_path = 'data/tmp/truly_unfiltered_before_preprocessing.csv'
        os.makedirs(os.path.dirname(truly_unfiltered_path), exist_ok=True)
        self.df.to_csv(truly_unfiltered_path, index=False)
        print(f"Truly unfiltered dataset saved: {truly_unfiltered_path} ({len(self.df)} rows)")
        
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df = self.df[self.df['DATE'] >= '2009-01-01']
        # Fix: Convert sex to string for comparison since it's stored as string in the data
        self.df = self.df[self.df['sex'].astype(str) == '2']
        
        
        # Auto-detect scaler path if not provided
        if scaler_path is None:
            # Try different possible paths
            possible_paths = [
                'saved_models/feature_scaler.joblib',
                '../saved_models/feature_scaler.joblib',
                '../../saved_models/feature_scaler.joblib'
            ]
            scaler_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    scaler_path = path
                    break
        
        try:
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = load(scaler_path)
            else:
                raise FileNotFoundError("Scaler file not found")
        except (ModuleNotFoundError, ImportError, FileNotFoundError) as e:
            print(f"Warning: Could not load scaler from {scaler_path}: {e}")
            print("Creating a new scaler...")
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        self._prepare_data()
        self.debug_data_split()

    def _prepare_data(self):
        latest = self.df['DATE'].max()
        cutoff = latest - timedelta(days=365)
        valid_cols = [c for c in getattr(self, 'importance_columns', []) if c in self.df.columns]
        self.df = self.df.dropna(subset=['win'])
        # Fix: Convert win column to integer since it's stored as string
        self.df['win'] = self.df['win'].astype(int)
        thresh = int(0.7 * len(valid_cols))
        self.df = self.df[self.df[valid_cols].isnull().sum(axis=1) < thresh]
        imp = SimpleImputer(strategy='median')
        self.df[valid_cols] = imp.fit_transform(self.df[valid_cols])
        
        # Save unfiltered dataset before filtering
        print("Saving unfiltered dataset...")
        unfiltered_df = self.df.copy()
        unfiltered_path = 'data/tmp/unfiltered_before_training.csv'
        os.makedirs(os.path.dirname(unfiltered_path), exist_ok=True)
        unfiltered_df.to_csv(unfiltered_path, index=False)
        print(f"Unfiltered dataset saved: {unfiltered_path} ({len(unfiltered_df)} rows)")
        
        # Apply filtering during training: exclude fights where precomp_boutcount < 1
        # This filters out first fights but keeps the full dataset for display purposes
        print("Applying precomp_boutcount filtering for training (min_fights=1)...")
        original_size = len(self.df)
        self.df = self.df[
            (self.df['precomp_boutcount'] >= 1) &
            (self.df['opp_precomp_boutcount'] >= 1)
        ]
        filtered_size = len(self.df)
        print(f"Filtering complete: {original_size} -> {filtered_size} rows ({original_size - filtered_size} removed)")
        
        # Save filtered dataset after filtering
        print("Saving filtered dataset...")
        filtered_path = 'data/tmp/filtered_after_training.csv'
        os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
        self.df.to_csv(filtered_path, index=False)
        print(f"Filtered dataset saved: {filtered_path} ({len(self.df)} rows)")
        
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
        # Use only elo features for MLP
        elo_features = self.importance_columns
        X_train_aligned = self.X_train[elo_features]
        X_test_aligned = self.X_test[elo_features]

        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import GridSearchCV

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train_aligned)
        X_test_scaled = scaler.transform(X_test_aligned)

        # Define a parameter grid for hidden_layer_sizes
        param_grid = {
            'hidden_layer_sizes': [
                (8, 8, 8),
                (16, 8, 4),
                (32, 16),
                (16, 16, 8),
                (32, 16, 8),
                (32, 32),
                (64, 32, 16),
                (16,),
                (32,),
                (64, 32)
            ],
            'alpha': [0.0001, 0.001, 0.01],
            'activation': ['relu', 'tanh'],
            'solver': ['adam']
        }

        mlp = MLPClassifier(max_iter=300, random_state=42)
        grid = GridSearchCV(
            mlp,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_scaled, self.y_train)
        best_model = grid.best_estimator_

        preds = (best_model.predict(X_test_scaled) > 0.5).astype("int32").flatten()
        acc = accuracy_score(self.y_test, preds)
        print("Best MLP params:", grid.best_params_)
        print(f"MLP Test accuracy: {acc:.3f}")
        return best_model, acc

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
            ('scaler', RobustScaler()),
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
        
        if shap is not None:
            model = best.named_steps['clf']
            imputed = best.named_steps['imputer'].transform(self.X_test)
            scaled  = best.named_steps['scaler'].transform(imputed)
            expl = shap.Explainer(model, scaled)
            sv = expl(scaled)
            shap.summary_plot(sv, scaled, feature_names=self.X_test.columns)
        else:
            print("SHAP not available - skipping feature importance analysis")
        self.best_model = best
        return best, acc
    
    def generate_odds_table(self):
        """
        Generate odds table from model predictions.
        Returns a DataFrame with DATE, EVENT, BOUT, FIGHTER, prob_norm, odds columns.
        """
        if not hasattr(self, 'probs'):
            raise RuntimeError("Run tune_logistic_regression() first.")
        
        # Use ALL test data (including 2025)
        # No date filtering - use all test data that has outcomes
        test_df_filtered = self.test_df.copy()
        
        if len(test_df_filtered) == 0:
            print("Warning: No test data found.")
            return pd.DataFrame()
        
        print(f"Generating odds table for {len(test_df_filtered)} test fights")
        
        # Get probabilities for filtered data
        # We need to get the corresponding probabilities for the filtered test data
        # Create a mapping from original test_df index to position in probs array
        test_df_positions = {idx: pos for pos, idx in enumerate(self.test_df.index)}
        filtered_positions = [test_df_positions[idx] for idx in test_df_filtered.index if idx in test_df_positions]
        probs_filtered = self.probs[filtered_positions]
        
        return make_consistent_odds_table(test_df_filtered, probs_filtered)
    
    def scrape_and_filter_odds(self, input_csv_path: str, output_csv_path: str = None):
        """
        Scrape odds from API and apply improved filtering with odds clamping.
        Combines functionality from odds_api.py and improved_odd_filter.py
        
        Args:
            input_csv_path: Path to input CSV file with fight data
            output_csv_path: Path to save processed CSV file (optional)
            
        Returns:
            DataFrame with processed odds data
        """
        import re
        import requests
        from datetime import timedelta
        from difflib import SequenceMatcher
        
        # API configuration
        API_KEY = os.getenv('ODDS_API_KEY')
        SPORT = 'mma_mixed_martial_arts'
        REGIONS = 'us'
        MARKETS = 'h2h'
        ODDS_FORMAT = 'american'
        DATE_FMT = 'iso'
        LOOKBACK_DAYS = 1200
        HIST_URL = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds'
        MAIN_BOOKMAKERS = ['draftkings', 'fanduel', 'betmgm', 'bet365', 'bovada']
        FUZZY_THRESHOLD = 0.8
        
        def normalize(name):
            return re.sub(r'\W+', '', (name or '').lower())
        
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()
        
        def fetch_snapshot(ts_iso):
            try:
                r = requests.get(HIST_URL, params={
                    'apiKey': API_KEY, 'regions': REGIONS,
                    'markets': MARKETS, 'oddsFormat': ODDS_FORMAT,
                    'dateFormat': DATE_FMT, 'date': ts_iso,
                })
                r.raise_for_status()
                return r.json().get('data', [])
            except Exception as e:
                print(f"Error fetching data for {ts_iso}: {e}")
                return []
        
        def find_best_event(fn, on, row_date, ev_list):
            """Return the best-matching event for fighter vs opponent on row_date (±1 day)."""
            candidates = []
            for ev in ev_list:
                ev_date = ev['commence_dt'].date()
                if abs((ev_date - row_date).days) > 1:
                    continue

                home, away = ev['home'], ev['away']
                # exact both-way match
                if {fn, on} == {home, away}:
                    return ev

                # fuzzy both-way match
                sim1 = similar(fn, home) + similar(on, away)
                sim2 = similar(fn, away) + similar(on, home)
                if max(sim1, sim2) / 2 >= FUZZY_THRESHOLD:
                    candidates.append((max(sim1, sim2) / 2, ev))

            # return highest-scoring fuzzy candidate, if any
            if candidates:
                return max(candidates, key=lambda x: x[0])[1]
            return None
        
        def clamp_odds_to_realistic_ranges(df):
            """Clamp extreme odds values to realistic ranges"""
            odds_columns = [col for col in df.columns if col.endswith('_odds')]
            
            print("=== APPLYING ODDS CLAMPING ===")
            
            for col in odds_columns:
                if col in df.columns:
                    # Convert to numeric, errors become NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Apply clamping logic (LESS AGGRESSIVE: Only clamp extreme values)
                    # Only clamp odds that are clearly unrealistic (> 200 or < -200)
                    mask_positive = (df[col] > 0) & (df[col] > 200)
                    mask_negative = (df[col] < 0) & (df[col] < -200)
                    
                    # Clamp positive odds > 200 to 200 (less aggressive)
                    df.loc[mask_positive, col] = 200
                    
                    # Clamp negative odds < -200 to -200 (less aggressive)
                    df.loc[mask_negative, col] = -200
                    
                    # Count changes
                    clamped_positive = mask_positive.sum()
                    clamped_negative = mask_negative.sum()
                    
                    if clamped_positive + clamped_negative > 0:
                        print(f"  {col}: Clamped {clamped_positive + clamped_negative} values")
            
            return df
        
        def improved_filter_sportsbook_odds(df, thresholds=None, handle_missing_odds="average_available"):
            """Improved filter for sportsbook odds that handles missing values intelligently"""
            if thresholds is None:
                thresholds = {
                    'draftkings_odds': 5000,
                    'fanduel_odds': 3500,
                    'betmgm_odds': 5000,
                    'bet365_odds': 5000,
                    'bovada_odds': 5000,
                }
            
            odds_columns = list(thresholds.keys())
            
            print("=== APPLYING IMPROVED ODDS FILTERING ===")
            print(f"Handling missing odds: {handle_missing_odds}")
            
            # Step 1: Filter out-of-bounds values
            for col, max_abs in thresholds.items():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    before_count = df[col].notna().sum()
                    df.loc[df[col].abs() > max_abs, col] = np.nan
                    after_count = df[col].notna().sum()
                    removed = before_count - after_count
                    if removed > 0:
                        print(f"  {col}: {removed} out-of-bounds values set to NaN")
            
            # Step 2: Handle missing odds based on strategy
            if handle_missing_odds == "average_available":
                print("\n=== AVERAGING AVAILABLE ODDS ===")
                
                # Calculate average odds for each fight using only available odds
                df['avg_odds_calculated'] = df[odds_columns].mean(axis=1, skipna=True)
                
                # Count fights with different numbers of available odds
                for i in range(1, len(odds_columns) + 1):
                    count = df[odds_columns].notna().sum(axis=1).eq(i).sum()
                    if count > 0:
                        print(f"  {count} fights with {i} available odds")
            
            # Step 3: Final statistics
            print(f"\n=== FINAL STATISTICS ===")
            for col in odds_columns:
                if col in df.columns:
                    total = len(df)
                    valid = df[col].notna().sum()
                    missing = total - valid
                    print(f"  {col}: {valid}/{total} valid ({missing} missing)")
            
            return df
        
        print("=== ODDS SCRAPING AND FILTERING ===")
        print(f"Loading data from: {input_csv_path}")
        
        # Load data
        df = pd.read_csv(input_csv_path, parse_dates=['DATE'])
        df['DATE'] = pd.to_datetime(df['DATE'], utc=True)
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=LOOKBACK_DAYS)
        df = df[df['DATE'] >= cutoff].copy()
        
        print(f"Processing {len(df)} fights from {df['DATE'].min().date()} to {df['DATE'].max().date()}")
        
        # Prepare for matching
        df['f_norm'] = df['FIGHTER'].apply(normalize)
        df['o_norm'] = df['opp_FIGHTER'].apply(normalize)
        df['date_str'] = df['DATE'].dt.strftime('%Y-%m-%d')
        
        # 1) Fetch and build ev_list
        print("\n=== FETCHING ODDS DATA ===")
        raw = []
        for d in sorted(df['date_str'].unique()):
            base = datetime.fromisoformat(d)
            for delta in (0, 1):
                ts = (base + timedelta(days=delta)).strftime('%Y-%m-%dT00:00:00Z')
                print(f"  Fetching {ts}")
                raw.extend(fetch_snapshot(ts))
        
        # Dedupe events
        seen = {}
        for e in raw:
            seen[e['id']] = e
        ev_list = []
        for e in seen.values():
            ct = pd.to_datetime(e['commence_time'], utc=True)
            ev_list.append({
                'commence_dt': ct,
                'home': normalize(e['home_team']),
                'away': normalize(e['away_team']),
                'bookmakers': e.get('bookmakers', []),
            })
        
        print(f"Found {len(ev_list)} unique events")
        
        # 2) Prepare output cols
        for bk in MAIN_BOOKMAKERS:
            df[f"{bk}_odds"] = None
        
        # 3) Row-by-row match with fuzzy fallback
        print("\n=== MATCHING FIGHTS TO ODDS ===")
        matches = 0
        for idx, row in df.iterrows():
            fn, on = row['f_norm'], row['o_norm']
            row_date = row['DATE'].date()
            ev = find_best_event(fn, on, row_date, ev_list)
            if not ev:
                continue
            
            matches += 1
            for bm in ev['bookmakers']:
                key = bm.get('key')
                if key not in MAIN_BOOKMAKERS:
                    continue
                h2h = next((m for m in bm['markets'] if m['key'] == 'h2h'), None)
                if not h2h:
                    continue
                # pull this row's fighter price
                price = next((o['price'] for o in h2h['outcomes'] 
                              if normalize(o['name']) == fn), None)
                df.at[idx, f"{key}_odds"] = price
        
        print(f"Matched {matches} fights to odds data")
        
        # Clean up temporary columns
        df.drop(columns=['f_norm', 'o_norm', 'date_str'], inplace=True)
        
        # 4) Apply data quality fixes
        print("\n=== APPLYING DATA QUALITY FIXES ===")
        
        # Apply odds clamping
        df = clamp_odds_to_realistic_ranges(df)
        
        # Apply improved filtering
        df = improved_filter_sportsbook_odds(df, handle_missing_odds="average_available")
        
        # 5) Save results
        if output_csv_path:
            df.to_csv(output_csv_path, index=False)
            print(f"\nProcessed data saved to: {output_csv_path}")
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total fights: {len(df)}")
        odds_cols = [col for col in df.columns if col.endswith('_odds')]
        fights_with_odds = df[odds_cols].notna().any(axis=1).sum()
        print(f"Fights with odds data: {fights_with_odds}")
        
        return df
    
    def filter_odds_outliers(self, df, odds_column, method='iqr', threshold=1.5, vegas_cols=None):
        """
        Filter out statistical outliers in odds data, and apply hard UFC industry limits.
        Args:
            df: DataFrame with odds data
            odds_column: Column name containing odds
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: Multiplier for IQR or standard deviations for z-score
            vegas_cols: List of sportsbook odds columns to check for hard limits
        """
        import numpy as np
        if vegas_cols is None:
            vegas_cols = ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bovada_odds']
        # Hard UFC limits
        upper_limit = 1300
        lower_limit = -1650
        before = len(df)
        mask = (df[vegas_cols] <= upper_limit).all(axis=1) & (df[vegas_cols] >= lower_limit).all(axis=1)
        df = df[mask].copy()
        after = len(df)
        print(f"Hard UFC odds filter: removed {before - after} fights outside +1300/-1650 range.")
        # Remove NaN values
        odds_data = df[odds_column].dropna()
        if method == 'iqr':
            Q1 = odds_data.quantile(0.25)
            Q3 = odds_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            filtered_df = df[(df[odds_column] >= lower_bound) & (df[odds_column] <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((odds_data - odds_data.mean()) / odds_data.std())
            filtered_df = df[z_scores < threshold]
        else:
            filtered_df = df
        print(f"Statistical outlier filter: removed {len(df) - len(filtered_df)} fights by {method}.")
        return filtered_df

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
        import numpy as np
        
        # 1. Load model odds and Vegas data
        df_model = pd.read_csv(odds_table_path, parse_dates=['DATE'])
        df_vegas = pd.read_csv(vegas_data_path, parse_dates=['DATE'])

        # 2. Drop timezone if present and ensure both are timezone-naive
        try:
            df_vegas['DATE'] = df_vegas['DATE'].dt.tz_convert(None)
        except Exception:
            pass
        
        try:
            df_model['DATE'] = df_model['DATE'].dt.tz_convert(None)
        except Exception:
            pass

        # 3. Use ALL available data (including 2025)
        # No date filtering - use all data that has outcomes
        
        print(f"=== USING ALL AVAILABLE DATA ===")
        print(f"Model odds: {df_model.shape[0]} rows")
        print(f"Vegas data: {df_vegas.shape[0]} rows")
        print(f"Date range: {df_model['DATE'].min()} to {df_model['DATE'].max()}")

        # 4. Prepare merge keys
        key_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER']
        # Default sportsbooks
        vegas_cols = vegas_cols or ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']

        # 5. Merge
        df = pd.merge(
            df_model[key_cols + ['odds']],
            df_vegas[key_cols + ['win'] + vegas_cols],
            on=key_cols,
            how='inner'
        )

        print(f"After merge: {df.shape[0]} rows")

        # 6. Average the Vegas odds
        df['avg_vegas_odds'] = df[vegas_cols].mean(axis=1, skipna=True)

        # 7. Filter realistic odds (remove extreme outliers)
        df = df[(df['avg_vegas_odds'] >= -500) & (df['avg_vegas_odds'] <= 500)]
        df = df.dropna(subset=['avg_vegas_odds', 'win', 'odds'])

        print(f"After filtering: {df.shape[0]} rows")

        # 8. Pick your model's favorite by lowest American odds (highest implied probability)
        picks = df.groupby('BOUT').apply(lambda x: x.loc[x['odds'].idxmin()]).reset_index(drop=True)

        print(f"Selected fights: {len(picks)}")

        # 9. Calculate profit per fight
        def calc_profit(row):
            ml = row['avg_vegas_odds']
            if row['win'] == 1:
                # Handle zero odds case
                if ml == 0:
                    return 0  # No profit or loss on a 0 odds bet
                # Correct profit calculation for American odds
                if ml > 0:
                    return stake * (ml / 100)
                else:
                    return stake * (100 / abs(ml))
            return -stake

        picks['stake'] = stake
        picks['profit'] = picks.apply(calc_profit, axis=1)

        # 10. Compute cumulative ROI
        picks = picks.sort_values('DATE')
        picks['cum_profit'] = picks['profit'].cumsum()
        picks['cum_stake'] = (picks.index + 1) * stake  # Fixed: cumulative stake calculation
        picks['cum_roi'] = picks['cum_profit'] / picks['cum_stake']

        # 11. Report results
        total_fights = len(picks)
        total_stake = total_fights * stake
        total_profit = picks['profit'].sum()
        final_roi = total_profit / total_stake
        win_rate = picks['win'].mean()

        print(f"\n=== CORRECTED ROI (HISTORICAL DATA ONLY) ===")
        print(f"Total fights: {total_fights}")
        print(f"Total stake: ${total_stake}")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"Final ROI: {final_roi:.2%}")
        print(f"Win rate: {win_rate:.2%}")

        # 12. Monthly analysis (FIXED: Use weighted averaging)
        picks['month'] = picks['DATE'].dt.to_period('M')
        monthly_stats = picks.groupby('month').agg({
            'profit': 'sum',
            'stake': 'sum'
        })
        monthly_roi = monthly_stats['profit'] / monthly_stats['stake']
        
        # Verify the calculation is correct
        total_monthly_profit = monthly_stats['profit'].sum()
        total_monthly_stake = monthly_stats['stake'].sum()
        weighted_avg_monthly_roi = total_monthly_profit / total_monthly_stake

        print(f"\n=== MONTHLY ROI (HISTORICAL) ===")
        for month, roi in monthly_roi.items():
            print(f"{month}: {roi:.2%}")
        
        print(f"\n=== VERIFICATION ===")
        print(f"Total ROI: {final_roi:.2%}")
        print(f"Weighted average monthly ROI: {weighted_avg_monthly_roi:.2%}")
        print(f"Difference: {abs(final_roi - weighted_avg_monthly_roi):.6f}")
        
        # Calculate what the simple average would be (this is what you were seeing)
        simple_avg_monthly_roi = monthly_roi.mean()
        print(f"\n=== EXPLANATION OF DISCREPANCY ===")
        print(f"Simple average of monthly ROIs: {simple_avg_monthly_roi:.2%}")
        print(f"This is what you get when you manually add up monthly ROIs")
        print(f"The correct calculation uses weighted averaging based on fights per month")
        print(f"Difference between simple and weighted: {abs(simple_avg_monthly_roi - weighted_avg_monthly_roi):.2%}")

        # 13. Visualization (disabled for performance)
        # try:
        #     import matplotlib.pyplot as plt
        #     # ... plotting code disabled for performance ...
        # except ImportError:
        #     print("Matplotlib not available - skipping visualization")

        return picks


    def tune_svm(self):
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
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
            'n_estimators': [10, 50, 100, 200],
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
        mlp_model, _ = self.build_mlp()
        print("mlp_model log probabilities", mlp_model)
        # Scale features for MLP
        svm_model, _ = self.tune_svm()
        print("svm_model log probabilities", svm_model)
        nb_model, _ = self.build_naive_bayes()
        print("nb_model log probabilities", nb_model)
        #X_test_scaled = self.scaler.transform(self.X_test)
        
        # Predict probabilities
        log_probs = log_model.predict_proba(self.X_test)[:, 1]
        print("log_probs", log_probs)
        xgb_probs = xgb_model.predict_proba(self.X_test)[:, 1]
        mlp_probs = mlp_model.predict_proba(self.X_test)[:, 1]

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
        if load_model is not None:
            mlp_model = load_model('../saved_models/best_model.h5')
        else:
            # Fallback to sklearn MLP if Keras model is not available
            mlp_model, _ = self.build_mlp()
        
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
    
    def analyze_elo_accuracy_by_event(self):
        """
        Analyze Elo prediction accuracy broken down by individual UFC events/nights.
        Shows how accuracy varies across the 2-year test period.
        """
        # Create predictions for each fight
        predictions = []
        
        for _, row in self.test_df.iterrows():
            fighter_elo = row['precomp_elo']
            opponent_elo = row['opp_precomp_elo']
            win = row['win']
            
            # Make prediction based on Elo comparison
            if fighter_elo > opponent_elo:
                pred = 1
            elif fighter_elo < opponent_elo:
                pred = 0
            else:
                pred = 0.5  # Tie case
                
            predictions.append({
                'DATE': row['DATE'],
                'EVENT': row['EVENT'],
                'BOUT': row['BOUT'],
                'FIGHTER': row['FIGHTER'],
                'fighter_elo': fighter_elo,
                'opponent_elo': opponent_elo,
                'elo_diff': fighter_elo - opponent_elo,
                'prediction': pred,
                'actual': win,
                'correct': (pred == win)
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(predictions)
        
        # Group by event and calculate accuracy per event
        event_accuracy = results_df.groupby(['DATE', 'EVENT']).agg({
            'correct': ['count', 'sum', 'mean'],
            'elo_diff': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        event_accuracy.columns = ['total_fights', 'correct_predictions', 'accuracy', 'avg_elo_diff', 'std_elo_diff']
        event_accuracy = event_accuracy.reset_index()
        
        # Sort by date
        event_accuracy = event_accuracy.sort_values('DATE')
        
        # Calculate overall statistics
        overall_accuracy = results_df['correct'].mean()
        total_fights = len(results_df)
        total_correct = results_df['correct'].sum()
        
        print(f"\n🎯 Overall Elo Prediction Performance:")
        print(f"Total Fights: {total_fights}")
        print(f"Correct Predictions: {total_correct}")
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        print(f"\n📊 Accuracy by Event (showing first 20 events):")
        print(event_accuracy.head(20).to_string(index=False))
        
        # Show events with highest and lowest accuracy
        print(f"\n🏆 Top 5 Events by Accuracy:")
        top_events = event_accuracy[event_accuracy['total_fights'] >= 3].nlargest(5, 'accuracy')
        print(top_events[['DATE', 'EVENT', 'accuracy', 'total_fights']].to_string(index=False))
        
        print(f"\n📉 Bottom 5 Events by Accuracy:")
        bottom_events = event_accuracy[event_accuracy['total_fights'] >= 3].nsmallest(5, 'accuracy')
        print(bottom_events[['DATE', 'EVENT', 'accuracy', 'total_fights']].to_string(index=False))
        
        # Time series analysis
        print(f"\n📈 Accuracy Trends Over Time:")
        monthly_accuracy = results_df.groupby(results_df['DATE'].dt.to_period('M')).agg({
            'correct': ['count', 'mean']
        }).round(4)
        monthly_accuracy.columns = ['fights', 'accuracy']
        monthly_accuracy = monthly_accuracy.reset_index()
        monthly_accuracy['DATE'] = monthly_accuracy['DATE'].astype(str)
        
        print("Monthly Accuracy:")
        print(monthly_accuracy.to_string(index=False))
        
        # Plotting
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Accuracy by event over time
        plt.subplot(2, 2, 1)
        plt.scatter(range(len(event_accuracy)), event_accuracy['accuracy'], alpha=0.7, s=50)
        plt.axhline(y=overall_accuracy, color='red', linestyle='--', label=f'Overall: {overall_accuracy:.3f}')
        plt.xlabel('Event Index (chronological)')
        plt.ylabel('Accuracy')
        plt.title('Elo Prediction Accuracy by Event')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of event accuracies
        plt.subplot(2, 2, 2)
        plt.hist(event_accuracy['accuracy'], bins=15, alpha=0.7, edgecolor='black')
        plt.axvline(x=overall_accuracy, color='red', linestyle='--', label=f'Overall: {overall_accuracy:.3f}')
        plt.xlabel('Event Accuracy')
        plt.ylabel('Number of Events')
        plt.title('Distribution of Event Accuracies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Monthly accuracy trend
        plt.subplot(2, 2, 3)
        monthly_accuracy['DATE'] = pd.to_datetime(monthly_accuracy['DATE'].astype(str))
        plt.plot(monthly_accuracy['DATE'], monthly_accuracy['accuracy'], marker='o', linewidth=2)
        plt.axhline(y=overall_accuracy, color='red', linestyle='--', label=f'Overall: {overall_accuracy:.3f}')
        plt.xlabel('Month')
        plt.ylabel('Accuracy')
        plt.title('Monthly Accuracy Trend')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy vs number of fights per event
        plt.subplot(2, 2, 4)
        plt.scatter(event_accuracy['total_fights'], event_accuracy['accuracy'], alpha=0.7)
        plt.xlabel('Number of Fights per Event')
        plt.ylabel('Event Accuracy')
        plt.title('Accuracy vs Event Size')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'overall_accuracy': overall_accuracy,
            'event_accuracy': event_accuracy,
            'monthly_accuracy': monthly_accuracy,
            'results_df': results_df
        }
    
    def find_best_feature_to_add(self, base_features=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        if base_features is None:
            base_features = self.importance_columns

        base_set = set(base_features)
        all_candidates = [f for f in self.main_stats_cols if f not in base_set]
        results = []

        print(f"Evaluating {len(all_candidates)} features not in importance_columns...\n")

        for candidate in all_candidates:
            current_features = base_features + [candidate]

            # Use the same data preprocessing as _prepare_data method
            # This ensures consistency with tune_logistic_regression
            sub_train = self.train_df.copy()
            sub_test = self.test_df.copy()
            
            # Apply the same imputation strategy as _prepare_data
            imp = SimpleImputer(strategy='median')
            sub_train[current_features] = imp.fit_transform(sub_train[current_features])
            sub_test[current_features] = imp.transform(sub_test[current_features])

            X_train = sub_train[current_features]
            y_train = sub_train['win']
            X_test = sub_test[current_features]
            y_test = sub_test['win']

            # Use the EXACT same pipeline configuration as tune_logistic_regression
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler()),
                ('clf', LogisticRegression(max_iter=10000, random_state=42))
            ])
            
            # Use the EXACT same parameter grid as tune_logistic_regression
            params = {
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__penalty': ['l2'],
                'clf__solver': ['liblinear', 'saga'],
                'clf__class_weight': [None, 'balanced']
            }
            
            # Use the EXACT same cross-validation strategy
            tscv = TimeSeriesSplit(n_splits=5)
            grid = GridSearchCV(pipeline, params, cv=tscv, scoring='accuracy', n_jobs=-1)

            try:
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                
                preds = best_model.predict(X_test)
                probs = best_model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, preds)
                loss = log_loss(y_test, probs)

                results.append({
                    'feature_added': candidate,
                    'accuracy': acc,
                    'log_loss': loss,
                    'best_params': grid.best_params_
                })
            except Exception as e:
                print(f"⚠️ Skipping {candidate} due to error: {e}")

        results_df = pd.DataFrame(results)
        
        # Sort by log loss (ascending) for best log loss
        results_df_log_loss = results_df.sort_values(by=['log_loss', 'accuracy'], ascending=[True, True])
        
        # Sort by accuracy (descending) for best accuracy
        results_df_accuracy = results_df.sort_values(by=['accuracy', 'log_loss'], ascending=[False, True])

        print("\n🏆 Top 5 candidates by LOWEST log loss:")
        print(results_df_log_loss.head(5))

        print("\n🎯 Top 5 candidates by HIGHEST accuracy:")
        print(results_df_accuracy.head(5))

        if not results_df.empty:
            # Return the best feature by log loss (primary metric)
            best_feature = results_df_log_loss.iloc[0]['feature_added']
            print(f"\n✅ Best feature to add (by log loss): {best_feature}")
            
            # Also show the best feature by accuracy
            best_accuracy_feature = results_df_accuracy.iloc[0]['feature_added']
            if best_accuracy_feature != best_feature:
                print(f"🎯 Best feature by accuracy: {best_accuracy_feature}")
            
            return best_feature, results_df
        else:
            print("\n❌ No valid features found to add")
            return None, pd.DataFrame()

    def roi_optimized_features(self, base_features=None, vegas_data_path=None, odds_table_path=None,
                           stake=100, top_n=5):
        """
        Add one extra feature at a time to base_features and pick what maximizes ROI.
        Baseline and candidates both go through calculate_roi on an odds table
        so ROI matches the main report.
        """
        import os, uuid
        import pandas as pd
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.metrics import accuracy_score, log_loss

        if vegas_data_path is None:
            raise ValueError("vegas_data_path is required")
        if odds_table_path is None:
            raise ValueError("odds_table_path is required to match the printed Final ROI")

        if base_features is None:
            base_features = list(self.importance_columns)

        base_set = set(base_features)
        candidates = [f for f in self.main_stats_cols if f not in base_set and f in self.df.columns]

        # Same model recipe as tune_logistic_regression
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('clf', LogisticRegression(max_iter=10000, random_state=42))
        ])
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'saga'],
            'clf__class_weight': [None, 'balanced']
        }
        tscv = TimeSeriesSplit(n_splits=5)

        # 1) Baseline ROI via your saved odds table (identical to banner)
        picks_df = self.calculate_roi(
            odds_table_path=odds_table_path,
            vegas_data_path=vegas_data_path
        )
        if not isinstance(picks_df, pd.DataFrame):
            raise RuntimeError("calculate_roi did not return a DataFrame")

        current_total_fights = int(len(picks_df))
        current_total_profit = float(picks_df['profit'].sum()) if current_total_fights else 0.0
        current_total_stake = float(picks_df['stake'].sum()) if 'stake' in picks_df.columns else current_total_fights * stake
        current_roi = (current_total_profit / current_total_stake) if current_total_stake else 0.0
        current_win_rate = float(picks_df['win'].mean()) if 'win' in picks_df.columns and current_total_fights else 0.0

        # Also report baseline Acc and LogLoss for context
        sub_train_b = self.train_df.copy()
        sub_test_b  = self.test_df.copy()
        imp_b = SimpleImputer(strategy='median')
        sub_train_b[base_features] = imp_b.fit_transform(sub_train_b[base_features])
        sub_test_b[base_features]  = imp_b.transform(sub_test_b[base_features])
        X_train_b, y_train_b = sub_train_b[base_features], sub_train_b['win']
        X_test_b,  y_test_b  = sub_test_b[base_features],  sub_test_b['win']

        grid_b = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1, refit=True)
        grid_b.fit(X_train_b, y_train_b)
        probs_b = grid_b.best_estimator_.predict_proba(X_test_b)[:, 1]
        preds_b = (probs_b >= 0.5).astype(int)
        acc_b = accuracy_score(y_test_b, preds_b)
        loss_b = log_loss(y_test_b, probs_b)

        print(f"Current model ROI: {current_roi:.4f} ({current_roi*100:.2f}%)  "
            f"Acc: {acc_b:.4f}  LogLoss: {loss_b:.4f}  "
            f"WinRate: {current_win_rate:.4f}  Fights: {current_total_fights}")

        # 2) Evaluate candidates, writing a temp odds table built exactly like generate_odds_table
        results = []
        print(f"Evaluating ROI for {len(candidates)} candidate features...")
        tmp_dir = os.path.join(os.path.dirname(odds_table_path), "roi_opt_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        for cand in candidates:
            feat_list = list(base_features) + [cand]
            try:
                sub_train = self.train_df.copy()
                sub_test = self.test_df.copy()

                imp = SimpleImputer(strategy='median')
                sub_train[feat_list] = imp.fit_transform(sub_train[feat_list])
                sub_test[feat_list] = imp.transform(sub_test[feat_list])

                X_train, y_train = sub_train[feat_list], sub_train['win']
                X_test,  y_test  = sub_test[feat_list],  sub_test['win']

                grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1, refit=True)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

                probs = best_model.predict_proba(X_test)[:, 1]
                preds = (probs >= 0.5).astype(int)
                acc = accuracy_score(y_test, preds)
                loss = log_loss(y_test, probs)

                # Build candidate odds table with the same normalization and rounding
                # as your generate_odds_table (uses make_consistent_odds_table).
                tmp_odds_df = make_consistent_odds_table(sub_test, probs)
                # Only columns calculate_roi actually uses
                tmp_odds_df = tmp_odds_df[['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'odds']]

                tmp_path = os.path.join(tmp_dir, f"candidate_{cand}_{uuid.uuid4().hex}.csv")
                tmp_odds_df.to_csv(tmp_path, index=False)

                # Compute ROI via the exact same path as baseline
                cand_picks = self.calculate_roi(
                    odds_table_path=tmp_path,
                    vegas_data_path=vegas_data_path
                )
                if isinstance(cand_picks, pd.DataFrame) and len(cand_picks) > 0:
                    cand_total_fights = int(len(cand_picks))
                    cand_total_profit = float(cand_picks['profit'].sum())
                    cand_total_stake = float(cand_picks['stake'].sum()) if 'stake' in cand_picks.columns else cand_total_fights * stake
                    cand_roi = (cand_total_profit / cand_total_stake) if cand_total_stake else 0.0
                    cand_win_rate = float(cand_picks['win'].mean()) if 'win' in cand_picks.columns else 0.0
                else:
                    cand_total_fights = 0
                    cand_total_profit = 0.0
                    cand_roi = 0.0
                    cand_win_rate = 0.0

                results.append({
                    'feature_added': cand,
                    'roi': cand_roi,
                    'roi_percent': cand_roi * 100.0,
                    'current_roi': current_roi,
                    'current_roi_percent': current_roi * 100.0,
                    'roi_improvement': cand_roi - current_roi,
                    'roi_improvement_percent': (cand_roi - current_roi) * 100.0,
                    'total_profit': cand_total_profit,
                    'current_total_profit': current_total_profit,
                    'win_rate': cand_win_rate,
                    'total_fights': cand_total_fights,
                    'current_win_rate': current_win_rate,
                    'current_total_fights': current_total_fights,
                    'accuracy': acc,
                    'log_loss': loss,
                    'baseline_accuracy': acc_b,
                    'baseline_log_loss': loss_b,
                    'best_params': grid.best_params_
                })
            except Exception as e:
                print(f"Skipping {cand} due to error: {e}")
                continue

        results_df = pd.DataFrame(results)
        if results_df.empty:
            print("No valid candidates evaluated.")
            return [], results_df

        results_df = results_df.sort_values(
            by=['roi', 'roi_improvement', 'win_rate', 'total_fights', 'log_loss'],
            ascending=[False, False, False, False, True]
        ).reset_index(drop=True)

        print("\nTop candidates by ROI (current and delta shown):")
        cols_to_show = ['feature_added', 'roi_percent', 'current_roi_percent',
                        'roi_improvement_percent', 'win_rate', 'total_fights',
                        'accuracy', 'log_loss']
        print(results_df[cols_to_show].head(top_n).to_string(index=False))

        best_features = results_df['feature_added'].head(top_n).tolist()
        return best_features, results_df






    def _calculate_roi_for_features(self, test_df, probs, vegas_data_path, stake):
        """
        Helper method to calculate ROI for feature evaluation.
        Uses the same logic as calculate_roi but simplified for feature selection.
        """
        import pandas as pd
        import numpy as np

        try:
            # Try to use the same odds table as calculate_roi method
            odds_table_path = 'data/tmp/odds_table.csv'
            
            try:
                # Load the odds table (same as calculate_roi)
                df_model = pd.read_csv(odds_table_path, parse_dates=['DATE'])
                df_vegas = pd.read_csv(vegas_data_path, parse_dates=['DATE'])
                
                # Handle timezone
                try:
                    df_vegas['DATE'] = df_vegas['DATE'].dt.tz_convert(None)
                except Exception:
                    pass
                try:
                    df_model['DATE'] = df_model['DATE'].dt.tz_convert(None)
                except Exception:
                    pass

                # Merge data (same as calculate_roi)
                key_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER']
                df = pd.merge(
                    df_model[key_cols + ['odds']],
                    df_vegas[key_cols + ['win', 'draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']],
                    on=key_cols,
                    how='inner'
                )
            except FileNotFoundError:
                # Fallback: create odds table from test data and probabilities (original method)
                df_vegas = pd.read_csv(vegas_data_path, parse_dates=['DATE'])
                
                # Handle timezone
                try:
                    df_vegas['DATE'] = df_vegas['DATE'].dt.tz_convert(None)
                except Exception:
                    pass

                # Create odds table from test data and probabilities
                odds_table = test_df[['DATE', 'EVENT', 'BOUT', 'FIGHTER']].copy()
                odds_table['prob_win'] = probs
                
                # Define prob_to_american_odds function locally
                def prob_to_american_odds(p):
                    """Convert win probability p (0 < p < 1) into American odds."""
                    if p <= 0 or p >= 1:
                        return np.nan
                    if p >= 0.5:
                        odds = - (p / (1 - p)) * 100
                    else:
                        odds = ((1 - p) / p) * 100
                    return int(np.sign(odds) * np.round(abs(odds)))
                
                odds_table['odds'] = odds_table['prob_win'].apply(prob_to_american_odds)

                # Merge with Vegas data
                df = pd.merge(
                    odds_table[['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'odds']],
                    df_vegas[['DATE', 'EVENT', 'BOUT', 'FIGHTER', 'win', 'draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']],
                    on=['DATE', 'EVENT', 'BOUT', 'FIGHTER'],
                    how='inner'
                )

            if len(df) == 0:
                return {'roi': 0, 'total_profit': 0, 'win_rate': 0, 'total_fights': 0}

            # Use EXACT same logic as calculate_roi method
            vegas_cols = ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bet365_odds', 'bovada_odds']
            
            # 6. Average the Vegas odds (same as calculate_roi)
            df['avg_vegas_odds'] = df[vegas_cols].mean(axis=1, skipna=True)

            # 7. Filter realistic odds (remove extreme outliers) - SAME as calculate_roi
            df = df[(df['avg_vegas_odds'] >= -500) & (df['avg_vegas_odds'] <= 500)]
            df = df.dropna(subset=['avg_vegas_odds', 'win', 'odds'])

            if len(df) == 0:
                return {'roi': 0, 'total_profit': 0, 'win_rate': 0, 'total_fights': 0}

            # Select model's favorite per fight (lowest odds = highest probability)
            picks = df.groupby('BOUT').apply(lambda x: x.loc[x['odds'].idxmin()]).reset_index(drop=True)

            if len(picks) == 0:
                return {'roi': 0, 'total_profit': 0, 'win_rate': 0, 'total_fights': 0}

            # Calculate profit for each bet
            def calculate_profit(vegas_odds, stake, won):
                if not won:
                    return -stake
                if vegas_odds == 0:  # Handle zero odds case
                    return 0  # No profit or loss on a 0 odds bet
                if vegas_odds > 0:
                    return (vegas_odds / 100) * stake
                else:
                    return (100 / abs(vegas_odds)) * stake

            picks['profit'] = picks.apply(lambda row: calculate_profit(row['avg_vegas_odds'], stake, row['win'] == 1), axis=1)

            # Calculate metrics
            total_profit = picks['profit'].sum()
            total_stake = len(picks) * stake
            roi = total_profit / total_stake if total_stake > 0 else 0
            win_rate = picks['win'].mean()

            return {
                'roi': roi,
                'total_profit': total_profit,
                'win_rate': win_rate,
                'total_fights': len(picks)
            }

        except Exception as e:
            print(f"Error calculating ROI: {e}")
            # Add debugging info
            try:
                if 'avg_vegas_odds' in df.columns:
                    zero_odds_count = (df['avg_vegas_odds'] == 0).sum()
                    nan_odds_count = df['avg_vegas_odds'].isna().sum()
                    print(f"Debug: {zero_odds_count} zero odds, {nan_odds_count} NaN odds")
            except:
                pass
            return {'roi': 0, 'total_profit': 0, 'win_rate': 0, 'total_fights': 0}

    def find_best_feature_by_metric(self, base_features=None, metric='log_loss'):
        """
        Find the best feature to add based on a specific metric.
        
        Args:
            base_features: List of base features to build upon (default: self.importance_columns)
            metric: 'log_loss' or 'accuracy' - which metric to optimize for
        
        Returns:
            best_feature, results_df
        """
        best_feature, results_df = self.find_best_feature_to_add(base_features)
        
        if results_df.empty:
            return None, results_df
        
        if metric == 'log_loss':
            # Already sorted by log loss in find_best_feature_to_add
            best_feature = results_df.sort_values(by=['log_loss', 'accuracy'], ascending=[True, True]).iloc[0]['feature_added']
            print(f"\n✅ Best feature by log loss: {best_feature}")
        elif metric == 'accuracy':
            best_feature = results_df.sort_values(by=['accuracy', 'log_loss'], ascending=[False, True]).iloc[0]['feature_added']
            print(f"\n✅ Best feature by accuracy: {best_feature}")
        else:
            raise ValueError("Metric must be 'log_loss' or 'accuracy'")
        
        return best_feature, results_df

    def compare_top_features(self, base_features=None, top_n=10):
        """
        Compare the top features by both log loss and accuracy metrics.
        Shows detailed comparison and recommendations.
        """
        best_feature, results_df = self.find_best_feature_to_add(base_features)
        
        if results_df.empty:
            print("❌ No valid features found to compare")
            return None
        
        print(f"\n📊 DETAILED FEATURE COMPARISON (Top {top_n})")
        print("="*80)
        
        # Sort by log loss
        top_log_loss = results_df.sort_values(by=['log_loss', 'accuracy'], ascending=[True, True]).head(top_n)
        
        # Sort by accuracy  
        top_accuracy = results_df.sort_values(by=['accuracy', 'log_loss'], ascending=[False, True]).head(top_n)
        
        print(f"\n🏆 TOP {top_n} BY LOG LOSS (Lower is Better):")
        print("-" * 60)
        for i, (_, row) in enumerate(top_log_loss.iterrows(), 1):
            print(f"{i:2d}. {row['feature_added']:30s} | Log Loss: {row['log_loss']:.4f} | Accuracy: {row['accuracy']:.4f}")
        
        print(f"\n🎯 TOP {top_n} BY ACCURACY (Higher is Better):")
        print("-" * 60)
        for i, (_, row) in enumerate(top_accuracy.iterrows(), 1):
            print(f"{i:2d}. {row['feature_added']:30s} | Accuracy: {row['accuracy']:.4f} | Log Loss: {row['log_loss']:.4f}")
        
        # Find features that are good in both metrics
        print(f"\n🌟 FEATURES GOOD IN BOTH METRICS:")
        print("-" * 60)
        
        # Get top 20% of each metric
        log_loss_threshold = results_df['log_loss'].quantile(0.2)
        accuracy_threshold = results_df['accuracy'].quantile(0.8)
        
        good_in_both = results_df[
            (results_df['log_loss'] <= log_loss_threshold) & 
            (results_df['accuracy'] >= accuracy_threshold)
        ].sort_values(by=['log_loss', 'accuracy'], ascending=[True, True])
        
        if not good_in_both.empty:
            for i, (_, row) in enumerate(good_in_both.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature_added']:30s} | Log Loss: {row['log_loss']:.4f} | Accuracy: {row['accuracy']:.4f}")
        else:
            print("No features found that are good in both metrics.")
        
        # Recommendations
        best_log_loss_feature = top_log_loss.iloc[0]['feature_added']
        best_accuracy_feature = top_accuracy.iloc[0]['feature_added']
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 60)
        print(f"🥇 For best log loss: {best_log_loss_feature}")
        print(f"🥇 For best accuracy: {best_accuracy_feature}")
        
        if best_log_loss_feature == best_accuracy_feature:
            print(f"🎉 Perfect! The same feature is best for both metrics!")
        else:
            print(f"🤔 Different features are best for different metrics.")
            print(f"   Consider your specific use case to choose between them.")
        
        return {
            'best_log_loss_feature': best_log_loss_feature,
            'best_accuracy_feature': best_accuracy_feature,
            'top_log_loss': top_log_loss,
            'top_accuracy': top_accuracy,
            'good_in_both': good_in_both
        }

    def validate_feature_addition(self, best_feature, base_features=None):
        """
        Validate that adding the best feature produces consistent results.
        This method runs tune_logistic_regression on the base features,
        then adds the best feature and runs it again to ensure consistency.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        if base_features is None:
            base_features = self.importance_columns

        print(f"\n🔍 Validating feature addition for: {best_feature}")
        print("="*60)

        # Step 1: Run tune_logistic_regression on base features
        print("📊 Step 1: Testing base features...")
        base_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('clf', LogisticRegression(max_iter=10000, random_state=42))
        ])
        base_params = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'saga'],
            'clf__class_weight': [None, 'balanced']
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        base_grid = GridSearchCV(base_pipeline, base_params, cv=tscv, scoring='accuracy', n_jobs=-1)
        
        # Prepare base features data
        base_imp = SimpleImputer(strategy='median')
        base_X_train = base_imp.fit_transform(self.train_df[base_features])
        base_X_test = base_imp.transform(self.test_df[base_features])
        
        base_grid.fit(base_X_train, self.y_train)
        base_best = base_grid.best_estimator_
        base_probs = base_best.predict_proba(base_X_test)[:, 1]
        base_preds = base_best.predict(base_X_test)
        base_acc = accuracy_score(self.y_test, base_preds)
        base_loss = log_loss(self.y_test, base_probs)
        
        print(f"Base features accuracy: {base_acc:.4f}")
        print(f"Base features log loss: {base_loss:.4f}")
        print(f"Base features best params: {base_grid.best_params_}")

        # Step 2: Add the best feature and run again
        print(f"\n📊 Step 2: Testing base features + {best_feature}...")
        extended_features = base_features + [best_feature]
        
        # Prepare extended features data
        ext_imp = SimpleImputer(strategy='median')
        ext_X_train = ext_imp.fit_transform(self.train_df[extended_features])
        ext_X_test = ext_imp.transform(self.test_df[extended_features])
        
        ext_grid = GridSearchCV(base_pipeline, base_params, cv=tscv, scoring='accuracy', n_jobs=-1)
        ext_grid.fit(ext_X_train, self.y_train)
        ext_best = ext_grid.best_estimator_
        ext_probs = ext_best.predict_proba(ext_X_test)[:, 1]
        ext_preds = ext_best.predict(ext_X_test)
        ext_acc = accuracy_score(self.y_test, ext_preds)
        ext_loss = log_loss(self.y_test, ext_probs)
        
        print(f"Extended features accuracy: {ext_acc:.4f}")
        print(f"Extended features log loss: {ext_loss:.4f}")
        print(f"Extended features best params: {ext_grid.best_params_}")

        # Step 3: Compare results
        print(f"\n📈 Comparison Results:")
        print(f"Accuracy improvement: {ext_acc - base_acc:.4f}")
        print(f"Log loss improvement: {base_loss - ext_loss:.4f}")
        
        if abs(ext_acc - base_acc) < 0.001 and abs(ext_loss - base_loss) < 0.001:
            print("⚠️  WARNING: Results are nearly identical - feature may not be adding value")
        elif ext_acc > base_acc and ext_loss < base_loss:
            print("✅ SUCCESS: Feature addition improved both accuracy and log loss")
        elif ext_acc > base_acc:
            print("✅ PARTIAL: Feature improved accuracy but not log loss")
        elif ext_loss < base_loss:
            print("✅ PARTIAL: Feature improved log loss but not accuracy")
        else:
            print("❌ FAILURE: Feature addition did not improve performance")

        return {
            'base_accuracy': base_acc,
            'base_log_loss': base_loss,
            'extended_accuracy': ext_acc,
            'extended_log_loss': ext_loss,
            'accuracy_improvement': ext_acc - base_acc,
            'log_loss_improvement': base_loss - ext_loss,
            'best_feature': best_feature
        }

    def roi_features_to_add(self, base_features=None, stake=100, volatility_weight=1.0, vegas_data_path=None, vegas_cols=None):
        """
        For each candidate feature, train a model, compute ROI and ROI volatility (mean absolute change in cumulative ROI),
        and return a DataFrame with feature, final ROI, volatility, and a combined score (ROI - volatility_weight * volatility).
        Accepts vegas_data_path for flexible CSV usage.
        """
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import accuracy_score, log_loss
        from tqdm import tqdm

        # Load vegas odds data
        vegas_df = pd.read_csv(vegas_data_path) if vegas_data_path is not None else None
        default_vegas_cols = ['draftkings_odds', 'fanduel_odds', 'betmgm_odds', 'bovada_odds']

        if base_features is None:
            base_features = self.importance_columns
        base_set = set(base_features)
        all_candidates = [f for f in self.main_stats_cols if f not in base_set]
        results = []
        print(f"Evaluating {len(all_candidates)} features for ROI and volatility...\n")
        for candidate in tqdm(all_candidates):
            current_features = base_features + [candidate]
            # Drop missing values specific to current feature set
            sub_train = self.train_df.copy()
            sub_test = self.test_df.copy()
            train_medians = sub_train[current_features].median()
            sub_train[current_features] = sub_train[current_features].fillna(train_medians)
            sub_test[current_features] = sub_test[current_features].fillna(train_medians)
            X_train = sub_train[current_features]
            y_train = sub_train['win']
            X_test = sub_test[current_features]
            y_test = sub_test['win']
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler()),
                ('clf', LogisticRegression(max_iter=10000, random_state=42))
            ])
            try:
                pipeline.fit(X_train, y_train)
                probs = pipeline.predict_proba(X_test)[:, 1]
                # Build model odds table in memory
                test_df = sub_test[['DATE', 'EVENT', 'BOUT', 'FIGHTER']].copy()
                odds_table = make_consistent_odds_table(test_df, probs)
                # Ensure both are timezone-naive datetime for merge
                odds_table['DATE'] = pd.to_datetime(odds_table['DATE']).dt.tz_localize(None)
                vegas_df['DATE'] = pd.to_datetime(vegas_df['DATE']).dt.tz_localize(None)
                # Merge with vegas odds
                key_cols = ['DATE', 'EVENT', 'BOUT', 'FIGHTER']
                if vegas_cols is not None:
                    vegas_cols_present = [col for col in vegas_cols if col in vegas_df.columns]
                elif vegas_df is not None:
                    vegas_cols_present = [col for col in default_vegas_cols if col in vegas_df.columns]
                else:
                    vegas_cols_present = []
                merge_cols = key_cols + ['win'] + vegas_cols_present
                if vegas_df is None or len(vegas_cols_present) == 0:
                    print(f"⚠️ Skipping {candidate} due to error: No vegas odds columns present in test set.")
                    continue
                merged = pd.merge(odds_table, vegas_df[merge_cols], on=key_cols, how='inner')
                merged['avg_vegas_odds'] = merged[vegas_cols_present].mean(axis=1)
                # Clamp avg_vegas_odds
                def clamp_vegas_odds(odds):
                    if 0 < odds < 100:
                        return 100
                    if -100 < odds < 0:
                        return -100
                    return odds
                merged['avg_vegas_odds'] = merged['avg_vegas_odds'].apply(clamp_vegas_odds)
                # Debug: merged shape and columns
                print(f"[DEBUG] merged shape for {candidate}: {merged.shape}, columns: {list(merged.columns)}")
                # Pick model's favorite by lowest odds per bout
                idx = merged.groupby('BOUT')['odds'].idxmin()
                picks = merged.loc[idx].copy()
                # Calculate profit per fight
                def calc_profit(row):
                    ml = row['avg_vegas_odds']
                    if row['win'] == 1:
                        # Handle zero odds case
                        if ml == 0:
                            return 0  # No profit or loss on a 0 odds bet
                        return stake * (ml / 100) if ml > 0 else stake * (100 / abs(ml))
                    return -stake
                picks['stake'] = stake
                picks['profit'] = picks.apply(calc_profit, axis=1)
                picks = picks.sort_values('DATE')
                picks['cum_profit'] = picks['profit'].cumsum()
                picks['cum_stake'] = picks['stake'].cumsum()
                picks['cum_roi'] = picks['cum_profit'] / picks['cum_stake']
                roi_series = picks['cum_roi'].values
                # Drop NaNs from roi_series for volatility calculation
                roi_series_nonan = roi_series[~np.isnan(roi_series)]
                if picks.empty or len(roi_series_nonan) == 0:
                    print(f"[DEBUG] picks is empty or ROI series is all NaN for {candidate}")
                    print(f"[DEBUG] picks length: {len(picks)}")
                    print(f"[DEBUG] picks head:\n{picks.head()}")
                    print(f"[DEBUG] roi_series: {roi_series}")
                    roi_volatility = 0.0
                    final_roi = 0.0
                    score = final_roi  # Always numeric
                else:
                    if len(roi_series_nonan) > 1:
                        roi_volatility = float(np.mean(np.abs(np.diff(roi_series_nonan))))
                    else:
                        roi_volatility = 0.0
                    final_roi = float(roi_series_nonan[-1])
                    score = final_roi - volatility_weight * roi_volatility
                # Ensure no NaNs in output
                if np.isnan(roi_volatility) or np.isnan(score):
                    print(f"[DEBUG] NaN detected in roi_volatility or score for {candidate}, setting to 0.0")
                    print(f"[DEBUG] picks length: {len(picks)}")
                    print(f"[DEBUG] picks head:\n{picks.head()}")
                    print(f"[DEBUG] roi_series: {roi_series}")
                    roi_volatility = 0.0
                    score = final_roi
                results.append({
                    'feature_added': candidate,
                    'final_roi': final_roi,
                    'roi_volatility': roi_volatility,
                    'score': score
                })
            except Exception as e:
                print(f"⚠️ Skipping {candidate} due to error: {e}")
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(by=['score', 'final_roi'], ascending=[False, False])
            print("\n🏆 Top 5 candidates by ROI score:")
            print(results_df.head(5))
        return results_df

    
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
        from sklearn.preprocessing import RobustScaler
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, log_loss

        imputer = SimpleImputer(strategy='median')
        scaler = RobustScaler()

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

    def advanced_feature_selection_methods(self, n_features_to_select=20, cv_folds=5):
        """
        Implement multiple advanced feature selection methods adapted for UFC fight prediction.
        Includes RFE, RFECV, SelectFromModel, Permutation Importance, and PCA-based selection.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import make_scorer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import RFE, RFECV, SelectFromModel
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import accuracy_score, log_loss
        import time
        
        print("🔬 ADVANCED FEATURE SELECTION METHODS")
        print("="*60)
        
        # Prepare data with imputation
        imputer = SimpleImputer(strategy='median')
        scaler = RobustScaler()
        
        # Use all available numeric features
        numeric_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check if requested features exceed available features
        if n_features_to_select > len(numeric_features):
            print(f"⚠️  Warning: Requested {n_features_to_select} features, but only {len(numeric_features)} available.")
            print(f"   Will use all available features ({len(numeric_features)}) instead.")
            n_features_to_select = len(numeric_features)
        X_train_processed = imputer.fit_transform(self.X_train[numeric_features])
        X_test_processed = imputer.transform(self.X_test[numeric_features])
        
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)
        
        # Create feature names after imputation
        feature_names = [numeric_features[i] for i in range(X_train_scaled.shape[1])]
        
        # Setup cross-validation and scoring
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        logloss_scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        
        results = {}
        
        # 1. Recursive Feature Elimination (RFE)
        print("\n1️⃣ Recursive Feature Elimination (RFE)")
        print("-" * 40)
        
        base_lr = LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000, random_state=42)
        # Ensure we don't request more features than available
        n_features_available = min(n_features_to_select, X_train_scaled.shape[1])
        rfe = RFE(estimator=base_lr, n_features_to_select=n_features_available, step=3)
        
        rfe_pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("rfe", rfe),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000, random_state=42))
        ])
        
        start_time = time.time()
        rfe_scores = cross_val_score(rfe_pipeline, X_train_scaled, self.y_train, cv=cv, scoring=logloss_scorer)
        rfe_time = time.time() - start_time
        
        # Fit on full data to get selected features
        rfe_pipeline.fit(X_train_scaled, self.y_train)
        rfe_features = [feature_names[i] for i, selected in enumerate(rfe_pipeline.named_steps["rfe"].get_support()) if selected]
        
        print(f"RFE Mean Log Loss: {rfe_scores.mean():.4f} ± {rfe_scores.std():.4f}")
        print(f"RFE Time: {rfe_time:.2f}s")
        print(f"Selected Features: {len(rfe_features)}")
        
        results['RFE'] = {
            'scores': rfe_scores,
            'mean_score': rfe_scores.mean(),
            'std_score': rfe_scores.std(),
            'time': rfe_time,
            'features': rfe_features,
            'n_features': len(rfe_features)
        }
        
        # 2. Recursive Feature Elimination with Cross-Validation (RFECV)
        print("\n2️⃣ RFE with Cross-Validation (RFECV)")
        print("-" * 40)
        
        # Ensure min_features_to_select doesn't exceed available features
        min_features = min(5, X_train_scaled.shape[1])
        rfecv = RFECV(estimator=base_lr, step=3, cv=cv, scoring=logloss_scorer, min_features_to_select=min_features)
        
        rfecv_pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("rfecv", rfecv),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000, random_state=42))
        ])
        
        start_time = time.time()
        rfecv_pipeline.fit(X_train_scaled, self.y_train)
        rfecv_time = time.time() - start_time
        
        rfecv_features = [feature_names[i] for i, selected in enumerate(rfecv_pipeline.named_steps["rfecv"].get_support()) if selected]
        
        # Evaluate with cross-validation
        rfecv_scores = cross_val_score(rfecv_pipeline, X_train_scaled, self.y_train, cv=cv, scoring=logloss_scorer)
        
        print(f"RFECV Mean Log Loss: {rfecv_scores.mean():.4f} ± {rfecv_scores.std():.4f}")
        print(f"RFECV Time: {rfecv_time:.2f}s")
        print(f"Optimal Features: {len(rfecv_features)}")
        print(f"CV Scores: {rfecv.cv_results_['mean_test_score']}")
        
        results['RFECV'] = {
            'scores': rfecv_scores,
            'mean_score': rfecv_scores.mean(),
            'std_score': rfecv_scores.std(),
            'time': rfecv_time,
            'features': rfecv_features,
            'n_features': len(rfecv_features),
            'cv_scores': rfecv.cv_results_['mean_test_score']
        }
        
        # 3. SelectFromModel with Random Forest
        print("\n3️⃣ SelectFromModel with Random Forest")
        print("-" * 40)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, self.y_train)
        
        # Use feature importances to select features
        # Ensure we don't request more features than available
        max_features_available = min(n_features_to_select, X_train_scaled.shape[1])
        selector = SelectFromModel(rf, max_features=max_features_available)
        selector.fit(X_train_scaled, self.y_train)
        
        sfm_features = [feature_names[i] for i, selected in enumerate(selector.get_support()) if selected]
        
        # Create pipeline with selected features
        sfm_pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("selector", selector),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000, random_state=42))
        ])
        
        start_time = time.time()
        sfm_scores = cross_val_score(sfm_pipeline, X_train_scaled, self.y_train, cv=cv, scoring=logloss_scorer)
        sfm_time = time.time() - start_time
        
        print(f"SelectFromModel Mean Log Loss: {sfm_scores.mean():.4f} ± {sfm_scores.std():.4f}")
        print(f"SelectFromModel Time: {sfm_time:.2f}s")
        print(f"Selected Features: {len(sfm_features)}")
        
        results['SelectFromModel'] = {
            'scores': sfm_scores,
            'mean_score': sfm_scores.mean(),
            'std_score': sfm_scores.std(),
            'time': sfm_time,
            'features': sfm_features,
            'n_features': len(sfm_features)
        }
        
        # 4. Permutation Importance
        print("\n4️⃣ Permutation Importance")
        print("-" * 40)
        
        # Train a model on all features
        full_model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000, random_state=42)
        full_model.fit(X_train_scaled, self.y_train)
        
        # Calculate permutation importance
        start_time = time.time()
        perm_importance = permutation_importance(full_model, X_test_scaled, self.y_test, 
                                                n_repeats=10, random_state=42, scoring='neg_log_loss')
        perm_time = time.time() - start_time
        
        # Select top features based on permutation importance
        importance_scores = perm_importance.importances_mean
        n_features_available = min(n_features_to_select, len(importance_scores))
        top_indices = np.argsort(importance_scores)[-n_features_available:]
        perm_features = [feature_names[i] for i in top_indices]
        
        # Evaluate with selected features
        X_train_selected = X_train_scaled[:, top_indices]
        X_test_selected = X_test_scaled[:, top_indices]
        
        perm_pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000, random_state=42))
        ])
        
        perm_scores = cross_val_score(perm_pipeline, X_train_selected, self.y_train, cv=cv, scoring=logloss_scorer)
        
        print(f"Permutation Importance Mean Log Loss: {perm_scores.mean():.4f} ± {perm_scores.std():.4f}")
        print(f"Permutation Importance Time: {perm_time:.2f}s")
        print(f"Selected Features: {len(perm_features)}")
        
        results['PermutationImportance'] = {
            'scores': perm_scores,
            'mean_score': perm_scores.mean(),
            'std_score': perm_scores.std(),
            'time': perm_time,
            'features': perm_features,
            'n_features': len(perm_features),
            'importance_scores': importance_scores
        }
        
        # 5. PCA-based Feature Selection
        print("\n5️⃣ PCA-based Feature Selection")
        print("-" * 40)
        
        # Apply PCA to reduce dimensionality
        n_components = min(n_features_to_select, X_train_scaled.shape[1])
        pca = PCA(n_components=n_components)
        
        pca_pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("pca", pca),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000, random_state=42))
        ])
        
        start_time = time.time()
        pca_scores = cross_val_score(pca_pipeline, X_train_scaled, self.y_train, cv=cv, scoring=logloss_scorer)
        pca_time = time.time() - start_time
        
        # Fit to get explained variance
        pca_pipeline.fit(X_train_scaled, self.y_train)
        explained_variance = pca.explained_variance_ratio_
        
        print(f"PCA Mean Log Loss: {pca_scores.mean():.4f} ± {pca_scores.std():.4f}")
        print(f"PCA Time: {pca_time:.2f}s")
        print(f"Components: {pca.n_components_}")
        print(f"Explained Variance: {explained_variance.sum():.4f}")
        
        results['PCA'] = {
            'scores': pca_scores,
            'mean_score': pca_scores.mean(),
            'std_score': pca_scores.std(),
            'time': pca_time,
            'n_components': pca.n_components_,
            'explained_variance': explained_variance.sum()
        }
        
        # 6. Comparison and Summary
        print("\n📊 FEATURE SELECTION COMPARISON")
        print("="*60)
        
        comparison_data = []
        for method, result in results.items():
            if method != 'PCA':  # PCA doesn't have traditional features
                comparison_data.append({
                    'Method': method,
                    'Log Loss': f"{result['mean_score']:.4f} ± {result['std_score']:.4f}",
                    'Time (s)': f"{result['time']:.2f}",
                    'Features': result['n_features']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best method
        best_method = min(results.keys(), key=lambda k: results[k]['mean_score'])
        print(f"\n🏆 Best Method: {best_method}")
        print(f"Best Log Loss: {results[best_method]['mean_score']:.4f}")
        
        # Feature overlap analysis
        if len(results) > 1:
            print(f"\n🔍 FEATURE OVERLAP ANALYSIS")
            print("-" * 40)
            
            feature_sets = {method: set(result['features']) for method, result in results.items() 
                          if 'features' in result}
            
            if len(feature_sets) > 1:
                # Find common features across methods
                common_features = set.intersection(*feature_sets.values())
                print(f"Common features across all methods: {len(common_features)}")
                if common_features:
                    print(f"Common features: {sorted(list(common_features))}")
                
                # Find unique features per method
                for method, features in feature_sets.items():
                    unique_features = features - set.union(*[fs for m, fs in feature_sets.items() if m != method])
                    print(f"{method} unique features: {len(unique_features)}")
                    if unique_features:
                        print(f"  {sorted(list(unique_features))}")
        
        return results

    def advanced_ensemble_feature_selection(self, n_features_to_select=20, cv_folds=5):
        """
        Advanced ensemble feature selection combining multiple methods with voting.
        """
        print("🎯 ADVANCED ENSEMBLE FEATURE SELECTION")
        print("="*60)
        
        # Get results from individual methods
        results = self.advanced_feature_selection_methods(n_features_to_select, cv_folds)
        
        # Create feature voting system
        feature_votes = {}
        
        for method, result in results.items():
            if 'features' in result:
                for feature in result['features']:
                    if feature not in feature_votes:
                        feature_votes[feature] = 0
                    feature_votes[feature] += 1
        
        # Sort features by vote count
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features by voting
        ensemble_features = [feature for feature, votes in sorted_features[:n_features_to_select]]
        
        print(f"\n🗳️ ENSEMBLE FEATURE SELECTION RESULTS")
        print("-" * 40)
        print(f"Total features considered: {len(feature_votes)}")
        print(f"Selected features: {len(ensemble_features)}")
        
        print(f"\nTop {min(10, len(ensemble_features))} features by voting:")
        for i, (feature, votes) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature:30s} - {votes} votes")
        
        # Evaluate ensemble selection
        imputer = SimpleImputer(strategy='median')
        scaler = RobustScaler()
        
        # Prepare data with ensemble features
        X_train_ensemble = imputer.fit_transform(self.X_train[ensemble_features])
        X_test_ensemble = imputer.transform(self.X_test[ensemble_features])
        
        X_train_scaled = scaler.fit_transform(X_train_ensemble)
        X_test_scaled = scaler.transform(X_test_ensemble)
        
        # Create pipeline
        ensemble_pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000, random_state=42))
        ])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        logloss_scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        
        ensemble_scores = cross_val_score(ensemble_pipeline, X_train_scaled, self.y_train, 
                                        cv=cv, scoring=logloss_scorer)
        
        print(f"\n📈 ENSEMBLE SELECTION PERFORMANCE")
        print(f"Mean Log Loss: {ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")
        
        # Compare with best individual method
        best_individual = min(results.keys(), key=lambda k: results[k]['mean_score'])
        print(f"Best individual method: {best_individual} ({results[best_individual]['mean_score']:.4f})")
        
        improvement = results[best_individual]['mean_score'] - ensemble_scores.mean()
        print(f"Ensemble improvement: {improvement:.4f}")
        
        return {
            'ensemble_features': ensemble_features,
            'feature_votes': feature_votes,
            'scores': ensemble_scores,
            'mean_score': ensemble_scores.mean(),
            'std_score': ensemble_scores.std(),
            'improvement': improvement
        }

    def rolling_backtest_roi(self, 
                           start_date='2024-01-01', 
                           end_date='2025-12-31',
                           test_period_months=6,
                           vegas_data_path=None,
                           stake=100,
                           model_type='logistic'):
        """
        Perform rolling backtesting with 6-month test periods over a 2-year period.
        
        Args:
            start_date: Start date for backtesting (e.g., '2024-01-01')
            end_date: End date for backtesting (e.g., '2025-12-31') 
            test_period_months: Number of months for each test period (default: 6)
            vegas_data_path: Path to Vegas odds data CSV
            stake: Betting stake per fight (default: $100)
            model_type: Type of model to use ('logistic', 'ensemble', 'xgboost')
            
        Returns:
            DataFrame with backtesting results for each 6-month period
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from dateutil.relativedelta import relativedelta
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.ensemble import VotingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        import warnings
        warnings.filterwarnings('ignore')
        
        print("🔄 ROLLING BACKTESTING WITH 6-MONTH PERIODS")
        print("="*60)
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create test period dates (6-month intervals)
        test_periods = []
        current_date = start_dt
        
        while current_date < end_dt:
            next_date = current_date + relativedelta(months=test_period_months)
            if next_date > end_dt:
                next_date = end_dt
            test_periods.append((current_date, next_date))
            current_date = next_date
        
        print(f"📅 Backtesting periods:")
        for i, (start, end) in enumerate(test_periods):
            print(f"  Period {i+1}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        
        results = []
        
        for period_idx, (test_start, test_end) in enumerate(test_periods):
            print(f"\n📊 PERIOD {period_idx + 1}: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            print("-" * 50)
            
            try:
                # Define training data: everything before test_start
                train_data = self.df[self.df['DATE'] < test_start].copy()
                test_data = self.df[(self.df['DATE'] >= test_start) & (self.df['DATE'] < test_end)].copy()
                
                if len(train_data) == 0 or len(test_data) == 0:
                    print(f"⚠️  Skipping period {period_idx + 1}: insufficient data")
                    continue
                
                print(f"Training data: {len(train_data)} fights ({train_data['DATE'].min().date()} to {train_data['DATE'].max().date()})")
                print(f"Test data: {len(test_data)} fights ({test_data['DATE'].min().date()} to {test_data['DATE'].max().date()})")
                
                # Use EXACT same feature selection and preprocessing as tune_logistic_regression
                valid_cols = [c for c in self.importance_columns if c in train_data.columns]
                
                # Apply same preprocessing as main model
                train_data = train_data.dropna(subset=['win'])
                train_data['win'] = train_data['win'].astype(int)
                
                test_data = test_data.dropna(subset=['win'])
                test_data['win'] = test_data['win'].astype(int)
                
                # Filter for fighters with at least 1 previous fight (same as _prepare_data)
                train_data = train_data[
                    (train_data['precomp_boutcount'] >= 1) &
                    (train_data['opp_precomp_boutcount'] >= 1)
                ]
                
                test_data = test_data[
                    (test_data['precomp_boutcount'] >= 1) &
                    (test_data['opp_precomp_boutcount'] >= 1)
                ]
                
                if len(train_data) == 0 or len(test_data) == 0:
                    print(f"⚠️  Skipping period {period_idx + 1}: no data after filtering")
                    continue
                
                # Apply same imputation strategy as _prepare_data
                thresh = int(0.7 * len(valid_cols))
                train_data = train_data[train_data[valid_cols].isnull().sum(axis=1) < thresh]
                test_data = test_data[test_data[valid_cols].isnull().sum(axis=1) < thresh]
                
                if len(train_data) == 0 or len(test_data) == 0:
                    print(f"⚠️  Skipping period {period_idx + 1}: no data after imputation filtering")
                    continue
                
                # Use same imputation strategy as _prepare_data
                imp = SimpleImputer(strategy='median')
                train_data[valid_cols] = imp.fit_transform(train_data[valid_cols])
                test_data[valid_cols] = imp.transform(test_data[valid_cols])
                
                X_train = train_data[valid_cols]
                y_train = train_data['win']
                X_test = test_data[valid_cols]
                y_test = test_data['win']
                
                print(f"Final training set: {len(X_train)} fights")
                print(f"Final test set: {len(X_test)} fights")
                
                # Train model based on type
                # Always use logistic regression
                if True:  # Always use logistic regression
                    # Use EXACT same pipeline and parameters as tune_logistic_regression
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', RobustScaler()),
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
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    probs = best_model.predict_proba(X_test)[:, 1]
                    
                    # Store the best model for this period
                    model = best_model
                    
                # elif model_type == 'ensemble':  # Removed - only using logistic regression
                    # Train ensemble model
                    log_model = LogisticRegression(max_iter=10000, random_state=42)
                    log_model.fit(X_train, y_train)
                    
                    if xgboost_available:
                        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                        xgb_model.fit(X_train, y_train)
                        
                        mlp_model = MLPClassifier(max_iter=300, random_state=42)
                        mlp_model.fit(X_train, y_train)
                        
                        ensemble = VotingClassifier(
                            estimators=[
                                ('logreg', log_model),
                                ('xgb', xgb_model),
                                ('mlp', mlp_model)
                            ],
                            voting='soft'
                        )
                        ensemble.fit(X_train, y_train)
                        probs = ensemble.predict_proba(X_test)[:, 1]
                    else:
                        # Fallback to just logistic regression
                        probs = log_model.predict_proba(X_test)[:, 1]
                        
                # elif model_type == 'xgboost':  # Removed - only using logistic regression
                    if xgboost_available:
                        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                        xgb_model.fit(X_train, y_train)
                        probs = xgb_model.predict_proba(X_test)[:, 1]
                    else:
                        print("XGBoost not available, falling back to logistic regression")
                        model = LogisticRegression(max_iter=10000, random_state=42)
                        model.fit(X_train, y_train)
                        probs = model.predict_proba(X_test)[:, 1]
                
                # Calculate model performance
                preds = (probs > 0.5).astype(int)
                acc = accuracy_score(y_test, preds)
                loss = log_loss(y_test, probs)
                
                print(f"Model accuracy: {acc:.4f}")
                print(f"Model log loss: {loss:.4f}")
                
                # Calculate ROI if Vegas data is available
                roi_result = None
                if vegas_data_path and os.path.exists(vegas_data_path):
                    try:
                        # Create odds table for this period
                        odds_table = make_consistent_odds_table(test_data, probs)
                        
                        # Save temporary odds table
                        temp_odds_path = f'data/tmp/rolling_backtest_odds_{period_idx}.csv'
                        os.makedirs(os.path.dirname(temp_odds_path), exist_ok=True)
                        odds_table.to_csv(temp_odds_path, index=False)
                        
                        # Calculate ROI
                        roi_result = self.calculate_roi(
                            odds_table_path=temp_odds_path,
                            vegas_data_path=vegas_data_path,
                            stake=stake
                        )
                        
                        if isinstance(roi_result, pd.DataFrame) and len(roi_result) > 0:
                            total_fights = len(roi_result)
                            total_profit = roi_result['profit'].sum()
                            total_stake = roi_result['stake'].sum()
                            final_roi = total_profit / total_stake if total_stake > 0 else 0
                            win_rate = roi_result['win'].mean()
                            
                            print(f"ROI: {final_roi:.4f} ({final_roi*100:.2f}%)")
                            print(f"Win rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
                            print(f"Total profit: ${total_profit:.2f}")
                            print(f"Total fights: {total_fights}")
                        else:
                            print("⚠️  No ROI data available for this period")
                            roi_result = None
                            
                    except Exception as e:
                        print(f"⚠️  ROI calculation failed: {e}")
                        roi_result = None
                else:
                    print("⚠️  No Vegas data path provided or file not found")
                
                # Store results
                period_result = {
                    'period': period_idx + 1,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_fights': len(train_data),
                    'test_fights': len(test_data),
                    'accuracy': acc,
                    'log_loss': loss,
                }
                
                if roi_result is not None and isinstance(roi_result, pd.DataFrame) and len(roi_result) > 0:
                    period_result.update({
                        'roi': final_roi,
                        'roi_percent': final_roi * 100,
                        'win_rate': win_rate,
                        'total_profit': total_profit,
                        'total_stake': total_stake,
                        'profitable_fights': total_fights
                    })
                else:
                    period_result.update({
                        'roi': None,
                        'roi_percent': None,
                        'win_rate': None,
                        'total_profit': None,
                        'total_stake': None,
                        'profitable_fights': None
                    })
                
                results.append(period_result)
                
            except Exception as e:
                print(f"❌ Error in period {period_idx + 1}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print("❌ No valid periods completed")
            return pd.DataFrame()
        
        # Summary statistics
        print(f"\n📈 ROLLING BACKTEST SUMMARY")
        print("="*60)
        print(f"Total periods: {len(results_df)}")
        print(f"Average accuracy: {results_df['accuracy'].mean():.4f}")
        print(f"Average log loss: {results_df['log_loss'].mean():.4f}")
        
        if 'roi' in results_df.columns and results_df['roi'].notna().any():
            valid_roi = results_df.dropna(subset=['roi'])
            if len(valid_roi) > 0:
                print(f"Average ROI: {valid_roi['roi'].mean():.4f} ({valid_roi['roi'].mean()*100:.2f}%)")
                print(f"Average win rate: {valid_roi['win_rate'].mean():.4f} ({valid_roi['win_rate'].mean()*100:.2f}%)")
                print(f"Total profit: ${valid_roi['total_profit'].sum():.2f}")
                print(f"Total stake: ${valid_roi['total_stake'].sum():.2f}")
                
                # Period-by-period breakdown
                print(f"\n📊 PERIOD-BY-PERIOD BREAKDOWN")
                print("-" * 60)
                for _, row in results_df.iterrows():
                    if pd.notna(row['roi']):
                        print(f"Period {row['period']}: {row['test_start'].strftime('%Y-%m-%d')} to {row['test_end'].strftime('%Y-%m-%d')}")
                        print(f"  ROI: {row['roi']:.4f} ({row['roi_percent']:.2f}%) | Win Rate: {row['win_rate']:.4f} | Profit: ${row['total_profit']:.2f}")
                    else:
                        print(f"Period {row['period']}: {row['test_start'].strftime('%Y-%m-%d')} to {row['test_end'].strftime('%Y-%m-%d')}")
                        print(f"  ROI: N/A | Accuracy: {row['accuracy']:.4f} | Log Loss: {row['log_loss']:.4f}")
        
        return results_df

    def run_rolling_backtest_default(self, vegas_data_path=None, stake=100):
        """
        Convenience method to run rolling backtest with default settings.
        Uses the same model configuration as tune_logistic_regression().
        
        Args:
            vegas_data_path: Path to Vegas odds data CSV
            stake: Betting stake per fight (default: $100)
            
        Returns:
            DataFrame with backtesting results
        """
        print("🔄 RUNNING ROLLING BACKTEST WITH DEFAULT SETTINGS")
        print("="*60)
        print("Using same model configuration as tune_logistic_regression()")
        print("6-month test periods over 2024-2025")
        
        return self.rolling_backtest_roi(
            start_date='2024-01-01',
            end_date='2025-12-31',
            test_period_months=6,
            vegas_data_path=vegas_data_path,
            stake=stake,
            model_type='logistic'  # Use the same model as tune_logistic_regression
        )

    def run_rolling_backtest_with_current_setup(self, vegas_data_path, stake=100):
        """
        Run rolling backtest using the same Vegas data path as your current setup.
        This method integrates seamlessly with your existing workflow.
        
        Args:
            vegas_data_path: Path to Vegas odds data CSV (e.g., 'final_with_odds_clamped.csv')
            stake: Betting stake per fight (default: $100)
            
        Returns:
            DataFrame with backtesting results
        """
        print("🔄 RUNNING ROLLING BACKTEST WITH YOUR CURRENT SETUP")
        print("="*60)
        print("Using same model configuration as tune_logistic_regression()")
        print("6-month test periods over 2024-2025")
        print(f"Vegas data: {vegas_data_path}")
        print(f"Stake per fight: ${stake}")
        
        return self.rolling_backtest_roi(
            start_date='2024-01-01',
            end_date='2025-12-31',
            test_period_months=6,
            vegas_data_path=vegas_data_path,
            stake=stake,
            model_type='logistic'
        )

    def backward_rolling_backtest_roi(self, 
                                    vegas_data_path=None, 
                                    stake=100, 
                                    test_period_months=6,
                                    num_periods=4,
                                    training_years=15,
                                    constant_window=True,
                                    test_period=0.5):
        """
        Perform backward rolling backtesting starting from the most recent date in the dataset.
        Trains on constant training window and tests on 6-month periods going backwards.
        
        Args:
            vegas_data_path: Path to Vegas odds data CSV
            stake: Betting stake per fight (default: $100)
            test_period_months: Number of months for each test period (default: 6)
            num_periods: Number of test periods to run (default: 4)
            training_years: Number of years for training window (default: 15)
            constant_window: If True, use constant training window; if False, use 2009-01-01 as start (default: True)
            test_period: Number of years for test period (default: 0.5 for 6 months)
            
        Returns:
            DataFrame with backtesting results and generates visualizations
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime, timedelta
        from dateutil.relativedelta import relativedelta
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.ensemble import VotingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        import warnings
        warnings.filterwarnings('ignore')
        
        print("🔄 BACKWARD ROLLING BACKTESTING FROM MOST RECENT DATE")
        print("="*70)
        
        # Find the most recent date in the dataset
        most_recent_date = self.df['DATE'].max()
        print(f"Most recent fight date in dataset: {most_recent_date.strftime('%Y-%m-%d')}")
        print(f"Constant window: {constant_window}")
        print(f"Training window: {training_years} years")
        print(f"Test period: {test_period} years")
        
        # Create test periods going backwards from most recent date
        test_periods = []
        current_end = most_recent_date
        
        for i in range(num_periods):
            test_start = current_end - relativedelta(years=test_period)
            test_periods.append((test_start, current_end))
            current_end = test_start
        
        # Reverse to get chronological order (oldest first)
        test_periods = test_periods[::-1]
        
        print(f"📅 Backtesting periods (going backwards from {most_recent_date.strftime('%Y-%m-%d')}):")
        for i, (start, end) in enumerate(test_periods):
            print(f"  Period {i+1}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        
        results = []
        
        for period_idx, (test_start, test_end) in enumerate(test_periods):
            print(f"\n📊 PERIOD {period_idx + 1}: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            print("-" * 60)
            
            try:
                # Define training data based on constant_window parameter
                if constant_window:
                    # Use constant training window (e.g., 15 years before test_start)
                    train_start = test_start - relativedelta(years=training_years)
                    train_end = test_start - timedelta(days=1)  # Exclude test period
                    train_data = self.df[
                        (self.df['DATE'] >= train_start) & 
                        (self.df['DATE'] <= train_end)
                    ].copy()
                else:
                    # Use expanding window starting from 2009-01-01
                    train_data = self.df[
                        (self.df['DATE'] >= pd.Timestamp('2009-01-01')) & 
                        (self.df['DATE'] < test_start)
                    ].copy()
                
                test_data = self.df[
                    (self.df['DATE'] >= test_start) & 
                    (self.df['DATE'] <= test_end)
                ].copy()
                
                if len(train_data) == 0 or len(test_data) == 0:
                    print(f"⚠️  Skipping period {period_idx + 1}: insufficient data")
                    continue
                
                print(f"Training data: {len(train_data)} fights ({train_data['DATE'].min().date()} to {train_data['DATE'].max().date()})")
                print(f"Test data: {len(test_data)} fights ({test_data['DATE'].min().date()} to {test_data['DATE'].max().date()})")
                if constant_window:
                    print(f"Training window: {training_years} years ending {test_start.strftime('%Y-%m-%d')}")
                else:
                    print(f"Training window: Expanding from 2009-01-01 to {test_start.strftime('%Y-%m-%d')}")
                
                # Use EXACT same feature selection and preprocessing as tune_logistic_regression
                valid_cols = [c for c in self.importance_columns if c in train_data.columns]
                
                # Apply same preprocessing as main model
                train_data = train_data.dropna(subset=['win'])
                train_data['win'] = train_data['win'].astype(int)
                
                test_data = test_data.dropna(subset=['win'])
                test_data['win'] = test_data['win'].astype(int)
                
                # Filter for fighters with at least 1 previous fight (same as _prepare_data)
                train_data = train_data[
                    (train_data['precomp_boutcount'] >= 1) &
                    (train_data['opp_precomp_boutcount'] >= 1)
                ]
                
                test_data = test_data[
                    (test_data['precomp_boutcount'] >= 1) &
                    (test_data['opp_precomp_boutcount'] >= 1)
                ]
                
                if len(train_data) == 0 or len(test_data) == 0:
                    print(f"⚠️  Skipping period {period_idx + 1}: no data after filtering")
                    continue
                
                # Apply same imputation strategy as _prepare_data
                thresh = int(0.7 * len(valid_cols))
                train_data = train_data[train_data[valid_cols].isnull().sum(axis=1) < thresh]
                test_data = test_data[test_data[valid_cols].isnull().sum(axis=1) < thresh]
                
                if len(train_data) == 0 or len(test_data) == 0:
                    print(f"⚠️  Skipping period {period_idx + 1}: no data after imputation filtering")
                    continue
                
                # Use same imputation strategy as _prepare_data
                imp = SimpleImputer(strategy='median')
                train_data[valid_cols] = imp.fit_transform(train_data[valid_cols])
                test_data[valid_cols] = imp.transform(test_data[valid_cols])
                
                X_train = train_data[valid_cols]
                y_train = train_data['win']
                X_test = test_data[valid_cols]
                y_test = test_data['win']
                
                print(f"Final training set: {len(X_train)} fights")
                print(f"Final test set: {len(X_test)} fights")
                
                # Train model based on type
                # Always use logistic regression
                if True:  # Always use logistic regression
                    # Use EXACT same pipeline and parameters as tune_logistic_regression
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', RobustScaler()),
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
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    probs = best_model.predict_proba(X_test)[:, 1]
                    
                    # Store the best model for this period
                    model = best_model
                    
                # elif model_type == 'ensemble':  # Removed - only using logistic regression
                    # Train ensemble model
                    log_model = LogisticRegression(max_iter=10000, random_state=42)
                    log_model.fit(X_train, y_train)
                    
                    if xgboost_available:
                        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                        xgb_model.fit(X_train, y_train)
                        
                        mlp_model = MLPClassifier(max_iter=300, random_state=42)
                        mlp_model.fit(X_train, y_train)
                        
                        ensemble = VotingClassifier(
                            estimators=[
                                ('logreg', log_model),
                                ('xgb', xgb_model),
                                ('mlp', mlp_model)
                            ],
                            voting='soft'
                        )
                        ensemble.fit(X_train, y_train)
                        probs = ensemble.predict_proba(X_test)[:, 1]
                    else:
                        # Fallback to just logistic regression
                        probs = log_model.predict_proba(X_test)[:, 1]
                        
                # elif model_type == 'xgboost':  # Removed - only using logistic regression
                    if xgboost_available:
                        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                        xgb_model.fit(X_train, y_train)
                        probs = xgb_model.predict_proba(X_test)[:, 1]
                    else:
                        print("XGBoost not available, falling back to logistic regression")
                        model = LogisticRegression(max_iter=10000, random_state=42)
                        model.fit(X_train, y_train)
                        probs = model.predict_proba(X_test)[:, 1]
                
                # Calculate model performance
                preds = (probs > 0.5).astype(int)
                acc = accuracy_score(y_test, preds)
                loss = log_loss(y_test, probs)
                
                print(f"Model accuracy: {acc:.4f}")
                print(f"Model log loss: {loss:.4f}")
                
                # Calculate ROI if Vegas data is available
                roi_result = None
                if vegas_data_path and os.path.exists(vegas_data_path):
                    try:
                        # Create odds table for this period
                        odds_table = make_consistent_odds_table(test_data, probs)
                        
                        # Save temporary odds table
                        temp_odds_path = f'data/tmp/backward_backtest_odds_{period_idx}.csv'
                        os.makedirs(os.path.dirname(temp_odds_path), exist_ok=True)
                        odds_table.to_csv(temp_odds_path, index=False)
                        
                        # Calculate ROI
                        roi_result = self.calculate_roi(
                            odds_table_path=temp_odds_path,
                            vegas_data_path=vegas_data_path,
                            stake=stake
                        )
                        
                        if isinstance(roi_result, pd.DataFrame) and len(roi_result) > 0:
                            total_fights = len(roi_result)
                            total_profit = roi_result['profit'].sum()
                            total_stake = roi_result['stake'].sum()
                            final_roi = total_profit / total_stake if total_stake > 0 else 0
                            win_rate = roi_result['win'].mean()
                            
                            print(f"ROI: {final_roi:.4f} ({final_roi*100:.2f}%)")
                            print(f"Win rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
                            print(f"Total profit: ${total_profit:.2f}")
                            print(f"Total fights: {total_fights}")
                        else:
                            print("⚠️  No ROI data available for this period")
                            roi_result = None
                            
                    except Exception as e:
                        print(f"⚠️  ROI calculation failed: {e}")
                        roi_result = None
                else:
                    print("⚠️  No Vegas data path provided or file not found")
                
                # Store results
                period_result = {
                    'period': period_idx + 1,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_fights': len(train_data),
                    'test_fights': len(test_data),
                    'accuracy': acc,
                    'log_loss': loss,
                }
                
                if roi_result is not None and isinstance(roi_result, pd.DataFrame) and len(roi_result) > 0:
                    period_result.update({
                        'roi': final_roi,
                        'roi_percent': final_roi * 100,
                        'win_rate': win_rate,
                        'total_profit': total_profit,
                        'total_stake': total_stake,
                        'profitable_fights': total_fights
                    })
                else:
                    period_result.update({
                        'roi': None,
                        'roi_percent': None,
                        'win_rate': None,
                        'total_profit': None,
                        'total_stake': None,
                        'profitable_fights': None
                    })
                
                results.append(period_result)
                
            except Exception as e:
                print(f"❌ Error in period {period_idx + 1}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print("❌ No valid periods completed")
            return pd.DataFrame()
        
        # Summary statistics
        print(f"\n📈 BACKWARD ROLLING BACKTEST SUMMARY")
        print("="*70)
        print(f"Total periods: {len(results_df)}")
        print(f"Average accuracy: {results_df['accuracy'].mean():.4f}")
        print(f"Average log loss: {results_df['log_loss'].mean():.4f}")
        
        if 'roi' in results_df.columns and results_df['roi'].notna().any():
            valid_roi = results_df.dropna(subset=['roi'])
            if len(valid_roi) > 0:
                print(f"Average ROI: {valid_roi['roi'].mean():.4f} ({valid_roi['roi'].mean()*100:.2f}%)")
                print(f"Average win rate: {valid_roi['win_rate'].mean():.4f} ({valid_roi['win_rate'].mean()*100:.2f}%)")
                print(f"Total profit: ${valid_roi['total_profit'].sum():.2f}")
                print(f"Total stake: ${valid_roi['total_stake'].sum():.2f}")
                
                # Period-by-period breakdown
                print(f"\n📊 PERIOD-BY-PERIOD BREAKDOWN")
                print("-" * 70)
                for _, row in results_df.iterrows():
                    if pd.notna(row['roi']):
                        print(f"Period {row['period']}: {row['test_start'].strftime('%Y-%m-%d')} to {row['test_end'].strftime('%Y-%m-%d')}")
                        print(f"  ROI: {row['roi']:.4f} ({row['roi_percent']:.2f}%) | Win Rate: {row['win_rate']:.4f} | Profit: ${row['total_profit']:.2f}")
                    else:
                        print(f"Period {row['period']}: {row['test_start'].strftime('%Y-%m-%d')} to {row['test_end'].strftime('%Y-%m-%d')}")
                        print(f"  ROI: N/A | Accuracy: {row['accuracy']:.4f} | Log Loss: {row['log_loss']:.4f}")
        
        # Create visualizations
        try:
            self._create_backtest_visualizations(results_df)
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
            print("Creating simple fallback visualization...")
            self._create_simple_backtest_visualization(results_df)
        
        return results_df

    def _create_backtest_visualizations(self, results_df):
        """
        Create comprehensive visualizations for backtest results.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        if results_df.empty:
            print("No data to visualize")
            return
        
        print(f"Creating visualizations for {len(results_df)} periods...")
        print(f"Results DataFrame columns: {list(results_df.columns)}")
        print(f"Results DataFrame shape: {results_df.shape}")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. ROI over time
        ax1 = plt.subplot(3, 3, 1)
        if 'roi' in results_df.columns and results_df['roi'].notna().any():
            valid_roi = results_df.dropna(subset=['roi'])
            if len(valid_roi) > 0:
                periods = valid_roi['period']
                rois = valid_roi['roi'] * 100  # Convert to percentage
                
                bars = ax1.bar(periods, rois, color=['green' if x > 0 else 'red' for x in rois], alpha=0.7)
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax1.set_xlabel('Period')
                ax1.set_ylabel('ROI (%)')
                ax1.set_title('ROI by Period')
                ax1.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, roi in zip(bars, rois):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                            f'{roi:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 2. Cumulative Profit
        ax2 = plt.subplot(3, 3, 2)
        if 'total_profit' in results_df.columns and results_df['total_profit'].notna().any():
            valid_profit = results_df.dropna(subset=['total_profit'])
            if len(valid_profit) > 0:
                cumulative_profit = valid_profit['total_profit'].cumsum()
                ax2.plot(valid_profit['period'], cumulative_profit, marker='o', linewidth=2, markersize=8)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_xlabel('Period')
                ax2.set_ylabel('Cumulative Profit ($)')
                ax2.set_title('Cumulative Profit Over Time')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (period, profit) in enumerate(zip(valid_profit['period'], cumulative_profit)):
                    ax2.annotate(f'${profit:.0f}', (period, profit), 
                               textcoords="offset points", xytext=(0,10), ha='center')
        
        # 3. Win Rate by Period
        ax3 = plt.subplot(3, 3, 3)
        if 'win_rate' in results_df.columns and results_df['win_rate'].notna().any():
            valid_winrate = results_df.dropna(subset=['win_rate'])
            if len(valid_winrate) > 0:
                bars = ax3.bar(valid_winrate['period'], valid_winrate['win_rate'] * 100, 
                              color='skyblue', alpha=0.7)
                ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
                ax3.set_xlabel('Period')
                ax3.set_ylabel('Win Rate (%)')
                ax3.set_title('Win Rate by Period')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, rate in zip(bars, valid_winrate['win_rate'] * 100):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{rate:.1f}%', ha='center', va='bottom')
        
        # 4. Accuracy by Period
        ax4 = plt.subplot(3, 3, 4)
        bars = ax4.bar(results_df['period'], results_df['accuracy'] * 100, 
                      color='lightcoral', alpha=0.7)
        ax4.set_xlabel('Period')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Model Accuracy by Period')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, results_df['accuracy'] * 100):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 5. Log Loss by Period
        ax5 = plt.subplot(3, 3, 5)
        bars = ax5.bar(results_df['period'], results_df['log_loss'], 
                      color='lightgreen', alpha=0.7)
        ax5.set_xlabel('Period')
        ax5.set_ylabel('Log Loss')
        ax5.set_title('Model Log Loss by Period')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, loss in zip(bars, results_df['log_loss']):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        # 6. Number of Fights by Period
        ax6 = plt.subplot(3, 3, 6)
        bars = ax6.bar(results_df['period'], results_df['test_fights'], 
                      color='gold', alpha=0.7)
        ax6.set_xlabel('Period')
        ax6.set_ylabel('Number of Fights')
        ax6.set_title('Test Fights by Period')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, fights in zip(bars, results_df['test_fights']):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{fights}', ha='center', va='bottom')
        
        # 7. ROI vs Win Rate Scatter
        ax7 = plt.subplot(3, 3, 7)
        if 'roi' in results_df.columns and 'win_rate' in results_df.columns:
            valid_data = results_df.dropna(subset=['roi', 'win_rate'])
            if len(valid_data) > 0:
                scatter = ax7.scatter(valid_data['win_rate'] * 100, valid_data['roi'] * 100, 
                                    c=valid_data['period'], s=100, alpha=0.7, cmap='viridis')
                ax7.set_xlabel('Win Rate (%)')
                ax7.set_ylabel('ROI (%)')
                ax7.set_title('ROI vs Win Rate')
                ax7.grid(True, alpha=0.3)
                
                # Add period labels
                for i, row in valid_data.iterrows():
                    ax7.annotate(f"P{row['period']}", 
                               (row['win_rate'] * 100, row['roi'] * 100),
                               xytext=(5, 5), textcoords='offset points')
                
                plt.colorbar(scatter, ax=ax7, label='Period')
        
        # 8. Performance Metrics Heatmap
        ax8 = plt.subplot(3, 3, 8)
        if 'roi' in results_df.columns and results_df['roi'].notna().any():
            valid_roi = results_df.dropna(subset=['roi'])
            if len(valid_roi) > 0:
                # Create a small heatmap of key metrics
                metrics_data = valid_roi[['roi', 'win_rate', 'accuracy']].T
                metrics_data.columns = [f"P{i+1}" for i in range(len(metrics_data.columns))]
                
                im = ax8.imshow(metrics_data.values, cmap='RdYlGn', aspect='auto')
                ax8.set_xticks(range(len(metrics_data.columns)))
                ax8.set_xticklabels(metrics_data.columns)
                ax8.set_yticks(range(len(metrics_data.index)))
                ax8.set_yticklabels(['ROI', 'Win Rate', 'Accuracy'])
                ax8.set_title('Performance Heatmap')
                
                # Add text annotations
                for i in range(len(metrics_data.index)):
                    for j in range(len(metrics_data.columns)):
                        value = metrics_data.iloc[i, j]
                        text = f'{value:.3f}' if 'roi' in metrics_data.index[i].lower() else f'{value:.2f}'
                        ax8.text(j, i, text, ha="center", va="center", color="black")
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate summary stats
        total_periods = len(results_df)
        avg_accuracy = results_df['accuracy'].mean()
        avg_log_loss = results_df['log_loss'].mean()
        
        summary_text = f"""
        BACKTEST SUMMARY
        
        Total Periods: {total_periods}
        Avg Accuracy: {avg_accuracy:.3f}
        Avg Log Loss: {avg_log_loss:.3f}
        """
        
        if 'roi' in results_df.columns and results_df['roi'].notna().any():
            valid_roi = results_df.dropna(subset=['roi'])
            if len(valid_roi) > 0:
                avg_roi = valid_roi['roi'].mean()
                total_profit = valid_roi['total_profit'].sum()
                avg_win_rate = valid_roi['win_rate'].mean()
                
                summary_text += f"""
        Avg ROI: {avg_roi:.3f} ({avg_roi*100:.1f}%)
        Avg Win Rate: {avg_win_rate:.3f} ({avg_win_rate*100:.1f}%)
        Total Profit: ${total_profit:.0f}
        """
        
        ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        import os
        os.makedirs('data/tmp', exist_ok=True)
        plt.savefig('data/tmp/backtest_visualizations.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualizations saved to: data/tmp/backtest_visualizations.png")

    def _create_simple_backtest_visualization(self, results_df):
        """
        Create a simple fallback visualization for backtest results.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if results_df.empty:
            print("No data to visualize")
            return
        
        print(f"Creating simple visualization for {len(results_df)} periods...")
        
        # Create a simple 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Backtest Results Summary', fontsize=16)
        
        # 1. Accuracy by Period
        ax1 = axes[0, 0]
        ax1.bar(results_df['period'], results_df['accuracy'] * 100, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy by Period')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (period, acc) in enumerate(zip(results_df['period'], results_df['accuracy'] * 100)):
            ax1.text(period, acc + 1, f'{acc:.1f}%', ha='center', va='bottom')
        
        # 2. Log Loss by Period
        ax2 = axes[0, 1]
        ax2.bar(results_df['period'], results_df['log_loss'], color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Log Loss')
        ax2.set_title('Model Log Loss by Period')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (period, loss) in enumerate(zip(results_df['period'], results_df['log_loss'])):
            ax2.text(period, loss + 0.01, f'{loss:.3f}', ha='center', va='bottom')
        
        # 3. ROI by Period (if available)
        ax3 = axes[1, 0]
        if 'roi' in results_df.columns and results_df['roi'].notna().any():
            valid_roi = results_df.dropna(subset=['roi'])
            if len(valid_roi) > 0:
                colors = ['green' if x > 0 else 'red' for x in valid_roi['roi']]
                ax3.bar(valid_roi['period'], valid_roi['roi'] * 100, color=colors, alpha=0.7)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.set_xlabel('Period')
                ax3.set_ylabel('ROI (%)')
                ax3.set_title('ROI by Period')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (period, roi) in enumerate(zip(valid_roi['period'], valid_roi['roi'] * 100)):
                    ax3.text(period, roi + (0.5 if roi >= 0 else -1.5), f'{roi:.1f}%', 
                           ha='center', va='bottom' if roi >= 0 else 'top')
            else:
                ax3.text(0.5, 0.5, 'No ROI data available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('ROI by Period (No Data)')
        else:
            ax3.text(0.5, 0.5, 'No ROI data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('ROI by Period (No Data)')
        
        # 4. Number of Fights by Period
        ax4 = axes[1, 1]
        ax4.bar(results_df['period'], results_df['test_fights'], color='gold', alpha=0.7)
        ax4.set_xlabel('Period')
        ax4.set_ylabel('Number of Fights')
        ax4.set_title('Test Fights by Period')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (period, fights) in enumerate(zip(results_df['period'], results_df['test_fights'])):
            ax4.text(period, fights + 0.5, f'{fights}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        import os
        os.makedirs('data/tmp', exist_ok=True)
        plt.savefig('data/tmp/backtest_visualizations.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Simple visualizations saved to: data/tmp/backtest_visualizations.png")
        plt.show()

    def run_constant_window_backtest(self, 
                                   vegas_data_path=None, 
                                   stake=100, 
                                   training_years=15,
                                   num_periods=4,
                                   constant_window=True,
                                   test_period=0.5):
        """
        Convenience method to run backtest with constant training window.
        
        Args:
            vegas_data_path: Path to Vegas odds data CSV
            stake: Betting stake per fight (default: $100)
            training_years: Number of years for training window (default: 15)
            num_periods: Number of 6-month periods to test (default: 4)
            
        Returns:
            DataFrame with backtesting results
        """
        print("🔄 RUNNING CONSTANT WINDOW BACKTEST")
        print("="*60)
        print(f"Training window: {training_years} years")
        print(f"Test periods: {num_periods} periods of {test_period} years each")
        print(f"Model: Logistic Regression")
        
        return self.backward_rolling_backtest_roi(
            vegas_data_path=vegas_data_path,
            stake=stake,
            test_period_months=6,
            training_years=training_years,
            num_periods=num_periods,
            constant_window=constant_window,
            test_period=test_period
        )
