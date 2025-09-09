import pandas as pd
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
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from statsmodels.stats.proportion import proportion_confint
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
            'precomp_elo','opp_precomp_elo', 'precomp_elo_change_3', 'opp_precomp_elo_change_3', 'precomp_elo_change_5', 'opp_precomp_elo_change_5', 
            'precomp_tdavg3', 'opp_precomp_tdavg3', 'precomp_tdavg5', 'opp_precomp_tdavg5', 'precomp_tddef3', 'opp_precomp_tddef3', 'precomp_tddef5', 'opp_precomp_tddef5',
            'precomp_totalacc_perc' , 'opp_precomp_totalacc_perc', 'precomp_totalacc_perc3', 'opp_precomp_totalacc_perc3', 'precomp_totalacc_perc5', 'opp_precomp_totalacc_perc5',
            'precomp_strdef', 'opp_precomp_strdef', 'precomp_strdef3', 'opp_precomp_strdef3', 'precomp_strdef5', 'opp_precomp_strdef5',
            'age_ratio_difference', 'opp_age_ratio_difference', 'precomp_strike_elo', 'opp_precomp_strike_elo', 'precomp_strike_elo_change_3', 'opp_precomp_strike_elo_change_3', 'precomp_strike_elo_change_5', 'opp_precomp_strike_elo_change_5',
            'opp_precomp_tdavg', 'precomp_tdavg','opp_precomp_tdacc_perc5', 'precomp_tdacc_perc5', 'REACH', 'opp_REACH', 'precomp_winsum3', 'opp_precomp_winsum3', 'weightindex', 'opp_weightindex', 'weight_of_fight', 'opp_weight_of_fight',
            'precomp_distacc_perc', 'opp_precomp_distacc_perc', 'precomp_tdacc_perc3', 'opp_precomp_tdacc_perc3', 'precomp_legacc_perc3', 'opp_precomp_legacc_perc3'
        ]

        # best log loss:
        """
        'age_ratio_difference','opp_age_ratio_difference','opp_precomp_elo_change_5', 'precomp_elo','opp_precomp_elo','precomp_tdavg', 'opp_precomp_tdavg','opp_precomp_tddef',
            'opp_precomp_sapm5','precomp_tddef','precomp_sapm5','precomp_headacc_perc3','opp_precomp_headacc_perc3','precomp_totalacc_perc3','precomp_elo_change_5','REACH','opp_REACH',
            'precomp_legacc_perc5','opp_precomp_totalacc_perc3','opp_precomp_legacc_perc5','opp_precomp_clinchacc_perc5','precomp_clinchacc_perc5','precomp_winsum3','opp_precomp_winsum3',
            'opp_precomp_sapm','precomp_sapm','opp_precomp_totalacc_perc','precomp_totalacc_perc','precomp_groundacc_perc5', 'opp_precomp_groundacc_perc5','precomp_losssum5','opp_precomp_losssum5',
            'age','opp_age','precomp_strike_elo', 'opp_precomp_strike_elo'
        """
        
        self.df = pd.read_csv(file_path, low_memory=False)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df = self.df[self.df['DATE'] >= '2009-01-01']
        # Fix: Convert sex to string for comparison since it's stored as string in the data
        self.df = self.df[self.df['sex'].astype(str) == '2']
        
        
        # Auto-detect scaler path if not provided
        if scaler_path is None:
            import os
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
        if not hasattr(self, 'probs'):
            raise RuntimeError("Run tune_logistic_regression() first.")
        return make_consistent_odds_table(self.test_df, self.probs)
    
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
        import sys
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

        # 5.5. Filter out statistical outliers in Vegas odds
        print("Filtering odds outliers...", flush=True)
        df_filtered = self.filter_odds_outliers(df, 'avg_vegas_odds', method='iqr', threshold=2.0, vegas_cols=vegas_cols)
        print(f"[DEBUG] df_filtered length: {len(df_filtered)}", flush=True)
        print(f"[DEBUG] df_filtered columns: {list(df_filtered.columns)}", flush=True)
        print(f"[DEBUG] First few rows of df_filtered:\n{df_filtered.head()}", flush=True)

        # 6. Pick your model's favorite by lowest American odds
        if len(df_filtered) == 0:
            print("[DEBUG] df_filtered is empty after filtering. Exiting early.", flush=True)
            return pd.DataFrame()
        idx = df_filtered.groupby('BOUT')['odds'].idxmin()
        if len(idx) == 0:
            print("[DEBUG] idx is empty after groupby. Exiting early.", flush=True)
            return pd.DataFrame()
        picks = df_filtered.loc[idx].copy()

        # Clamp avg_vegas_odds to avoid unrealistic profits
        def clamp_vegas_odds(odds):
            if 0 < odds < 100:
                return 100
            if -100 < odds < 0:
                return -100
            return odds
        picks['avg_vegas_odds'] = picks['avg_vegas_odds'].apply(clamp_vegas_odds)

        # --- ROI Calculation Section ---
        if picks is not None:
            if picks.empty:
                print("[DEBUG] WARNING: picks DataFrame is empty after filtering! Printing the DataFrame being filtered:")
                try:
                    print("[DEBUG] DataFrame before filtering (head):\n", df_filtered.head())
                    print("[DEBUG] DataFrame before filtering (columns):", df_filtered.columns.tolist())
                except Exception as e:
                    print("[DEBUG] Could not print df_filtered:", e)
            else:
                print("[DEBUG] picks DataFrame shape:", picks.shape)
                print("[DEBUG] picks DataFrame columns:", picks.columns.tolist())
                print("[DEBUG] picks DataFrame head:\n", picks.head())
                print("[DEBUG] picks DataFrame tail:\n", picks.tail())
                print("[DEBUG] picks DataFrame non-null counts:\n", picks.count())
                print("[DEBUG] picks['DATE'] unique count:", picks['DATE'].nunique() if 'DATE' in picks.columns else 'DATE column missing')
                print("[DEBUG] picks['profit'] describe:\n", picks['profit'].describe() if 'profit' in picks.columns else 'profit column missing')
        else:
            print("[DEBUG] picks is None after filtering!")

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

        # Debug: print columns and head of picks after filtering
        print("\n[DEBUG] Columns in picks after filtering:", list(picks.columns), flush=True)
        print("[DEBUG] First few rows of picks after filtering:\n", picks.head(), flush=True)

        # Identify the night (date) with the most money gained
        night_profit = picks.groupby('DATE')['profit'].sum()
        best_night = night_profit.idxmax()
        best_night_profit = night_profit.max()
        best_night_fights = picks[picks['DATE'] == best_night].shape[0]
        print(f"\nMost profitable night: {best_night.date()} | Profit: ${best_night_profit:.2f} | Fights: {best_night_fights}", flush=True)

        # Print all fights from the most profitable night with detailed stats
        fights_night = picks[picks['DATE'] == best_night].copy()
        # Fix artificial odds: if avg_vegas_odds is between -100 and +100, set odds to -100 if negative, +100 if positive
        def fix_artificial_odds(row):
            if -100 < row['avg_vegas_odds'] < 100:
                return 100 if row['avg_vegas_odds'] > 0 else -100
            return row['odds']
        fights_night['odds'] = fights_night.apply(fix_artificial_odds, axis=1)
        print("[DEBUG] Columns in fights_night:", list(fights_night.columns), flush=True)
        print("[DEBUG] First few rows of fights_night:\n", fights_night.head(), flush=True)
        # Columns to show: BOUT, FIGHTER, win, avg_vegas_odds, odds (model), profit
        cols_to_show = ['BOUT', 'FIGHTER', 'win', 'avg_vegas_odds', 'odds', 'profit']
        try:
            print("\nFight-by-fight stats for the most profitable night (with fixed artificial odds):", flush=True)
            print(fights_night[cols_to_show].to_string(index=False), flush=True)
            print("\nAverages for this night:", flush=True)
            print(fights_night[cols_to_show].mean(numeric_only=True).to_string(), flush=True)
        except Exception as e:
            print("\n[DEBUG] Could not print fight table. Available columns:", list(fights_night.columns), flush=True)
            print("Error:", e, flush=True)

        # 9. Report
        overall = picks['cum_roi'].iloc[-1]
        print(f"Overall ROI: {overall:.4%}")

        # --- After cumulative ROI calculation, before visualization ---
        # Debug: Print picks DataFrame info before best night/month analysis
        print("\n[DEBUG] picks DataFrame shape:", picks.shape)
        print("[DEBUG] picks DataFrame columns:", picks.columns.tolist())
        print("[DEBUG] picks DataFrame head:\n", picks.head())
        print("[DEBUG] picks DataFrame tail:\n", picks.tail())
        print("[DEBUG] picks DataFrame non-null counts:\n", picks.count())
        print("[DEBUG] picks['DATE'] unique count:", picks['DATE'].nunique() if 'DATE' in picks.columns else 'DATE column missing')
        print("[DEBUG] picks['profit'] describe:\n", picks['profit'].describe() if 'profit' in picks.columns else 'profit column missing')

        # Identify the night (date) with the most money gained
        night_profit = picks.groupby('DATE')['profit'].sum()
        print("[DEBUG] night_profit (sum by DATE):\n", night_profit)
        if not night_profit.empty:
            best_night = night_profit.idxmax()
            best_night_profit = night_profit.max()
            best_night_fights = picks[picks['DATE'] == best_night].shape[0]
            print(f"\nMost profitable night: {best_night.date()} | Profit: ${best_night_profit:.2f} | Fights: {best_night_fights}")
        else:
            print("[DEBUG] night_profit is empty. No profitable nights found.")
            best_night = None
            best_night_profit = None
            best_night_fights = 0

        # Identify the month with the most money gained
        if 'DATE' in picks.columns:
            picks['MONTH'] = picks['DATE'].dt.to_period('M')
            month_profit = picks.groupby('MONTH')['profit'].sum()
            print("[DEBUG] month_profit (sum by MONTH):\n", month_profit)
            if not month_profit.empty:
                best_month = month_profit.idxmax()
                best_month_profit = month_profit.max()
                best_month_fights = picks[picks['MONTH'] == best_month].shape[0]
                print(f"Most profitable month: {best_month} | Profit: ${best_month_profit:.2f} | Fights: {best_month_fights}")
            else:
                print("[DEBUG] month_profit is empty. No profitable months found.")
                best_month = None
                best_month_profit = None
                best_month_fights = 0
        else:
            print("[DEBUG] 'DATE' column missing from picks. Cannot compute month_profit.")
            best_month = None
            best_month_profit = None
            best_month_fights = 0
        # --- End best night/month calculation ---

        # --- Visualization Section ---
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot 1: Cumulative Profit Over Time
        axs[0].plot(picks['DATE'], picks['cum_profit'], marker='o', label='Cumulative Profit')
        axs[0].set_ylabel('Cumulative Profit ($)')
        axs[0].set_title('Cumulative Profit Over Time')
        axs[0].grid(True)
        axs[0].legend()
        fig.savefig('roi_cumulative_profit.png')

        # Plot 2: Cumulative ROI Over Time
        axs[1].plot(picks['DATE'], picks['cum_roi'], marker='o', color='orange', label='Cumulative ROI')
        axs[1].set_ylabel('Cumulative ROI')
        axs[1].set_title('Cumulative ROI Over Time')
        axs[1].grid(True)
        axs[1].legend()
        fig.savefig('roi_cumulative_roi.png')

        # Plot 3: Profit Per Fight
        axs[2].bar(picks['DATE'], picks['profit'], color='green', alpha=0.6, label='Profit per Fight')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Profit per Fight ($)')
        axs[2].set_title('Profit per Fight')
        axs[2].grid(True)
        axs[2].legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig('roi_profit_per_fight.png')
        plt.show()
        # --- End Visualization Section ---

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
                ('scaler', RobustScaler()),
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
