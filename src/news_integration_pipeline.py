"""
News Integration Pipeline

This module integrates news features with the existing UFC fight prediction pipeline.
It extends the current model with news-based features while maintaining compatibility.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
from pathlib import Path

from enhanced_news_features import EnhancedNewsFeatureExtractor, EnhancedNewsFeatures
from ufc_news_scraper import UFCNewsScraper

logger = logging.getLogger(__name__)

class NewsIntegratedPipeline:
    """
    Pipeline that integrates news features with existing fight prediction model
    """
    
    def __init__(self, data_path: str = "data/final.csv", 
                 news_cache_dir: str = "news_cache",
                 model_cache_dir: str = "model_cache"):
        """
        Initialize the news-integrated pipeline
        
        Args:
            data_path: Path to the main fight dataset
            news_cache_dir: Directory for news feature caching
            model_cache_dir: Directory for model caching
        """
        self.data_path = data_path
        self.news_cache_dir = news_cache_dir
        self.model_cache_dir = model_cache_dir
        
        # Initialize components
        self.news_extractor = EnhancedNewsFeatureExtractor(news_cache_dir)
        self.news_scraper = UFCNewsScraper()
        
        # Load base dataset
        self.df = self._load_base_dataset()
        
        # Define news feature columns
        self.news_feature_columns = [
            'short_notice', 'short_notice_duration', 'injury_risk', 'camp_status',
            'news_confidence', 'media_sentiment', 'controversy_score',
            'training_mentions', 'injury_mentions', 'camp_mentions',
            'weight_cut_issues', 'mental_state', 'opponent_analysis',
            'fight_prediction_mentions', 'news_frequency', 'early_news_sentiment',
            'late_news_sentiment', 'sentiment_trend'
        ]
        
        # Define existing model feature columns (from your ensemble model)
        self.existing_feature_columns = [
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
        
        # Combined feature set
        self.all_feature_columns = self.existing_feature_columns + self.news_feature_columns

    def _load_base_dataset(self) -> pd.DataFrame:
        """Load the base fight dataset"""
        try:
            df = pd.read_csv(self.data_path, parse_dates=['DATE'])
            logger.info(f"Loaded dataset with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def add_news_features_to_dataset(self, df: Optional[pd.DataFrame] = None, 
                                    sample_size: Optional[int] = None,
                                    output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Add news features to the dataset
        
        Args:
            df: Optional DataFrame to process (uses self.df if None)
            sample_size: Optional sample size for testing
            output_file: Optional file to save results
            
        Returns:
            DataFrame with added news features
        """
        if df is None:
            df = self.df.copy()
        
        if sample_size:
            # Sample for testing
            unique_fights = df[['FIGHTER', 'DATE']].drop_duplicates()
            sample_fights = unique_fights.sample(n=min(sample_size, len(unique_fights)), random_state=42)
            df = df.merge(sample_fights, on=['FIGHTER', 'DATE'])
            logger.info(f"Sampling {len(sample_fights)} fights for news feature extraction")
        
        # Add news feature columns
        for col in self.news_feature_columns:
            if col not in df.columns:
                df[col] = 0.0 if 'sentiment' in col or 'frequency' in col else 0
        
        # Process unique fights
        unique_fights = df[['FIGHTER', 'DATE']].drop_duplicates()
        logger.info(f"Processing {len(unique_fights)} unique fights for news features")
        
        for idx, row in unique_fights.iterrows():
            try:
                # Extract enhanced features
                features = self.news_extractor.extract_enhanced_features(
                    row['FIGHTER'], 
                    row['DATE'].strftime('%Y-%m-%d')
                )
                
                # Update all rows for this fighter and date
                mask = (df['FIGHTER'] == row['FIGHTER']) & (df['DATE'] == row['DATE'])
                
                for col in self.news_feature_columns:
                    df.loc[mask, col] = getattr(features, col)
                
                logger.info(f"Processed {row['FIGHTER']} - {row['DATE']}")
                
            except Exception as e:
                logger.error(f"Error processing {row['FIGHTER']}: {str(e)}")
                continue
        
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Dataset with news features saved to {output_file}")
        
        return df

    def create_enhanced_model_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create enhanced model features by combining existing and news features
        
        Args:
            df: Optional DataFrame to process
            
        Returns:
            DataFrame with enhanced features
        """
        if df is None:
            df = self.df.copy()
        
        # Ensure news features are present
        if not all(col in df.columns for col in self.news_feature_columns):
            logger.info("Adding news features to dataset")
            df = self.add_news_features_to_dataset(df)
        
        # Create enhanced features by combining existing and news features
        enhanced_features = []
        
        # 1. Interaction features between news and existing features
        if 'injury_risk' in df.columns and 'precomp_elo' in df.columns:
            df['injury_elo_interaction'] = df['injury_risk'] * df['precomp_elo']
            enhanced_features.append('injury_elo_interaction')
        
        if 'camp_status' in df.columns and 'precomp_tdavg' in df.columns:
            df['camp_takedown_interaction'] = df['camp_status'] * df['precomp_tdavg']
            enhanced_features.append('camp_takedown_interaction')
        
        if 'short_notice' in df.columns and 'precomp_elo_change_5' in df.columns:
            df['short_notice_elo_change'] = df['short_notice'].astype(int) * df['precomp_elo_change_5']
            enhanced_features.append('short_notice_elo_change')
        
        # 2. News-based composite features
        if all(col in df.columns for col in ['injury_risk', 'camp_status', 'weight_cut_issues']):
            df['overall_preparation_score'] = (
                df['injury_risk'] + df['camp_status'] + df['weight_cut_issues']
            ) / 3
            enhanced_features.append('overall_preparation_score')
        
        if all(col in df.columns for col in ['media_sentiment', 'mental_state', 'controversy_score']):
            df['psychological_factors'] = (
                df['media_sentiment'] + df['mental_state'] - df['controversy_score']
            ) / 3
            enhanced_features.append('psychological_factors')
        
        # 3. Temporal news features
        if all(col in df.columns for col in ['early_news_sentiment', 'late_news_sentiment']):
            df['sentiment_momentum'] = df['late_news_sentiment'] - df['early_news_sentiment']
            enhanced_features.append('sentiment_momentum')
        
        # 4. News frequency features
        if 'news_frequency' in df.columns:
            df['news_attention_score'] = np.log1p(df['news_frequency'])
            enhanced_features.append('news_attention_score')
        
        logger.info(f"Created {len(enhanced_features)} enhanced features")
        
        return df

    def get_feature_importance_with_news(self, df: Optional[pd.DataFrame] = None,
                                       target_col: str = 'win',
                                       test_size: float = 0.2) -> pd.DataFrame:
        """
        Analyze feature importance including news features
        
        Args:
            df: Optional DataFrame to analyze
            target_col: Target column name
            test_size: Test set size for validation
            
        Returns:
            DataFrame with feature importance analysis
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        if df is None:
            df = self.df.copy()
        
        # Ensure news features are present
        if not all(col in df.columns for col in self.news_feature_columns):
            logger.info("Adding news features for importance analysis")
            df = self.add_news_features_to_dataset(df)
        
        # Select all available features
        available_features = [col for col in self.all_feature_columns if col in df.columns]
        
        if not available_features:
            logger.warning("No features available for importance analysis")
            return pd.DataFrame()
        
        # Prepare data
        X = df[available_features].fillna(0)
        y = df[target_col]
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            logger.warning("No valid data for importance analysis")
            return pd.DataFrame()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Get predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_,
            'accuracy': accuracy,
            'auc': auc
        }).sort_values('importance', ascending=False)
        
        # Categorize features
        importance_df['feature_type'] = importance_df['feature'].apply(
            lambda x: 'news' if x in self.news_feature_columns else 'existing'
        )
        
        logger.info(f"Feature importance analysis completed. Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        return importance_df

    def create_news_enhanced_predictions(self, df: Optional[pd.DataFrame] = None,
                                       model_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create predictions using news-enhanced features
        
        Args:
            df: Optional DataFrame to predict on
            model_path: Optional path to saved model
            
        Returns:
            DataFrame with predictions
        """
        if df is None:
            df = self.df.copy()
        
        # Ensure news features are present
        if not all(col in df.columns for col in self.news_feature_columns):
            logger.info("Adding news features for predictions")
            df = self.add_news_features_to_dataset(df)
        
        # Create enhanced features
        df = self.create_enhanced_model_features(df)
        
        # Select features for prediction
        prediction_features = [col for col in self.all_feature_columns if col in df.columns]
        
        # Add any enhanced features that were created
        enhanced_features = [col for col in df.columns if col not in self.all_feature_columns and col not in ['DATE', 'FIGHTER', 'EVENT', 'BOUT', 'win']]
        prediction_features.extend(enhanced_features)
        
        # Prepare data for prediction
        X = df[prediction_features].fillna(0)
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            # Load existing model
            from joblib import load
            model = load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            # Create new model (simplified for example)
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train on available data with target
            if 'win' in df.columns:
                y = df['win'].fillna(0)
                model.fit(X, y)
                logger.info("Trained new model with news features")
            else:
                logger.warning("No target column found, cannot train model")
                return df
        
        # Make predictions
        predictions = model.predict_proba(X)[:, 1]
        df['news_enhanced_prediction'] = predictions
        
        # Add confidence scores
        df['prediction_confidence'] = np.abs(predictions - 0.5) * 2
        
        logger.info(f"Created predictions for {len(df)} fights")
        
        return df

    def compare_model_performance(self, df: Optional[pd.DataFrame] = None,
                                 target_col: str = 'win',
                                 test_size: float = 0.2) -> Dict[str, float]:
        """
        Compare performance of model with and without news features
        
        Args:
            df: Optional DataFrame to analyze
            target_col: Target column name
            test_size: Test set size for validation
            
        Returns:
            Dictionary with performance metrics
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
        
        if df is None:
            df = self.df.copy()
        
        # Ensure news features are present
        if not all(col in df.columns for col in self.news_feature_columns):
            logger.info("Adding news features for performance comparison")
            df = self.add_news_features_to_dataset(df)
        
        # Prepare data
        y = df[target_col].fillna(0)
        mask = ~y.isna()
        df = df[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Model 1: Existing features only
        existing_features = [col for col in self.existing_feature_columns if col in df.columns]
        X_existing_train = X_train[existing_features].fillna(0)
        X_existing_test = X_test[existing_features].fillna(0)
        
        model_existing = RandomForestClassifier(n_estimators=100, random_state=42)
        model_existing.fit(X_existing_train, y_train)
        
        y_pred_existing = model_existing.predict(X_existing_test)
        y_pred_proba_existing = model_existing.predict_proba(X_existing_test)[:, 1]
        
        # Model 2: All features (existing + news)
        all_features = [col for col in self.all_feature_columns if col in df.columns]
        X_all_train = X_train[all_features].fillna(0)
        X_all_test = X_test[all_features].fillna(0)
        
        model_all = RandomForestClassifier(n_estimators=100, random_state=42)
        model_all.fit(X_all_train, y_train)
        
        y_pred_all = model_all.predict(X_all_test)
        y_pred_proba_all = model_all.predict_proba(X_all_test)[:, 1]
        
        # Calculate metrics
        results = {
            'existing_features': {
                'accuracy': accuracy_score(y_test, y_pred_existing),
                'auc': roc_auc_score(y_test, y_pred_proba_existing),
                'precision': precision_score(y_test, y_pred_existing),
                'recall': recall_score(y_test, y_pred_existing)
            },
            'all_features': {
                'accuracy': accuracy_score(y_test, y_pred_all),
                'auc': roc_auc_score(y_test, y_pred_proba_all),
                'precision': precision_score(y_test, y_pred_all),
                'recall': recall_score(y_test, y_pred_all)
            }
        }
        
        # Calculate improvements
        for metric in ['accuracy', 'auc', 'precision', 'recall']:
            improvement = results['all_features'][metric] - results['existing_features'][metric]
            results[f'{metric}_improvement'] = improvement
        
        logger.info(f"Performance comparison completed")
        logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.3f}")
        logger.info(f"AUC improvement: {results['auc_improvement']:.3f}")
        
        return results

    def generate_news_feature_report(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate a comprehensive report on news features
        
        Args:
            df: Optional DataFrame to analyze
            
        Returns:
            Dictionary with news feature analysis
        """
        if df is None:
            df = self.df.copy()
        
        # Ensure news features are present
        if not all(col in df.columns for col in self.news_feature_columns):
            logger.info("Adding news features for report generation")
            df = self.add_news_features_to_dataset(df)
        
        report = {}
        
        # Basic statistics
        report['total_fights'] = len(df)
        report['fights_with_news'] = len(df[df['news_confidence'] > 0])
        report['news_coverage_rate'] = report['fights_with_news'] / report['total_fights']
        
        # News feature statistics
        for col in self.news_feature_columns:
            if col in df.columns:
                report[f'{col}_stats'] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'non_zero': (df[col] != 0).sum()
                }
        
        # Short notice analysis
        if 'short_notice' in df.columns:
            report['short_notice_stats'] = {
                'total_short_notice': df['short_notice'].sum(),
                'short_notice_rate': df['short_notice'].mean(),
                'avg_duration': df[df['short_notice']]['short_notice_duration'].mean()
            }
        
        # Injury risk analysis
        if 'injury_risk' in df.columns:
            report['injury_risk_stats'] = {
                'high_injury_risk': (df['injury_risk'] >= 7).sum(),
                'injury_risk_rate': (df['injury_risk'] >= 7).mean(),
                'avg_injury_risk': df['injury_risk'].mean()
            }
        
        # Camp status analysis
        if 'camp_status' in df.columns:
            report['camp_status_stats'] = {
                'camp_issues': (df['camp_status'] >= 7).sum(),
                'camp_issue_rate': (df['camp_status'] >= 7).mean(),
                'avg_camp_status': df['camp_status'].mean()
            }
        
        logger.info("News feature report generated")
        
        return report


def main():
    """Example usage of the News Integration Pipeline"""
    pipeline = NewsIntegratedPipeline()
    
    # Add news features to dataset
    df_with_news = pipeline.add_news_features_to_dataset(sample_size=10)
    
    # Create enhanced features
    df_enhanced = pipeline.create_enhanced_model_features(df_with_news)
    
    # Analyze feature importance
    importance_df = pipeline.get_feature_importance_with_news(df_enhanced)
    print("Top 10 most important features:")
    print(importance_df.head(10))
    
    # Compare model performance
    performance = pipeline.compare_model_performance(df_enhanced)
    print(f"Performance comparison: {performance}")
    
    # Generate report
    report = pipeline.generate_news_feature_report(df_enhanced)
    print(f"News feature report: {report}")


if __name__ == "__main__":
    main()
