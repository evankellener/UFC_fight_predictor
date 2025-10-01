"""
Enhanced News Features Module

This module provides advanced feature extraction from UFC news sources,
integrating with the existing fight prediction pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import re
from collections import Counter
import json
import os

from ufc_news_scraper import UFCNewsScraper, NewsFeatures

logger = logging.getLogger(__name__)

@dataclass
class EnhancedNewsFeatures:
    """Enhanced news features with additional metrics"""
    # Basic features
    short_notice: bool = False
    short_notice_duration: int = 0
    injury_risk: int = 0
    camp_status: int = 0
    confidence_score: float = 0.0
    
    # Enhanced features
    media_sentiment: float = 0.0  # -1 to 1 scale
    controversy_score: int = 0    # 0-10 scale
    training_mentions: int = 0   # Number of training-related mentions
    injury_mentions: int = 0     # Number of injury-related mentions
    camp_mentions: int = 0       # Number of camp-related mentions
    weight_cut_issues: int = 0   # 0-10 scale for weight cut problems
    mental_state: int = 0        # 0-10 scale for mental/psychological state
    opponent_analysis: int = 0   # 0-10 scale for opponent analysis mentions
    fight_prediction_mentions: int = 0  # Number of prediction mentions
    
    # Temporal features
    news_frequency: float = 0.0  # Articles per day leading to fight
    early_news_sentiment: float = 0.0  # Sentiment in early news
    late_news_sentiment: float = 0.0   # Sentiment in late news
    sentiment_trend: float = 0.0       # Trend in sentiment over time

class EnhancedNewsFeatureExtractor:
    """
    Enhanced feature extractor that builds upon the basic news scraper
    """
    
    def __init__(self, cache_dir: str = "news_cache"):
        """
        Initialize the enhanced feature extractor
        
        Args:
            cache_dir: Directory to cache news data
        """
        self.scraper = UFCNewsScraper()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Enhanced keyword sets
        self.sentiment_positive = [
            'confident', 'ready', 'prepared', 'focused', 'determined',
            'excited', 'motivated', 'sharp', 'in shape', 'peak condition',
            'training hard', 'looking good', 'feeling great', 'strong',
            'dominant', 'impressive', 'excellent', 'outstanding'
        ]
        
        self.sentiment_negative = [
            'concerned', 'worried', 'struggling', 'difficult', 'challenging',
            'problem', 'issue', 'setback', 'disappointing', 'poor',
            'weak', 'sluggish', 'tired', 'exhausted', 'injured',
            'hurt', 'pain', 'sore', 'recovering', 'struggling'
        ]
        
        self.controversy_keywords = [
            'controversy', 'drama', 'beef', 'feud', 'rivalry', 'trash talk',
            'disrespect', 'insult', 'argument', 'conflict', 'tension',
            'bad blood', 'heated', 'intense', 'personal', 'grudge'
        ]
        
        self.weight_cut_keywords = [
            'weight cut', 'cutting weight', 'weight cutting', 'dehydration',
            'weight issues', 'weight problems', 'struggling with weight',
            'weight concerns', 'weight management', 'making weight',
            'weight class', 'moving up', 'moving down', 'weight advantage'
        ]
        
        self.mental_state_keywords = [
            'mental', 'psychological', 'mindset', 'confidence', 'nerves',
            'pressure', 'stress', 'anxiety', 'focused', 'distracted',
            'motivated', 'demotivated', 'mental toughness', 'mental game',
            'head space', 'mental preparation', 'mental state'
        ]
        
        self.opponent_analysis_keywords = [
            'opponent', 'matchup', 'style', 'game plan', 'strategy',
            'strengths', 'weaknesses', 'advantages', 'disadvantages',
            'tough opponent', 'difficult matchup', 'favorable matchup',
            'unfavorable matchup', 'style clash', 'technical', 'striking',
            'grappling', 'wrestling', 'jiu-jitsu', 'standup', 'ground'
        ]

    def _load_cached_features(self, fighter_name: str, fight_date: str) -> Optional[EnhancedNewsFeatures]:
        """Load cached features if available"""
        cache_file = os.path.join(
            self.cache_dir, 
            f"{fighter_name.replace(' ', '_')}_{fight_date}.json"
        )
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return EnhancedNewsFeatures(**data)
            except Exception as e:
                logger.warning(f"Error loading cache for {fighter_name}: {str(e)}")
        
        return None

    def _save_cached_features(self, fighter_name: str, fight_date: str, 
                            features: EnhancedNewsFeatures):
        """Save features to cache"""
        cache_file = os.path.join(
            self.cache_dir, 
            f"{fighter_name.replace(' ', '_')}_{fight_date}.json"
        )
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(features.__dict__, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache for {fighter_name}: {str(e)}")

    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text (-1 to 1)"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.sentiment_positive if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_negative if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        
        # Calculate net sentiment
        sentiment = positive_score - negative_score
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, sentiment * 10))

    def _calculate_controversy_score(self, text: str) -> int:
        """Calculate controversy score (0-10)"""
        text_lower = text.lower()
        controversy_count = sum(1 for word in self.controversy_keywords if word in text_lower)
        
        # Normalize to 0-10 scale
        return min(10, controversy_count)

    def _calculate_weight_cut_issues(self, text: str) -> int:
        """Calculate weight cut issues score (0-10)"""
        text_lower = text.lower()
        weight_count = sum(1 for word in self.weight_cut_keywords if word in text_lower)
        
        # Look for specific weight cut problems
        problem_indicators = [
            'struggling', 'difficult', 'hard', 'tough', 'problem',
            'issue', 'concern', 'worry', 'trouble'
        ]
        
        problem_count = sum(1 for word in problem_indicators if word in text_lower)
        
        # Combine weight mentions with problem indicators
        score = weight_count + problem_count
        return min(10, score)

    def _calculate_mental_state(self, text: str) -> int:
        """Calculate mental state score (0-10)"""
        text_lower = text.lower()
        mental_count = sum(1 for word in self.mental_state_keywords if word in text_lower)
        
        # Look for positive mental indicators
        positive_mental = ['confident', 'focused', 'motivated', 'determined', 'ready']
        positive_count = sum(1 for word in positive_mental if word in text_lower)
        
        # Look for negative mental indicators
        negative_mental = ['nervous', 'anxious', 'worried', 'distracted', 'unfocused']
        negative_count = sum(1 for word in negative_mental if word in text_lower)
        
        # Calculate net mental state
        score = mental_count + positive_count - negative_count
        return max(0, min(10, score))

    def _calculate_opponent_analysis(self, text: str) -> int:
        """Calculate opponent analysis mentions (0-10)"""
        text_lower = text.lower()
        analysis_count = sum(1 for word in self.opponent_analysis_keywords if word in text_lower)
        
        return min(10, analysis_count)

    def _calculate_temporal_features(self, articles: List[Dict], fight_date: str) -> Tuple[float, float, float, float]:
        """Calculate temporal features from articles"""
        fight_dt = datetime.strptime(fight_date, '%Y-%m-%d')
        
        # Sort articles by date (assuming they're in chronological order)
        articles_by_date = sorted(articles, key=lambda x: x.get('date', fight_dt))
        
        if not articles_by_date:
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculate news frequency (articles per day)
        days_span = (fight_dt - articles_by_date[0].get('date', fight_dt)).days
        news_frequency = len(articles) / max(1, days_span)
        
        # Split articles into early and late periods
        mid_point = len(articles_by_date) // 2
        early_articles = articles_by_date[:mid_point]
        late_articles = articles_by_date[mid_point:]
        
        # Calculate sentiment for early and late periods
        early_sentiment = np.mean([self._calculate_sentiment(art['full_text']) for art in early_articles]) if early_articles else 0.0
        late_sentiment = np.mean([self._calculate_sentiment(art['full_text']) for art in late_articles]) if late_articles else 0.0
        
        # Calculate sentiment trend
        sentiment_trend = late_sentiment - early_sentiment
        
        return news_frequency, early_sentiment, late_sentiment, sentiment_trend

    def extract_enhanced_features(self, fighter_name: str, fight_date: str, 
                                 days_before: int = 90) -> EnhancedNewsFeatures:
        """
        Extract enhanced news features for a fighter and fight
        
        Args:
            fighter_name: Name of the fighter
            fight_date: Date of the fight (YYYY-MM-DD format)
            days_before: Number of days before fight to search
            
        Returns:
            EnhancedNewsFeatures object with all extracted features
        """
        # Check cache first
        cached_features = self._load_cached_features(fighter_name, fight_date)
        if cached_features:
            logger.info(f"Using cached features for {fighter_name}")
            return cached_features
        
        logger.info(f"Extracting enhanced features for {fighter_name} on {fight_date}")
        
        # Get basic features from scraper
        basic_features = self.scraper.extract_fight_features(fighter_name, fight_date, days_before)
        
        # Get articles for enhanced analysis
        articles = self.scraper._search_fighter_news(fighter_name, fight_date, days_before)
        
        if not articles:
            logger.warning(f"No articles found for {fighter_name}")
            return EnhancedNewsFeatures()
        
        # Calculate enhanced features
        all_text = ' '.join([art['full_text'] for art in articles])
        
        # Basic features from scraper
        short_notice = basic_features.short_notice
        short_notice_duration = basic_features.short_notice_duration
        injury_risk = basic_features.injury_risk
        camp_status = basic_features.camp_status
        confidence_score = basic_features.confidence_score
        
        # Enhanced features
        media_sentiment = self._calculate_sentiment(all_text)
        controversy_score = self._calculate_controversy_score(all_text)
        training_mentions = all_text.count('training') + all_text.count('camp')
        injury_mentions = all_text.count('injury') + all_text.count('injured')
        camp_mentions = all_text.count('camp') + all_text.count('training')
        weight_cut_issues = self._calculate_weight_cut_issues(all_text)
        mental_state = self._calculate_mental_state(all_text)
        opponent_analysis = self._calculate_opponent_analysis(all_text)
        fight_prediction_mentions = all_text.count('prediction') + all_text.count('predict')
        
        # Temporal features
        news_frequency, early_sentiment, late_sentiment, sentiment_trend = self._calculate_temporal_features(articles, fight_date)
        
        # Create enhanced features object
        enhanced_features = EnhancedNewsFeatures(
            short_notice=short_notice,
            short_notice_duration=short_notice_duration,
            injury_risk=injury_risk,
            camp_status=camp_status,
            confidence_score=confidence_score,
            media_sentiment=media_sentiment,
            controversy_score=controversy_score,
            training_mentions=training_mentions,
            injury_mentions=injury_mentions,
            camp_mentions=camp_mentions,
            weight_cut_issues=weight_cut_issues,
            mental_state=mental_state,
            opponent_analysis=opponent_analysis,
            fight_prediction_mentions=fight_prediction_mentions,
            news_frequency=news_frequency,
            early_news_sentiment=early_sentiment,
            late_news_sentiment=late_sentiment,
            sentiment_trend=sentiment_trend
        )
        
        # Cache the results
        self._save_cached_features(fighter_name, fight_date, enhanced_features)
        
        return enhanced_features

    def process_dataset_with_enhanced_features(self, df: pd.DataFrame, 
                                              fighter_col: str = 'FIGHTER',
                                              date_col: str = 'DATE',
                                              output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process entire dataset with enhanced news features
        
        Args:
            df: DataFrame with fight data
            fighter_col: Column name containing fighter names
            date_col: Column name containing fight dates
            output_file: Optional file to save results
            
        Returns:
            DataFrame with added enhanced news features
        """
        logger.info(f"Processing {len(df)} fights with enhanced news features")
        
        # Add all enhanced feature columns
        enhanced_columns = [
            'short_notice', 'short_notice_duration', 'injury_risk', 'camp_status',
            'news_confidence', 'media_sentiment', 'controversy_score',
            'training_mentions', 'injury_mentions', 'camp_mentions',
            'weight_cut_issues', 'mental_state', 'opponent_analysis',
            'fight_prediction_mentions', 'news_frequency', 'early_news_sentiment',
            'late_news_sentiment', 'sentiment_trend'
        ]
        
        for col in enhanced_columns:
            df[col] = 0.0 if 'sentiment' in col or 'frequency' in col else 0
        
        # Process each unique fight
        unique_fights = df[[fighter_col, date_col]].drop_duplicates()
        
        for idx, row in unique_fights.iterrows():
            try:
                features = self.extract_enhanced_features(
                    row[fighter_col], 
                    row[date_col].strftime('%Y-%m-%d')
                )
                
                # Update all rows for this fighter and date
                mask = (df[fighter_col] == row[fighter_col]) & (df[date_col] == row[date_col])
                
                for col in enhanced_columns:
                    df.loc[mask, col] = getattr(features, col)
                
                logger.info(f"Processed {row[fighter_col]} - {row[date_col]}")
                
            except Exception as e:
                logger.error(f"Error processing {row[fighter_col]}: {str(e)}")
                continue
        
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Enhanced features saved to {output_file}")
        
        return df

    def get_feature_importance_analysis(self, df: pd.DataFrame, target_col: str = 'win') -> pd.DataFrame:
        """
        Analyze feature importance for news features
        
        Args:
            df: DataFrame with news features and target
            target_col: Target column name
            
        Returns:
            DataFrame with feature importance analysis
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Select news feature columns
        news_columns = [
            'short_notice', 'short_notice_duration', 'injury_risk', 'camp_status',
            'media_sentiment', 'controversy_score', 'training_mentions',
            'injury_mentions', 'camp_mentions', 'weight_cut_issues',
            'mental_state', 'opponent_analysis', 'fight_prediction_mentions',
            'news_frequency', 'early_news_sentiment', 'late_news_sentiment',
            'sentiment_trend'
        ]
        
        # Filter available columns
        available_columns = [col for col in news_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("No news features found in dataset")
            return pd.DataFrame()
        
        # Prepare data
        X = df[available_columns].fillna(0)
        y = df[target_col]
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            logger.warning("No valid data for feature importance analysis")
            return pd.DataFrame()
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': available_columns,
            'importance': rf.feature_importances_,
            'accuracy': [accuracy_score(y_test, rf.predict(X_test))] * len(available_columns)
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """Example usage of the Enhanced News Feature Extractor"""
    extractor = EnhancedNewsFeatureExtractor()
    
    # Example: Extract enhanced features for a specific fighter
    fighter_name = "Jon Jones"
    fight_date = "2023-03-04"
    
    features = extractor.extract_enhanced_features(fighter_name, fight_date)
    print(f"Enhanced features for {fighter_name}:")
    print(f"  Short Notice: {features.short_notice}")
    print(f"  Injury Risk: {features.injury_risk}")
    print(f"  Camp Status: {features.camp_status}")
    print(f"  Media Sentiment: {features.media_sentiment}")
    print(f"  Controversy Score: {features.controversy_score}")
    print(f"  Mental State: {features.mental_state}")
    print(f"  News Frequency: {features.news_frequency}")


if __name__ == "__main__":
    main()
