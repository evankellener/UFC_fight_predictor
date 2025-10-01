"""
Test script for the UFC News Scraper

This script demonstrates how to use the news scraper with sample fights from the dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add src directory to path
sys.path.append('src')

from ufc_news_scraper import UFCNewsScraper, NewsFeatures
from enhanced_news_features import EnhancedNewsFeatureExtractor, EnhancedNewsFeatures
from news_integration_pipeline import NewsIntegratedPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_news_scraper():
    """Test the basic news scraper functionality"""
    print("=" * 60)
    print("TESTING BASIC NEWS SCRAPER")
    print("=" * 60)
    
    scraper = UFCNewsScraper()
    
    # Test with a well-known fighter
    fighter_name = "Jon Jones"
    fight_date = "2023-03-04"  # UFC 285
    
    print(f"Testing news scraper for {fighter_name} on {fight_date}")
    
    try:
        features = scraper.extract_fight_features(fighter_name, fight_date)
        
        print(f"\nBasic Features for {fighter_name}:")
        print(f"  Short Notice: {features.short_notice}")
        print(f"  Short Notice Duration: {features.short_notice_duration} days")
        print(f"  Injury Risk: {features.injury_risk}")
        print(f"  Camp Status: {features.camp_status}")
        print(f"  Confidence Score: {features.confidence_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing basic scraper: {str(e)}")
        return False

def test_enhanced_news_features():
    """Test the enhanced news features extractor"""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED NEWS FEATURES")
    print("=" * 60)
    
    extractor = EnhancedNewsFeatureExtractor()
    
    # Test with a well-known fighter
    fighter_name = "Jon Jones"
    fight_date = "2023-03-04"
    
    print(f"Testing enhanced features for {fighter_name} on {fight_date}")
    
    try:
        features = extractor.extract_enhanced_features(fighter_name, fight_date)
        
        print(f"\nEnhanced Features for {fighter_name}:")
        print(f"  Basic Features:")
        print(f"    Short Notice: {features.short_notice}")
        print(f"    Injury Risk: {features.injury_risk}")
        print(f"    Camp Status: {features.camp_status}")
        
        print(f"  Enhanced Features:")
        print(f"    Media Sentiment: {features.media_sentiment:.3f}")
        print(f"    Controversy Score: {features.controversy_score}")
        print(f"    Mental State: {features.mental_state}")
        print(f"    Weight Cut Issues: {features.weight_cut_issues}")
        print(f"    News Frequency: {features.news_frequency:.3f}")
        print(f"    Sentiment Trend: {features.sentiment_trend:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing enhanced features: {str(e)}")
        return False

def test_integration_pipeline():
    """Test the news integration pipeline"""
    print("\n" + "=" * 60)
    print("TESTING NEWS INTEGRATION PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = NewsIntegratedPipeline(data_path="data/final.csv")
        
        print("Pipeline initialized successfully")
        
        # Test with a small sample
        print("Testing with sample data...")
        df_sample = pipeline.add_news_features_to_dataset(sample_size=5)
        
        print(f"Sample dataset shape: {df_sample.shape}")
        print(f"News feature columns: {[col for col in df_sample.columns if col in pipeline.news_feature_columns]}")
        
        # Check if news features were added
        news_cols = [col for col in pipeline.news_feature_columns if col in df_sample.columns]
        if news_cols:
            print(f"\nNews features added successfully:")
            for col in news_cols:
                non_zero = (df_sample[col] != 0).sum()
                print(f"  {col}: {non_zero} non-zero values")
        
        return True
        
    except Exception as e:
        print(f"Error testing integration pipeline: {str(e)}")
        return False

def test_with_sample_fights():
    """Test with specific sample fights from the dataset"""
    print("\n" + "=" * 60)
    print("TESTING WITH SAMPLE FIGHTS FROM DATASET")
    print("=" * 60)
    
    try:
        # Load the dataset
        df = pd.read_csv("data/final.csv", parse_dates=['DATE'])
        
        # Get recent fights (last 2 years)
        recent_date = datetime.now() - timedelta(days=730)
        recent_fights = df[df['DATE'] >= recent_date]
        
        # Get unique fights
        unique_fights = recent_fights[['FIGHTER', 'DATE']].drop_duplicates()
        
        print(f"Found {len(unique_fights)} recent fights")
        
        # Test with a few sample fights
        sample_fights = unique_fights.head(3)
        
        scraper = UFCNewsScraper()
        
        for idx, row in sample_fights.iterrows():
            fighter_name = row['FIGHTER']
            fight_date = row['DATE'].strftime('%Y-%m-%d')
            
            print(f"\nTesting {fighter_name} on {fight_date}")
            
            try:
                features = scraper.extract_fight_features(fighter_name, fight_date)
                
                print(f"  Short Notice: {features.short_notice}")
                print(f"  Injury Risk: {features.injury_risk}")
                print(f"  Camp Status: {features.camp_status}")
                print(f"  Confidence: {features.confidence_score:.3f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing with sample fights: {str(e)}")
        return False

def test_feature_importance():
    """Test feature importance analysis"""
    print("\n" + "=" * 60)
    print("TESTING FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    try:
        pipeline = NewsIntegratedPipeline(data_path="data/final.csv")
        
        # Create a small sample with news features
        print("Creating sample dataset with news features...")
        df_sample = pipeline.add_news_features_to_dataset(sample_size=10)
        
        # Create enhanced features
        df_enhanced = pipeline.create_enhanced_model_features(df_sample)
        
        # Analyze feature importance
        print("Analyzing feature importance...")
        importance_df = pipeline.get_feature_importance_with_news(df_enhanced)
        
        if not importance_df.empty:
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10)[['feature', 'importance', 'feature_type']])
            
            # Show news features specifically
            news_features = importance_df[importance_df['feature_type'] == 'news']
            if not news_features.empty:
                print(f"\nNews Features Importance:")
                print(news_features[['feature', 'importance']])
        
        return True
        
    except Exception as e:
        print(f"Error testing feature importance: {str(e)}")
        return False

def test_performance_comparison():
    """Test model performance comparison"""
    print("\n" + "=" * 60)
    print("TESTING MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    try:
        pipeline = NewsIntegratedPipeline(data_path="data/final.csv")
        
        # Create sample dataset
        print("Creating sample dataset...")
        df_sample = pipeline.add_news_features_to_dataset(sample_size=20)
        
        # Create enhanced features
        df_enhanced = pipeline.create_enhanced_model_features(df_sample)
        
        # Compare performance
        print("Comparing model performance...")
        performance = pipeline.compare_model_performance(df_enhanced)
        
        print(f"\nPerformance Comparison:")
        print(f"  Existing Features Accuracy: {performance['existing_features']['accuracy']:.3f}")
        print(f"  All Features Accuracy: {performance['all_features']['accuracy']:.3f}")
        print(f"  Accuracy Improvement: {performance['accuracy_improvement']:.3f}")
        
        print(f"  Existing Features AUC: {performance['existing_features']['auc']:.3f}")
        print(f"  All Features AUC: {performance['all_features']['auc']:.3f}")
        print(f"  AUC Improvement: {performance['auc_improvement']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing performance comparison: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("UFC NEWS SCRAPER TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic News Scraper", test_basic_news_scraper),
        ("Enhanced News Features", test_enhanced_news_features),
        ("Integration Pipeline", test_integration_pipeline),
        ("Sample Fights", test_with_sample_fights),
        ("Feature Importance", test_feature_importance),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"Test {test_name} failed with error: {str(e)}")
            results[test_name] = "ERROR"
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status_icon = "✓" if result == "PASSED" else "✗"
        print(f"{status_icon} {test_name}: {result}")
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The news scraper is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
