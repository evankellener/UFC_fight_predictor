"""
Notebook Example: Using UFC News Scraper

This script shows how to use the news scraper in a Jupyter notebook environment.
"""

import sys
import os
sys.path.append('src')

from ufc_news_scraper import scrape_fight_news, NewsFeatures
import pandas as pd

# Example 1: Scrape news for a single fighter
print("Example 1: Single Fighter News Scraping")
print("=" * 50)

fighter_name = "Jon Jones"
fight_date = "2023-03-04"

# This is the function you can now import
features = scrape_fight_news(fighter_name, fight_date)

print(f"Fighter: {fighter_name}")
print(f"Fight Date: {fight_date}")
print(f"Short Notice: {features.short_notice}")
print(f"Short Notice Duration: {features.short_notice_duration} days")
print(f"Injury Risk: {features.injury_risk}")
print(f"Camp Status: {features.camp_status}")
print(f"Confidence Score: {features.confidence_score}")

# Example 2: Scrape news for multiple fighters
print("\nExample 2: Multiple Fighters News Scraping")
print("=" * 50)

from ufc_news_scraper import scrape_multiple_fights

fights = [
    ("Jon Jones", "2023-03-04"),
    ("Amanda Nunes", "2023-06-10"),
    ("Conor McGregor", "2021-01-23")
]

results = scrape_multiple_fights(fights)

for (fighter, date), features in results.items():
    print(f"\n{fighter} ({date}):")
    print(f"  Short Notice: {features.short_notice}")
    print(f"  Injury Risk: {features.injury_risk}")
    print(f"  Camp Status: {features.camp_status}")

# Example 3: Add news features to a dataset
print("\nExample 3: Adding News Features to Dataset")
print("=" * 50)

from ufc_news_scraper import add_news_features_to_dataset

# Load your existing dataset
try:
    df = pd.read_csv("data/final.csv", parse_dates=['DATE'])
    print(f"Loaded dataset with {len(df)} rows")
    
    # Add news features (use a small sample for demo)
    df_sample = df.head(10)  # Just first 10 rows for demo
    df_with_news = add_news_features_to_dataset(df_sample)
    
    print(f"Dataset with news features shape: {df_with_news.shape}")
    
    # Show the news features that were added
    news_columns = ['short_notice', 'short_notice_duration', 'injury_risk', 'camp_status', 'news_confidence']
    available_news_cols = [col for col in news_columns if col in df_with_news.columns]
    
    if available_news_cols:
        print(f"News features added: {available_news_cols}")
        print("\nSample news features:")
        print(df_with_news[['FIGHTER', 'DATE'] + available_news_cols].head())
    
except FileNotFoundError:
    print("final.csv not found. Please ensure the data file exists.")

# Example 4: Using the enhanced features
print("\nExample 4: Enhanced News Features")
print("=" * 50)

try:
    from enhanced_news_features import EnhancedNewsFeatureExtractor
    
    extractor = EnhancedNewsFeatureExtractor()
    enhanced_features = extractor.extract_enhanced_features(fighter_name, fight_date)
    
    print(f"Enhanced features for {fighter_name}:")
    print(f"  Media Sentiment: {enhanced_features.media_sentiment}")
    print(f"  Controversy Score: {enhanced_features.controversy_score}")
    print(f"  Mental State: {enhanced_features.mental_state}")
    print(f"  Weight Cut Issues: {enhanced_features.weight_cut_issues}")
    print(f"  News Frequency: {enhanced_features.news_frequency}")
    
except ImportError:
    print("Enhanced features module not available")

print("\n" + "=" * 50)
print("Examples completed!")
print("\nYou can now use these functions in your notebook:")
print("1. scrape_fight_news(fighter_name, fight_date)")
print("2. scrape_multiple_fights(fights_list)")
print("3. add_news_features_to_dataset(dataframe)")
