"""
Mock UFC News Scraper

This module provides mock news features for demonstration purposes when
real news scraping is blocked or unavailable.
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class NewsFeatures:
    """Data class to hold extracted news features"""
    short_notice: bool = False
    short_notice_duration: int = 0
    injury_risk: int = 0
    camp_status: int = 0
    confidence_score: float = 0.0

def scrape_fight_news(fighter_name: str, fight_date: str, days_before: int = 90) -> NewsFeatures:
    """
    Mock function to scrape news for a single fight
    
    Args:
        fighter_name: Name of the fighter
        fight_date: Date of the fight (YYYY-MM-DD format)
        days_before: Number of days before fight to search
        
    Returns:
        NewsFeatures object with mock extracted features
    """
    print(f"Mock scraping news for {fighter_name} on {fight_date}")
    
    # Generate realistic mock features based on fighter name patterns
    name_lower = fighter_name.lower()
    
    # Short notice heuristics
    short_notice = random.random() < 0.15  # 15% chance of short notice
    short_notice_duration = random.randint(3, 14) if short_notice else 0
    
    # Injury risk based on fighter profile
    injury_risk = 0
    if any(word in name_lower for word in ['jones', 'mcgregor', 'diaz', 'lesnar', 'silva', 'gsp']):
        injury_risk = random.randint(2, 6)  # Higher profile fighters
    elif any(word in name_lower for word in ['rookie', 'debut', 'new']):
        injury_risk = random.randint(0, 3)  # Newer fighters
    else:
        injury_risk = random.randint(0, 4)  # Average fighters
    
    # Camp status based on fighter activity
    camp_status = 0
    if any(word in name_lower for word in ['jones', 'mcgregor', 'diaz', 'lesnar']):
        camp_status = random.randint(1, 5)  # High-profile fighters
    elif any(word in name_lower for word in ['champion', 'champ', 'title']):
        camp_status = random.randint(0, 4)  # Champions
    else:
        camp_status = random.randint(0, 3)  # Regular fighters
    
    # Add some randomness
    injury_risk += random.randint(-1, 1)
    camp_status += random.randint(-1, 1)
    
    # Normalize scores
    injury_risk = max(0, min(10, injury_risk))
    camp_status = max(0, min(10, camp_status))
    
    # Confidence based on fighter profile
    confidence = 0.7 if any(word in name_lower for word in ['jones', 'mcgregor', 'diaz', 'lesnar']) else 0.5
    
    return NewsFeatures(
        short_notice=short_notice,
        short_notice_duration=short_notice_duration,
        injury_risk=injury_risk,
        camp_status=camp_status,
        confidence_score=confidence
    )

def scrape_multiple_fights(fights: List[Tuple[str, str]], max_workers: int = 3) -> Dict[Tuple[str, str], NewsFeatures]:
    """
    Mock function to scrape news for multiple fights
    
    Args:
        fights: List of (fighter_name, fight_date) tuples
        max_workers: Maximum number of parallel workers (ignored in mock)
        
    Returns:
        Dictionary mapping (fighter_name, fight_date) to NewsFeatures
    """
    results = {}
    
    for fighter_name, fight_date in fights:
        features = scrape_fight_news(fighter_name, fight_date)
        results[(fighter_name, fight_date)] = features
    
    return results

def add_news_features_to_dataset(df: pd.DataFrame, 
                                fighter_col: str = 'FIGHTER',
                                date_col: str = 'DATE',
                                output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Mock function to add news features to a dataset
    
    Args:
        df: DataFrame with fight data
        fighter_col: Column name containing fighter names
        date_col: Column name containing fight dates
        output_file: Optional file to save results
        
    Returns:
        DataFrame with added news features
    """
    print(f"Mock adding news features to {len(df)} fights")
    
    # Add news feature columns
    df['short_notice'] = False
    df['short_notice_duration'] = 0
    df['injury_risk'] = 0
    df['camp_status'] = 0
    df['news_confidence'] = 0.0
    
    # Process unique fights
    unique_fights = df[[fighter_col, date_col]].drop_duplicates()
    
    for idx, row in unique_fights.iterrows():
        features = scrape_fight_news(row[fighter_col], row[date_col].strftime('%Y-%m-%d'))
        
        # Update all rows for this fighter and date
        mask = (df[fighter_col] == row[fighter_col]) & (df[date_col] == row[date_col])
        df.loc[mask, 'short_notice'] = features.short_notice
        df.loc[mask, 'short_notice_duration'] = features.short_notice_duration
        df.loc[mask, 'injury_risk'] = features.injury_risk
        df.loc[mask, 'camp_status'] = features.camp_status
        df.loc[mask, 'news_confidence'] = features.confidence_score
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Mock results saved to {output_file}")
    
    return df

def main():
    """Example usage of the mock news scraper"""
    print("Mock UFC News Scraper Example")
    print("=" * 40)
    
    # Test single fighter
    features = scrape_fight_news("Jon Jones", "2023-03-04")
    print(f"Jon Jones features:")
    print(f"  Short Notice: {features.short_notice}")
    print(f"  Injury Risk: {features.injury_risk}")
    print(f"  Camp Status: {features.camp_status}")
    print(f"  Confidence: {features.confidence_score}")
    
    # Test multiple fighters
    fights = [("Jon Jones", "2023-03-04"), ("Amanda Nunes", "2023-06-10")]
    results = scrape_multiple_fights(fights)
    
    for (fighter, date), features in results.items():
        print(f"\n{fighter} ({date}):")
        print(f"  Short Notice: {features.short_notice}")
        print(f"  Injury Risk: {features.injury_risk}")
        print(f"  Camp Status: {features.camp_status}")

if __name__ == "__main__":
    main()
