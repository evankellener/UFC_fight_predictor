"""
Real UFC News Scraper - Actually scrapes real news articles
No fake data, no demo mode - only real news or failure
"""

import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealUFCNewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Real RSS feeds for MMA news
        self.rss_feeds = {
            'bloody_elbow': 'https://www.bloodyelbow.com/rss',
            'mma_fighting': 'https://www.mmafighting.com/rss',
            'cageside_seats': 'https://www.cagesideseats.com/rss',
            'mmamania': 'https://www.mmamania.com/rss',
            'lowkick_mma': 'https://www.lowkickmma.com/feed/'
        }
        
        # UFC-related keywords
        self.ufc_keywords = [
            'ufc', 'ultimate fighting championship', 'dana white',
            'conor mcgregor', 'jon jones', 'amanda nunes', 'valentina shevchenko',
            'khabib nurmagomedov', 'islam makhachev', 'alexander volkanovski',
            'champion', 'title fight', 'main event', 'co-main event'
        ]
        
        # Fighter name patterns (to be updated with actual fighter names)
        self.fighter_patterns = []

    def add_fighter_names(self, fighter_names: List[str]):
        """Add fighter names to search for in articles"""
        self.fighter_patterns.extend([name.lower() for name in fighter_names])
        
    def fetch_rss_articles(self, days_back: int = 7) -> List[Dict]:
        """Fetch articles from RSS feeds"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for source, rss_url in self.rss_feeds.items():
            try:
                logger.info(f"Fetching RSS feed: {source}")
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    try:
                        # Parse publication date
                        pub_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        
                        # Skip old articles
                        if pub_date and pub_date < cutoff_date:
                            continue
                            
                        article = {
                            'title': entry.get('title', ''),
                            'link': entry.get('link', ''),
                            'published': pub_date,
                            'summary': entry.get('summary', ''),
                            'source': source
                        }
                        
                        # Check if article is UFC-related
                        if self._is_ufc_related(article):
                            articles.append(article)
                            
                    except Exception as e:
                        logger.warning(f"Error processing RSS entry: {e}")
                        continue
                        
                logger.info(f"Found {len([a for a in articles if a['source'] == source])} UFC articles from {source}")
                
            except Exception as e:
                logger.error(f"Error fetching RSS feed {source}: {e}")
                continue
                
            # Be respectful - small delay between feeds
            time.sleep(1)
            
        return articles

    def _is_ufc_related(self, article: Dict) -> bool:
        """Check if article is UFC-related"""
        text = f"{article['title']} {article['summary']}".lower()
        
        # Check for UFC keywords
        ufc_match = any(keyword in text for keyword in self.ufc_keywords)
        
        # Check for fighter names if provided
        fighter_match = any(fighter in text for fighter in self.fighter_patterns)
        
        return ufc_match or fighter_match

    def scrape_fight_news(self, fighter_names: List[str], event_date: str, days_before: int = 14) -> pd.DataFrame:
        """
        Scrape real news articles for specific fighters
        
        Args:
            fighter_names: List of fighter names to search for
            event_date: Date of the fight (YYYY-MM-DD)
            days_before: How many days before the fight to search
            
        Returns:
            DataFrame with real news articles
        """
        logger.info(f"🔍 Scraping REAL news for: {', '.join(fighter_names)}")
        logger.info(f"📅 Fight date: {event_date}")
        
        # Add fighter names to search patterns
        self.add_fighter_names(fighter_names)
        
        # Fetch articles from RSS feeds
        articles = self.fetch_rss_articles(days_before)
        
        if not articles:
            logger.warning("❌ No articles found from RSS feeds")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Filter for articles containing fighter names
        if fighter_names:
            fighter_articles = []
            for fighter in fighter_names:
                fighter_lower = fighter.lower()
                fighter_matches = df[
                    df['title'].str.lower().str.contains(fighter_lower, na=False) |
                    df['summary'].str.lower().str.contains(fighter_lower, na=False)
                ].copy()
                fighter_matches['fighter_mentioned'] = fighter
                fighter_articles.append(fighter_matches)
            
            if fighter_articles:
                df = pd.concat(fighter_articles, ignore_index=True)
                df = df.drop_duplicates(subset=['link'])  # Remove duplicates
            else:
                logger.warning(f"❌ No articles found mentioning fighters: {fighter_names}")
                return pd.DataFrame()
        
        logger.info(f"✅ Found {len(df)} real news articles")
        
        # Add basic features
        df['sentiment_score'] = self._calculate_sentiment(df)
        df['injury_mentioned'] = df['title'].str.contains('injury|hurt|damage', case=False, na=False)
        df['weight_mentioned'] = df['title'].str.contains('weight|cut|missed', case=False, na=False)
        df['training_mentioned'] = df['title'].str.contains('training|camp|preparation', case=False, na=False)
        
        return df

    def _calculate_sentiment(self, df: pd.DataFrame) -> float:
        """Simple sentiment calculation based on keywords"""
        positive_words = ['great', 'excellent', 'ready', 'confident', 'prepared', 'healthy', 'strong']
        negative_words = ['injured', 'hurt', 'problem', 'concern', 'struggle', 'difficult', 'issue']
        
        def get_sentiment(text):
            if pd.isna(text):
                return 0.0
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            return (pos_count - neg_count) / max(pos_count + neg_count, 1)
        
        # Calculate sentiment for each article
        df['article_sentiment'] = df['title'].apply(get_sentiment)
        
        # Return average sentiment
        return df['article_sentiment'].mean() if len(df) > 0 else 0.0

    def generate_prediction_features(self, fighter_names: List[str], event_date: str) -> Dict:
        """
        Generate prediction features from real news data
        
        Returns:
            Dictionary with real features or empty dict if no data found
        """
        news_df = self.scrape_fight_news(fighter_names, event_date)
        
        if news_df.empty:
            logger.warning("❌ No real news data available - cannot generate features")
            return {}
        
        # Group by fighter mentioned
        features = {}
        
        for i, fighter in enumerate(fighter_names):
            fighter_key = 'FIGHTER' if i == 0 else 'opp_FIGHTER'
            
            fighter_articles = news_df[news_df['fighter_mentioned'].str.contains(fighter, case=False, na=False)]
            
            if len(fighter_articles) > 0:
                features[f'{fighter_key}_news_sentiment'] = fighter_articles['article_sentiment'].mean()
                features[f'{fighter_key}_news_count'] = len(fighter_articles)
                features[f'{fighter_key}_injury_risk'] = fighter_articles['injury_mentioned'].mean()
                features[f'{fighter_key}_weight_cut_issues'] = fighter_articles['weight_mentioned'].mean()
                features[f'{fighter_key}_training_disruption'] = 1 - fighter_articles['training_mentioned'].mean()
            else:
                # No news found for this fighter
                features[f'{fighter_key}_news_sentiment'] = 0.0
                features[f'{fighter_key}_news_count'] = 0.0
                features[f'{fighter_key}_injury_risk'] = 0.0
                features[f'{fighter_key}_weight_cut_issues'] = 0.0
                features[f'{fighter_key}_training_disruption'] = 0.0
        
        logger.info(f"✅ Generated {len(features)} real prediction features")
        return features

def create_real_news_scraping_method():
    """Create and return a real news scraping function"""
    return RealUFCNewsScraper()

# Test function
if __name__ == "__main__":
    scraper = create_real_news_scraping_method()
    
    # Test with real fighters
    test_fighters = ['Jon Jones', 'Stipe Miocic']
    features = scraper.generate_prediction_features(test_fighters, '2024-03-02')
    
    print("Real news features generated:")
    for key, value in features.items():
        print(f"  {key}: {value}")
