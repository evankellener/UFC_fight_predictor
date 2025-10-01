"""
UFC News Scraper Module

This module scrapes UFC news websites to extract features for fight prediction models.
Features extracted:
- short_notice: Boolean indicating if fight was short notice
- short_notice_duration: Days between fight announcement and fight date
- injury_risk: Integer score for injury concerns mentioned in news
- camp_status: Integer score for training camp issues mentioned in news
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
from typing import Dict, List, Tuple, Optional
import logging
from urllib.parse import urljoin, quote
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsFeatures:
    """Data class to hold extracted news features"""
    short_notice: bool = False
    short_notice_duration: int = 0
    injury_risk: int = 0
    camp_status: int = 0
    confidence_score: float = 0.0

class UFCNewsScraper:
    """
    Main class for scraping UFC news and extracting features
    """
    
    def __init__(self, delay_range=(2, 5)):
        """
        Initialize the scraper with configurable delay between requests
        
        Args:
            delay_range: Tuple of (min, max) seconds to wait between requests
        """
        self.delay_range = delay_range
        self.session = requests.Session()
        
        # Rotate between different user agents to avoid detection
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        
        # Set initial headers
        self._update_headers()
        
        # UFC news sources - using more reliable sources and direct URLs
        self.news_sources = {
            'mmafighting': {
                'base_url': 'https://www.mmafighting.com',
                'search_url': 'https://www.mmafighting.com/search',
                'article_selector': 'a[href*="/"]',
                'content_selector': '.c-entry-content, .entry-content, .post-content'
            },
            'espn_mma': {
                'base_url': 'https://www.espn.com/mma',
                'search_url': 'https://www.espn.com/mma',
                'article_selector': 'a[href*="/mma/"]',
                'content_selector': '.article-body, .StoryBody, .content'
            },
            'mmajunkie': {
                'base_url': 'https://mmajunkie.usatoday.com',
                'search_url': 'https://mmajunkie.usatoday.com',
                'article_selector': 'a[href*="/"]',
                'content_selector': '.c-entry-content, .entry-content, .post-content'
            },
            'sherdog': {
                'base_url': 'https://www.sherdog.com',
                'search_url': 'https://www.sherdog.com/news',
                'article_selector': 'a[href*="/news/"]',
                'content_selector': '.news-content, .content, .article-content'
            }
        }
        
        # Keywords for feature extraction
        self.short_notice_keywords = [
            'short notice', 'replacement', 'fill-in', 'last minute', 'emergency',
            'stepped in', 'stepping in', 'replacement fighter', 'late replacement',
            'fight week', 'days notice', 'weeks notice', 'announced today',
            'just announced', 'recently announced', 'fight announcement'
        ]
        
        self.injury_keywords = [
            'injury', 'injured', 'hurt', 'pain', 'sore', 'nagging injury',
            'recovering', 'rehabilitation', 'physical therapy', 'medical',
            'health concern', 'injury concern', 'injury scare', 'injury report',
            'injury update', 'injury status', 'injury news', 'injury problems',
            'injury issues', 'injury setback', 'injury recovery', 'injury rehab'
        ]
        
        self.camp_keywords = [
            'training camp', 'camp', 'training', 'preparation', 'prep',
            'training issues', 'camp problems', 'training problems',
            'camp issues', 'training concerns', 'camp concerns',
            'training setback', 'camp setback', 'training difficulties',
            'camp difficulties', 'training struggles', 'camp struggles',
            'training disruption', 'camp disruption', 'training interruption',
            'camp interruption', 'training schedule', 'camp schedule',
            'training routine', 'camp routine', 'training environment',
            'camp environment', 'training facility', 'camp facility'
        ]
        
        # Negative indicators (reduce scores)
        self.negative_keywords = [
            'no injury', 'healthy', 'injury free', 'fully healthy',
            'no problems', 'smooth', 'excellent', 'great', 'perfect',
            'no issues', 'no concerns', 'no problems', 'no setbacks'
        ]

    def _update_headers(self):
        """Update session headers with random user agent"""
        user_agent = random.choice(self.user_agents)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1'
        })

    def _delay(self):
        """Add random delay between requests to be respectful"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

    def _search_fighter_news(self, fighter_name: str, fight_date: str, days_before: int = 90) -> List[Dict]:
        """
        Search for news articles about a fighter leading up to a fight
        
        Args:
            fighter_name: Name of the fighter
            fight_date: Date of the fight (YYYY-MM-DD format)
            days_before: Number of days before fight to search
            
        Returns:
            List of article dictionaries with title, content, date, url
        """
        articles = []
        fight_dt = datetime.strptime(fight_date, '%Y-%m-%d')
        search_start = fight_dt - timedelta(days=days_before)
        
        # Clean fighter name for search
        search_name = self._clean_fighter_name(fighter_name)
        
        for source_name, source_config in self.news_sources.items():
            try:
                self._delay()
                source_articles = self._search_source(
                    source_name, source_config, search_name, fight_dt, search_start
                )
                articles.extend(source_articles)
                logger.info(f"Found {len(source_articles)} articles from {source_name}")
            except Exception as e:
                logger.warning(f"Error searching {source_name}: {str(e)}")
                continue
        
        return articles

    def _clean_fighter_name(self, name: str) -> str:
        """Clean fighter name for search queries"""
        # Remove common prefixes and suffixes
        name = re.sub(r'\b(Champ|Champion|Former|Ex-|Retired)\b', '', name, flags=re.IGNORECASE)
        name = name.strip()
        return name

    def _search_source(self, source_name: str, source_config: Dict, 
                      fighter_name: str, fight_date: datetime, 
                      search_start: datetime) -> List[Dict]:
        """Search a specific news source"""
        articles = []
        
        try:
            # Update headers before each request
            self._update_headers()
            self._delay()
            
            # Try direct page access first
            base_url = source_config['base_url']
            response = self.session.get(base_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.select(source_config['article_selector'])
            
            # Filter links that might contain fighter name
            fighter_keywords = fighter_name.lower().split()
            relevant_links = []
            
            for link in article_links[:20]:  # Check more links
                try:
                    link_text = link.get_text().lower()
                    link_href = link.get('href', '')
                    
                    # Check if link text or href contains fighter keywords
                    if any(keyword in link_text or keyword in link_href for keyword in fighter_keywords):
                        relevant_links.append(link)
                except:
                    continue
            
            # If no relevant links found, use all links
            if not relevant_links:
                relevant_links = article_links[:10]
            
            for link in relevant_links[:5]:  # Limit to first 5 results
                try:
                    article_url = urljoin(source_config['base_url'], link.get('href'))
                    article_data = self._scrape_article(article_url, source_config, fight_date, search_start)
                    if article_data:
                        articles.append(article_data)
                except Exception as e:
                    logger.warning(f"Error scraping article: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error searching {source_name}: {str(e)}")
            
        return articles

    def _construct_search_url(self, source_config: Dict, fighter_name: str) -> str:
        """Construct search URL for a specific source"""
        base_url = source_config['search_url']
        
        # Different sources may have different search parameter formats
        if 'mmajunkie' in base_url or 'mmafighting' in base_url:
            return f"{base_url}?q={quote(fighter_name)}"
        elif 'sherdog' in base_url:
            return f"{base_url}?q={quote(fighter_name)}"
        elif 'espn' in base_url:
            return f"{base_url}?q={quote(fighter_name)}&type=article"
        else:
            return f"{base_url}?q={quote(fighter_name)}"

    def _scrape_article(self, url: str, source_config: Dict, 
                       fight_date: datetime, search_start: datetime) -> Optional[Dict]:
        """Scrape individual article content"""
        try:
            self._delay()
            self._update_headers()  # Update headers for each request
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple content selectors
            content_elem = None
            for selector in source_config['content_selector'].split(', '):
                content_elem = soup.select_one(selector)
                if content_elem:
                    break
            
            if not content_elem:
                # Fallback: get all text from body
                content_elem = soup.find('body')
                if not content_elem:
                    return None
                
            content = content_elem.get_text().lower()
            
            # Get title
            title = soup.find('title')
            title_text = title.get_text().lower() if title else ""
            
            # Get meta description as additional content
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_text = meta_desc.get('content', '').lower() if meta_desc else ""
            
            # Extract date (simplified - in real implementation, you'd parse actual dates)
            article_date = self._extract_article_date(soup, fight_date)
            
            # Check if article is within our time window
            if article_date and article_date < search_start:
                return None
                
            # Combine all text
            full_text = f"{title_text} {meta_text} {content}"
            
            return {
                'url': url,
                'title': title_text,
                'content': content,
                'date': article_date,
                'full_text': full_text
            }
            
        except Exception as e:
            logger.warning(f"Error scraping article {url}: {str(e)}")
            return None

    def _extract_article_date(self, soup: BeautifulSoup, fight_date: datetime) -> Optional[datetime]:
        """Extract article date (simplified implementation)"""
        # This is a simplified version - in practice, you'd parse actual dates
        # For now, we'll assume articles are within the search window
        if isinstance(fight_date, str):
            fight_date = datetime.strptime(fight_date, '%Y-%m-%d')
        return fight_date - timedelta(days=random.randint(1, 30))

    def _extract_short_notice_features(self, articles: List[Dict]) -> Tuple[bool, int]:
        """
        Extract short notice features from articles
        
        Returns:
            Tuple of (is_short_notice, duration_in_days)
        """
        short_notice_score = 0
        duration_mentions = []
        
        for article in articles:
            text = article['full_text']
            
            # Check for short notice keywords
            for keyword in self.short_notice_keywords:
                if keyword in text:
                    short_notice_score += 1
                    
            # Look for duration mentions (simplified)
            duration_patterns = [
                r'(\d+)\s*days?\s*notice',
                r'(\d+)\s*weeks?\s*notice',
                r'(\d+)\s*days?\s*to\s*prepare',
                r'(\d+)\s*weeks?\s*to\s*prepare'
            ]
            
            for pattern in duration_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    days = int(match)
                    if 'week' in pattern:
                        days *= 7
                    duration_mentions.append(days)
        
        is_short_notice = short_notice_score >= 2
        avg_duration = int(np.mean(duration_mentions)) if duration_mentions else 0
        
        return is_short_notice, avg_duration

    def _extract_injury_risk(self, articles: List[Dict]) -> int:
        """Extract injury risk score from articles"""
        injury_score = 0
        
        for article in articles:
            text = article['full_text']
            
            # Count injury-related mentions
            for keyword in self.injury_keywords:
                count = text.count(keyword)
                injury_score += count
                
            # Reduce score for negative indicators
            for keyword in self.negative_keywords:
                if keyword in text:
                    injury_score = max(0, injury_score - 1)
        
        # Normalize score (0-10 scale)
        return min(10, max(0, injury_score))

    def _extract_camp_status(self, articles: List[Dict]) -> int:
        """Extract camp status score from articles"""
        camp_score = 0
        
        for article in articles:
            text = article['full_text']
            
            # Count camp-related mentions
            for keyword in self.camp_keywords:
                count = text.count(keyword)
                camp_score += count
                
            # Reduce score for positive indicators
            for keyword in self.negative_keywords:
                if keyword in text:
                    camp_score = max(0, camp_score - 1)
        
        # Normalize score (0-10 scale)
        return min(10, max(0, camp_score))

    def extract_fight_features(self, fighter_name: str, fight_date: str, 
                             days_before: int = 90) -> NewsFeatures:
        """
        Extract all news features for a fighter and fight
        
        Args:
            fighter_name: Name of the fighter
            fight_date: Date of the fight (YYYY-MM-DD format)
            days_before: Number of days before fight to search for news
            
        Returns:
            NewsFeatures object with extracted features
        """
        logger.info(f"Extracting features for {fighter_name} on {fight_date}")
        
        # Search for news articles
        articles = self._search_fighter_news(fighter_name, fight_date, days_before)
        
        if not articles:
            logger.warning(f"No articles found for {fighter_name}")
            return NewsFeatures()
        
        # Extract features
        short_notice, duration = self._extract_short_notice_features(articles)
        injury_risk = self._extract_injury_risk(articles)
        camp_status = self._extract_camp_status(articles)
        
        # Calculate confidence score based on number of articles
        confidence = min(1.0, len(articles) / 10.0)
        
        return NewsFeatures(
            short_notice=short_notice,
            short_notice_duration=duration,
            injury_risk=injury_risk,
            camp_status=camp_status,
            confidence_score=confidence
        )

    def process_fight_dataset(self, df: pd.DataFrame, 
                            fighter_col: str = 'FIGHTER',
                            date_col: str = 'DATE',
                            output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process entire fight dataset to extract news features
        
        Args:
            df: DataFrame with fight data
            fighter_col: Column name containing fighter names
            date_col: Column name containing fight dates
            output_file: Optional file to save results
            
        Returns:
            DataFrame with added news features
        """
        logger.info(f"Processing {len(df)} fights for news features")
        
        # Create new columns for news features
        df['short_notice'] = False
        df['short_notice_duration'] = 0
        df['injury_risk'] = 0
        df['camp_status'] = 0
        df['news_confidence'] = 0.0
        
        # Process each unique fight
        unique_fights = df[[fighter_col, date_col]].drop_duplicates()
        
        for idx, row in unique_fights.iterrows():
            try:
                features = self.extract_fight_features(
                    row[fighter_col], 
                    row[date_col].strftime('%Y-%m-%d')
                )
                
                # Update all rows for this fighter and date
                mask = (df[fighter_col] == row[fighter_col]) & (df[date_col] == row[date_col])
                df.loc[mask, 'short_notice'] = features.short_notice
                df.loc[mask, 'short_notice_duration'] = features.short_notice_duration
                df.loc[mask, 'injury_risk'] = features.injury_risk
                df.loc[mask, 'camp_status'] = features.camp_status
                df.loc[mask, 'news_confidence'] = features.confidence_score
                
                logger.info(f"Processed {row[fighter_col]} - {row[date_col]}")
                
            except Exception as e:
                logger.error(f"Error processing {row[fighter_col]}: {str(e)}")
                continue
        
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return df

    def batch_process_fights(self, fights: List[Tuple[str, str]], 
                           max_workers: int = 3) -> Dict[Tuple[str, str], NewsFeatures]:
        """
        Process multiple fights in parallel
        
        Args:
            fights: List of (fighter_name, fight_date) tuples
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping (fighter_name, fight_date) to NewsFeatures
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_fight = {
                executor.submit(self.extract_fight_features, fighter, date): (fighter, date)
                for fighter, date in fights
            }
            
            # Collect results
            for future in as_completed(future_to_fight):
                fight_key = future_to_fight[future]
                try:
                    features = future.result()
                    results[fight_key] = features
                except Exception as e:
                    logger.error(f"Error processing {fight_key}: {str(e)}")
                    results[fight_key] = NewsFeatures()
        
        return results



def scrape_fight_news(fighter_name: str, fight_date: str, days_before: int = 90) -> NewsFeatures:
    """
    Convenience function to scrape news for a single fight
    
    Args:
        fighter_name: Name of the fighter
        fight_date: Date of the fight (YYYY-MM-DD format)
        days_before: Number of days before fight to search
        
    Returns:
        NewsFeatures object with extracted features
    """
    scraper = UFCNewsScraper()
    return scraper.extract_fight_features(fighter_name, fight_date, days_before)


def scrape_multiple_fights(fights: List[Tuple[str, str]], max_workers: int = 3) -> Dict[Tuple[str, str], NewsFeatures]:
    """
    Convenience function to scrape news for multiple fights
    
    Args:
        fights: List of (fighter_name, fight_date) tuples
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary mapping (fighter_name, fight_date) to NewsFeatures
    """
    scraper = UFCNewsScraper()
    return scraper.batch_process_fights(fights, max_workers)


def add_news_features_to_dataset(df: pd.DataFrame, 
                                fighter_col: str = 'FIGHTER',
                                date_col: str = 'DATE',
                                output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to add news features to a dataset
    
    Args:
        df: DataFrame with fight data
        fighter_col: Column name containing fighter names
        date_col: Column name containing fight dates
        output_file: Optional file to save results
        
    Returns:
        DataFrame with added news features
    """
    scraper = UFCNewsScraper()
    return scraper.process_fight_dataset(df, fighter_col, date_col, output_file)


def main():
    """Example usage of the UFC News Scraper"""
    scraper = UFCNewsScraper()
    
    # Example: Extract features for a specific fighter
    fighter_name = "Jon Jones"
    fight_date = "2023-03-04"
    
    features = scraper.extract_fight_features(fighter_name, fight_date)
    print(f"Features for {fighter_name}:")
    print(f"  Short Notice: {features.short_notice}")
    print(f"  Duration: {features.short_notice_duration} days")
    print(f"  Injury Risk: {features.injury_risk}")
    print(f"  Camp Status: {features.camp_status}")
    print(f"  Confidence: {features.confidence_score}")


if __name__ == "__main__":
    main()
