"""
UFC Fight Article Scraper
Scrapes MMA news articles about specific UFC fights and provides content summarization
"""

import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import re
from urllib.parse import urljoin, urlparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Article:
    """Represents a news article with its content"""
    
    def __init__(self, title: str, url: str, published_date: Optional[datetime], 
                 source: str, summary: str = ""):
        self.title = title
        self.url = url
        self.published_date = published_date
        self.source = source
        self.summary = summary
        self.full_content = ""
        self.fighters_mentioned = []
        
    def __repr__(self):
        return f"Article(title='{self.title[:50]}...', source={self.source})"


class UFCFightArticleScraper:
    """
    Scraper for MMA news articles about specific UFC fights
    
    Usage:
        scraper = UFCFightArticleScraper()
        articles = scraper.scrape_fight_articles("Jon Jones", "Stipe Miocic", "2024-11-16")
        
        for article in articles:
            print(article.title)
            print(article.full_content)
            print(article.get_summary(max_sentences=3))
    """
    
    def __init__(self, delay_between_requests: float = 2.0):
        """
        Initialize the scraper
        
        Args:
            delay_between_requests: Seconds to wait between requests (be respectful)
        """
        self.delay = delay_between_requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # RSS feeds for major MMA news sources
        self.rss_feeds = {
            'MMA Fighting': 'https://www.mmafighting.com/rss/index.xml',
            'Bloody Elbow': 'https://www.bloodyelbow.com/rss/index.xml',
            'MMA Mania': 'https://www.mmamania.com/rss/index.xml',
            'Cageside Seats': 'https://www.cagesideseats.com/rss/index.xml',
            'Low Kick MMA': 'https://www.lowkickmma.com/feed/',
        }
        
        # Search URLs for news sites (fallback if RSS doesn't work)
        self.search_urls = {
            'MMA Fighting': 'https://www.mmafighting.com/search?q=',
            'Bloody Elbow': 'https://www.bloodyelbow.com/search?q=',
        }
        
    def scrape_fight_articles(self, fighter1_name: str, fighter2_name: str, 
                              fight_date: str, days_before: int = 60, 
                              max_articles: int = 10) -> List[Article]:
        """
        Scrape articles about a specific fight
        
        Args:
            fighter1_name: First fighter's name (e.g., "Jon Jones")
            fighter2_name: Second fighter's name (e.g., "Stipe Miocic")
            fight_date: Fight date in YYYY-MM-DD format
            days_before: How many days before the fight to search for articles (default: 60)
            max_articles: Maximum number of articles to return (default: 10, prioritizes most recent)
            
        Returns:
            List of Article objects containing article information (sorted by date, most recent first)
        """
        logger.info(f"🔍 Searching for articles about {fighter1_name} vs {fighter2_name}")
        logger.info(f"📅 Fight date: {fight_date}, searching {days_before} days before")
        
        # Parse fight date
        try:
            event_date = datetime.strptime(fight_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {fight_date}. Use YYYY-MM-DD")
            return []
        
        cutoff_date = event_date - timedelta(days=days_before)
        
        all_articles = []
        
        # Fetch from RSS feeds
        for source, feed_url in self.rss_feeds.items():
            try:
                logger.info(f"📰 Fetching from {source}...")
                articles = self._fetch_from_rss(
                    feed_url, source, fighter1_name, fighter2_name, 
                    cutoff_date, event_date
                )
                all_articles.extend(articles)
                time.sleep(self.delay)
            except Exception as e:
                logger.warning(f"⚠️  Error fetching from {source}: {e}")
                continue
        
        # Remove duplicates based on URL
        unique_articles = self._remove_duplicates(all_articles)
        
        # Sort by date (most recent first) and limit to max_articles
        sorted_articles = sorted(
            unique_articles, 
            key=lambda x: x.published_date if x.published_date else datetime.min,
            reverse=True
        )
        
        # Apply article limit
        articles_to_fetch = sorted_articles[:max_articles] if max_articles else sorted_articles
        
        logger.info(f"📄 Fetching full content for {len(articles_to_fetch)} articles (limited to {max_articles})...")
        
        # Fetch full content for each article
        for article in articles_to_fetch:
            try:
                self._fetch_article_content(article)
                time.sleep(self.delay)
            except Exception as e:
                logger.warning(f"⚠️  Error fetching content for {article.url}: {e}")
                continue
        
        logger.info(f"✅ Found {len(articles_to_fetch)} articles (sorted by date, most recent first)")
        return articles_to_fetch
    
    def _fetch_from_rss(self, feed_url: str, source: str, fighter1: str, 
                        fighter2: str, cutoff_date: datetime, 
                        event_date: datetime) -> List[Article]:
        """Fetch articles from an RSS feed"""
        articles = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                # Parse publication date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])
                    except:
                        pass
                
                # Check if article is within date range
                if pub_date and (pub_date < cutoff_date or pub_date > event_date):
                    continue
                
                title = entry.get('title', '')
                link = entry.get('link', '')
                summary = entry.get('summary', '')
                
                # Check if article mentions both fighters
                text = f"{title} {summary}".lower()
                fighter1_lower = fighter1.lower()
                fighter2_lower = fighter2.lower()
                
                # Also check for last names only
                fighter1_last = fighter1.split()[-1].lower()
                fighter2_last = fighter2.split()[-1].lower()
                
                if ((fighter1_lower in text or fighter1_last in text) and 
                    (fighter2_lower in text or fighter2_last in text)):
                    
                    article = Article(title, link, pub_date, source, summary)
                    article.fighters_mentioned = [fighter1, fighter2]
                    articles.append(article)
                    logger.debug(f"  ✓ Found: {title[:60]}...")
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        
        return articles
    
    def _fetch_article_content(self, article: Article) -> None:
        """Fetch the full content of an article from its URL"""
        try:
            response = self.session.get(article.url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 
                                         'footer', 'aside', 'iframe']):
                element.decompose()
            
            # Try different content selectors based on common MMA news site structures
            content = self._extract_article_body(soup)
            
            if content:
                article.full_content = content
                logger.debug(f"  ✓ Fetched content ({len(content)} chars): {article.title[:50]}")
            else:
                logger.warning(f"  ⚠️  No content extracted from: {article.url}")
                
        except Exception as e:
            logger.error(f"Error fetching article content: {e}")
    
    def _extract_article_body(self, soup: BeautifulSoup) -> str:
        """Extract the main article body from HTML"""
        # Try common article body selectors
        selectors = [
            'article',
            '.article-content',
            '.entry-content', 
            '.post-content',
            '.article-body',
            '[class*="article-content"]',
            '[class*="entry-content"]',
            'div[itemprop="articleBody"]',
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                # Get all paragraphs from the article
                paragraphs = elements[0].find_all('p')
                text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                if len(text) > 200:  # Minimum reasonable article length
                    return text
        
        # Fallback: get all paragraphs from main content
        paragraphs = soup.find_all('p')
        text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        return text if len(text) > 200 else ""
    
    def _remove_duplicates(self, articles: List[Article]) -> List[Article]:
        """Remove duplicate articles based on URL"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles
    
    @staticmethod
    def summarize_article(article: Article, max_sentences: int = 5) -> str:
        """
        Create a simple extractive summary of the article
        
        Args:
            article: Article object with full_content
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Summary string
        """
        if not article.full_content:
            return article.summary
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', article.full_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return article.full_content
        
        # Simple extractive summary: take first few sentences and any with fighter names
        summary_sentences = sentences[:2]  # Always include first 2 sentences
        
        # Find sentences mentioning both fighters
        for sentence in sentences[2:]:
            if len(summary_sentences) >= max_sentences:
                break
            
            sentence_lower = sentence.lower()
            if article.fighters_mentioned:
                fighter_mentions = sum(
                    1 for fighter in article.fighters_mentioned 
                    if fighter.lower() in sentence_lower
                )
                if fighter_mentions > 0:
                    summary_sentences.append(sentence)
        
        # Fill remaining slots with next sentences if needed
        remaining = max_sentences - len(summary_sentences)
        if remaining > 0:
            idx = 2
            while remaining > 0 and idx < len(sentences):
                if sentences[idx] not in summary_sentences:
                    summary_sentences.append(sentences[idx])
                    remaining -= 1
                idx += 1
        
        return '. '.join(summary_sentences) + '.'
    
    @staticmethod
    def get_article_stats(articles: List[Article]) -> Dict:
        """Get statistics about the scraped articles"""
        if not articles:
            return {
                'total_articles': 0,
                'sources': [],
                'date_range': None,
                'avg_content_length': 0
            }
        
        sources = {}
        for article in articles:
            sources[article.source] = sources.get(article.source, 0) + 1
        
        dates = [a.published_date for a in articles if a.published_date]
        date_range = None
        if dates:
            date_range = {
                'earliest': min(dates),
                'latest': max(dates)
            }
        
        content_lengths = [len(a.full_content) for a in articles if a.full_content]
        avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        
        return {
            'total_articles': len(articles),
            'sources': sources,
            'date_range': date_range,
            'avg_content_length': int(avg_length),
            'articles_with_content': len(content_lengths)
        }


def format_article_report(article: Article, include_full_content: bool = False) -> str:
    """Format an article into a readable report"""
    report = []
    report.append("=" * 80)
    report.append(f"TITLE: {article.title}")
    report.append(f"SOURCE: {article.source}")
    report.append(f"URL: {article.url}")
    
    if article.published_date:
        report.append(f"PUBLISHED: {article.published_date.strftime('%Y-%m-%d %H:%M')}")
    
    if article.fighters_mentioned:
        report.append(f"FIGHTERS: {', '.join(article.fighters_mentioned)}")
    
    report.append("=" * 80)
    
    if include_full_content and article.full_content:
        report.append("\nFULL CONTENT:")
        report.append("-" * 80)
        report.append(article.full_content)
    else:
        report.append("\nSUMMARY:")
        report.append("-" * 80)
        summary = UFCFightArticleScraper.summarize_article(article, max_sentences=5)
        report.append(summary)
    
    report.append("\n")
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    scraper = UFCFightArticleScraper(delay_between_requests=2.0)
    
    # Example: Jon Jones vs Stipe Miocic
    fighter1 = "Jon Jones"
    fighter2 = "Stipe Miocic"
    fight_date = "2024-11-16"
    
    print(f"Searching for articles about {fighter1} vs {fighter2}")
    print(f"Fight date: {fight_date}\n")
    
    articles = scraper.scrape_fight_articles(fighter1, fighter2, fight_date, days_before=30)
    
    if articles:
        # Print statistics
        stats = scraper.get_article_stats(articles)
        print(f"\n📊 STATISTICS:")
        print(f"  Total articles: {stats['total_articles']}")
        print(f"  Articles with content: {stats['articles_with_content']}")
        print(f"  Average content length: {stats['avg_content_length']} characters")
        print(f"\n  Sources:")
        for source, count in stats['sources'].items():
            print(f"    {source}: {count} articles")
        
        if stats['date_range']:
            print(f"\n  Date range: {stats['date_range']['earliest'].strftime('%Y-%m-%d')} to {stats['date_range']['latest'].strftime('%Y-%m-%d')}")
        
        # Print article summaries
        print(f"\n\n📰 ARTICLES:\n")
        for i, article in enumerate(articles[:5], 1):  # Show first 5
            print(format_article_report(article, include_full_content=False))
    else:
        print("❌ No articles found")

