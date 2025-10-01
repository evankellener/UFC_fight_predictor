#!/usr/bin/env python3
"""
Historical Fight Article Scraper

For fights that already happened - uses web scraping instead of RSS feeds.
RSS feeds only keep articles for 2-4 weeks, this searches article archives.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from datetime import datetime
from typing import List
import time

from article_scraper import Article, UFCFightArticleScraper


class HistoricalFightScraper:
    """
    Scraper for articles about fights that already happened.
    
    Uses Google/Bing search instead of RSS feeds to find archived articles.
    """
    
    def __init__(self, delay_between_requests: float = 2.0):
        self.delay = delay_between_requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Known MMA news domains to prioritize
        self.mma_domains = [
            'mmafighting.com',
            'bloodyelbow.com',
            'mmamania.com',
            'espn.com/mma',
            'sherdog.com',
            'lowkickmma.com',
            'mmanews.com',
            'bjpenn.com'
        ]
    
    def search_fight_articles(self, fighter1: str, fighter2: str, 
                             fight_date: str, max_articles: int = 10) -> List[Article]:
        """
        Search for historical fight articles using web search
        
        Args:
            fighter1: First fighter name
            fighter2: Second fighter name
            fight_date: Fight date (YYYY-MM-DD)
            max_articles: Maximum articles to return
            
        Returns:
            List of Article objects
        """
        print(f"🔍 Searching historical articles for: {fighter1} vs {fighter2}")
        print(f"📅 Fight date: {fight_date}")
        print(f"🌐 Using web search (not RSS feeds)")
        print()
        
        articles = []
        
        # Build search queries
        queries = [
            f'"{fighter1}" vs "{fighter2}" UFC',
            f'{fighter1} {fighter2} fight {fight_date}',
            f'{fighter1} {fighter2} UFC result',
        ]
        
        for query in queries:
            print(f"Searching: {query}")
            results = self._search_google(query, max_results=5)
            
            for result in results:
                if len(articles) >= max_articles:
                    break
                
                # Check if it's from an MMA site
                if any(domain in result['link'] for domain in self.mma_domains):
                    # Try to create article
                    article = Article(
                        title=result['title'],
                        url=result['link'],
                        published_date=None,  # Will try to parse from page
                        source=self._get_source_from_url(result['link']),
                        summary=result.get('snippet', '')
                    )
                    article.fighters_mentioned = [fighter1, fighter2]
                    
                    # Fetch full content
                    print(f"  📄 Fetching: {article.title[:60]}...")
                    self._fetch_content(article)
                    
                    if article.full_content:
                        articles.append(article)
                    
                    time.sleep(self.delay)
            
            if len(articles) >= max_articles:
                break
            
            time.sleep(self.delay)
        
        print(f"\n✅ Found {len(articles)} historical articles")
        return articles[:max_articles]
    
    def _search_google(self, query: str, max_results: int = 5) -> List[dict]:
        """
        Search Google for articles (simplified - uses DuckDuckGo HTML search)
        
        Note: This is a basic implementation. For production, consider using:
        - Google Custom Search API
        - Bing Search API  
        - SerpAPI
        """
        results = []
        
        try:
            # Use DuckDuckGo HTML search (no API key needed)
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse search results
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    # Extract actual URL from DuckDuckGo redirect
                    href = title_elem.get('href', '')
                    actual_url = self._extract_url_from_redirect(href)
                    
                    if actual_url:
                        results.append({
                            'title': title_elem.get_text(),
                            'link': actual_url,
                            'snippet': snippet_elem.get_text() if snippet_elem else ''
                        })
                    
        except Exception as e:
            print(f"  ⚠️  Search error: {e}")
        
        return results
    
    def _extract_url_from_redirect(self, redirect_url: str) -> str:
        """Extract actual URL from DuckDuckGo redirect"""
        if not redirect_url:
            return ''
        
        # DuckDuckGo uses //duckduckgo.com/l/?uddg=ENCODED_URL
        if 'uddg=' in redirect_url:
            try:
                from urllib.parse import unquote, parse_qs, urlparse
                
                # Handle relative URLs
                if redirect_url.startswith('//'):
                    redirect_url = 'https:' + redirect_url
                
                # Parse the redirect URL
                parsed = urlparse(redirect_url)
                params = parse_qs(parsed.query)
                
                if 'uddg' in params:
                    actual_url = unquote(params['uddg'][0])
                    return actual_url
            except:
                pass
        
        return redirect_url
    
    def _get_source_from_url(self, url: str) -> str:
        """Extract source name from URL"""
        if 'mmafighting.com' in url:
            return 'MMA Fighting'
        elif 'bloodyelbow.com' in url:
            return 'Bloody Elbow'
        elif 'mmamania.com' in url:
            return 'MMA Mania'
        elif 'espn.com' in url:
            return 'ESPN MMA'
        elif 'sherdog.com' in url:
            return 'Sherdog'
        elif 'lowkickmma.com' in url:
            return 'Low Kick MMA'
        else:
            return 'Unknown Source'
    
    def _fetch_content(self, article: Article):
        """Fetch article content - reuse logic from main scraper"""
        scraper = UFCFightArticleScraper()
        scraper._fetch_article_content(article)


def scrape_historical_fight(fighter1: str, fighter2: str, fight_date: str, 
                            max_articles: int = 10):
    """
    Main function to scrape historical fight articles
    
    Args:
        fighter1: First fighter name
        fighter2: Second fighter name  
        fight_date: Fight date (YYYY-MM-DD)
        max_articles: Max articles to return
        
    Returns:
        List of Article objects with full content
    """
    scraper = HistoricalFightScraper(delay_between_requests=2.0)
    articles = scraper.search_fight_articles(fighter1, fighter2, fight_date, max_articles)
    
    if articles:
        print(f"\n📊 Found {len(articles)} articles:")
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article.title}")
            print(f"   Source: {article.source}")
            print(f"   URL: {article.url}")
            if article.full_content:
                print(f"   Content: {len(article.full_content)} characters")
                
                # Generate summary
                main_scraper = UFCFightArticleScraper()
                summary = main_scraper.summarize_article(article, max_sentences=3)
                print(f"   Summary: {summary[:150]}...")
    
    return articles


if __name__ == "__main__":
    # Test with Reyes vs Ulberg
    print("=" * 80)
    print("🔍 HISTORICAL FIGHT SCRAPER TEST")
    print("=" * 80)
    print("\nThis scraper finds articles about fights that already happened.")
    print("It uses web search instead of RSS feeds.\n")
    
    articles = scrape_historical_fight(
        fighter1="Dominic Reyes",
        fighter2="Carlos Ulberg",
        fight_date="2025-09-27",
        max_articles=10
    )
    
    if not articles:
        print("\n💡 Try:")
        print("  • Check if fighter names are spelled correctly")
        print("  • Verify the fight actually happened")
        print("  • The fight might not have media coverage yet")

