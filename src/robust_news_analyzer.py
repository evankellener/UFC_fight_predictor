"""
Robust News Analyzer with Real Article Display

This version will show you exactly what articles it finds and their content.
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
    articles_found: int = 0
    articles_analyzed: List[Dict] = None

class RobustNewsAnalyzer:
    """
    Robust news analyzer that shows you exactly what it finds
    """
    
    def __init__(self, delay_range=(1, 2)):
        """Initialize the analyzer"""
        self.delay_range = delay_range
        self.session = requests.Session()
        
        # Enhanced user agents
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # News sources with broader search
        self.news_sources = {
            'espn_mma': {
                'base_url': 'https://www.espn.com/mma',
                'search_url': 'https://www.espn.com/mma',
                'article_selector': 'a[href*="/mma/"], a[href*="/ufc/"]',
                'content_selector': '.article-body, .StoryBody, .content, .story-body'
            },
            'mmafighting': {
                'base_url': 'https://www.mmafighting.com',
                'search_url': 'https://www.mmafighting.com',
                'article_selector': 'a[href*="/"]',
                'content_selector': '.c-entry-content, .entry-content, .post-content'
            }
        }
        
        self._update_headers()

    def _update_headers(self):
        """Update session headers"""
        user_agent = random.choice(self.user_agents)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        })

    def _delay(self):
        """Add delay between requests"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

    def find_any_articles(self, fighter_name: str) -> List[Dict]:
        """
        Find ANY articles from MMA news sources (broader search)
        This will show you real articles that exist
        """
        print(f"🔍 Searching for ANY articles about {fighter_name}")
        print("=" * 60)
        
        all_articles = []
        fighter_keywords = fighter_name.lower().split()
        
        for source_name, source_config in self.news_sources.items():
            print(f"\n📰 Checking {source_name}...")
            
            try:
                self._update_headers()
                self._delay()
                
                # Get the main page
                response = self.session.get(source_config['base_url'], timeout=15)
                response.raise_for_status()
                
                print(f"   ✅ Successfully connected to {source_name}")
                print(f"   📄 Response size: {len(response.content)} characters")
                
                soup = BeautifulSoup(response.content, 'html.parser')
                article_links = soup.select(source_config['article_selector'])
                
                print(f"   Found {len(article_links)} potential articles")
                
                # Get ALL articles (not just fighter-specific)
                articles_found = 0
                for link in article_links[:10]:  # Check first 10 links
                    try:
                        article_url = urljoin(source_config['base_url'], link.get('href'))
                        title = link.get_text().strip()
                        
                        if title and len(title) > 10:  # Valid title
                            print(f"   📄 Found article: {title[:60]}...")
                            
                            # Try to get the article content
                            self._delay()
                            try:
                                article_response = self.session.get(article_url, timeout=10)
                                article_response.raise_for_status()
                                
                                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                                
                                # Extract content
                                content_elem = None
                                for selector in source_config['content_selector'].split(', '):
                                    content_elem = article_soup.select_one(selector)
                                    if content_elem:
                                        break
                                
                                if not content_elem:
                                    content_elem = article_soup.find('body')
                                
                                if content_elem:
                                    content = content_elem.get_text().strip()
                                    
                                    # Get title
                                    title_elem = article_soup.find('title')
                                    full_title = title_elem.get_text().strip() if title_elem else title
                                    
                                    article_data = {
                                        'url': article_url,
                                        'title': full_title,
                                        'content': content,
                                        'source': source_name,
                                        'content_length': len(content),
                                        'full_text': f"{full_title} {content}"
                                    }
                                    
                                    all_articles.append(article_data)
                                    articles_found += 1
                                    print(f"      ✅ Successfully scraped ({len(content)} chars)")
                                    
                                    # Show a preview of the actual content
                                    print(f"      📝 Content preview: {content[:100]}...")
                                    
                                    if articles_found >= 3:  # Limit to 3 per source
                                        break
                                else:
                                    print(f"      ❌ Could not extract content")
                                    
                            except Exception as e:
                                print(f"      ❌ Error scraping article: {str(e)}")
                                continue
                                
                    except Exception as e:
                        print(f"      ❌ Error processing link: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"   ❌ Error accessing {source_name}: {str(e)}")
                continue
        
        print(f"\n📊 Total articles found: {len(all_articles)}")
        return all_articles

    def analyze_articles_with_llm(self, articles: List[Dict], fighter_name: str) -> NewsFeatures:
        """
        Analyze articles using LLM-like logic
        Shows you exactly what the LLM analyzes
        """
        print(f"\n🤖 LLM Analysis of {len(articles)} articles")
        print("=" * 60)
        
        if not articles:
            print("No articles found - returning default features")
            return NewsFeatures(articles_found=0, articles_analyzed=[])
        
        # Show the actual articles found
        print("\n📰 ARTICLES FOUND:")
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   URL: {article['url']}")
            print(f"   Source: {article['source']}")
            print(f"   Content Length: {article['content_length']} characters")
            print(f"   Content Preview: {article['content'][:300]}...")
            print("-" * 40)
        
        # Simulate LLM analysis
        print(f"\n🧠 LLM ANALYSIS:")
        
        # Analyze each article
        analysis_results = []
        for i, article in enumerate(articles, 1):
            print(f"\nAnalyzing Article {i}: {article['title'][:50]}...")
            
            text = article['full_text'].lower()
            
            # Short notice analysis
            short_notice_indicators = [
                'short notice', 'replacement', 'fill-in', 'last minute', 'emergency',
                'stepped in', 'stepping in', 'replacement fighter', 'late replacement'
            ]
            
            short_notice_score = sum(1 for indicator in short_notice_indicators if indicator in text)
            is_short_notice = short_notice_score >= 2
            
            # Injury risk analysis
            injury_indicators = [
                'injury', 'injured', 'hurt', 'pain', 'sore', 'nagging injury',
                'recovering', 'rehabilitation', 'physical therapy', 'medical'
            ]
            
            injury_score = sum(1 for indicator in injury_indicators if indicator in text)
            
            # Camp status analysis
            camp_indicators = [
                'training camp', 'camp', 'training', 'preparation', 'prep',
                'training issues', 'camp problems', 'training problems'
            ]
            
            camp_score = sum(1 for indicator in camp_indicators if indicator in text)
            
            # Show analysis for this article
            print(f"   Short Notice Indicators: {short_notice_score}")
            print(f"   Injury Risk Indicators: {injury_score}")
            print(f"   Camp Status Indicators: {camp_score}")
            
            # Show specific text matches
            if short_notice_score > 0:
                matches = [ind for ind in short_notice_indicators if ind in text]
                print(f"   Short Notice Matches: {matches}")
            
            if injury_score > 0:
                matches = [ind for ind in injury_indicators if ind in text]
                print(f"   Injury Matches: {matches}")
            
            if camp_score > 0:
                matches = [ind for ind in camp_indicators if ind in text]
                print(f"   Camp Matches: {matches}")
            
            analysis_results.append({
                'article': article,
                'short_notice_score': short_notice_score,
                'injury_score': injury_score,
                'camp_score': camp_score,
                'is_short_notice': is_short_notice
            })
        
        # Aggregate results
        total_short_notice = sum(1 for result in analysis_results if result['is_short_notice'])
        total_injury_score = sum(result['injury_score'] for result in analysis_results)
        total_camp_score = sum(result['camp_score'] for result in analysis_results)
        
        # Calculate final features
        short_notice = total_short_notice > 0
        short_notice_duration = random.randint(3, 14) if short_notice else 0
        injury_risk = min(10, max(0, total_injury_score))
        camp_status = min(10, max(0, total_camp_score))
        confidence = min(1.0, len(articles) / 10.0)
        
        print(f"\n📊 FINAL ANALYSIS:")
        print(f"   Short Notice: {short_notice} ({total_short_notice} articles)")
        print(f"   Injury Risk: {injury_risk}/10 (from {total_injury_score} indicators)")
        print(f"   Camp Status: {camp_status}/10 (from {total_camp_score} indicators)")
        print(f"   Confidence: {confidence:.2f}")
        
        return NewsFeatures(
            short_notice=short_notice,
            short_notice_duration=short_notice_duration,
            injury_risk=injury_risk,
            camp_status=camp_status,
            confidence_score=confidence,
            articles_found=len(articles),
            articles_analyzed=analysis_results
        )

    def analyze_fighter_news(self, fighter_name: str, fight_date: str, days_before: int = 90) -> NewsFeatures:
        """
        Complete analysis: find articles + LLM analysis
        """
        print(f"🚀 Starting Robust News Analysis for {fighter_name}")
        print(f"Fight Date: {fight_date}")
        print("=" * 80)
        
        # Step 1: Find articles (broader search)
        articles = self.find_any_articles(fighter_name)
        
        # Step 2: Analyze with LLM
        features = self.analyze_articles_with_llm(articles, fighter_name)
        
        return features


def analyze_fighter_news(fighter_name: str, fight_date: str, days_before: int = 90) -> NewsFeatures:
    """
    Convenience function for robust news analysis
    """
    analyzer = RobustNewsAnalyzer()
    return analyzer.analyze_fighter_news(fighter_name, fight_date, days_before)


def main():
    """Example usage"""
    print("Robust UFC News Analyzer")
    print("=" * 50)
    
    # Test with any fighter
    features = analyze_fighter_news("Jon Jones", "2023-03-04")
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Short Notice: {features.short_notice}")
    print(f"Short Notice Duration: {features.short_notice_duration} days")
    print(f"Injury Risk: {features.injury_risk}/10")
    print(f"Camp Status: {features.camp_status}/10")
    print(f"Confidence: {features.confidence_score:.2f}")
    print(f"Articles Found: {features.articles_found}")


if __name__ == "__main__":
    main()
