"""
Test the improved UFC News Scraper

This script tests the enhanced scraper with better headers, user agent rotation,
and improved article detection.
"""

import sys
import os
sys.path.append('src')

from ufc_news_scraper import scrape_fight_news, UFCNewsScraper
import time

def test_improved_scraper():
    """Test the improved scraper functionality"""
    print("Testing Improved UFC News Scraper")
    print("=" * 50)
    
    # Test with a well-known fighter
    fighter_name = "Jon Jones"
    fight_date = "2023-03-04"
    
    print(f"Testing with {fighter_name} on {fight_date}")
    print("This may take a few minutes due to delays between requests...")
    
    try:
        # Test the convenience function
        print("\n1. Testing convenience function...")
        features = scrape_fight_news(fighter_name, fight_date)
        
        print(f"Results:")
        print(f"  Short Notice: {features.short_notice}")
        print(f"  Short Notice Duration: {features.short_notice_duration}")
        print(f"  Injury Risk: {features.injury_risk}")
        print(f"  Camp Status: {features.camp_status}")
        print(f"  Confidence: {features.confidence_score}")
        
        # Test the class directly
        print("\n2. Testing scraper class directly...")
        scraper = UFCNewsScraper(delay_range=(1, 2))  # Faster for testing
        
        # Test individual methods
        print("Testing news search...")
        articles = scraper._search_fighter_news(fighter_name, fight_date, days_before=30)
        print(f"Found {len(articles)} articles")
        
        if articles:
            print("Sample article data:")
            for i, article in enumerate(articles[:2]):  # Show first 2 articles
                print(f"  Article {i+1}:")
                print(f"    URL: {article['url']}")
                print(f"    Title: {article['title'][:100]}...")
                print(f"    Content length: {len(article['content'])}")
        
        return True
        
    except Exception as e:
        print(f"Error testing improved scraper: {str(e)}")
        return False

def test_multiple_sources():
    """Test scraping from multiple sources"""
    print("\n" + "=" * 50)
    print("Testing Multiple News Sources")
    print("=" * 50)
    
    scraper = UFCNewsScraper(delay_range=(1, 2))
    
    # Test each source individually
    sources = ['mmafighting', 'espn_mma', 'mmajunkie', 'sherdog']
    
    for source_name in sources:
        print(f"\nTesting {source_name}...")
        try:
            source_config = scraper.news_sources[source_name]
            articles = scraper._search_source(
                source_name, 
                source_config, 
                "Jon Jones", 
                "2023-03-04", 
                "2023-01-01"
            )
            print(f"  Found {len(articles)} articles from {source_name}")
            
            if articles:
                print(f"  Sample article: {articles[0]['url']}")
                
        except Exception as e:
            print(f"  Error with {source_name}: {str(e)}")

def test_different_fighters():
    """Test with different fighters"""
    print("\n" + "=" * 50)
    print("Testing Different Fighters")
    print("=" * 50)
    
    fighters = [
        ("Jon Jones", "2023-03-04"),
        ("Amanda Nunes", "2023-06-10"),
        ("Conor McGregor", "2021-01-23")
    ]
    
    for fighter_name, fight_date in fighters:
        print(f"\nTesting {fighter_name} on {fight_date}...")
        try:
            features = scrape_fight_news(fighter_name, fight_date)
            print(f"  Short Notice: {features.short_notice}")
            print(f"  Injury Risk: {features.injury_risk}")
            print(f"  Camp Status: {features.camp_status}")
            print(f"  Confidence: {features.confidence_score}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")

def main():
    """Run all tests"""
    print("Improved UFC News Scraper Test Suite")
    print("=" * 60)
    
    # Test 1: Basic functionality
    success1 = test_improved_scraper()
    
    # Test 2: Multiple sources
    test_multiple_sources()
    
    # Test 3: Different fighters
    test_different_fighters()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Basic scraper test: {'✓ PASS' if success1 else '✗ FAIL'}")
    
    print("\nNotes:")
    print("- The scraper now uses rotating user agents and better headers")
    print("- It tries multiple content selectors for better article detection")
    print("- Delays between requests help avoid rate limiting")
    print("- If no articles are found, it returns empty features (not synthetic)")
    
    print("\nIf you're still getting blocked:")
    print("1. Try running from a different IP address")
    print("2. Use a VPN to change your location")
    print("3. Consider using a proxy service")
    print("4. Try running at different times of day")

if __name__ == "__main__":
    main()
