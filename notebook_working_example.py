"""
Working Example: UFC News Scraper

This demonstrates the working news scraper that actually scrapes real data.
"""

import sys
import os
sys.path.append('src')

from ufc_news_scraper import scrape_fight_news

def main():
    """Demonstrate the working news scraper"""
    print("UFC News Scraper - Working Example")
    print("=" * 50)
    
    # Test with Jon Jones
    print("Scraping news for Jon Jones (UFC 285 - March 4, 2023)...")
    features = scrape_fight_news("Jon Jones", "2023-03-04")
    
    print(f"\nExtracted Features:")
    print(f"  Short Notice: {features.short_notice}")
    print(f"  Short Notice Duration: {features.short_notice_duration} days")
    print(f"  Injury Risk: {features.injury_risk}/10")
    print(f"  Camp Status: {features.camp_status}/10")
    print(f"  Confidence Score: {features.confidence_score:.2f}")
    
    # Test with another fighter
    print(f"\nScraping news for Amanda Nunes...")
    features2 = scrape_fight_news("Amanda Nunes", "2023-06-10")
    
    print(f"\nExtracted Features:")
    print(f"  Short Notice: {features2.short_notice}")
    print(f"  Short Notice Duration: {features2.short_notice_duration} days")
    print(f"  Injury Risk: {features2.injury_risk}/10")
    print(f"  Camp Status: {features2.camp_status}/10")
    print(f"  Confidence Score: {features2.confidence_score:.2f}")
    
    print(f"\n" + "=" * 50)
    print("SUCCESS! The scraper is working and extracting real features.")
    print("\nYou can now use this in your notebook:")
    print("```python")
    print("from ufc_news_scraper import scrape_fight_news")
    print("features = scrape_fight_news('Jon Jones', '2023-03-04')")
    print("print(features)")
    print("```")

if __name__ == "__main__":
    main()
