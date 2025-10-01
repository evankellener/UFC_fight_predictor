#!/usr/bin/env python3
"""
Smart UFC Article Scraper

Automatically chooses the best scraping method based on fight timing:
- RSS feeds for upcoming/current fights (fast, reliable)
- Web search for historical fights (finds archived articles)
"""

from datetime import datetime, timedelta
from article_scraper import UFCFightArticleScraper
from historical_scraper import HistoricalFightScraper


def scrape_fight_articles(fighter1_name: str, fighter2_name: str, fight_date: str,
                          days_before: int = 60, max_articles: int = 10):
    """
    Smart scraper that automatically chooses RSS or historical search
    
    Args:
        fighter1_name: First fighter's name
        fighter2_name: Second fighter's name  
        fight_date: Fight date (YYYY-MM-DD or MM/DD/YY)
        days_before: Days before fight to search (default 60)
        max_articles: Max articles to return (default 10)
        
    Returns:
        List of Article objects
    """
    
    # Convert date format if needed
    if '/' in fight_date:
        parts = fight_date.split('/')
        month, day, year = parts[0], parts[1], parts[2]
        if len(year) == 2:
            year = '20' + year
        fight_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Parse fight date
    fight_datetime = datetime.strptime(fight_date, "%Y-%m-%d")
    days_from_now = (fight_datetime - datetime.now()).days
    
    print("=" * 80)
    print("🥊 SMART UFC ARTICLE SCRAPER")
    print("=" * 80)
    print(f"\nFight: {fighter1_name} vs {fighter2_name}")
    print(f"Date: {fight_date}")
    
    # Decide which scraper to use
    # Use RSS for upcoming fights or fights within last 3 days (should still be in feeds)
    if days_from_now >= -3:  # Fight is upcoming or very recent (within 3 days)
        if days_from_now >= 0:
            print(f"\n📡 Using RSS FEED scraper (fight is {days_from_now} days away)")
        else:
            print(f"\n📡 Using RSS FEED scraper (fight was {abs(days_from_now)} days ago - should be in RSS)")
        print("   → Best for upcoming/recent fights")
        print(f"   → Searching {days_before} days before fight")
        print(f"   → Limited to {max_articles} most recent articles\n")
        
        scraper = UFCFightArticleScraper(delay_between_requests=2.0)
        articles = scraper.scrape_fight_articles(
            fighter1_name=fighter1_name,
            fighter2_name=fighter2_name,
            fight_date=fight_date,
            days_before=days_before,
            max_articles=max_articles
        )
        
    else:  # Fight is in the past (>3 days ago)
        print(f"\n🌐 Using HISTORICAL scraper (fight was {abs(days_from_now)} days ago)")
        print("   → Best for past fights")
        print("   → Uses web search to find archived articles")
        print(f"   → Limited to {max_articles} articles\n")
        
        scraper = HistoricalFightScraper(delay_between_requests=2.0)
        articles = scraper.search_fight_articles(
            fighter1=fighter1_name,
            fighter2=fighter2_name,
            fight_date=fight_date,
            max_articles=max_articles
        )
    
    # Display results
    print("\n" + "=" * 80)
    
    if articles:
        print(f"✅ SUCCESS! Found {len(articles)} articles\n")
        
        print("📰 ARTICLES:\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article.title}")
            print(f"   Source: {article.source}")
            print(f"   URL: {article.url}")
            
            if article.full_content:
                print(f"   ✅ Full content: {len(article.full_content)} chars")
                
                # Generate summary
                summarizer = UFCFightArticleScraper()
                summary = summarizer.summarize_article(article, max_sentences=3)
                print(f"   Summary: {summary[:200]}...")
            else:
                print(f"   Summary: {article.summary[:200]}...")
            
            print()
    else:
        print("❌ No articles found")
        
        if days_from_now > 30:
            print("\n💡 Tips:")
            print("   • Fight might not be officially announced yet")
            print("   • Try again closer to the fight date")
        elif days_from_now < -30:
            print("\n💡 Tips:")
            print("   • Historical articles may have limited availability")
            print("   • Some sites block automated access")
            print("   • Try searching manually on MMA news sites")
    
    print("=" * 80)
    return articles


if __name__ == "__main__":
    import sys
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    SMART UFC ARTICLE SCRAPER                               ║
╚════════════════════════════════════════════════════════════════════════════╝

This tool automatically chooses the best scraping method:
  • RSS feeds for upcoming/current fights (fast, reliable)
  • Web search for historical fights (finds archived articles)

""")
    
    # Interactive mode
    fighter1 = input("Fighter 1 name: ").strip()
    if not fighter1:
        print("Fighter 1 required!")
        sys.exit(1)
    
    fighter2 = input("Fighter 2 name: ").strip()
    if not fighter2:
        print("Fighter 2 required!")
        sys.exit(1)
    
    fight_date = input("Fight date (MM/DD/YY or YYYY-MM-DD): ").strip()
    if not fight_date:
        print("Fight date required!")
        sys.exit(1)
    
    max_articles = input("Max articles (default 10): ").strip()
    max_articles = int(max_articles) if max_articles.isdigit() else 10
    
    # Scrape
    articles = scrape_fight_articles(fighter1, fighter2, fight_date, 
                                     days_before=60, max_articles=max_articles)
    
    # Save option
    if articles:
        save = input("\n💾 Save articles to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"{fighter1.replace(' ', '_')}_vs_{fighter2.replace(' ', '_')}.txt"
            with open(filename, 'w') as f:
                for i, article in enumerate(articles, 1):
                    f.write(f"\nARTICLE {i}\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Title: {article.title}\n")
                    f.write(f"Source: {article.source}\n")
                    f.write(f"URL: {article.url}\n\n")
                    f.write(f"Content:\n{article.full_content}\n\n")
            
            print(f"✅ Saved to: {filename}")

