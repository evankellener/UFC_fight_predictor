#!/usr/bin/env python3
"""
UFC Fight Article Scraper - Interactive Input

Query specific upcoming fights and get articles from the last 2 months.
"""

from article_scraper import UFCFightArticleScraper, format_article_report
from summarizer import create_comprehensive_summary
import sys


def scrape_fight():
    """Interactive scraper for specific fights"""
    
    print("=" * 80)
    print("🥊 UFC FIGHT ARTICLE SCRAPER")
    print("=" * 80)
    print("\nEnter fight details to scrape articles from the last 2 months:\n")
    
    # Get fighter 1
    fighter1 = input("Fighter 1 name (e.g., 'Dominic Reyes'): ").strip()
    if not fighter1:
        print("❌ Fighter 1 name is required!")
        return
    
    # Get fighter 2
    fighter2 = input("Fighter 2 name (e.g., 'Carlos Ulberg'): ").strip()
    if not fighter2:
        print("❌ Fighter 2 name is required!")
        return
    
    # Get fight date
    fight_date = input("Fight date (MM/DD/YY or YYYY-MM-DD, e.g., '09/27/25'): ").strip()
    if not fight_date:
        print("❌ Fight date is required!")
        return
    
    # Convert date format if needed (MM/DD/YY -> YYYY-MM-DD)
    if '/' in fight_date:
        try:
            parts = fight_date.split('/')
            month, day, year = parts[0], parts[1], parts[2]
            # Handle 2-digit year
            if len(year) == 2:
                year = '20' + year
            fight_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except:
            print(f"❌ Invalid date format: {fight_date}")
            return
    
    # Get max articles (optional)
    max_articles_input = input("Max articles to scrape (default 10, press Enter to use default): ").strip()
    max_articles = int(max_articles_input) if max_articles_input.isdigit() else 10
    
    # Get lookback period (optional)
    days_input = input("Days before fight to search (default 60 for ~2 months, press Enter to use default): ").strip()
    days_before = int(days_input) if days_input.isdigit() else 60
    
    print("\n" + "=" * 80)
    print(f"🔍 Searching for: {fighter1} vs {fighter2}")
    print(f"📅 Fight date: {fight_date}")
    print(f"⏰ Looking back {days_before} days ({days_before // 30} months)")
    print(f"📊 Max articles: {max_articles}")
    print(f"\n📰 Checking news sources (this may take 15-20 seconds)...")
    print("=" * 80 + "\n")
    
    # Initialize scraper
    scraper = UFCFightArticleScraper(delay_between_requests=2.0)
    
    # Scrape articles
    articles = scraper.scrape_fight_articles(
        fighter1_name=fighter1,
        fighter2_name=fighter2,
        fight_date=fight_date,
        days_before=days_before,
        max_articles=max_articles
    )
    
    # Display results
    if not articles:
        print("\n❌ NO ARTICLES FOUND")
        print("\n🔍 Possible reasons:")
        print(f"   • No recent news about {fighter1} vs {fighter2}")
        print("   • Fight might not be officially announced yet")
        print("   • RSS feeds may not have articles for this matchup")
        print("\n💡 Try:")
        print("   • Increase days_before (e.g., 90 days)")
        print("   • Check fighter name spelling")
        print("   • Try a different, more publicized fight")
        return
    
    # Show statistics
    stats = scraper.get_article_stats(articles)
    
    print(f"\n✅ SUCCESS! Found {len(articles)} articles\n")
    print("=" * 80)
    print("📊 SCRAPING RESULTS")
    print("=" * 80)
    print(f"Total articles: {stats['total_articles']}")
    print(f"Articles with full content: {stats['articles_with_content']}")
    print(f"Average content length: {stats['avg_content_length']:,} characters")
    
    print(f"\n📰 Sources:")
    for source, count in stats['sources'].items():
        print(f"   • {source}: {count} articles")
    
    if stats['date_range']:
        earliest = stats['date_range']['earliest'].strftime('%Y-%m-%d')
        latest = stats['date_range']['latest'].strftime('%Y-%m-%d')
        print(f"\n📅 Article date range: {earliest} to {latest}")
        print(f"   (Most recent articles prioritized)")
    
    # Ask what to display
    print("\n" + "=" * 80)
    print("What would you like to see?")
    print("=" * 80)
    print("  1. Article summaries only")
    print("  2. Comprehensive analysis (summaries + key points)")
    print("  3. Full article content")
    print("  4. Save to file")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    print("\n" + "=" * 80)
    
    if choice == '1':
        # Simple summaries
        print("📰 ARTICLE SUMMARIES")
        print("=" * 80 + "\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article.title}")
            print(f"   Source: {article.source} | Published: {article.published_date.strftime('%Y-%m-%d') if article.published_date else 'Unknown'}")
            print(f"   URL: {article.url}")
            if article.full_content:
                summary = scraper.summarize_article(article, max_sentences=3)
                print(f"\n   Summary: {summary}\n")
            else:
                print(f"\n   Summary: {article.summary}\n")
            print("-" * 80)
    
    elif choice == '2':
        # Comprehensive analysis
        print("🔬 COMPREHENSIVE ANALYSIS")
        print("=" * 80 + "\n")
        for i, article in enumerate(articles, 1):
            print(f"\nARTICLE {i}:")
            print("=" * 80)
            if article.full_content:
                summary = create_comprehensive_summary(
                    article.full_content,
                    [fighter1, fighter2],
                    include_bullets=True
                )
                print(summary)
            else:
                print(f"Title: {article.title}")
                print(f"Summary: {article.summary}")
            print()
    
    elif choice == '3':
        # Full content
        print("📄 FULL ARTICLE CONTENT")
        print("=" * 80 + "\n")
        for i, article in enumerate(articles, 1):
            print(format_article_report(article, include_full_content=True))
    
    elif choice == '4':
        # Save to file
        filename = f"{fighter1.replace(' ', '_')}_vs_{fighter2.replace(' ', '_')}_{fight_date}.txt"
        with open(filename, 'w') as f:
            f.write(f"UFC Fight Articles: {fighter1} vs {fighter2}\n")
            f.write(f"Fight Date: {fight_date}\n")
            f.write(f"Scraped: {stats['total_articles']} articles\n")
            f.write("=" * 80 + "\n\n")
            
            for i, article in enumerate(articles, 1):
                f.write(f"\nARTICLE {i}:\n")
                f.write(format_article_report(article, include_full_content=True))
                f.write("\n" + "=" * 80 + "\n")
        
        print(f"✅ Saved {len(articles)} articles to: {filename}")
    
    else:
        print("Invalid choice. Showing summaries...")
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article.title}")
            if article.full_content:
                summary = scraper.summarize_article(article, max_sentences=3)
                print(f"   {summary}")
    
    print("\n" + "=" * 80)
    print("✅ SCRAPING COMPLETE!")
    print("=" * 80)
    print(f"\n📝 Summary:")
    print(f"   • Found {len(articles)} articles about {fighter1} vs {fighter2}")
    print(f"   • Fight date: {fight_date}")
    print(f"   • Articles from last {days_before} days before the fight")
    print(f"   • Prioritized most recent articles")


def main():
    """Main entry point"""
    try:
        scrape_fight()
    except KeyboardInterrupt:
        print("\n\n❌ Scraping cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

