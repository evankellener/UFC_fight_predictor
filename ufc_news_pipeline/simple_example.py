#!/usr/bin/env python3
"""
Simple Example - How to use the UFC Fight Article Scraper

This is the simplest way to get started with the scraper.
"""

from article_scraper import UFCFightArticleScraper
from datetime import datetime, timedelta

# 1. Create the scraper
scraper = UFCFightArticleScraper(delay_between_requests=2.0)

# 2. Define your fight
# NOTE: RSS feeds only keep articles from the last few weeks!
# For best results, use fights happening within the next 2 months or in the last 2 weeks

# Example 1: Recent fight (more likely to have results)
fighter1 = "Jon Jones"
fighter2 = "Stipe Miocic"
fight_date = "2024-11-16"  # YYYY-MM-DD format

# Example 2: Try with a current/upcoming fight for better results
# Uncomment these lines and update with a current fight:
# fighter1 = "Islam Makhachev"
# fighter2 = "Arman Tsarukyan"
# fight_date = "2025-01-18"  # Update with actual upcoming fight date

print("=" * 80)
print("UFC FIGHT ARTICLE SCRAPER - SIMPLE EXAMPLE")
print("=" * 80)

# 3. Scrape articles
print(f"\n🔍 Searching for articles about: {fighter1} vs {fighter2}")
print(f"📅 Fight date: {fight_date}")
print(f"⏰ Searching 60 days (~2 months) before the fight...")
print(f"📊 Max articles: 10 (prioritizing most recent)")
print(f"\n📰 Checking news sources (this may take 10-15 seconds)...\n")

articles = scraper.scrape_fight_articles(
    fighter1_name=fighter1,
    fighter2_name=fighter2,
    fight_date=fight_date,
    days_before=60,  # Search 2 months before the fight
    max_articles=10  # Limit to 10 most recent articles
)

print("\n" + "=" * 80)

# 4. Use the results
if articles:
    print(f"\n✅ SUCCESS! Found {len(articles)} articles with scrapable content!\n")
    
    # Show statistics
    stats = scraper.get_article_stats(articles)
    print("📊 SCRAPING RESULTS:")
    print(f"   • Total articles scraped: {stats['total_articles']}")
    print(f"   • Articles with full content: {stats['articles_with_content']}")
    print(f"   • Average article length: {stats['avg_content_length']:,} characters")
    print(f"\n   📰 Sources:")
    for source, count in stats['sources'].items():
        print(f"      - {source}: {count} articles")
    
    if stats['date_range']:
        print(f"\n   📅 Article dates: {stats['date_range']['earliest'].strftime('%Y-%m-%d')} to {stats['date_range']['latest'].strftime('%Y-%m-%d')}")
    
    print("\n" + "-" * 80)
    print("📄 ARTICLE DETAILS & SUMMARIES:")
    print("-" * 80 + "\n")
    
    for i, article in enumerate(articles[:5], 1):  # Show first 5
        print(f"{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   URL: {article.url}")
        
        # Access full article content
        if article.full_content:
            print(f"   ✅ Full content scraped: {len(article.full_content)} characters")
            
            # Show preview of content
            print(f"\n   📄 Content Preview:")
            preview = article.full_content[:300].replace('\n', ' ')
            print(f"   {preview}...")
            
            # Get a quick summary
            print(f"\n   📝 AI-Generated Summary:")
            summary = scraper.summarize_article(article, max_sentences=3)
            print(f"   {summary}")
        else:
            print(f"   ⚠️  RSS summary only: {article.summary[:200]}...")
        
        print()
    
    if len(articles) > 5:
        print(f"   ... and {len(articles) - 5} more articles\n")
    
    print("=" * 80)
    print("✅ SCRAPING COMPLETE!")
    print("=" * 80)
    
else:
    print("\n❌ NO ARTICLES FOUND")
    print("\n🔍 Why this might happen:")
    print("   1. RSS feeds only keep articles from the last few weeks")
    print(f"   2. The fight date ({fight_date}) might be too old")
    print("   3. Not enough media coverage for this specific matchup")
    
    print("\n💡 Try this instead:")
    print("   1. Find an upcoming UFC fight from: https://www.ufc.com/events")
    print("   2. Update fighter names and fight_date in this script")
    print("   3. Run again with a current/upcoming fight")
    print("\n   Example of upcoming fights to try:")
    print("   - Check UFC website for the next main event")
    print("   - Look for fights announced in the last week")
    print("   - Use fights happening in the next 1-2 months")


print("\n" + "="*80)
print("📚 HOW TO USE THE SCRAPED DATA")
print("="*80)

if articles:
    print("\n✅ You now have articles with full content to analyze!")
    print("\nEach article object provides:")
    print("  • article.title          - The article headline")
    print("  • article.url            - Direct link to the article")
    print("  • article.full_content   - ✨ COMPLETE ARTICLE TEXT ✨")
    print("  • article.source         - News source name")
    print("  • article.published_date - When it was published")
    print("  • article.fighters_mentioned - List of fighters mentioned")
    
    print("\n💡 Next steps:")
    print("  1. Use article.full_content for detailed analysis")
    print("  2. Feed content into your prediction model")
    print("  3. Extract features (injuries, training updates, etc.)")
    print("  4. Run sentiment analysis on the text")
else:
    print("\n💡 To get articles:")
    print("  1. Find an upcoming UFC event")
    print("  2. Update fighter1, fighter2, and fight_date variables")
    print("  3. Re-run this script")
    
print("\n📖 For more examples:")
print("  • Run: python quick_demo.py (interactive)")
print("  • Run: python example_usage.py (comprehensive examples)")
print("  • Read: README.md (full documentation)")

