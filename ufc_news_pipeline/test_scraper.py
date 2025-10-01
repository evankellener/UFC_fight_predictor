#!/usr/bin/env python3
"""
Test the scraper with the example you provided
"""

from article_scraper import UFCFightArticleScraper

# Your example: Dominic Reyes vs Carlos Ulberg 09/27/25
fighter1 = "Dominic Reyes"
fighter2 = "Carlos Ulberg"
fight_date = "2025-09-27"  # Converted from 09/27/25

print("=" * 80)
print("🥊 TESTING: Dominic Reyes vs Carlos Ulberg (09/27/25)")
print("=" * 80)
print(f"\nFighter 1: {fighter1}")
print(f"Fighter 2: {fighter2}")
print(f"Fight Date: {fight_date}")
print(f"Lookback: 60 days (~2 months)")
print(f"Max articles: 10 (prioritizing most recent)")
print("\n📰 Starting scrape...")
print("=" * 80 + "\n")

# Initialize scraper
scraper = UFCFightArticleScraper(delay_between_requests=2.0)

# Scrape with new parameters
articles = scraper.scrape_fight_articles(
    fighter1_name=fighter1,
    fighter2_name=fighter2,
    fight_date=fight_date,
    days_before=60,      # 2 months lookback
    max_articles=10      # Cap at 10 articles
)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if articles:
    stats = scraper.get_article_stats(articles)
    
    print(f"\n✅ Found {len(articles)} articles (limited to 10 most recent)")
    print(f"\n📊 Statistics:")
    print(f"   • Total scraped: {stats['total_articles']}")
    print(f"   • With full content: {stats['articles_with_content']}")
    print(f"   • Avg length: {stats['avg_content_length']:,} chars")
    
    if stats['sources']:
        print(f"\n   📰 Sources:")
        for source, count in stats['sources'].items():
            print(f"      - {source}: {count} articles")
    
    if stats['date_range']:
        print(f"\n   📅 Date range: {stats['date_range']['earliest'].strftime('%Y-%m-%d')} to {stats['date_range']['latest'].strftime('%Y-%m-%d')}")
        print(f"      (Most recent articles prioritized)")
    
    print(f"\n📰 Articles (sorted by date, newest first):")
    print("-" * 80)
    for i, article in enumerate(articles, 1):
        date_str = article.published_date.strftime('%Y-%m-%d') if article.published_date else 'Unknown'
        print(f"\n{i}. [{date_str}] {article.title}")
        print(f"   Source: {article.source}")
        print(f"   URL: {article.url}")
        if article.full_content:
            print(f"   ✅ Full content: {len(article.full_content)} chars")
            summary = scraper.summarize_article(article, max_sentences=2)
            print(f"   Summary: {summary[:200]}...")
        else:
            print(f"   ⚠️  RSS only: {article.summary[:150]}...")
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS - Scraping complete!")
    
else:
    print(f"\n❌ No articles found for {fighter1} vs {fighter2}")
    print("\n🔍 This might be because:")
    print("   • The fight isn't officially announced yet")
    print("   • No recent media coverage in RSS feeds")
    print("   • Fighter names might need adjustment")
    print("\n💡 For upcoming fights, the scraper will find articles as they're published")

print("\n" + "=" * 80)
print("📝 USAGE EXAMPLE")
print("=" * 80)
print("\nTo use in your code:")
print("""
from article_scraper import UFCFightArticleScraper

scraper = UFCFightArticleScraper()
articles = scraper.scrape_fight_articles(
    fighter1_name="Dominic Reyes",
    fighter2_name="Carlos Ulberg",
    fight_date="2025-09-27",
    days_before=60,      # 2 months before fight
    max_articles=10      # Limit to 10 most recent
)

# Access articles
for article in articles:
    print(article.title)
    print(article.full_content)  # Complete article text
""")

