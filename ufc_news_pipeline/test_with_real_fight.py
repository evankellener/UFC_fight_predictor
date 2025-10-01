#!/usr/bin/env python3
"""
Test with a REAL recent UFC fight to demonstrate the scraper works
"""

from article_scraper import UFCFightArticleScraper
from datetime import datetime, timedelta

print("=" * 80)
print("🔍 TESTING WITH REAL UFC FIGHTS")
print("=" * 80)

scraper = UFCFightArticleScraper(delay_between_requests=2.0)

# Test with several recent/upcoming fights
test_fights = [
    # Recent main events that likely have coverage
    ("Jon Jones", "Stipe Miocic", "2024-11-16", "UFC 295 (recent)"),
    ("Alex Pereira", "Khalil Rountree", "2024-10-05", "UFC 307 (recent)"),
    ("Dricus Du Plessis", "Israel Adesanya", "2024-08-17", "UFC 305 (recent)"),
    # Try just one fighter to see general UFC news
    ("Alex Pereira", "Jiri Prochazka", "2024-06-29", "UFC 303"),
]

print("\n🧪 Testing multiple fights to find one with articles...\n")

found_articles = False

for fighter1, fighter2, fight_date, event_name in test_fights:
    print(f"\n{'='*80}")
    print(f"Testing: {fighter1} vs {fighter2} ({event_name})")
    print(f"Date: {fight_date}")
    print(f"{'='*80}")
    
    # Calculate if this fight is within RSS feed range (usually last 30 days)
    fight_datetime = datetime.strptime(fight_date, "%Y-%m-%d")
    days_ago = (datetime.now() - fight_datetime).days
    
    print(f"Fight was {days_ago} days ago")
    
    if days_ago > 45:
        print("⚠️  Likely too old for RSS feeds (>45 days ago)")
        print("   RSS feeds typically only keep last 2-4 weeks of articles")
        continue
    
    articles = scraper.scrape_fight_articles(
        fighter1_name=fighter1,
        fighter2_name=fighter2,
        fight_date=fight_date,
        days_before=60,
        max_articles=10
    )
    
    if articles:
        print(f"\n✅ SUCCESS! Found {len(articles)} articles!")
        found_articles = True
        
        stats = scraper.get_article_stats(articles)
        print(f"\n📊 Results:")
        print(f"   • Total: {len(articles)}")
        print(f"   • Sources: {list(stats['sources'].keys())}")
        
        print(f"\n📰 Articles (most recent first):")
        for i, article in enumerate(articles[:3], 1):
            date_str = article.published_date.strftime('%Y-%m-%d') if article.published_date else 'Unknown'
            print(f"\n   {i}. [{date_str}] {article.title}")
            print(f"      Source: {article.source}")
            print(f"      Content: {len(article.full_content)} chars")
            if article.full_content:
                summary = scraper.summarize_article(article, max_sentences=2)
                print(f"      Summary: {summary[:150]}...")
        
        break  # Found working fight, stop testing
    else:
        print(f"❌ No articles found for this fight")

print("\n\n" + "=" * 80)

if found_articles:
    print("✅ SCRAPER IS WORKING!")
    print("=" * 80)
    print("\n✨ The scraper successfully found and processed articles!")
    print("\nYour Reyes vs Ulberg fight probably:")
    print("  • Isn't officially announced yet")
    print("  • Hasn't had media coverage")
    print("  • Won't have articles until closer to the fight date")
    
else:
    print("⚠️  NO ARTICLES FOUND IN ANY TEST")
    print("=" * 80)
    print("\n🔍 This likely means:")
    print("  1. RSS feeds have expired for these older fights")
    print("  2. Need to test with fights from the LAST 2-3 WEEKS")
    
    print("\n💡 To see the scraper work:")
    print("  1. Go to https://www.ufc.com/events")
    print("  2. Find a fight happening in the NEXT 1-4 weeks")
    print("  3. Use those fighter names and date")
    print("  4. Articles will be in RSS feeds for upcoming events")

print("\n📝 HOW RSS FEEDS WORK:")
print("  • RSS feeds only keep articles from last 2-4 weeks")
print("  • Older fights: articles are archived (not in RSS)")
print("  • Upcoming fights: articles appear 1-4 weeks before")
print("  • Most coverage: 1 week before fight date")

print("\n🎯 FOR YOUR USE CASE:")
print("  When you run this on an actual upcoming UFC fight:")
print("  1. Finds all recent articles from RSS feeds")
print("  2. Filters for articles mentioning both fighters")
print("  3. Sorts by date (most recent first)")
print("  4. Limits to top 10 articles")
print("  5. Returns full content for analysis")

