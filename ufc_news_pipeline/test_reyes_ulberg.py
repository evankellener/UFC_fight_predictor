#!/usr/bin/env python3
"""
Debug why we're not finding Reyes vs Ulberg articles
"""

import feedparser
from datetime import datetime

print("=" * 80)
print("🔍 DEBUGGING: Reyes vs Ulberg Article Search")
print("=" * 80)

# Check RSS feeds for ANY mention of these fighters
feeds = {
    'MMA Fighting': 'https://www.mmafighting.com/rss/index.xml',
    'Bloody Elbow': 'https://www.bloodyelbow.com/rss/index.xml',
    'MMA Mania': 'https://www.mmamania.com/rss/index.xml',
    'Cageside Seats': 'https://www.cagesideseats.com/rss/index.xml',
    'Low Kick MMA': 'https://www.lowkickmma.com/feed/',
}

fighters = ['reyes', 'ulberg', 'dominic', 'carlos', 'perth']
fight_keywords = ['ufc perth', 'ufc 305', 'september 27', 'sept 27']

print("\nSearching for articles mentioning:")
print(f"  Fighters: {fighters}")
print(f"  Keywords: {fight_keywords}")
print("\n" + "=" * 80)

total_articles = 0
matching_articles = []

for source, url in feeds.items():
    print(f"\n📰 Checking {source}...")
    
    try:
        feed = feedparser.parse(url)
        
        if not feed.entries:
            print(f"   ⚠️  No articles in feed")
            continue
        
        print(f"   Found {len(feed.entries)} total articles in feed")
        
        # Check each article
        for entry in feed.entries:
            title = entry.get('title', '').lower()
            summary = entry.get('summary', '').lower()
            text = f"{title} {summary}"
            
            # Check for fighter mentions
            mentions = []
            for fighter in fighters:
                if fighter in text:
                    mentions.append(fighter)
            
            # Check for keywords
            keywords_found = []
            for keyword in fight_keywords:
                if keyword in text:
                    keywords_found.append(keyword)
            
            if mentions or keywords_found:
                total_articles += 1
                article_info = {
                    'title': entry.get('title', ''),
                    'source': source,
                    'link': entry.get('link', ''),
                    'mentions': mentions,
                    'keywords': keywords_found
                }
                matching_articles.append(article_info)
                
                print(f"\n   ✅ FOUND: {entry.get('title', '')[:70]}")
                if mentions:
                    print(f"      Fighters mentioned: {', '.join(mentions)}")
                if keywords_found:
                    print(f"      Keywords found: {', '.join(keywords_found)}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n\n" + "=" * 80)
print("📊 RESULTS")
print("=" * 80)

if matching_articles:
    print(f"\n✅ Found {len(matching_articles)} articles mentioning fighters or keywords!")
    
    # Group by how many fighters mentioned
    both_fighters = [a for a in matching_articles if 'reyes' in a['mentions'] and 'ulberg' in a['mentions']]
    one_fighter = [a for a in matching_articles if len(a['mentions']) > 0 and a not in both_fighters]
    keyword_only = [a for a in matching_articles if len(a['mentions']) == 0]
    
    print(f"\n   📋 Breakdown:")
    print(f"      Both fighters: {len(both_fighters)}")
    print(f"      One fighter: {len(one_fighter)}")
    print(f"      Keywords only: {len(keyword_only)}")
    
    if both_fighters:
        print(f"\n   ✅ Articles mentioning BOTH fighters:")
        for article in both_fighters:
            print(f"      • {article['title']}")
            print(f"        Source: {article['source']}")
            print(f"        Link: {article['link']}")
    
    if one_fighter:
        print(f"\n   ⚠️  Articles mentioning ONE fighter:")
        for article in one_fighter[:3]:  # Show first 3
            print(f"      • {article['title']}")
            print(f"        Mentions: {', '.join(article['mentions'])}")
            print(f"        Source: {article['source']}")
    
    print("\n" + "=" * 80)
    print("💡 DIAGNOSIS")
    print("=" * 80)
    
    if both_fighters:
        print("\n✅ Articles exist that mention BOTH fighters!")
        print("   The scraper SHOULD find these.")
        print("\n   Possible issues:")
        print("   • Date range might be excluding them")
        print("   • Name matching might be too strict")
    else:
        print("\n⚠️  NO articles mention both fighters together")
        print("   Articles might:")
        print("   • Only mention the winner")
        print("   • Use different name formats")
        print("   • Focus on the event, not fighters")
        
        print("\n   💡 Solution: Make scraper more flexible:")
        print("   • Search for either fighter + event keywords")
        print("   • Accept articles mentioning just one fighter")
        print("   • Search for 'UFC Perth' or event name")

else:
    print(f"\n❌ No articles found mentioning fighters or keywords")
    print("\n   This means:")
    print("   • Fight coverage not in current RSS feeds")
    print("   • Articles might be older than RSS retention")
    print("   • Different naming used in articles")

print("\n" + "=" * 80)

