#!/usr/bin/env python3
"""
Demo with Mock Data - Shows what the scraper output looks like with actual results

This creates mock articles to demonstrate what you'll see when the scraper finds real articles.
"""

from article_scraper import Article, UFCFightArticleScraper
from datetime import datetime

# Create mock articles to show what real results look like
mock_articles = [
    Article(
        title="Jon Jones vs Stipe Miocic: Everything You Need to Know About UFC 295",
        url="https://www.mmafighting.com/2024/11/10/jones-miocic-preview",
        published_date=datetime(2024, 11, 10),
        source="MMA Fighting",
        summary="Jones defends heavyweight title against former champion Miocic at MSG"
    ),
    Article(
        title="Stipe Miocic: 'Jon Jones Has Never Faced Anyone Like Me'",
        url="https://www.bloodyelbow.com/2024/11/8/miocic-jones-interview",
        published_date=datetime(2024, 11, 8),
        source="Bloody Elbow",
        summary="Former heavyweight champion discusses his training camp and gameplan"
    ),
    Article(
        title="UFC 295: Jones vs Miocic Fight Card, Start Time, Predictions",
        url="https://www.mmamania.com/2024/11/12/ufc-295-preview",
        published_date=datetime(2024, 11, 12),
        source="MMA Mania",
        summary="Complete breakdown of the heavyweight title fight in New York"
    )
]

# Add mock full content
mock_articles[0].full_content = """
Jon Jones will defend his heavyweight title against former champion Stipe Miocic at UFC 295 on November 16th at Madison Square Garden. This highly anticipated matchup pits two of the greatest heavyweights in UFC history against each other.

Jones captured the heavyweight title earlier this year with a first-round submission of Ciryl Gane. The former light heavyweight champion has been dominant throughout his career, with only one loss on his record (a disqualification). Now 36 years old, Jones is looking to cement his legacy as the greatest fighter of all time.

Miocic, 41, is coming off a long layoff after losing his title to Francis Ngannou in 2021. The former firefighter from Cleveland has been training in secret for this comeback fight. He holds wins over Daniel Cormier, Fabricio Werdum, and Junior dos Santos during his championship reign.

According to sources close to the training camps, both fighters are in excellent shape. Jones has been working extensively on his wrestling at Jackson-Wink MMA in Albuquerque. His coach Greg Jackson reports that Jones has looked sharp in sparring sessions and is fully recovered from a minor injury sustained earlier in camp.

Miocic has been preparing at Strong Style Fight Team in Cleveland. His longtime coach Marcus Marinelli says the former champion has rediscovered his passion for fighting. "Stipe looks as good as he did when he was 30," Marinelli told reporters last week.

Oddsmakers have installed Jones as a significant favorite, with most sportsbooks listing him around -300. However, Miocic's knockout power cannot be discounted. The Croatian-American has 15 knockout victories in his career and has never been submitted.

The fight is expected to be contested primarily on the feet, though Jones has superior wrestling credentials. Many analysts predict Jones will look to control distance with his reach advantage and potentially take Miocic down in later rounds if necessary.
"""

mock_articles[1].full_content = """
Stipe Miocic spoke to media members this week about his upcoming heavyweight title challenge against Jon Jones at UFC 295. The 41-year-old former champion sounded confident about his chances despite being the underdog.

"Jon has never faced anyone like me," Miocic said. "He's fought tall guys, he's fought wrestlers, but he hasn't fought someone with my combination of power, wrestling, and experience. I've been here before. I know what it takes to win."

Miocic addressed concerns about his age and layoff. "People wrote me off before, and I proved them wrong. Age is just a number. I feel better now than I did five years ago. The time off allowed me to heal up completely and get my mind right."

The former champion revealed he's been studying Jones extensively. "We know he's going to try to use his reach. We've got a gameplan for that. I'm not going to stand on the outside and let him pick me apart. I'm going to bring the pressure and make it a fight."

Training partners say Miocic has looked exceptional in camp. "He's been finishing guys in sparring," one teammate reported. "His boxing is crisp, his cardio is great, and he's hungry. This isn't the same Stipe who lost to Ngannou."

When asked about Jones's wrestling, Miocic was dismissive. "I'm a Division I wrestler myself. I've defended takedowns from Daniel Cormier, who's an Olympic wrestler. I'm not worried about Jones's wrestling. If he wants to wrestle, we can wrestle. If he wants to box, we can box."

The fight represents potentially the final opportunity for Miocic to reclaim heavyweight gold. A win would make him only the second fighter to become a three-time UFC heavyweight champion, joining Randy Couture in that exclusive club.
"""

mock_articles[2].full_content = """
UFC 295 takes place on November 16th at Madison Square Garden in New York City. The event is headlined by a heavyweight title fight between champion Jon Jones and former champion Stipe Miocic.

The main card begins at 10 PM ET / 7 PM PT. The preliminary card starts at 8 PM ET on ESPN+. This marks Jones's first title defense since winning the belt in March.

Fight analysts are divided on the outcome. Some favor Jones's versatility and fight IQ, while others believe Miocic's power and experience could prove decisive. ESPN's panel has Jones winning by decision, while several former fighters have picked Miocic by knockout.

The co-main event features a light heavyweight clash between Jiri Prochazka and Alex Pereira. Both fighters are known for their striking ability and this bout could determine the next title challenger.

Betting odds heavily favor Jones, who opened as a -280 favorite. However, money has been coming in on Miocic, moving the line slightly. The fight is expected to generate over 500,000 pay-per-view buys.

Training reports from both camps have been positive. Neither fighter has reported significant injuries. Both successfully made weight at Friday's weigh-in, with Jones coming in at 239 pounds and Miocic at 241 pounds.

MMA experts predict the fight will be competitive for the first two rounds before Jones potentially takes over with his superior cardio and technique. However, Miocic has shown the ability to land fight-ending shots at any moment, making this a high-stakes battle between two legends of the sport.
"""

# Set fighters mentioned
for article in mock_articles:
    article.fighters_mentioned = ["Jon Jones", "Stipe Miocic"]

# Now demonstrate the scraper output
scraper = UFCFightArticleScraper()

print("=" * 80)
print("🎬 DEMO: What The Scraper Output Looks Like With REAL Articles")
print("=" * 80)
print("\nThis shows what you'll see when the scraper successfully finds articles.")
print("(Using mock data to demonstrate)\n")

# Simulate the same output as the real scraper
fighter1 = "Jon Jones"
fighter2 = "Stipe Miocic"
fight_date = "2024-11-16"

print(f"🔍 Searching for articles about: {fighter1} vs {fighter2}")
print(f"📅 Fight date: {fight_date}")
print(f"⏰ Searching 30 days before the fight...")
print(f"\n📰 Checking news sources (this may take 10-15 seconds)...\n")

articles = mock_articles

print("=" * 80)

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
    
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   URL: {article.url}")
        
        if article.full_content:
            print(f"   ✅ Full content scraped: {len(article.full_content)} characters")
            
            # Show preview
            print(f"\n   📄 Content Preview:")
            preview = article.full_content[:300].replace('\n', ' ')
            print(f"   {preview}...")
            
            # Get summary
            print(f"\n   📝 AI-Generated Summary:")
            summary = scraper.summarize_article(article, max_sentences=3)
            print(f"   {summary}")
        
        print()
    
    print("=" * 80)
    print("✅ SCRAPING COMPLETE!")
    print("=" * 80)

print("\n" + "="*80)
print("📚 HOW TO USE THE SCRAPED DATA")
print("="*80)

print("\n✅ This is what you'll see when real articles are found!")
print("\nEach article object provides:")
print("  • article.title          - The article headline")
print("  • article.url            - Direct link to the article")
print("  • article.full_content   - ✨ COMPLETE ARTICLE TEXT ✨")
print("  • article.source         - News source name")
print("  • article.published_date - When it was published")
print("  • article.fighters_mentioned - List of fighters mentioned")

print("\n💡 To get REAL articles like this:")
print("  1. Find an upcoming UFC event at: https://www.ufc.com/events")
print("  2. Update fighter names and date in simple_example.py")
print("  3. Run the scraper with a current/upcoming fight")

print("\n📖 Try the interactive demo:")
print("  • Run: python quick_demo.py")
print("  • Enter an upcoming fight when prompted")
print("  • See real articles scraped and summarized!")

