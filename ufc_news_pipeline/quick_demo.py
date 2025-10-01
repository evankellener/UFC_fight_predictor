#!/usr/bin/env python3
"""
Quick Demo - UFC Fight Article Scraper

Simple script to quickly test the scraper with a specific fight.
"""

from article_scraper import UFCFightArticleScraper, format_article_report
from summarizer import ArticleSummarizer, create_comprehensive_summary


def main():
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║              UFC FIGHT ARTICLE SCRAPER - QUICK DEMO                        ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get fight details from user
    print("\n📋 Enter fight details:\n")
    
    fighter1 = input("Fighter 1 name: ").strip()
    if not fighter1:
        print("❌ Fighter 1 name is required!")
        return
    
    fighter2 = input("Fighter 2 name: ").strip()
    if not fighter2:
        print("❌ Fighter 2 name is required!")
        return
    
    fight_date = input("Fight date (YYYY-MM-DD): ").strip()
    if not fight_date:
        print("❌ Fight date is required!")
        return
    
    days_before = input("Days before fight to search (default 30): ").strip()
    days_before = int(days_before) if days_before.isdigit() else 30
    
    # Initialize scraper
    print(f"\n🔍 Searching for articles about {fighter1} vs {fighter2}...")
    print(f"📅 Fight date: {fight_date}")
    print(f"⏰ Searching {days_before} days before the fight\n")
    
    scraper = UFCFightArticleScraper(delay_between_requests=2.0)
    
    # Scrape articles
    articles = scraper.scrape_fight_articles(
        fighter1_name=fighter1,
        fighter2_name=fighter2,
        fight_date=fight_date,
        days_before=days_before
    )
    
    # Display results
    if not articles:
        print("\n❌ No articles found for this fight")
        print("\nTips:")
        print("  • Try increasing the days_before parameter")
        print("  • Make sure fighter names are spelled correctly")
        print("  • Recent/upcoming fights have more coverage")
        return
    
    print(f"\n✅ Found {len(articles)} articles!\n")
    
    # Show statistics
    stats = scraper.get_article_stats(articles)
    print("📊 STATISTICS")
    print("=" * 80)
    print(f"Total articles: {stats['total_articles']}")
    print(f"Articles with full content: {stats['articles_with_content']}")
    print(f"Average content length: {stats['avg_content_length']:,} characters")
    
    print("\n📰 Sources:")
    for source, count in stats['sources'].items():
        print(f"  • {source}: {count} articles")
    
    if stats['date_range']:
        print(f"\n📅 Date range:")
        print(f"  {stats['date_range']['earliest'].strftime('%Y-%m-%d')} to {stats['date_range']['latest'].strftime('%Y-%m-%d')}")
    
    # Ask user what to display
    print("\n\n" + "=" * 80)
    print("What would you like to see?")
    print("  1. Article summaries (5 sentences each)")
    print("  2. Comprehensive analysis (with bullet points, categories)")
    print("  3. Full article content")
    print("  4. Fighter-specific summaries")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    summarizer = ArticleSummarizer()
    
    if choice == '1':
        # Simple summaries
        print("\n\n📰 ARTICLE SUMMARIES")
        print("=" * 80)
        for i, article in enumerate(articles[:10], 1):  # Max 10 articles
            print(f"\n{i}. {article.title}")
            print(f"   Source: {article.source} | URL: {article.url}")
            print(f"   {'-' * 76}")
            if article.full_content:
                summary = summarizer.extractive_summary(
                    article.full_content,
                    num_sentences=5,
                    fighter_names=[fighter1, fighter2]
                )
                print(f"   {summary}")
            else:
                print(f"   {article.summary}")
    
    elif choice == '2':
        # Comprehensive analysis
        print("\n\n🔬 COMPREHENSIVE ANALYSIS")
        for i, article in enumerate(articles[:5], 1):  # Max 5 for comprehensive
            print(f"\n{'=' * 80}")
            print(f"ARTICLE {i}: {article.title}")
            print(f"{'=' * 80}")
            if article.full_content:
                comprehensive = create_comprehensive_summary(
                    article.full_content,
                    [fighter1, fighter2],
                    include_bullets=True
                )
                print(comprehensive)
            else:
                print(f"Summary: {article.summary}")
    
    elif choice == '3':
        # Full content
        print("\n\n📄 FULL ARTICLE CONTENT")
        for i, article in enumerate(articles[:3], 1):  # Max 3 for full content
            print(format_article_report(article, include_full_content=True))
    
    elif choice == '4':
        # Fighter-specific
        print(f"\n\n👤 FIGHTER-SPECIFIC SUMMARIES")
        print("=" * 80)
        
        # Combine all content
        all_content = "\n\n".join([a.full_content for a in articles if a.full_content])
        
        if all_content:
            fighter_summaries = summarizer.fighter_focused_summary(
                all_content,
                [fighter1, fighter2],
                num_sentences=7
            )
            
            for fighter, summary in fighter_summaries.items():
                print(f"\n{fighter.upper()}:")
                print("-" * 80)
                print(summary)
        else:
            print("No article content available for fighter-specific analysis")
    
    else:
        print("\n❌ Invalid choice. Showing simple summaries...")
        for i, article in enumerate(articles[:5], 1):
            print(f"\n{i}. {article.title}")
            print(f"   {article.url}")
            if article.full_content:
                summary = UFCFightArticleScraper.summarize_article(article, max_sentences=3)
                print(f"   {summary[:200]}...")
    
    print("\n\n✅ Demo complete!")
    print("\nTo use this in your own code:")
    print("  from article_scraper import UFCFightArticleScraper")
    print("  scraper = UFCFightArticleScraper()")
    print(f"  articles = scraper.scrape_fight_articles('{fighter1}', '{fighter2}', '{fight_date}')")


if __name__ == "__main__":
    main()

