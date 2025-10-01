"""
Example Usage of UFC Fight Article Scraper

This script demonstrates how to use the scraper to fetch and analyze
articles about upcoming UFC fights.
"""

from article_scraper import UFCFightArticleScraper, format_article_report
from summarizer import ArticleSummarizer, create_comprehensive_summary
import sys
from datetime import datetime


def example_basic_scraping():
    """Basic example: Scrape articles for a specific fight"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Article Scraping")
    print("=" * 80)
    
    # Initialize the scraper
    scraper = UFCFightArticleScraper(delay_between_requests=2.0)
    
    # Define the fight
    fighter1 = "Jon Jones"
    fighter2 = "Stipe Miocic"
    fight_date = "2024-11-16"  # Format: YYYY-MM-DD
    
    print(f"\nSearching for articles about: {fighter1} vs {fighter2}")
    print(f"Fight date: {fight_date}")
    print(f"Searching 30 days before the fight...\n")
    
    # Scrape articles
    articles = scraper.scrape_fight_articles(
        fighter1_name=fighter1,
        fighter2_name=fighter2,
        fight_date=fight_date,
        days_before=30
    )
    
    # Display results
    if articles:
        print(f"\n✅ Found {len(articles)} articles!")
        
        # Show first article in detail
        print("\n" + "=" * 80)
        print("FIRST ARTICLE (Full Content):")
        print(format_article_report(articles[0], include_full_content=True))
    else:
        print("\n❌ No articles found for this fight")


def example_with_summarization():
    """Example: Scrape and summarize articles"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Scraping with Summarization")
    print("=" * 80)
    
    scraper = UFCFightArticleScraper(delay_between_requests=2.0)
    summarizer = ArticleSummarizer()
    
    # Different fight example
    fighter1 = "Alex Pereira"
    fighter2 = "Jamahal Hill"
    fight_date = "2024-04-13"
    
    print(f"\nAnalyzing articles for: {fighter1} vs {fighter2}")
    print(f"Fight date: {fight_date}\n")
    
    articles = scraper.scrape_fight_articles(fighter1, fighter2, fight_date, days_before=21)
    
    if articles:
        print(f"Found {len(articles)} articles\n")
        
        # Summarize each article
        for i, article in enumerate(articles[:3], 1):  # Show first 3
            print("=" * 80)
            print(f"ARTICLE {i}: {article.title}")
            print("=" * 80)
            
            if article.full_content:
                # Create comprehensive summary
                summary = create_comprehensive_summary(
                    article.full_content,
                    [fighter1, fighter2],
                    include_bullets=True
                )
                print(summary)
            print("\n")
    else:
        print("No articles found")


def example_detailed_analysis():
    """Example: Detailed analysis of all articles for a fight"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Detailed Fight Analysis")
    print("=" * 80)
    
    scraper = UFCFightArticleScraper(delay_between_requests=2.0)
    summarizer = ArticleSummarizer()
    
    fighter1 = "Islam Makhachev"
    fighter2 = "Alexander Volkanovski"
    fight_date = "2023-10-21"
    
    print(f"\nDetailed analysis for: {fighter1} vs {fighter2}")
    print(f"Fight date: {fight_date}\n")
    
    articles = scraper.scrape_fight_articles(fighter1, fighter2, fight_date, days_before=30)
    
    if not articles:
        print("No articles found")
        return
    
    # Get statistics
    stats = scraper.get_article_stats(articles)
    
    print("\n📊 ARTICLE STATISTICS")
    print("=" * 80)
    print(f"Total articles found: {stats['total_articles']}")
    print(f"Articles with full content: {stats['articles_with_content']}")
    print(f"Average content length: {stats['avg_content_length']:,} characters")
    
    print("\n📰 Articles by source:")
    for source, count in stats['sources'].items():
        print(f"  • {source}: {count} articles")
    
    if stats['date_range']:
        print(f"\n📅 Date range:")
        print(f"  Earliest: {stats['date_range']['earliest'].strftime('%Y-%m-%d')}")
        print(f"  Latest: {stats['date_range']['latest'].strftime('%Y-%m-%d')}")
    
    # Analyze content across all articles
    print("\n\n🔍 AGGREGATED ANALYSIS")
    print("=" * 80)
    
    all_injury_mentions = []
    all_training_mentions = []
    all_predictions = []
    
    for article in articles:
        if article.full_content:
            key_points = summarizer.get_key_points(article.full_content, [fighter1, fighter2])
            all_injury_mentions.extend(key_points['injury_concerns'])
            all_training_mentions.extend(key_points['training_updates'])
            all_predictions.extend(key_points['predictions'])
    
    print(f"\nInjury mentions across all articles: {len(all_injury_mentions)}")
    if all_injury_mentions:
        print("Examples:")
        for mention in all_injury_mentions[:3]:
            print(f"  • {mention[:120]}...")
    
    print(f"\nTraining updates: {len(all_training_mentions)}")
    if all_training_mentions:
        print("Examples:")
        for mention in all_training_mentions[:3]:
            print(f"  • {mention[:120]}...")
    
    print(f"\nPredictions/analysis: {len(all_predictions)}")
    if all_predictions:
        print("Examples:")
        for mention in all_predictions[:3]:
            print(f"  • {mention[:120]}...")
    
    # Fighter-focused summary from all content
    print("\n\n👤 FIGHTER-SPECIFIC INSIGHTS")
    print("=" * 80)
    
    # Combine all article content
    all_content = "\n\n".join([a.full_content for a in articles if a.full_content])
    
    if all_content:
        fighter_insights = summarizer.fighter_focused_summary(
            all_content,
            [fighter1, fighter2],
            num_sentences=5
        )
        
        for fighter, summary in fighter_insights.items():
            print(f"\n{fighter.upper()}:")
            print("-" * 80)
            print(summary)


def example_custom_fight():
    """Example: Scrape articles for a user-specified fight"""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Custom Fight Search")
    print("=" * 80)
    
    # Get user input (or use defaults)
    print("\nEnter fight details (or press Enter to use defaults):")
    
    fighter1 = input("Fighter 1 name (default: Jon Jones): ").strip()
    if not fighter1:
        fighter1 = "Jon Jones"
    
    fighter2 = input("Fighter 2 name (default: Stipe Miocic): ").strip()
    if not fighter2:
        fighter2 = "Stipe Miocic"
    
    fight_date = input("Fight date YYYY-MM-DD (default: 2024-11-16): ").strip()
    if not fight_date:
        fight_date = "2024-11-16"
    
    days_before = input("Days before fight to search (default: 30): ").strip()
    days_before = int(days_before) if days_before.isdigit() else 30
    
    print(f"\n🔍 Searching for: {fighter1} vs {fighter2}")
    print(f"📅 Fight date: {fight_date}")
    print(f"⏰ Searching {days_before} days before fight\n")
    
    # Scrape
    scraper = UFCFightArticleScraper(delay_between_requests=2.0)
    articles = scraper.scrape_fight_articles(fighter1, fighter2, fight_date, days_before)
    
    if articles:
        print(f"\n✅ Found {len(articles)} articles\n")
        
        # Show summaries
        for i, article in enumerate(articles[:5], 1):
            print(f"\n{'=' * 80}")
            print(f"ARTICLE {i}: {article.title}")
            print(f"Source: {article.source}")
            print(f"URL: {article.url}")
            print(f"{'=' * 80}")
            
            if article.full_content:
                summary = UFCFightArticleScraper.summarize_article(article, max_sentences=5)
                print(f"\nSummary:\n{summary}\n")
            else:
                print(f"\nSummary:\n{article.summary}\n")
    else:
        print(f"\n❌ No articles found for {fighter1} vs {fighter2}")
        print("\nTips:")
        print("  • Try increasing the days_before parameter")
        print("  • Check that fighter names are spelled correctly")
        print("  • Recent fights have more coverage than older ones")


def main():
    """Run all examples"""
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                  UFC FIGHT ARTICLE SCRAPER - EXAMPLES                      ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nThis script demonstrates different ways to use the UFC article scraper.")
    print("\nSelect an example to run:")
    print("  1. Basic article scraping")
    print("  2. Scraping with summarization")
    print("  3. Detailed fight analysis")
    print("  4. Custom fight search (interactive)")
    print("  5. Run all examples")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    examples = {
        '1': example_basic_scraping,
        '2': example_with_summarization,
        '3': example_detailed_analysis,
        '4': example_custom_fight,
    }
    
    if choice in examples:
        examples[choice]()
    elif choice == '5':
        for example_func in examples.values():
            example_func()
            print("\n\n" + "="*80 + "\n\n")
    else:
        print("Invalid choice. Running basic example...")
        example_basic_scraping()
    
    print("\n\n✅ Examples complete!")
    print("\nNext steps:")
    print("  • Modify the examples to search for different fights")
    print("  • Integrate the scraper into your prediction pipeline")
    print("  • Use the summarization features to extract key insights")


if __name__ == "__main__":
    main()

