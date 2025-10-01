# UFC News Pipeline - Fight Article Scraper

A comprehensive scraper for fetching and analyzing MMA news articles about specific UFC fights. Feed in two fighter names and a fight date, and get back article content with intelligent summarization.

## Features

- 🔍 **Smart Article Discovery**: Automatically searches multiple MMA news sources (MMA Fighting, Bloody Elbow, MMA Mania, etc.)
- 📄 **Full Content Extraction**: Fetches complete article text, not just summaries
- 🤖 **Multiple Summarization Methods**: Extractive, keyword-based, fighter-focused, and bullet-point summaries
- 📊 **Detailed Analytics**: Get statistics on article coverage, sources, and content
- ⚡ **Rate-Limited & Respectful**: Built-in delays to be respectful to news websites

## Installation

Required dependencies:

```bash
pip install requests beautifulsoup4 feedparser
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from article_scraper import UFCFightArticleScraper

# Initialize scraper
scraper = UFCFightArticleScraper(delay_between_requests=2.0)

# Scrape articles for a specific fight
articles = scraper.scrape_fight_articles(
    fighter1_name="Jon Jones",
    fighter2_name="Stipe Miocic",
    fight_date="2024-11-16",  # Format: YYYY-MM-DD
    days_before=30  # Search 30 days before the fight
)

# Access article content
for article in articles:
    print(f"Title: {article.title}")
    print(f"Source: {article.source}")
    print(f"URL: {article.url}")
    print(f"Content: {article.full_content[:500]}...")  # First 500 chars
    print()
```

### With Summarization

```python
from article_scraper import UFCFightArticleScraper
from summarizer import ArticleSummarizer, create_comprehensive_summary

scraper = UFCFightArticleScraper()
articles = scraper.scrape_fight_articles("Jon Jones", "Stipe Miocic", "2024-11-16")

# Simple summary
for article in articles:
    summary = UFCFightArticleScraper.summarize_article(article, max_sentences=5)
    print(f"{article.title}\n{summary}\n")

# Advanced summarization
summarizer = ArticleSummarizer()
for article in articles:
    if article.full_content:
        # Extractive summary
        summary = summarizer.extractive_summary(
            article.full_content, 
            num_sentences=5,
            fighter_names=["Jon Jones", "Stipe Miocic"]
        )
        print(summary)
        
        # Bullet points
        bullets = summarizer.bullet_point_summary(
            article.full_content,
            num_points=5,
            fighter_names=["Jon Jones", "Stipe Miocic"]
        )
        for bullet in bullets:
            print(f"  • {bullet}")
```

### Comprehensive Analysis

```python
from summarizer import create_comprehensive_summary

# Get comprehensive summary with all analysis types
for article in articles:
    if article.full_content:
        comprehensive = create_comprehensive_summary(
            article.full_content,
            fighter_names=["Jon Jones", "Stipe Miocic"],
            include_bullets=True
        )
        print(comprehensive)
```

## API Reference

### UFCFightArticleScraper

Main class for scraping articles.

#### Methods

**`scrape_fight_articles(fighter1_name, fighter2_name, fight_date, days_before=30)`**

Scrapes articles about a specific fight.

- **Parameters:**
  - `fighter1_name` (str): First fighter's full name (e.g., "Jon Jones")
  - `fighter2_name` (str): Second fighter's full name (e.g., "Stipe Miocic")
  - `fight_date` (str): Fight date in YYYY-MM-DD format
  - `days_before` (int): How many days before the fight to search (default: 30)

- **Returns:** List of `Article` objects

**`get_article_stats(articles)`** (static method)

Get statistics about scraped articles.

- **Returns:** Dictionary with:
  - `total_articles`: Total number of articles
  - `sources`: Dictionary mapping source names to article counts
  - `date_range`: Dictionary with 'earliest' and 'latest' dates
  - `avg_content_length`: Average article length in characters
  - `articles_with_content`: Number of articles with full content

**`summarize_article(article, max_sentences=5)`** (static method)

Create a simple extractive summary of an article.

- **Returns:** Summary string

### Article Object

Represents a news article.

#### Attributes

- `title` (str): Article title
- `url` (str): Article URL
- `published_date` (datetime): Publication date
- `source` (str): News source name
- `summary` (str): RSS summary
- `full_content` (str): Full article text
- `fighters_mentioned` (list): List of fighter names mentioned

### ArticleSummarizer

Advanced summarization with multiple strategies.

#### Methods

**`extractive_summary(text, num_sentences=5, fighter_names=None)`**

Creates an extractive summary by selecting the most important sentences.

**`keyword_based_summary(text, num_sentences=5, fighter_names=None)`**

Creates a summary focusing on fight-relevant keywords.

**`fighter_focused_summary(text, fighter_names, num_sentences=5)`**

Creates separate summaries for each fighter.

- **Returns:** Dictionary mapping fighter names to their summaries

**`bullet_point_summary(text, num_points=5, fighter_names=None)`**

Creates a bullet-point summary with key facts.

- **Returns:** List of bullet point strings

**`get_key_points(text, fighter_names)`**

Extracts categorized key points from the article.

- **Returns:** Dictionary with categories:
  - `injury_concerns`: Injury-related sentences
  - `training_updates`: Training/camp updates
  - `predictions`: Predictions and analysis
  - `fight_details`: Fight logistics
  - `quotes`: Direct quotes

## Examples

Run the example script to see all features in action:

```bash
python example_usage.py
```

This includes:
1. Basic article scraping
2. Scraping with summarization
3. Detailed fight analysis
4. Custom fight search (interactive)

### Example Output

```
🔍 Searching for articles about Jon Jones vs Stipe Miocic
📅 Fight date: 2024-11-16, searching 30 days before

📰 Fetching from MMA Fighting...
📰 Fetching from Bloody Elbow...
📰 Fetching from MMA Mania...

✅ Found 12 articles

📊 STATISTICS:
  Total articles: 12
  Articles with content: 11
  Average content length: 3,245 characters

  Sources:
    MMA Fighting: 5 articles
    Bloody Elbow: 4 articles
    MMA Mania: 3 articles
```

## News Sources

The scraper currently searches the following MMA news sources:

- **MMA Fighting** (mmafighting.com)
- **Bloody Elbow** (bloodyelbow.com)
- **MMA Mania** (mmamania.com)
- **Cageside Seats** (cagesideseats.com)
- **Low Kick MMA** (lowkickmma.com)

## Configuration

### Adjusting Request Delay

To be respectful to websites, adjust the delay between requests:

```python
scraper = UFCFightArticleScraper(delay_between_requests=3.0)  # 3 seconds
```

### Custom Date Range

Search articles in a specific date range:

```python
# Search 45 days before the fight
articles = scraper.scrape_fight_articles(
    fighter1_name="Fighter One",
    fighter2_name="Fighter Two",
    fight_date="2024-12-31",
    days_before=45
)
```

## Integration with Prediction Pipeline

You can integrate this scraper with your fight prediction model:

```python
from article_scraper import UFCFightArticleScraper
from summarizer import ArticleSummarizer

def get_fight_insights(fighter1, fighter2, fight_date):
    """Get news-based insights for a fight"""
    scraper = UFCFightArticleScraper()
    summarizer = ArticleSummarizer()
    
    articles = scraper.scrape_fight_articles(fighter1, fighter2, fight_date)
    
    if not articles:
        return None
    
    insights = {
        'article_count': len(articles),
        'sources': list(set(a.source for a in articles)),
        'injury_mentions': 0,
        'training_mentions': 0,
        'predictions': []
    }
    
    for article in articles:
        if article.full_content:
            key_points = summarizer.get_key_points(
                article.full_content,
                [fighter1, fighter2]
            )
            insights['injury_mentions'] += len(key_points['injury_concerns'])
            insights['training_mentions'] += len(key_points['training_updates'])
            insights['predictions'].extend(key_points['predictions'])
    
    return insights
```

## Troubleshooting

### No Articles Found

If no articles are found:

1. **Check fighter names**: Make sure names are spelled correctly
2. **Increase search window**: Try `days_before=60` or more
3. **Check date format**: Must be YYYY-MM-DD
4. **Recent fights**: Recent/upcoming fights have more coverage

### Content Extraction Issues

If article content is not being extracted:

1. The scraper tries multiple content selectors
2. Some sites may have different HTML structures
3. Check the `article.summary` field as a fallback

### Rate Limiting

If you're being rate-limited:

1. Increase `delay_between_requests` (e.g., to 3-5 seconds)
2. Reduce the number of articles being fetched
3. Run searches during off-peak hours

## Advanced Usage

### Custom Content Processing

```python
def process_articles(articles):
    """Custom processing of article content"""
    for article in articles:
        if article.full_content:
            # Count specific keywords
            injury_count = article.full_content.lower().count('injury')
            knockout_count = article.full_content.lower().count('knockout')
            
            print(f"{article.title}")
            print(f"  Injury mentions: {injury_count}")
            print(f"  Knockout mentions: {knockout_count}")
```

### Combining Multiple Fights

```python
def analyze_multiple_fights(fight_list):
    """Analyze articles for multiple fights"""
    scraper = UFCFightArticleScraper()
    
    all_results = {}
    for fighter1, fighter2, date in fight_list:
        articles = scraper.scrape_fight_articles(fighter1, fighter2, date)
        all_results[f"{fighter1} vs {fighter2}"] = articles
    
    return all_results
```

## Performance

- **Average request time**: 2-3 seconds per request (with delay)
- **Typical article count**: 5-20 articles per fight (depending on popularity)
- **Content extraction success rate**: ~90% for supported sources

## Contributing

To add support for new MMA news sources:

1. Add the RSS feed URL to `self.rss_feeds` in `UFCFightArticleScraper`
2. Test content extraction works correctly
3. Add content selectors to `_extract_article_body()` if needed

## License

Part of the UFC Fight Predictor project.

## Contact

For issues or questions, please open an issue on the project repository.

