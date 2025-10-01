# How to Use the UFC Fight Article Scraper

## ✅ New Features Added

Your scraper now has all the features you requested:

1. ✅ **Input specific upcoming fights** - Interactive prompt or programmatic input
2. ✅ **2-month lookback** - Scrapes articles from 60 days before the fight
3. ✅ **Article limit** - Caps at 10 articles by default
4. ✅ **Prioritizes recent articles** - Sorts by date, most recent first

## 🚀 Quick Start

### Method 1: Interactive Input (Easiest)

```bash
python scrape_fight.py
```

Then enter when prompted:
- Fighter 1: `Dominic Reyes`
- Fighter 2: `Carlos Ulberg`
- Fight date: `09/27/25` (or `2025-09-27`)
- Max articles: `10` (or press Enter for default)
- Days before: `60` (or press Enter for default)

### Method 2: In Your Code

```python
from article_scraper import UFCFightArticleScraper

scraper = UFCFightArticleScraper()

# Your example: Dominic Reyes vs Carlos Ulberg 09/27/25
articles = scraper.scrape_fight_articles(
    fighter1_name="Dominic Reyes",
    fighter2_name="Carlos Ulberg",
    fight_date="2025-09-27",
    days_before=60,      # 2 months before fight
    max_articles=10      # Limit to 10 most recent articles
)

# Articles are sorted by date (most recent first)
for article in articles:
    print(f"Title: {article.title}")
    print(f"Date: {article.published_date}")
    print(f"Content: {article.full_content}")
    print()
```

## 📋 Parameters Explained

### `scrape_fight_articles()` Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fighter1_name` | str | Required | First fighter's full name |
| `fighter2_name` | str | Required | Second fighter's full name |
| `fight_date` | str | Required | Fight date in YYYY-MM-DD format |
| `days_before` | int | `60` | Days before fight to search (2 months) |
| `max_articles` | int | `10` | Maximum articles to return |

### How It Works:

1. **Searches from fight date backwards** - If fight is on 09/27/25, searches from 09/27/25 back to ~07/28/25
2. **Finds all matching articles** - Articles mentioning BOTH fighters
3. **Sorts by date** - Most recent articles first
4. **Limits results** - Returns only the top N most recent articles
5. **Fetches full content** - Gets complete article text from each URL

## 🎯 Your Exact Use Case

### Example: Dominic Reyes vs Carlos Ulberg (09/27/25)

```python
from article_scraper import UFCFightArticleScraper

scraper = UFCFightArticleScraper(delay_between_requests=2.0)

# Scrape articles about the fight
articles = scraper.scrape_fight_articles(
    fighter1_name="Dominic Reyes",
    fighter2_name="Carlos Ulberg",
    fight_date="2025-09-27",  # Converted from 09/27/25
    days_before=60,           # ~2 months before (starting ~07/28/25)
    max_articles=10           # Cap at 10 most recent articles
)

if articles:
    print(f"Found {len(articles)} articles (most recent first):")
    
    for i, article in enumerate(articles, 1):
        date = article.published_date.strftime('%Y-%m-%d')
        print(f"\n{i}. [{date}] {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Length: {len(article.full_content)} characters")
        
        # Get summary
        summary = scraper.summarize_article(article, max_sentences=3)
        print(f"   Summary: {summary}")
else:
    print("No articles found yet - check back closer to the fight date")
```

## 📅 Date Formats Supported

The scraper accepts dates in multiple formats:

- **YYYY-MM-DD**: `2025-09-27` ✅ (preferred)
- **MM/DD/YY**: `09/27/25` ✅ (auto-converted)
- **MM/DD/YYYY**: `09/27/2025` ✅ (auto-converted)

The interactive script (`scrape_fight.py`) handles conversion automatically.

## 🔧 Customization

### Change Article Limit

```python
# Get more articles
articles = scraper.scrape_fight_articles(
    fighter1_name="Fighter A",
    fighter2_name="Fighter B",
    fight_date="2025-09-27",
    max_articles=20  # Get top 20 most recent
)

# Get all articles (no limit)
articles = scraper.scrape_fight_articles(
    fighter1_name="Fighter A",
    fighter2_name="Fighter B",
    fight_date="2025-09-27",
    max_articles=None  # No limit
)
```

### Change Lookback Period

```python
# 1 month lookback
articles = scraper.scrape_fight_articles(
    fighter1_name="Fighter A",
    fighter2_name="Fighter B",
    fight_date="2025-09-27",
    days_before=30  # 1 month
)

# 3 months lookback
articles = scraper.scrape_fight_articles(
    fighter1_name="Fighter A",
    fighter2_name="Fighter B",
    fight_date="2025-09-27",
    days_before=90  # 3 months
)
```

## 📊 What You Get

### Article Object Properties:

```python
for article in articles:
    # Basic info
    article.title              # "Reyes vs Ulberg: Fight Preview"
    article.url                # "https://mmafighting.com/..."
    article.source             # "MMA Fighting"
    article.published_date     # datetime(2025, 9, 15, ...)
    
    # Content
    article.full_content       # Complete article text (1000-3000+ chars)
    article.summary            # RSS summary (short)
    
    # Metadata
    article.fighters_mentioned # ["Dominic Reyes", "Carlos Ulberg"]
```

### Article Sorting:

Articles are **automatically sorted by date** (most recent first):

```python
articles = scraper.scrape_fight_articles(...)

# Articles are already sorted:
# articles[0] = Most recent article
# articles[1] = Second most recent
# articles[9] = 10th most recent
```

## 💡 Pro Tips

### 1. For Upcoming Fights

The scraper works best when run **closer to the fight date**:

```python
# If fight is on 09/27/25, run the scraper:
# - 2 months before (07/27/25) = May find 0-2 articles
# - 1 month before (08/27/25) = May find 3-5 articles
# - 1 week before (09/20/25) = Will find 8-15+ articles
```

### 2. Batch Processing

Scrape multiple fights at once:

```python
fights = [
    ("Dominic Reyes", "Carlos Ulberg", "2025-09-27"),
    ("Fighter A", "Fighter B", "2025-10-15"),
    ("Fighter C", "Fighter D", "2025-11-20"),
]

scraper = UFCFightArticleScraper()

for fighter1, fighter2, date in fights:
    articles = scraper.scrape_fight_articles(
        fighter1, fighter2, date,
        days_before=60,
        max_articles=10
    )
    print(f"{fighter1} vs {fighter2}: Found {len(articles)} articles")
```

### 3. Save Results

```python
import json

articles = scraper.scrape_fight_articles(...)

# Save to JSON
articles_data = [{
    'title': a.title,
    'url': a.url,
    'source': a.source,
    'date': a.published_date.isoformat() if a.published_date else None,
    'content': a.full_content,
    'summary': scraper.summarize_article(a, 3)
} for a in articles]

with open('fight_articles.json', 'w') as f:
    json.dump(articles_data, f, indent=2)
```

## 🎬 Complete Example

```python
#!/usr/bin/env python3
from article_scraper import UFCFightArticleScraper
from summarizer import create_comprehensive_summary

# Initialize
scraper = UFCFightArticleScraper(delay_between_requests=2.0)

# Your query: Dominic Reyes vs Carlos Ulberg 09/27/25
fighter1 = "Dominic Reyes"
fighter2 = "Carlos Ulberg"
fight_date = "2025-09-27"

print(f"Scraping articles for {fighter1} vs {fighter2}")

# Scrape with your requirements:
# - 2 months lookback (60 days)
# - Max 10 articles
# - Prioritize most recent
articles = scraper.scrape_fight_articles(
    fighter1_name=fighter1,
    fighter2_name=fighter2,
    fight_date=fight_date,
    days_before=60,
    max_articles=10
)

if articles:
    print(f"\n✅ Found {len(articles)} articles\n")
    
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article.title}")
        print(f"   Date: {article.published_date.strftime('%Y-%m-%d')}")
        print(f"   Source: {article.source}")
        print(f"   URL: {article.url}")
        
        # Get comprehensive summary
        if article.full_content:
            summary = create_comprehensive_summary(
                article.full_content,
                [fighter1, fighter2]
            )
            print(f"\n{summary}\n")
            print("-" * 80)
else:
    print("No articles found - fight may not be announced yet")
```

## 🚀 Run the Interactive Version

The easiest way to use this:

```bash
cd ufc_news_pipeline
source ../ufc_env/bin/activate
python scrape_fight.py
```

Then enter your fight details when prompted!

---

## ✅ Summary

You now have a scraper that:

✅ Takes **2 fighter names + fight date** as input  
✅ Searches **60 days (2 months) before the fight**  
✅ **Caps at 10 articles** by default  
✅ **Prioritizes most recent** articles first  
✅ Returns **full article content** for analysis  

Perfect for your use case! 🥊📰

