# UFC News Scraper Module

This module provides comprehensive news scraping and feature extraction capabilities for UFC fight prediction models. It scrapes news from major UFC outlets and extracts features that can enhance your existing prediction models.

## Features Extracted

### Basic Features
- **short_notice**: Boolean indicating if the fight was short notice
- **short_notice_duration**: Number of days between fight announcement and fight date
- **injury_risk**: Integer score (0-10) for injury concerns mentioned in news
- **camp_status**: Integer score (0-10) for training camp issues mentioned in news

### Enhanced Features
- **media_sentiment**: Sentiment score (-1 to 1) from news articles
- **controversy_score**: Controversy level (0-10) based on news mentions
- **training_mentions**: Number of training-related mentions
- **injury_mentions**: Number of injury-related mentions
- **camp_mentions**: Number of camp-related mentions
- **weight_cut_issues**: Score (0-10) for weight cutting problems
- **mental_state**: Mental/psychological state score (0-10)
- **opponent_analysis**: Level of opponent analysis in news (0-10)
- **fight_prediction_mentions**: Number of prediction mentions
- **news_frequency**: Articles per day leading to fight
- **early_news_sentiment**: Sentiment in early news coverage
- **late_news_sentiment**: Sentiment in late news coverage
- **sentiment_trend**: Trend in sentiment over time

## Installation

The module requires the following dependencies:

```bash
pip install requests beautifulsoup4 pandas numpy scikit-learn
```

## Quick Start

### Basic Usage

```python
from src.ufc_news_scraper import UFCNewsScraper

# Initialize scraper
scraper = UFCNewsScraper()

# Extract features for a specific fighter
fighter_name = "Jon Jones"
fight_date = "2023-03-04"

features = scraper.extract_fight_features(fighter_name, fight_date)

print(f"Short Notice: {features.short_notice}")
print(f"Injury Risk: {features.injury_risk}")
print(f"Camp Status: {features.camp_status}")
```

### Enhanced Features

```python
from src.enhanced_news_features import EnhancedNewsFeatureExtractor

# Initialize enhanced extractor
extractor = EnhancedNewsFeatureExtractor()

# Extract enhanced features
features = extractor.extract_enhanced_features(fighter_name, fight_date)

print(f"Media Sentiment: {features.media_sentiment}")
print(f"Controversy Score: {features.controversy_score}")
print(f"Mental State: {features.mental_state}")
```

### Integration with Existing Pipeline

```python
from src.news_integration_pipeline import NewsIntegratedPipeline

# Initialize pipeline
pipeline = NewsIntegratedPipeline(data_path="data/final.csv")

# Add news features to your dataset
df_with_news = pipeline.add_news_features_to_dataset(sample_size=100)

# Create enhanced features
df_enhanced = pipeline.create_enhanced_model_features(df_with_news)

# Analyze feature importance
importance_df = pipeline.get_feature_importance_with_news(df_enhanced)
```

## Detailed Usage

### 1. Basic News Scraper

The `UFCNewsScraper` class provides the core functionality for scraping news and extracting basic features.

```python
from src.ufc_news_scraper import UFCNewsScraper

scraper = UFCNewsScraper(delay_range=(1, 3))  # 1-3 second delay between requests

# Extract features for a single fight
features = scraper.extract_fight_features("Conor McGregor", "2021-01-23")

# Process multiple fights
fights = [("Jon Jones", "2023-03-04"), ("Amanda Nunes", "2023-06-10")]
results = scraper.batch_process_fights(fights, max_workers=3)

# Process entire dataset
df = pd.read_csv("data/final.csv")
df_with_news = scraper.process_fight_dataset(df, output_file="data/final_with_news.csv")
```

### 2. Enhanced News Features

The `EnhancedNewsFeatureExtractor` provides advanced feature extraction with sentiment analysis and temporal features.

```python
from src.enhanced_news_features import EnhancedNewsFeatureExtractor

extractor = EnhancedNewsFeatureExtractor(cache_dir="news_cache")

# Extract enhanced features
features = extractor.extract_enhanced_features("Jon Jones", "2023-03-04")

# Process dataset with enhanced features
df = pd.read_csv("data/final.csv")
df_enhanced = extractor.process_dataset_with_enhanced_features(
    df, 
    output_file="data/final_with_enhanced_news.csv"
)
```

### 3. Integration Pipeline

The `NewsIntegratedPipeline` provides a complete solution for integrating news features with your existing model.

```python
from src.news_integration_pipeline import NewsIntegratedPipeline

# Initialize pipeline
pipeline = NewsIntegratedPipeline(data_path="data/final.csv")

# Add news features
df_with_news = pipeline.add_news_features_to_dataset(sample_size=1000)

# Create enhanced features
df_enhanced = pipeline.create_enhanced_model_features(df_with_news)

# Analyze feature importance
importance_df = pipeline.get_feature_importance_with_news(df_enhanced)

# Compare model performance
performance = pipeline.compare_model_performance(df_enhanced)

# Generate comprehensive report
report = pipeline.generate_news_feature_report(df_enhanced)
```

## Configuration

### News Sources

The scraper supports multiple UFC news sources:

- MMA Fighting (mmafighting.com)
- MMA Junkie (mmajunkie.usatoday.com)
- Sherdog (sherdog.com)
- ESPN MMA (espn.com/mma)
- Bloody Elbow (bloodyelbow.com)

### Customization

You can customize the scraper by modifying the keyword sets:

```python
# Add custom keywords for feature extraction
scraper.injury_keywords.extend(['custom_injury_term'])
scraper.camp_keywords.extend(['custom_camp_term'])
```

### Caching

The module supports caching to avoid re-scraping the same data:

```python
# Enable caching
extractor = EnhancedNewsFeatureExtractor(cache_dir="news_cache")
```

## Testing

Run the test suite to verify everything is working:

```bash
python test_news_scraper.py
```

This will test:
- Basic news scraper functionality
- Enhanced feature extraction
- Integration pipeline
- Feature importance analysis
- Model performance comparison

## Performance Considerations

### Rate Limiting

The scraper includes built-in rate limiting to be respectful to news websites:

```python
scraper = UFCNewsScraper(delay_range=(1, 3))  # 1-3 second delay
```

### Parallel Processing

For large datasets, use parallel processing:

```python
# Process multiple fights in parallel
results = scraper.batch_process_fights(fights, max_workers=3)
```

### Caching

Enable caching to avoid re-scraping:

```python
extractor = EnhancedNewsFeatureExtractor(cache_dir="news_cache")
```

## Integration with Existing Models

### Adding News Features to Your Model

```python
# Load your existing model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# Add news features to your dataset
pipeline = NewsIntegratedPipeline(data_path="data/final.csv")
df_with_news = pipeline.add_news_features_to_dataset()

# Select features for training
feature_columns = [
    # Your existing features
    'precomp_elo', 'opp_precomp_elo', 'age', 'opp_age',
    # News features
    'short_notice', 'injury_risk', 'camp_status', 'media_sentiment'
]

X = df_with_news[feature_columns].fillna(0)
y = df_with_news['win']

# Train model
model.fit(X, y)
```

### Feature Importance Analysis

```python
# Analyze which news features are most important
importance_df = pipeline.get_feature_importance_with_news(df_with_news)

# Show top news features
news_features = importance_df[importance_df['feature_type'] == 'news']
print(news_features.head(10))
```

## Troubleshooting

### Common Issues

1. **No articles found**: This can happen for older fights or less popular fighters. The scraper will return default values.

2. **Rate limiting**: If you get blocked, increase the delay between requests:
   ```python
   scraper = UFCNewsScraper(delay_range=(3, 5))
   ```

3. **Memory issues**: For large datasets, process in batches:
   ```python
   df_sample = pipeline.add_news_features_to_dataset(sample_size=100)
   ```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

### Example 1: Basic Feature Extraction

```python
from src.ufc_news_scraper import UFCNewsScraper

scraper = UFCNewsScraper()
features = scraper.extract_fight_features("Jon Jones", "2023-03-04")

print(f"Short Notice: {features.short_notice}")
print(f"Injury Risk: {features.injury_risk}")
print(f"Camp Status: {features.camp_status}")
```

### Example 2: Enhanced Features

```python
from src.enhanced_news_features import EnhancedNewsFeatureExtractor

extractor = EnhancedNewsFeatureExtractor()
features = extractor.extract_enhanced_features("Jon Jones", "2023-03-04")

print(f"Media Sentiment: {features.media_sentiment}")
print(f"Controversy Score: {features.controversy_score}")
print(f"Mental State: {features.mental_state}")
```

### Example 3: Full Pipeline Integration

```python
from src.news_integration_pipeline import NewsIntegratedPipeline

pipeline = NewsIntegratedPipeline(data_path="data/final.csv")
df_with_news = pipeline.add_news_features_to_dataset(sample_size=1000)
df_enhanced = pipeline.create_enhanced_model_features(df_with_news)

# Analyze performance
performance = pipeline.compare_model_performance(df_enhanced)
print(f"Accuracy improvement: {performance['accuracy_improvement']:.3f}")
```

## Contributing

To add new news sources or improve feature extraction:

1. Add new sources to the `news_sources` dictionary in `UFCNewsScraper`
2. Add new keywords to the appropriate keyword sets
3. Implement new feature extraction methods in `EnhancedNewsFeatureExtractor`

## License

This module is part of the UFC Fight Predictor project and follows the same license terms.
