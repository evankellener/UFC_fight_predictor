"""
Example: Integrating News Features with UFC Fight Predictor

This script demonstrates how to use the news scraper to enhance your existing
fight prediction model with news-based features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append('src')

from news_integration_pipeline import NewsIntegratedPipeline

def main():
    """Main example function"""
    print("UFC News Integration Example")
    print("=" * 50)
    
    # Initialize the news integration pipeline
    print("Initializing news integration pipeline...")
    pipeline = NewsIntegratedPipeline(data_path="data/final.csv")
    
    # Step 1: Add news features to a sample of fights
    print("\nStep 1: Adding news features to sample fights...")
    print("Note: This will scrape news for recent fights. It may take a while.")
    
    # Use a small sample for demonstration
    df_with_news = pipeline.add_news_features_to_dataset(sample_size=5)
    
    print(f"Dataset shape: {df_with_news.shape}")
    print(f"News features added: {[col for col in df_with_news.columns if col in pipeline.news_feature_columns]}")
    
    # Step 2: Create enhanced features
    print("\nStep 2: Creating enhanced features...")
    df_enhanced = pipeline.create_enhanced_model_features(df_with_news)
    
    # Show some sample data
    print("\nSample news features:")
    news_cols = [col for col in pipeline.news_feature_columns if col in df_enhanced.columns]
    if news_cols:
        sample_data = df_enhanced[['FIGHTER', 'DATE'] + news_cols].head(3)
        print(sample_data.to_string())
    
    # Step 3: Analyze feature importance
    print("\nStep 3: Analyzing feature importance...")
    try:
        importance_df = pipeline.get_feature_importance_with_news(df_enhanced)
        
        if not importance_df.empty:
            print("\nTop 5 Most Important Features:")
            top_features = importance_df.head(5)
            for _, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f} ({row['feature_type']})")
            
            # Show news features specifically
            news_features = importance_df[importance_df['feature_type'] == 'news']
            if not news_features.empty:
                print(f"\nNews Features Importance:")
                for _, row in news_features.head(3).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.3f}")
        else:
            print("No feature importance data available (insufficient data)")
            
    except Exception as e:
        print(f"Feature importance analysis failed: {str(e)}")
    
    # Step 4: Compare model performance
    print("\nStep 4: Comparing model performance...")
    try:
        performance = pipeline.compare_model_performance(df_enhanced)
        
        print(f"\nPerformance Comparison:")
        print(f"  Existing Features:")
        print(f"    Accuracy: {performance['existing_features']['accuracy']:.3f}")
        print(f"    AUC: {performance['existing_features']['auc']:.3f}")
        
        print(f"  With News Features:")
        print(f"    Accuracy: {performance['all_features']['accuracy']:.3f}")
        print(f"    AUC: {performance['all_features']['auc']:.3f}")
        
        print(f"  Improvements:")
        print(f"    Accuracy: {performance['accuracy_improvement']:+.3f}")
        print(f"    AUC: {performance['auc_improvement']:+.3f}")
        
    except Exception as e:
        print(f"Performance comparison failed: {str(e)}")
    
    # Step 5: Generate news feature report
    print("\nStep 5: Generating news feature report...")
    try:
        report = pipeline.generate_news_feature_report(df_enhanced)
        
        print(f"\nNews Feature Report:")
        print(f"  Total fights: {report['total_fights']}")
        print(f"  Fights with news: {report['fights_with_news']}")
        print(f"  News coverage rate: {report['news_coverage_rate']:.1%}")
        
        if 'short_notice_stats' in report:
            print(f"  Short notice fights: {report['short_notice_stats']['total_short_notice']}")
            print(f"  Short notice rate: {report['short_notice_stats']['short_notice_rate']:.1%}")
        
        if 'injury_risk_stats' in report:
            print(f"  High injury risk fights: {report['injury_risk_stats']['high_injury_risk']}")
            print(f"  Injury risk rate: {report['injury_risk_stats']['injury_risk_rate']:.1%}")
        
    except Exception as e:
        print(f"Report generation failed: {str(e)}")
    
    # Step 6: Save results
    print("\nStep 6: Saving results...")
    try:
        output_file = "data/final_with_news_features.csv"
        df_enhanced.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        # Also save a summary
        summary_file = "data/news_features_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("UFC News Features Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total fights processed: {len(df_enhanced)}\n")
            f.write(f"News features added: {len([col for col in df_enhanced.columns if col in pipeline.news_feature_columns])}\n")
            f.write(f"Enhanced features created: {len([col for col in df_enhanced.columns if col not in pipeline.all_feature_columns and col not in ['DATE', 'FIGHTER', 'EVENT', 'BOUT', 'win']])}\n")
            
            if not importance_df.empty:
                f.write(f"\nTop 5 Most Important Features:\n")
                for _, row in importance_df.head(5).iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.3f}\n")
        
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Review the generated CSV file with news features")
    print("2. Integrate news features into your existing model")
    print("3. Retrain your model with the enhanced feature set")
    print("4. Evaluate performance improvements")

if __name__ == "__main__":
    main()
