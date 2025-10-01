"""
Test different news scraping approaches

This script demonstrates both the real news scraper and the mock scraper
to help you choose the best approach for your use case.
"""

import sys
import os
sys.path.append('src')

def test_real_scraper():
    """Test the real news scraper"""
    print("Testing Real News Scraper")
    print("=" * 40)
    
    try:
        from ufc_news_scraper import scrape_fight_news
        
        # Test with a well-known fighter
        features = scrape_fight_news("Jon Jones", "2023-03-04")
        
        print(f"Real scraper results:")
        print(f"  Short Notice: {features.short_notice}")
        print(f"  Short Notice Duration: {features.short_notice_duration}")
        print(f"  Injury Risk: {features.injury_risk}")
        print(f"  Camp Status: {features.camp_status}")
        print(f"  Confidence: {features.confidence_score}")
        
        return True
        
    except Exception as e:
        print(f"Real scraper failed: {str(e)}")
        return False

def test_mock_scraper():
    """Test the mock news scraper"""
    print("\nTesting Mock News Scraper")
    print("=" * 40)
    
    try:
        from mock_news_scraper import scrape_fight_news
        
        # Test with the same fighter
        features = scrape_fight_news("Jon Jones", "2023-03-04")
        
        print(f"Mock scraper results:")
        print(f"  Short Notice: {features.short_notice}")
        print(f"  Short Notice Duration: {features.short_notice_duration}")
        print(f"  Injury Risk: {features.injury_risk}")
        print(f"  Camp Status: {features.camp_status}")
        print(f"  Confidence: {features.confidence_score}")
        
        return True
        
    except Exception as e:
        print(f"Mock scraper failed: {str(e)}")
        return False

def test_multiple_fighters():
    """Test with multiple fighters"""
    print("\nTesting Multiple Fighters")
    print("=" * 40)
    
    try:
        from mock_news_scraper import scrape_multiple_fights
        
        fights = [
            ("Jon Jones", "2023-03-04"),
            ("Amanda Nunes", "2023-06-10"),
            ("Conor McGregor", "2021-01-23"),
            ("Nate Diaz", "2022-09-10")
        ]
        
        results = scrape_multiple_fights(fights)
        
        for (fighter, date), features in results.items():
            print(f"\n{fighter} ({date}):")
            print(f"  Short Notice: {features.short_notice}")
            print(f"  Injury Risk: {features.injury_risk}")
            print(f"  Camp Status: {features.camp_status}")
            print(f"  Confidence: {features.confidence_score}")
        
        return True
        
    except Exception as e:
        print(f"Multiple fighters test failed: {str(e)}")
        return False

def test_dataset_integration():
    """Test integration with a dataset"""
    print("\nTesting Dataset Integration")
    print("=" * 40)
    
    try:
        import pandas as pd
        from mock_news_scraper import add_news_features_to_dataset
        
        # Create a small sample dataset
        sample_data = {
            'FIGHTER': ['Jon Jones', 'Amanda Nunes', 'Conor McGregor', 'Nate Diaz'],
            'DATE': ['2023-03-04', '2023-06-10', '2021-01-23', '2022-09-10'],
            'win': [1, 0, 1, 0]
        }
        
        df = pd.DataFrame(sample_data)
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        print(f"Original dataset shape: {df.shape}")
        
        # Add news features
        df_with_news = add_news_features_to_dataset(df)
        
        print(f"Dataset with news features shape: {df_with_news.shape}")
        print("\nNews features added:")
        news_cols = ['short_notice', 'short_notice_duration', 'injury_risk', 'camp_status', 'news_confidence']
        for col in news_cols:
            if col in df_with_news.columns:
                print(f"  {col}: {df_with_news[col].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"Dataset integration test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("News Scraper Testing Suite")
    print("=" * 50)
    
    # Test real scraper
    real_success = test_real_scraper()
    
    # Test mock scraper
    mock_success = test_mock_scraper()
    
    # Test multiple fighters
    multi_success = test_multiple_fighters()
    
    # Test dataset integration
    dataset_success = test_dataset_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Real Scraper: {'✓ PASS' if real_success else '✗ FAIL'}")
    print(f"Mock Scraper: {'✓ PASS' if mock_success else '✗ FAIL'}")
    print(f"Multiple Fighters: {'✓ PASS' if multi_success else '✗ FAIL'}")
    print(f"Dataset Integration: {'✓ PASS' if dataset_success else '✗ FAIL'}")
    
    print("\nRecommendations:")
    if real_success:
        print("✓ Real scraper is working - use for production")
    else:
        print("⚠ Real scraper is blocked - use mock scraper for development")
    
    if mock_success:
        print("✓ Mock scraper is working - good for testing and development")
    
    print("\nNext steps:")
    print("1. Use mock_news_scraper for development and testing")
    print("2. Try real scraper with different fighters/dates")
    print("3. Integrate with your existing model pipeline")

if __name__ == "__main__":
    main()
