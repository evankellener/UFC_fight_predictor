#!/usr/bin/env python3
"""
Debug script to test what search queries are generated and if they work
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ufc_news_features import CONFIG, search_web

def test_search_queries():
    fighter = "Dominick Reyes"
    opponent = "Carlos Ulberg"
    
    # Test the actual queries the scraper would use
    q_base = [
        f'"{fighter}" vs "{opponent}"',
        f'"{fighter}" "{opponent}" booked',
        f'"{fighter}" "{opponent}" announced',
        f'"{fighter}" "{opponent}" finalized',
        f'"{fighter}" short notice',
        f'"{fighter}" replaces',
        f'"{fighter}" withdraws',
        f'"{fighter}" camp',
        f'"{fighter}" injury',
    ]
    
    sites = [
        "mmafighting.com", "mmajunkie.usatoday.com", "mmamania.com",
        "bloodyelbow.com", "espn.com", "ufc.com", "sherdog.com", "tapology.com",
        "sports.yahoo.com", "si.com", "cbssports.com"
    ]
    q_site = [f'site:{s} "{fighter}" "{opponent}"' for s in sites]
    queries = q_base + q_site
    
    print("Search queries being generated:")
    for i, q in enumerate(queries[:10]):  # Show first 10
        print(f"{i+1}. {q}")
    
    print(f"\nTotal queries: {len(queries)}")
    print(f"Per query results: {CONFIG['search_results_per_query']}")
    print(f"Max articles per fighter: {CONFIG['max_articles_per_fighter']}")
    
    # Test a simple search
    print("\nTesting a simple search...")
    try:
        results = search_web([f'"{fighter}" "{opponent}"'], per_query=3)
        print(f"Found {len(results)} results")
        for i, r in enumerate(results[:3]):
            print(f"  {i+1}. {r.get('title', 'No title')[:80]}...")
            print(f"     URL: {r.get('url', 'No URL')[:80]}...")
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    test_search_queries()
