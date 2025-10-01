#!/usr/bin/env python3
"""
Test the actual output format of the improved scraper
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ufc_news_features import FighterFeatures, write_features_csv

# Simulate what the actual scraper would produce with minimal/no data found
# This is more realistic - when no articles are found, it returns baseline values
sample_features = [
    FighterFeatures(
        fighter="Dominick Reyes",
        opponent="Carlos Ulberg", 
        event_date="2025-09-27",
        event="UFC Fight Night Perth",
        short_notice=0,  # No short notice found
        short_notice_duration_days=None,  # No duration found
        injury_risk=3,  # Baseline when no injury mentions found
        camp_status=5,  # Baseline when no camp mentions found
        weight_cut_risk=3,  # Baseline when no weight cut mentions found
        personal_issues=2,  # Baseline when no personal issues found
        evidence_urls=[]
    ),
    FighterFeatures(
        fighter="Carlos Ulberg",
        opponent="Dominick Reyes",
        event_date="2025-09-27", 
        event="UFC Fight Night Perth",
        short_notice=0,  # No short notice found
        short_notice_duration_days=None,  # No duration found
        injury_risk=3,  # Baseline when no injury mentions found
        camp_status=5,  # Baseline when no camp mentions found
        weight_cut_risk=3,  # Baseline when no weight cut mentions found
        personal_issues=2,  # Baseline when no personal issues found
        evidence_urls=[]
    )
]

# Write the realistic output
write_features_csv(sample_features, "realistic_output.csv")
print("Realistic output when scraper finds minimal/no data:")
print("This shows the baseline values that would be returned")
