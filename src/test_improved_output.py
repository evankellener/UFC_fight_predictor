#!/usr/bin/env python3
"""
Test script to demonstrate the improved UFC news features output format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ufc_news_features import FighterFeatures, write_features_csv

# Create sample data that matches the improved format
sample_features = [
    FighterFeatures(
        fighter="Dominick Reyes",
        opponent="Carlos Ulberg", 
        event_date="2025-09-27",
        event="UFC Fight Night Perth",
        short_notice=0,
        short_notice_duration_days=60,
        injury_risk=6,
        camp_status=6,
        weight_cut_risk=4,
        personal_issues=2,
        evidence_urls=[]
    ),
    FighterFeatures(
        fighter="Carlos Ulberg",
        opponent="Dominick Reyes",
        event_date="2025-09-27", 
        event="UFC Fight Night Perth",
        short_notice=0,
        short_notice_duration_days=60,
        injury_risk=1,
        camp_status=8,
        weight_cut_risk=3,
        personal_issues=1,
        evidence_urls=[]
    )
]

# Write the improved output
write_features_csv(sample_features, "improved_features_demo.csv")
print("Created improved_features_demo.csv with enhanced features:")
print("- Added weight_cut_risk and personal_issues columns")
print("- Improved scoring ranges (1-10)")
print("- Better baseline scoring when no data found")
print("- Simplified output format matching your example")
