#!/usr/bin/env python3
"""
Patches for fixing ROI calculation issues in ensemble_model_best.py
"""

def fix_profit_calculation(row, stake=100):
    """
    Fixed profit calculation for American odds
    """
    vegas_odds = row['avg_vegas_odds']
    won = row['win'] == 1
    
    if not won:
        return -stake
    
    if vegas_odds > 0:
        # Positive odds: win (odds/100) * stake
        return (vegas_odds / 100) * stake
    else:
        # Negative odds: win (100/abs(odds)) * stake
        return (100 / abs(vegas_odds)) * stake

def fix_odds_clamping(odds):
    """
    More reasonable odds clamping
    """
    if pd.isna(odds):
        return odds
    
    # Clamp to reasonable range but not too restrictive
    if odds > 0:
        return min(odds, 1000)  # Max +1000
    else:
        return max(odds, -1000)  # Min -1000

def improved_odds_filtering(df, vegas_cols, min_odds=-500, max_odds=500):
    """
    Improved odds filtering that's less aggressive
    """
    print(f"Original dataset size: {len(df)}")
    
    for col in vegas_cols:
        if col in df.columns:
            before = len(df)
            # Only filter extreme outliers, keep reasonable range
            df = df[(df[col] >= min_odds) & (df[col] <= max_odds) | df[col].isna()]
            after = len(df)
            print(f"Filtered {col}: {before} -> {after} rows")
    
    print(f"After odds filtering: {len(df)}")
    return df

def calculate_edge_metrics(picks):
    """
    Calculate additional edge metrics for better analysis
    """
    def american_odds_to_prob(odds):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    picks['implied_prob_vegas'] = picks['avg_vegas_odds'].apply(american_odds_to_prob)
    picks['implied_prob_model'] = picks['odds'].apply(american_odds_to_prob)
    picks['edge'] = picks['implied_prob_model'] - picks['implied_prob_vegas']
    
    return picks
