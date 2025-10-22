"""
Predict UFC Fight Night: de Ridder vs. Allen - August 18, 2025
CORRECTED VERSION: Uses postcomp stats from last fight as precomp stats for upcoming fight
"""
import sys
sys.path.insert(0, '/Users/evankellener/Desktop/UFC_fight_predictor/src')

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

print("="*80)
print("UFC FIGHT NIGHT: de Ridder vs. Allen - August 18, 2025")
print("CORRECTED BETTING PREDICTIONS")
print("="*80)
print()

# Configuration
BANKROLL = 10000
KELLY_FRACTION = 0.25
STRATEGY = 'balanced'

# Load upcoming fights
print("Loading upcoming fight card...")
upcoming_df = pd.read_csv('data/tmp/ufc_fight_night_08_18_25.csv')

fights = []
for _, row in upcoming_df.iterrows():
    if pd.notna(row['Fighter 1']) and pd.notna(row['Fighter 2']):
        odds_a = pd.to_numeric(row['Fighter 1 Odds'], errors='coerce') if pd.notna(row.get('Fighter 1 Odds')) else None
        odds_b = pd.to_numeric(row['Fighter 2 Odds'], errors='coerce') if pd.notna(row.get('Fighter 2 Odds')) else None
        
        fights.append({
            'card': row['Card'],
            'weight_class': row['Weight Class'],
            'fighter_a': row['Fighter 1'].strip(),
            'fighter_b': row['Fighter 2'].strip(),
            'odds_a': odds_a,
            'odds_b': odds_b
        })

print(f"✅ Loaded {len(fights)} fights")
print()

# Load historical database
print("="*80)
print("LOADING HISTORICAL DATA")
print("="*80)
print()

final_df = pd.read_csv('data/tmp/final.csv', low_memory=False)
final_df['DATE'] = pd.to_datetime(final_df['DATE'])

# Convert numeric columns
numeric_cols = [col for col in final_df.columns if col not in ['FIGHTER', 'EVENT', 'DATE', 'win', 'BOUT']]
for col in numeric_cols:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

print(f"✅ Loaded {len(final_df)} historical fight records")
print(f"   Date range: {final_df['DATE'].min().date()} to {final_df['DATE'].max().date()}")
print()

# Calculate rolling_ema
print("Calculating rolling EMA...")
df_sorted = final_df.sort_values('DATE').reset_index(drop=True)
df_sorted['win_numeric'] = pd.to_numeric(df_sorted['win'], errors='coerce')
rolling_ema_full = df_sorted['win_numeric'].ewm(span=200, min_periods=20).mean()
df_sorted['postcomp_rolling_ema'] = rolling_ema_full

current_rolling_ema = df_sorted['postcomp_rolling_ema'].iloc[-1]
print(f"✅ Current rolling EMA: {current_rolling_ema:.4f}")
print()

# Load production model
print("="*80)
print("LOADING PRODUCTION MODEL")
print("="*80)
print()

model = joblib.load('saved_models/production_xgboost_champion_ema.joblib')
with open('saved_models/production_features.json', 'r') as f:
    feature_config = json.load(f)
    required_features = feature_config['features']

print(f"✅ Model loaded")
print(f"✅ Required features: {len(required_features)}")
print()

# Function to get fighter's most recent postcomp stats
def get_fighter_postcomp_stats(fighter_name, df):
    """Get most recent POSTCOMP stats for a fighter (these become precomp for next fight)"""
    fighter_rows = df[df['FIGHTER'] == fighter_name].copy()
    
    if len(fighter_rows) == 0:
        fighter_rows = df[df['FIGHTER'].str.lower() == fighter_name.lower()].copy()
    
    if len(fighter_rows) == 0:
        fighter_rows = df[df['FIGHTER'].str.contains(fighter_name, case=False, na=False)].copy()
    
    if len(fighter_rows) == 0:
        return None, None
    
    fighter_rows = fighter_rows.sort_values('DATE', ascending=False)
    most_recent = fighter_rows.iloc[0]
    
    # Extract POSTCOMP stats (these become PRECOMP for next fight)
    postcomp_stats = {}
    for col in most_recent.index:
        if col.startswith('postcomp_'):
            postcomp_stats[col] = most_recent[col]
    
    return most_recent, postcomp_stats

# Helper functions
def american_to_implied_prob(odds):
    if odds is None or pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def calculate_kelly_bet(model_prob, vegas_odds, bankroll, kelly_fraction=0.25):
    if vegas_odds is None or pd.isna(vegas_odds):
        return 0
    
    if vegas_odds > 0:
        decimal_odds = 1 + (vegas_odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(vegas_odds))
    
    implied_prob = 1 / decimal_odds
    edge = model_prob - implied_prob
    
    if edge <= 0:
        return 0
    
    kelly = edge / (decimal_odds - 1)
    bet_size = bankroll * kelly * kelly_fraction
    bet_size = min(bet_size, bankroll * 0.05)
    bet_size = max(bet_size, 0)
    
    return bet_size

# Generate predictions
print("="*80)
print("GENERATING PREDICTIONS (WITH FULL STAT TRANSPARENCY)")
print("="*80)
print()

predictions = []
fighters_not_found = []

for fight_num, fight in enumerate(fights, 1):
    fighter_a_name = fight['fighter_a']
    fighter_b_name = fight['fighter_b']
    
    print("="*80)
    print(f"FIGHT {fight_num}: {fighter_a_name} vs {fighter_b_name}")
    print(f"Weight Class: {fight['weight_class']}")
    print("="*80)
    print()
    
    # Get stats for both fighters
    last_fight_a, postcomp_a = get_fighter_postcomp_stats(fighter_a_name, df_sorted)
    last_fight_b, postcomp_b = get_fighter_postcomp_stats(fighter_b_name, df_sorted)
    
    if last_fight_a is None:
        fighters_not_found.append(fighter_a_name)
        print(f"⚠️  {fighter_a_name} not found in database")
        print()
        continue
    
    if last_fight_b is None:
        fighters_not_found.append(fighter_b_name)
        print(f"⚠️  {fighter_b_name} not found in database")
        print()
        continue
    
    # Check for sufficient history
    if last_fight_a.get('postcomp_boutcount', 0) < 1:
        print(f"⚠️  {fighter_a_name} has insufficient fight history")
        print()
        continue
    
    if last_fight_b.get('postcomp_boutcount', 0) < 1:
        print(f"⚠️  {fighter_b_name} has insufficient fight history")
        print()
        continue
    
    # Print fighter stats
    print(f"📊 {fighter_a_name} (Last Fight: {last_fight_a['DATE'].date()}):")
    print(f"   postcomp_elo: {postcomp_a.get('postcomp_elo', 'N/A'):.1f}")
    print(f"   postcomp_strike_elo: {postcomp_a.get('postcomp_strike_elo', 'N/A'):.1f}")
    print(f"   postcomp_grapple_elo: {postcomp_a.get('postcomp_grapple_elo', 'N/A'):.1f}")
    print(f"   postcomp_boutcount: {postcomp_a.get('postcomp_boutcount', 'N/A'):.0f}")
    print(f"   postcomp_winsum: {postcomp_a.get('postcomp_winsum', 'N/A'):.0f}")
    print()
    
    print(f"📊 {fighter_b_name} (Last Fight: {last_fight_b['DATE'].date()}):")
    print(f"   postcomp_elo: {postcomp_b.get('postcomp_elo', 'N/A'):.1f}")
    print(f"   postcomp_strike_elo: {postcomp_b.get('postcomp_strike_elo', 'N/A'):.1f}")
    print(f"   postcomp_grapple_elo: {postcomp_b.get('postcomp_grapple_elo', 'N/A'):.1f}")
    print(f"   postcomp_boutcount: {postcomp_b.get('postcomp_boutcount', 'N/A'):.0f}")
    print(f"   postcomp_winsum: {postcomp_b.get('postcomp_winsum', 'N/A'):.0f}")
    print()
    
    # Build feature vectors for Fighter A's perspective
    features_a = {}
    
    # Differential features (A - B)
    features_a['precomp_elo_diff'] = postcomp_a.get('postcomp_elo', 0) - postcomp_b.get('postcomp_elo', 0)
    features_a['precomp_strike_elo_diff'] = postcomp_a.get('postcomp_strike_elo', 0) - postcomp_b.get('postcomp_strike_elo', 0)
    features_a['precomp_grapple_elo_diff'] = postcomp_a.get('postcomp_grapple_elo', 0) - postcomp_b.get('postcomp_grapple_elo', 0)
    
    # Fighter A's own stats (postcomp → precomp)
    for key in postcomp_a.keys():
        precomp_key = key.replace('postcomp_', 'precomp_')
        if precomp_key in required_features and not precomp_key.startswith('opp_'):
            features_a[precomp_key] = postcomp_a[key]
    
    # Opponent stats (Fighter B's postcomp → opp_precomp for A)
    for key in postcomp_b.keys():
        opp_key = key.replace('postcomp_', 'opp_precomp_')
        if opp_key in required_features:
            features_a[opp_key] = postcomp_b[key]
    
    # Add static features (REACH, age_ratio_difference, etc.)
    if 'age_ratio_difference' in required_features:
        features_a['age_ratio_difference'] = last_fight_a.get('age_ratio_difference', 0)
    if 'opp_age_ratio_difference' in required_features:
        features_a['opp_age_ratio_difference'] = last_fight_b.get('age_ratio_difference', 0)
    if 'opp_REACH' in required_features:
        features_a['opp_REACH'] = last_fight_b.get('REACH', 0)
    
    # Add rolling EMA
    features_a['precomp_rolling_ema'] = current_rolling_ema
    
    # Build feature vectors for Fighter B's perspective (reversed)
    features_b = {}
    
    features_b['precomp_elo_diff'] = postcomp_b.get('postcomp_elo', 0) - postcomp_a.get('postcomp_elo', 0)
    features_b['precomp_strike_elo_diff'] = postcomp_b.get('postcomp_strike_elo', 0) - postcomp_a.get('postcomp_strike_elo', 0)
    features_b['precomp_grapple_elo_diff'] = postcomp_b.get('postcomp_grapple_elo', 0) - postcomp_a.get('postcomp_grapple_elo', 0)
    
    for key in postcomp_b.keys():
        precomp_key = key.replace('postcomp_', 'precomp_')
        if precomp_key in required_features and not precomp_key.startswith('opp_'):
            features_b[precomp_key] = postcomp_b[key]
    
    for key in postcomp_a.keys():
        opp_key = key.replace('postcomp_', 'opp_precomp_')
        if opp_key in required_features:
            features_b[opp_key] = postcomp_a[key]
    
    # Add static features for Fighter B
    if 'age_ratio_difference' in required_features:
        features_b['age_ratio_difference'] = last_fight_b.get('age_ratio_difference', 0)
    if 'opp_age_ratio_difference' in required_features:
        features_b['opp_age_ratio_difference'] = last_fight_a.get('age_ratio_difference', 0)
    if 'opp_REACH' in required_features:
        features_b['opp_REACH'] = last_fight_a.get('REACH', 0)
    
    features_b['precomp_rolling_ema'] = current_rolling_ema
    
    # Print calculated matchup features
    print("🔢 Calculated Matchup Features:")
    print(f"   precomp_elo_diff (A-B): {features_a['precomp_elo_diff']:+.1f}")
    print(f"   precomp_strike_elo_diff (A-B): {features_a['precomp_strike_elo_diff']:+.1f}")
    print(f"   precomp_grapple_elo_diff (A-B): {features_a['precomp_grapple_elo_diff']:+.1f}")
    print(f"   precomp_rolling_ema: {current_rolling_ema:.4f}")
    print()
    
    # Create DataFrames
    X_a = pd.DataFrame([features_a])[required_features]
    X_b = pd.DataFrame([features_b])[required_features]
    
    # Check for missing features
    missing_a = X_a.isna().sum().sum()
    missing_b = X_b.isna().sum().sum()
    
    if missing_a > 0 or missing_b > 0:
        print(f"⚠️  Missing features: {missing_a} for {fighter_a_name}, {missing_b} for {fighter_b_name}")
        print()
        continue
    
    # Make predictions
    prob_a = model.predict_proba(X_a)[0, 1]
    prob_b = model.predict_proba(X_b)[0, 1]
    
    print("🎯 PREDICTIONS:")
    print(f"   {fighter_a_name}: {prob_a*100:.1f}%")
    print(f"   {fighter_b_name}: {prob_b*100:.1f}%")
    print(f"   Sum: {(prob_a + prob_b)*100:.1f}% {'✅' if abs((prob_a + prob_b) - 1.0) < 0.01 else '❌ PROBLEM!'}")
    print()
    
    # Store predictions
    predictions.append({
        'fighter': fighter_a_name,
        'opponent': fighter_b_name,
        'weight_class': fight['weight_class'],
        'model_prob': prob_a,
        'odds': fight['odds_a'],
        'last_fight_date': last_fight_a['DATE'].date(),
        'elo': postcomp_a.get('postcomp_elo', 0),
        'total_fights': int(postcomp_a.get('postcomp_boutcount', 0))
    })
    
    predictions.append({
        'fighter': fighter_b_name,
        'opponent': fighter_a_name,
        'weight_class': fight['weight_class'],
        'model_prob': prob_b,
        'odds': fight['odds_b'],
        'last_fight_date': last_fight_b['DATE'].date(),
        'elo': postcomp_b.get('postcomp_elo', 0),
        'total_fights': int(postcomp_b.get('postcomp_boutcount', 0))
    })

print("="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Generated predictions for {len(predictions)//2} fights")
print()

if fighters_not_found:
    print("⚠️  Fighters not found:")
    for fighter in fighters_not_found:
        print(f"   - {fighter}")
    print()

if len(predictions) == 0:
    print("❌ No valid predictions generated")
    sys.exit(0)

# Betting analysis
predictions_df = pd.DataFrame(predictions)
predictions_df['implied_prob'] = predictions_df['odds'].apply(american_to_implied_prob)
predictions_df['edge'] = predictions_df['model_prob'] - predictions_df['implied_prob']
predictions_df['disagreement'] = predictions_df['edge'].abs()
predictions_df['is_underdog'] = predictions_df['odds'] > 0
predictions_df['bet_size'] = predictions_df.apply(
    lambda row: calculate_kelly_bet(row['model_prob'], row['odds'], BANKROLL, KELLY_FRACTION),
    axis=1
)

# Apply strategy
recommendations = []

for _, row in predictions_df.iterrows():
    if row['odds'] is None or pd.isna(row['odds']):
        continue
    
    should_bet = False
    reason = ""
    
    if STRATEGY == 'aggressive':
        if row['model_prob'] > 0.50 and row['edge'] > 0:
            should_bet = True
            reason = "High confidence pick"
    elif STRATEGY == 'conservative':
        if row['is_underdog'] and row['disagreement'] > 0.08 and row['model_prob'] > 0.50 and row['edge'] > 0:
            should_bet = True
            reason = "Underdog with >8% disagreement"
    else:  # balanced
        if row['model_prob'] > 0.70 and row['edge'] > 0:
            should_bet = True
            reason = "High confidence (>70%)"
        elif row['is_underdog'] and row['disagreement'] > 0.08 and row['model_prob'] > 0.50 and row['edge'] > 0:
            should_bet = True
            reason = "Underdog disagreement"
    
    if should_bet and row['bet_size'] >= 10:
        recommendations.append({
            'fighter': row['fighter'],
            'opponent': row['opponent'],
            'weight_class': row['weight_class'],
            'model_prob': row['model_prob'],
            'odds': row['odds'],
            'implied_prob': row['implied_prob'],
            'edge': row['edge'],
            'bet_size': row['bet_size'],
            'reason': reason,
            'is_underdog': row['is_underdog']
        })

# Display recommendations
print("="*80)
print("🎯 BETTING RECOMMENDATIONS")
print("="*80)
print()

if len(recommendations) == 0:
    print("❌ NO BETS RECOMMENDED")
    print()
    print("Reasons:")
    print("  • No high-confidence picks (>70%)")
    print("  • No underdog opportunities with >8% disagreement")
    print()
else:
    for i, rec in enumerate(recommendations, 1):
        print(f"{'='*80}")
        print(f"BET #{i}: {rec['fighter']}")
        print(f"{'='*80}")
        print(f"Opponent:        {rec['opponent']}")
        print(f"Weight Class:    {rec['weight_class']}")
        print(f"Model Prob:      {rec['model_prob']:.1%}")
        print(f"Vegas Odds:      {rec['odds']:+.0f} (implied {rec['implied_prob']:.1%})")
        print(f"Edge:            {rec['edge']:+.1%}")
        print(f"Type:            {'UNDERDOG 🔸' if rec['is_underdog'] else 'FAVORITE ⭐'}")
        print(f"Reason:          {rec['reason']}")
        print()
        print(f"💰 RECOMMENDED BET: ${rec['bet_size']:.2f}")
        if rec['odds'] > 0:
            potential_win = rec['bet_size'] * (rec['odds']/100)
        else:
            potential_win = rec['bet_size'] * (100/abs(rec['odds']))
        print(f"   Potential Win: ${potential_win:.2f}")
        print()
    
    total_stake = sum(rec['bet_size'] for rec in recommendations)
    expected_wins = sum(rec['model_prob'] for rec in recommendations)
    
    print("="*80)
    print("BETTING SUMMARY")
    print("="*80)
    print(f"Total Bets:      {len(recommendations)}")
    print(f"Total Stake:     ${total_stake:.2f}")
    print(f"% of Bankroll:   {total_stake/BANKROLL*100:.2f}%")
    print(f"Expected Wins:   {expected_wins:.1f} / {len(recommendations)} ({expected_wins/len(recommendations)*100:.1f}%)")
    print()

print("="*80)

