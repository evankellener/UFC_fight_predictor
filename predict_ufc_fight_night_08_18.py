"""
Predict UFC Fight Night: de Ridder vs. Allen - August 18, 2025
Generate betting recommendations for upcoming fights
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
print("BETTING PREDICTIONS")
print("="*80)
print()

# Configuration
BANKROLL = 10000  # Default bankroll
KELLY_FRACTION = 0.25  # 1/4 Kelly (conservative)
STRATEGY = 'balanced'  # 'aggressive', 'conservative', or 'balanced'

# Load upcoming fights
print("Loading upcoming fight card...")
upcoming_df = pd.read_csv('data/tmp/ufc_fight_night_08_18_25.csv')
print(f"✅ Loaded {len(upcoming_df)} fights")
print()

# Clean up the data
fights = []
for _, row in upcoming_df.iterrows():
    if pd.notna(row['Fighter 1']) and pd.notna(row['Fighter 2']):
        # Parse odds
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

print("Fight Card:")
print("-" * 80)
for i, fight in enumerate(fights, 1):
    print(f"{i:2d}. {fight['fighter_a']:25s} vs {fight['fighter_b']:25s} ({fight['weight_class']})")
print()

# Load historical database
print("="*80)
print("LOADING HISTORICAL DATA")
print("="*80)
print()

final_df = pd.read_csv('data/tmp/final.csv', low_memory=False)
final_df['DATE'] = pd.to_datetime(final_df['DATE'])

# Convert numeric columns to proper types
numeric_cols = [col for col in final_df.columns if col not in ['FIGHTER', 'EVENT', 'DATE', 'win', 'BOUT']]
for col in numeric_cols:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

print(f"✅ Loaded {len(final_df)} historical fight records")
print(f"   Date range: {final_df['DATE'].min().date()} to {final_df['DATE'].max().date()}")
print()

# Calculate rolling_ema for the full dataset
print("Calculating rolling EMA...")
df_sorted = final_df.sort_values('DATE').reset_index(drop=True)
df_sorted['win_numeric'] = pd.to_numeric(df_sorted['win'], errors='coerce')
rolling_ema_full = df_sorted['win_numeric'].ewm(span=200, min_periods=20).mean()
df_sorted['precomp_rolling_ema'] = rolling_ema_full.shift(1)
df_sorted['postcomp_rolling_ema'] = rolling_ema_full

# Get the most recent EMA value (this is what upcoming fights will use)
current_rolling_ema = df_sorted['postcomp_rolling_ema'].iloc[-1]
print(f"✅ Current rolling EMA: {current_rolling_ema:.4f}")
print()

# Create derived features
print("Creating derived features...")
df_sorted['precomp_strike_elo_diff'] = df_sorted['precomp_strike_elo'] - df_sorted['opp_precomp_strike_elo']
df_sorted['precomp_grapple_elo_diff'] = df_sorted['precomp_grapple_elo'] - df_sorted['opp_precomp_grapple_elo']
df_sorted['precomp_elo_diff'] = df_sorted['precomp_elo'] - df_sorted['opp_precomp_elo']
print("✅ Derived features created")
print()

# Load production model
print("="*80)
print("LOADING PRODUCTION MODEL")
print("="*80)
print()

model = joblib.load('saved_models/production_xgboost_champion_ema.joblib')
with open('saved_models/production_features.json', 'r') as f:
    feature_config = json.load(f)
    features = feature_config['features']

print(f"✅ Model loaded")
print(f"✅ Features loaded: {len(features)}")
print()

# Function to get fighter's most recent stats
def get_fighter_stats(fighter_name, df):
    """Get most recent stats for a fighter"""
    # Try exact match first
    fighter_rows = df[df['FIGHTER'] == fighter_name].copy()
    
    if len(fighter_rows) == 0:
        # Try case-insensitive match
        fighter_rows = df[df['FIGHTER'].str.lower() == fighter_name.lower()].copy()
    
    if len(fighter_rows) == 0:
        # Try partial match
        fighter_rows = df[df['FIGHTER'].str.contains(fighter_name, case=False, na=False)].copy()
    
    if len(fighter_rows) == 0:
        return None
    
    # Get most recent fight
    fighter_rows = fighter_rows.sort_values('DATE', ascending=False)
    most_recent = fighter_rows.iloc[0]
    
    return most_recent

# Generate predictions
print("="*80)
print("GENERATING PREDICTIONS")
print("="*80)
print()

predictions = []
fighters_not_found = []

for fight in fights:
    fighter_a_name = fight['fighter_a']
    fighter_b_name = fight['fighter_b']
    
    # Get stats for both fighters
    stats_a = get_fighter_stats(fighter_a_name, df_sorted)
    stats_b = get_fighter_stats(fighter_b_name, df_sorted)
    
    if stats_a is None:
        fighters_not_found.append(fighter_a_name)
        print(f"⚠️  Fighter not found: {fighter_a_name}")
        continue
    
    if stats_b is None:
        fighters_not_found.append(fighter_b_name)
        print(f"⚠️  Fighter not found: {fighter_b_name}")
        continue
    
    # Check if fighters have enough fights
    if stats_a.get('precomp_boutcount', 0) < 1:
        print(f"⚠️  {fighter_a_name} has insufficient fight history")
        continue
    
    if stats_b.get('precomp_boutcount', 0) < 1:
        print(f"⚠️  {fighter_b_name} has insufficient fight history")
        continue
    
    # Build feature vectors for both fighters
    # We need to create TWO rows: one for each fighter's perspective
    
    def build_features(stats_self, stats_opp):
        """Build feature vector from fighter's perspective"""
        feature_dict = {}
        
        # Differential features
        feature_dict['precomp_elo_diff'] = stats_self['precomp_elo'] - stats_opp['precomp_elo']
        feature_dict['precomp_strike_elo_diff'] = stats_self['precomp_strike_elo'] - stats_opp['precomp_strike_elo']
        feature_dict['precomp_grapple_elo_diff'] = stats_self['precomp_grapple_elo'] - stats_opp['precomp_grapple_elo']
        
        # Self stats
        feature_dict['precomp_legacc_perc5'] = stats_self['precomp_legacc_perc5']
        feature_dict['precomp_distacc_perc'] = stats_self['precomp_distacc_perc']
        feature_dict['precomp_tdavg3'] = stats_self['precomp_tdavg3']
        feature_dict['precomp_winsum'] = stats_self['precomp_winsum']
        feature_dict['precomp_groundacc_perc'] = stats_self['precomp_groundacc_perc']
        feature_dict['precomp_winsum5'] = stats_self['precomp_winsum5']
        feature_dict['precomp_ctrl_per_min'] = stats_self['precomp_ctrl_per_min']
        feature_dict['precomp_elo_change_5'] = stats_self['precomp_elo_change_5']
        feature_dict['age_ratio_difference'] = stats_self['age_ratio_difference']
        
        # Opponent stats (with opp_ prefix)
        feature_dict['opp_precomp_sigstr_pm5'] = stats_opp['precomp_sigstr_pm5']
        feature_dict['opp_precomp_grapple_strike_mix'] = stats_opp['precomp_grapple_strike_mix']
        feature_dict['opp_precomp_clinchacc_perc'] = stats_opp['precomp_clinchacc_perc']
        feature_dict['opp_age_ratio_difference'] = stats_opp['age_ratio_difference']
        feature_dict['opp_precomp_elo'] = stats_opp['precomp_elo']
        feature_dict['opp_precomp_winsum'] = stats_opp['precomp_winsum']
        feature_dict['opp_precomp_legacc_perc3'] = stats_opp['precomp_legacc_perc3']
        feature_dict['opp_precomp_str_eff_diff3'] = stats_opp['precomp_str_eff_diff3']
        feature_dict['opp_precomp_sapm3'] = stats_opp['precomp_sapm3']
        feature_dict['opp_precomp_ctrl_per_min'] = stats_opp['precomp_ctrl_per_min']
        feature_dict['opp_REACH'] = stats_opp['REACH']
        feature_dict['opp_precomp_strdef5'] = stats_opp['precomp_strdef5']
        feature_dict['opp_precomp_tdavg5'] = stats_opp['precomp_tdavg5']
        feature_dict['opp_precomp_headacc_perc5'] = stats_opp['precomp_headacc_perc5']
        feature_dict['opp_precomp_winsum3'] = stats_opp['precomp_winsum3']
        feature_dict['opp_precomp_groundacc_perc5'] = stats_opp['precomp_groundacc_perc5']
        
        # Temporal feature (same for all fights on this date)
        feature_dict['precomp_rolling_ema'] = current_rolling_ema
        
        return feature_dict
    
    # Build features for both fighters
    features_a = build_features(stats_a, stats_b)
    features_b = build_features(stats_b, stats_a)
    
    # Create dataframes
    X_a = pd.DataFrame([features_a])[features]
    X_b = pd.DataFrame([features_b])[features]
    
    # Get predictions
    prob_a = model.predict_proba(X_a)[0, 1]
    prob_b = model.predict_proba(X_b)[0, 1]
    
    # Store predictions
    predictions.append({
        'fighter': fighter_a_name,
        'opponent': fighter_b_name,
        'weight_class': fight['weight_class'],
        'model_prob': prob_a,
        'odds': fight['odds_a'],
        'last_fight_date': stats_a['DATE'].date(),
        'elo': stats_a['precomp_elo'],
        'total_fights': int(stats_a['precomp_boutcount'])
    })
    
    predictions.append({
        'fighter': fighter_b_name,
        'opponent': fighter_a_name,
        'weight_class': fight['weight_class'],
        'model_prob': prob_b,
        'odds': fight['odds_b'],
        'last_fight_date': stats_b['DATE'].date(),
        'elo': stats_b['precomp_elo'],
        'total_fights': int(stats_b['precomp_boutcount'])
    })

print(f"✅ Generated predictions for {len(predictions)//2} fights")
print()

if fighters_not_found:
    print("⚠️  Fighters not found in database:")
    for fighter in fighters_not_found:
        print(f"   - {fighter}")
    print()

# Helper functions
def american_to_implied_prob(odds):
    """Convert American odds to implied probability"""
    if odds is None or pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def calculate_kelly_bet(model_prob, vegas_odds, bankroll, kelly_fraction=0.25):
    """Calculate bet size using Kelly Criterion"""
    if vegas_odds is None or pd.isna(vegas_odds):
        return 0
    
    # Convert to decimal odds
    if vegas_odds > 0:
        decimal_odds = 1 + (vegas_odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(vegas_odds))
    
    # Calculate edge
    implied_prob = 1 / decimal_odds
    edge = model_prob - implied_prob
    
    if edge <= 0:
        return 0
    
    # Kelly formula
    kelly = edge / (decimal_odds - 1)
    bet_size = bankroll * kelly * kelly_fraction
    
    # Safety caps
    bet_size = min(bet_size, bankroll * 0.05)  # Max 5% per bet
    bet_size = max(bet_size, 0)
    
    return bet_size

# Betting analysis
print("="*80)
print("BETTING ANALYSIS WITH ODDS")
print("="*80)
print()

predictions_df = pd.DataFrame(predictions)

# Calculate implied probabilities and edges
predictions_df['implied_prob'] = predictions_df['odds'].apply(american_to_implied_prob)
predictions_df['edge'] = predictions_df['model_prob'] - predictions_df['implied_prob']
predictions_df['disagreement'] = predictions_df['edge'].abs()

# Identify underdogs and favorites
predictions_df['is_underdog'] = predictions_df['odds'] > 0

# Calculate bet sizes
predictions_df['bet_size'] = predictions_df.apply(
    lambda row: calculate_kelly_bet(row['model_prob'], row['odds'], BANKROLL, KELLY_FRACTION),
    axis=1
)

# Apply strategy filters
print(f"Strategy: {STRATEGY.upper()}")
print()

recommendations = []

for _, row in predictions_df.iterrows():
    if row['odds'] is None or pd.isna(row['odds']):
        continue
    
    should_bet = False
    reason = ""
    
    if STRATEGY == 'aggressive':
        # Bet all high confidence picks
        if row['model_prob'] > 0.50 and row['edge'] > 0:
            should_bet = True
            reason = "High confidence pick"
    
    elif STRATEGY == 'conservative':
        # Only bet underdogs with high disagreement
        if row['is_underdog'] and row['disagreement'] > 0.08 and row['model_prob'] > 0.50 and row['edge'] > 0:
            should_bet = True
            reason = "Underdog with >8% disagreement"
    
    else:  # balanced
        # Bet high confidence (>70%) OR underdog disagreements
        if row['model_prob'] > 0.70 and row['edge'] > 0:
            should_bet = True
            reason = "High confidence (>70%)"
        elif row['is_underdog'] and row['disagreement'] > 0.08 and row['model_prob'] > 0.50 and row['edge'] > 0:
            should_bet = True
            reason = "Underdog disagreement"
    
    if should_bet and row['bet_size'] >= 10:  # Minimum $10 bet
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
            'elo': row['elo'],
            'is_underdog': row['is_underdog']
        })

# Display all fights with analysis
print("="*80)
print("ALL FIGHTS ANALYSIS")
print("="*80)
print()

for i in range(0, len(predictions), 2):
    if i+1 < len(predictions):
        pred_a = predictions[i]
        pred_b = predictions[i+1]
        
        print(f"Fight {i//2 + 1}: {pred_a['fighter']} vs {pred_b['fighter']}")
        print(f"  Weight Class: {pred_a['weight_class']}")
        print()
        
        # Fighter A
        print(f"  {pred_a['fighter']:30s}")
        print(f"    Model Prob: {pred_a['model_prob']*100:5.1f}%", end="")
        if pred_a['odds'] is not None:
            implied = american_to_implied_prob(pred_a['odds'])
            edge_a = pred_a['model_prob'] - implied
            print(f" | Odds: {pred_a['odds']:+4.0f} (implied {implied*100:5.1f}%) | Edge: {edge_a:+.1%}")
        else:
            print(" | No odds")
        
        # Fighter B  
        print(f"  {pred_b['fighter']:30s}")
        print(f"    Model Prob: {pred_b['model_prob']*100:5.1f}%", end="")
        if pred_b['odds'] is not None:
            implied = american_to_implied_prob(pred_b['odds'])
            edge_b = pred_b['model_prob'] - implied
            print(f" | Odds: {pred_b['odds']:+4.0f} (implied {implied*100:5.1f}%) | Edge: {edge_b:+.1%}")
        else:
            print(" | No odds")
        
        print()
        print("-" * 80)
        print()

# Display betting recommendations
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
    print("  • Model probabilities are generally low for this card")
    print()
    print("💡 Recommendation: SKIP THIS EVENT or wait for better odds")
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
        print(f"   Potential Win: ${rec['bet_size'] * (abs(rec['odds'])/100 if rec['odds'] > 0 else 100/abs(rec['odds'])):.2f}")
        print()
    
    # Summary
    total_stake = sum(rec['bet_size'] for rec in recommendations)
    expected_wins = sum(rec['model_prob'] for rec in recommendations)
    
    print("="*80)
    print("BETTING SUMMARY")
    print("="*80)
    print(f"Total Bets:      {len(recommendations)}")
    print(f"Total Stake:     ${total_stake:.2f}")
    print(f"% of Bankroll:   {total_stake/BANKROLL*100:.2f}%")
    print(f"Avg Bet Size:    ${total_stake/len(recommendations):.2f}")
    print()
    print(f"Expected Wins:   {expected_wins:.1f} / {len(recommendations)} ({expected_wins/len(recommendations)*100:.1f}%)")
    print(f"Avg Edge:        {sum(rec['edge'] for rec in recommendations)/len(recommendations):.1%}")
    print()
    
    if total_stake > BANKROLL * 0.20:
        print("⚠️  WARNING: Total stake exceeds 20% of bankroll")
    elif total_stake > BANKROLL * 0.10:
        print("⚠️  CAUTION: Total stake is 10-20% of bankroll")
    else:
        print("✅ GOOD: Total stake is under 10% of bankroll")

print()
print("="*80)

