"""
Predict Upcoming Fights - CLEAN VERSION
Uses the production_clean model (no rolling_ema, Version B filter)
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

print("="*80)
print("CLEAN UFC PREDICTIONS")
print("="*80)
print()

# Load upcoming fights
upcoming_df = pd.read_csv('data/tmp/ufc_fight_night_08_18_25.csv')

fights = []
for _, row in upcoming_df.iterrows():
    if pd.notna(row['Fighter 1']) and pd.notna(row['Fighter 2']):
        odds_a = pd.to_numeric(row.get('Fighter 1 Odds'), errors='coerce') if pd.notna(row.get('Fighter 1 Odds')) else None
        odds_b = pd.to_numeric(row.get('Fighter 2 Odds'), errors='coerce') if pd.notna(row.get('Fighter 2 Odds')) else None
        
        fights.append({
            'card': row['Card'],
            'weight_class': row['Weight Class'],
            'fighter_a': row['Fighter 1'].strip(),
            'fighter_b': row['Fighter 2'].strip(),
            'odds_a': odds_a,
            'odds_b': odds_b
        })

print(f"Upcoming fights: {len(fights)}")
print()

# Load historical database
final_df = pd.read_csv('data/tmp/final.csv', low_memory=False)
final_df['DATE'] = pd.to_datetime(final_df['DATE'])

# Convert numeric columns
numeric_cols = [col for col in final_df.columns if col not in ['FIGHTER', 'EVENT', 'DATE', 'win', 'BOUT', 'sex']]
for col in numeric_cols:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

print(f"Historical database: {len(final_df)} rows")
print(f"Date range: {final_df['DATE'].min().date()} to {final_df['DATE'].max().date()}")
print()

# Load production model
model = joblib.load('saved_models/production_clean_xgboost.joblib')
with open('saved_models/production_clean_features.json', 'r') as f:
    feature_config = json.load(f)
    required_features = feature_config['features']

print(f"Model loaded: {len(required_features)} features")
print()

# Function to get fighter's most recent stats
def get_fighter_last_fight_stats(fighter_name, df):
    """Get fighter's most recent postcomp stats"""
    fighter_rows = df[df['FIGHTER'].str.contains(fighter_name, case=False, na=False, regex=False)]
    
    if len(fighter_rows) == 0:
        return None
    
    fighter_rows = fighter_rows.sort_values('DATE', ascending=False)
    return fighter_rows.iloc[0]

# Function to apply ELO decay
def apply_elo_decay(elo, days_since_fight):
    """Apply 2.2% decay if >274 days since last fight"""
    if days_since_fight > 274:
        return elo * 0.978
    return elo

# Generate predictions
print("="*80)
print("PREDICTIONS")
print("="*80)
print()

predictions = []
prediction_date = pd.Timestamp('2025-08-18')

for fight_num, fight in enumerate(fights, 1):
    fighter_a_name = fight['fighter_a']
    fighter_b_name = fight['fighter_b']
    
    print(f"\n{'='*80}")
    print(f"FIGHT {fight_num}: {fighter_a_name} vs {fighter_b_name}")
    print(f"Weight Class: {fight['weight_class']}")
    print(f"{'='*80}\n")
    
    # Get last fights
    last_fight_a = get_fighter_last_fight_stats(fighter_a_name, final_df)
    last_fight_b = get_fighter_last_fight_stats(fighter_b_name, final_df)
    
    if last_fight_a is None:
        print(f"❌ {fighter_a_name} not found in database\n")
        continue
    
    if last_fight_b is None:
        print(f"❌ {fighter_b_name} not found in database\n")
        continue
    
    # Check bout count
    if last_fight_a.get('postcomp_boutcount', 0) < 1:
        print(f"❌ {fighter_a_name} has insufficient fight history\n")
        continue
    
    if last_fight_b.get('postcomp_boutcount', 0) < 1:
        print(f"❌ {fighter_b_name} has insufficient fight history\n")
        continue
    
    # Calculate days since last fight
    days_since_a = (prediction_date - last_fight_a['DATE']).days
    days_since_b = (prediction_date - last_fight_b['DATE']).days
    
    # Apply ELO decay
    elo_a = apply_elo_decay(last_fight_a['postcomp_elo'], days_since_a)
    elo_b = apply_elo_decay(last_fight_b['postcomp_elo'], days_since_b)
    
    strike_elo_a = apply_elo_decay(last_fight_a['postcomp_strike_elo'], days_since_a)
    strike_elo_b = apply_elo_decay(last_fight_b['postcomp_strike_elo'], days_since_b)
    
    grapple_elo_a = apply_elo_decay(last_fight_a['postcomp_grapple_elo'], days_since_a)
    grapple_elo_b = apply_elo_decay(last_fight_b['postcomp_grapple_elo'], days_since_b)
    
    print(f"📊 {fighter_a_name}:")
    print(f"   Last fight: {last_fight_a['DATE'].date()} ({days_since_a} days ago)")
    print(f"   ELO: {elo_a:.1f} (decay applied: {days_since_a > 274})")
    print(f"   Strike ELO: {strike_elo_a:.1f}")
    print(f"   Grapple ELO: {grapple_elo_a:.1f}")
    print(f"   Career: {int(last_fight_a['postcomp_boutcount'])} fights, {int(last_fight_a['postcomp_winsum'])} wins\n")
    
    print(f"📊 {fighter_b_name}:")
    print(f"   Last fight: {last_fight_b['DATE'].date()} ({days_since_b} days ago)")
    print(f"   ELO: {elo_b:.1f} (decay applied: {days_since_b > 274})")
    print(f"   Strike ELO: {strike_elo_b:.1f}")
    print(f"   Grapple ELO: {grapple_elo_b:.1f}")
    print(f"   Career: {int(last_fight_b['postcomp_boutcount'])} fights, {int(last_fight_b['postcomp_winsum'])} wins\n")
    
    # Build feature vectors
    features_a = {}
    features_b = {}
    
    # Differential features
    features_a['precomp_elo_diff'] = elo_a - elo_b
    features_a['precomp_strike_elo_diff'] = strike_elo_a - strike_elo_b
    features_a['precomp_grapple_elo_diff'] = grapple_elo_a - grapple_elo_b
    
    features_b['precomp_elo_diff'] = elo_b - elo_a
    features_b['precomp_strike_elo_diff'] = strike_elo_b - strike_elo_a
    features_b['precomp_grapple_elo_diff'] = grapple_elo_b - grapple_elo_a
    
    # Extract other features from postcomp stats
    for feature in required_features:
        if feature in ['precomp_elo_diff', 'precomp_strike_elo_diff', 'precomp_grapple_elo_diff']:
            continue  # Already calculated
        
        # Fighter A's features
        if feature.startswith('opp_'):
            # Opponent feature = Fighter B's stat
            base_feature = feature.replace('opp_', '')
            postcomp_feature = base_feature.replace('precomp_', 'postcomp_')
            
            if base_feature in last_fight_b.index:
                features_a[feature] = last_fight_b[base_feature]
            elif postcomp_feature in last_fight_b.index:
                features_a[feature] = last_fight_b[postcomp_feature]
            else:
                features_a[feature] = 0
        else:
            # Own feature
            postcomp_feature = feature.replace('precomp_', 'postcomp_')
            if feature in last_fight_a.index:
                features_a[feature] = last_fight_a[feature]
            elif postcomp_feature in last_fight_a.index:
                features_a[feature] = last_fight_a[postcomp_feature]
            else:
                features_a[feature] = 0
        
        # Fighter B's features (reversed)
        if feature.startswith('opp_'):
            base_feature = feature.replace('opp_', '')
            postcomp_feature = base_feature.replace('precomp_', 'postcomp_')
            
            if base_feature in last_fight_a.index:
                features_b[feature] = last_fight_a[base_feature]
            elif postcomp_feature in last_fight_a.index:
                features_b[feature] = last_fight_a[postcomp_feature]
            else:
                features_b[feature] = 0
        else:
            postcomp_feature = feature.replace('precomp_', 'postcomp_')
            if feature in last_fight_b.index:
                features_b[feature] = last_fight_b[feature]
            elif postcomp_feature in last_fight_b.index:
                features_b[feature] = last_fight_b[postcomp_feature]
            else:
                features_b[feature] = 0
    
    # Create DataFrames
    X_a = pd.DataFrame([features_a])[required_features]
    X_b = pd.DataFrame([features_b])[required_features]
    
    # Make predictions
    prob_a = model.predict_proba(X_a)[0, 1]
    prob_b = model.predict_proba(X_b)[0, 1]
    
    # Normalize to sum to 100%
    total = prob_a + prob_b
    prob_a_norm = prob_a / total
    prob_b_norm = prob_b / total
    
    print(f"🎯 PREDICTIONS:")
    print(f"   {fighter_a_name}: {prob_a_norm*100:.1f}%")
    print(f"   {fighter_b_name}: {prob_b_norm*100:.1f}%")
    print(f"   Sum: {(prob_a_norm + prob_b_norm)*100:.1f}% {'✅' if abs((prob_a_norm + prob_b_norm) - 1.0) < 0.01 else '❌'}\n")
    
    # Store predictions
    predictions.append({
        'fighter': fighter_a_name,
        'opponent': fighter_b_name,
        'model_prob': prob_a_norm,
        'odds': fight['odds_a']
    })
    
    predictions.append({
        'fighter': fighter_b_name,
        'opponent': fighter_a_name,
        'model_prob': prob_b_norm,
        'odds': fight['odds_b']
    })

print("\n" + "="*80)
print("PREDICTION SUMMARY")
print("="*80)
print(f"Successfully predicted {len(predictions)//2} fights")
print()
print("Model: Clean XGBoost (NO rolling_ema, Version B filter)")
print("Features: 28 champion features")
print("Training: 9,612 fights (2009-2025)")
print("="*80)

