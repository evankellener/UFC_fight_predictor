"""
Predict all fights in a UFC event at once.

This script shows how to inference an entire UFC event (all fights)
using the rolling_ema feature consistently.

Key insight: All fights in the same event use the SAME rolling_ema value
because they happen on the same date!
"""

import pandas as pd
import numpy as np
from datetime import datetime

def predict_ufc_event(event_name, event_date, fights_list, historical_df, model, features):
    """
    Predict all fights in a UFC event at once.
    
    Args:
        event_name: Name of the event (e.g., "UFC 300: Pereira vs. Hill")
        event_date: Date of the event (string or datetime)
        fights_list: List of tuples [(fighter_a, fighter_b), ...]
        historical_df: Historical fight data with rolling_ema
        model: Trained model (e.g., from tune_xgboost_with_rolling_ema)
        features: List of feature names the model expects
        
    Returns:
        DataFrame with predictions for all fights in the event
    """
    event_date = pd.to_datetime(event_date)
    
    # Step 1: Get the rolling_ema value for this event
    # All fights in the event use the SAME value
    historical_df_sorted = historical_df.sort_values('DATE')
    past_data = historical_df_sorted[historical_df_sorted['DATE'] < event_date]
    
    if len(past_data) == 0:
        print(f"Warning: No historical data before {event_date}")
        event_rolling_ema = 0.5
    else:
        # Use the most recent postcomp_rolling_ema
        event_rolling_ema = past_data['postcomp_rolling_ema'].iloc[-1]
    
    print(f"\n{'='*80}")
    print(f"EVENT: {event_name}")
    print(f"DATE: {event_date.strftime('%Y-%m-%d')}")
    print(f"FIGHTS: {len(fights_list)}")
    print(f"{'='*80}")
    print(f"Using rolling_ema = {event_rolling_ema:.4f} for ALL fights in this event")
    print(f"{'='*80}\n")
    
    # Step 2: Build prediction rows for all fights
    all_predictions = []
    
    for fight_num, (fighter_a, fighter_b) in enumerate(fights_list, 1):
        # Get each fighter's most recent stats
        fighter_a_data = historical_df[historical_df['FIGHTER'] == fighter_a].sort_values('DATE')
        fighter_b_data = historical_df[historical_df['FIGHTER'] == fighter_b].sort_values('DATE')
        
        if len(fighter_a_data) == 0:
            print(f"⚠️  Warning: No historical data for {fighter_a}")
            continue
        if len(fighter_b_data) == 0:
            print(f"⚠️  Warning: No historical data for {fighter_b}")
            continue
        
        fighter_a_last = fighter_a_data.iloc[-1]
        fighter_b_last = fighter_b_data.iloc[-1]
        
        # Build feature row for Fighter A
        row_a = {
            'BOUT': f"{event_name}_Fight{fight_num}",
            'EVENT': event_name,
            'DATE': event_date,
            'FIGHTER': fighter_a,
            'OPPONENT': fighter_b,
        }
        
        # Add all precomp features for Fighter A
        for feature in features:
            if feature == 'precomp_rolling_ema' or feature == 'rolling_ema':
                # Global meta-game (SAME for all fighters)
                row_a[feature] = event_rolling_ema
            elif feature.startswith('opp_'):
                # Opponent features (Fighter B's stats)
                base_feature = feature.replace('opp_', '')
                postcomp_feature = base_feature.replace('precomp_', 'postcomp_')
                if postcomp_feature in fighter_b_last:
                    row_a[feature] = fighter_b_last[postcomp_feature]
                elif base_feature in fighter_b_last:
                    row_a[feature] = fighter_b_last[base_feature]
                else:
                    row_a[feature] = np.nan
            else:
                # Fighter A's own features
                postcomp_feature = feature.replace('precomp_', 'postcomp_')
                if postcomp_feature in fighter_a_last:
                    row_a[feature] = fighter_a_last[postcomp_feature]
                elif feature in fighter_a_last:
                    row_a[feature] = fighter_a_last[feature]
                else:
                    row_a[feature] = np.nan
        
        # Build feature row for Fighter B (reversed)
        row_b = {
            'BOUT': f"{event_name}_Fight{fight_num}",
            'EVENT': event_name,
            'DATE': event_date,
            'FIGHTER': fighter_b,
            'OPPONENT': fighter_a,
        }
        
        # Add all precomp features for Fighter B
        for feature in features:
            if feature == 'precomp_rolling_ema' or feature == 'rolling_ema':
                # Global meta-game (SAME for all fighters)
                row_b[feature] = event_rolling_ema
            elif feature.startswith('opp_'):
                # Opponent features (Fighter A's stats)
                base_feature = feature.replace('opp_', '')
                postcomp_feature = base_feature.replace('precomp_', 'postcomp_')
                if postcomp_feature in fighter_a_last:
                    row_b[feature] = fighter_a_last[postcomp_feature]
                elif base_feature in fighter_a_last:
                    row_b[feature] = fighter_a_last[base_feature]
                else:
                    row_b[feature] = np.nan
            else:
                # Fighter B's own features
                postcomp_feature = feature.replace('precomp_', 'postcomp_')
                if postcomp_feature in fighter_b_last:
                    row_b[feature] = fighter_b_last[postcomp_feature]
                elif feature in fighter_b_last:
                    row_b[feature] = fighter_b_last[feature]
                else:
                    row_b[feature] = np.nan
        
        all_predictions.append((row_a, row_b, fighter_a, fighter_b))
    
    # Step 3: Create DataFrame for all predictions
    all_rows = []
    for row_a, row_b, _, _ in all_predictions:
        all_rows.append(row_a)
        all_rows.append(row_b)
    
    prediction_df = pd.DataFrame(all_rows)
    
    # Step 4: Make predictions for ALL fights at once
    X_pred = prediction_df[features]
    probabilities = model.predict_proba(X_pred)[:, 1]
    
    # Step 5: Format results
    results = []
    for i, (row_a, row_b, fighter_a, fighter_b) in enumerate(all_predictions):
        prob_a = probabilities[i * 2]
        prob_b = probabilities[i * 2 + 1]
        
        # Normalize probabilities
        total = prob_a + prob_b
        prob_a_norm = prob_a / total
        prob_b_norm = prob_b / total
        
        results.append({
            'event': event_name,
            'date': event_date,
            'bout_num': i + 1,
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'prob_a_wins': prob_a_norm,
            'prob_b_wins': prob_b_norm,
            'favorite': fighter_a if prob_a_norm > prob_b_norm else fighter_b,
            'favorite_prob': max(prob_a_norm, prob_b_norm),
            'rolling_ema': event_rolling_ema  # Same for all
        })
    
    return pd.DataFrame(results)


def predict_ufc_event_simple(event_name, event_date, fights_list, fight_model):
    """
    Simplified version using FightOutcomeModel directly.
    
    Args:
        event_name: Name of the event
        event_date: Date of the event
        fights_list: List of tuples [(fighter_a, fighter_b), ...]
        fight_model: FightOutcomeModel instance (already trained)
        
    Returns:
        DataFrame with predictions
    """
    # Get the latest rolling_ema
    df = fight_model.df.sort_values('DATE')
    event_date_dt = pd.to_datetime(event_date)
    past_data = df[df['DATE'] < event_date_dt]
    
    if 'postcomp_rolling_ema' in df.columns:
        event_ema = past_data['postcomp_rolling_ema'].iloc[-1]
    elif 'rolling_ema' in df.columns:
        event_ema = past_data['rolling_ema'].iloc[-1]
    else:
        event_ema = 0.5
    
    print(f"\n{'='*80}")
    print(f"EVENT PREDICTION: {event_name}")
    print(f"{'='*80}")
    print(f"Date: {event_date}")
    print(f"Fights: {len(fights_list)}")
    print(f"rolling_ema for this event: {event_ema:.4f}")
    print(f"{'='*80}\n")
    
    results = []
    for i, (fighter_a, fighter_b) in enumerate(fights_list, 1):
        print(f"\nFight {i}: {fighter_a} vs {fighter_b}")
        print(f"  rolling_ema: {event_ema:.4f} (same for both fighters)")
        
        # Get each fighter's most recent data
        fa_data = df[df['FIGHTER'] == fighter_a].sort_values('DATE')
        fb_data = df[df['FIGHTER'] == fighter_b].sort_values('DATE')
        
        if len(fa_data) == 0 or len(fb_data) == 0:
            print(f"  ⚠️  Skipping: Missing historical data")
            continue
        
        fa_last = fa_data.iloc[-1]
        fb_last = fb_data.iloc[-1]
        
        # Show example of how both fighters get same rolling_ema
        if 'postcomp_elo' in fa_last:
            print(f"  {fighter_a}: precomp_elo={fa_last.get('postcomp_elo', 'N/A'):.0f}, rolling_ema={event_ema:.4f}")
            print(f"  {fighter_b}: precomp_elo={fb_last.get('postcomp_elo', 'N/A'):.0f}, rolling_ema={event_ema:.4f}")
        
        # Note: In practice, you'd build full feature vectors here
        # For now, we just demonstrate the concept
        results.append({
            'bout_num': i,
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'event_rolling_ema': event_ema
        })
    
    return pd.DataFrame(results)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("UFC EVENT PREDICTION EXAMPLE")
    print("="*80)
    
    # Example: UFC 300 (hypothetical)
    event_name = "UFC 300: Pereira vs. Hill"
    event_date = "2024-04-13"
    
    # Define all fights in the event
    fights = [
        ("Alex Pereira", "Jamahal Hill"),  # Main event
        ("Zhang Weili", "Yan Xiaonan"),    # Co-main
        ("Justin Gaethje", "Max Holloway"),
        ("Charles Oliveira", "Arman Tsarukyan"),
        ("Bo Nickal", "Cody Brundage"),
    ]
    
    print(f"\nEvent: {event_name}")
    print(f"Date: {event_date}")
    print(f"Number of fights: {len(fights)}")
    print("\nFights:")
    for i, (fa, fb) in enumerate(fights, 1):
        print(f"  {i}. {fa} vs {fb}")
    
    # Load historical data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv('data/tmp/final_with_rolling_ema_prepost.csv', 
                     parse_dates=['DATE'], 
                     low_memory=False)
    
    # Get the rolling_ema value for this event
    event_date_dt = pd.to_datetime(event_date)
    past_data = df[df['DATE'] < event_date_dt].sort_values('DATE')
    
    if len(past_data) > 0:
        event_ema = past_data['postcomp_rolling_ema'].iloc[-1]
        print(f"\nrolling_ema for {event_date}: {event_ema:.4f}")
        print(f"This value will be used for ALL {len(fights)} fights in the event!")
        
        # Show consistency
        print("\n" + "="*80)
        print("CONSISTENCY CHECK")
        print("="*80)
        print("\nAll fighters in all fights get the SAME rolling_ema:")
        for i, (fa, fb) in enumerate(fights, 1):
            print(f"\nFight {i}: {fa} vs {fb}")
            print(f"  {fa}: precomp_rolling_ema = {event_ema:.4f}")
            print(f"  {fb}: precomp_rolling_ema = {event_ema:.4f} (SAME)")
        
        print("\n" + "="*80)
        print("KEY INSIGHT")
        print("="*80)
        print("""
When predicting an entire UFC event:

1. ✅ ALL fights happen on the SAME date
2. ✅ ALL fighters use the SAME rolling_ema value
3. ✅ Each fighter still gets their own precomp_elo, precomp_strike_elo, etc.
4. ✅ This is MORE consistent than predicting one-by-one!

You can batch predict all fights at once - it's actually EASIER and FASTER!
        """)
        
        print("\n" + "="*80)
        print("IMPLEMENTATION")
        print("="*80)
        print("""
# Option 1: Use the predict_ufc_event function
from predict_ufc_event import predict_ufc_event

results = predict_ufc_event(
    event_name="UFC 300: Pereira vs. Hill",
    event_date="2024-04-13",
    fights_list=[("Alex Pereira", "Jamahal Hill"), ...],
    historical_df=df,
    model=trained_model,
    features=feature_list
)

# Option 2: Manual batch prediction
event_ema = df[df['DATE'] < event_date]['postcomp_rolling_ema'].iloc[-1]

# Build all prediction rows at once
all_rows = []
for fighter_a, fighter_b in fights_list:
    # Get each fighter's stats
    fa_last = df[df['FIGHTER'] == fighter_a].iloc[-1]
    fb_last = df[df['FIGHTER'] == fighter_b].iloc[-1]
    
    # Fighter A row
    all_rows.append({
        'precomp_elo': fa_last['postcomp_elo'],
        'opp_precomp_elo': fb_last['postcomp_elo'],
        # ... all other features ...
        'precomp_rolling_ema': event_ema  # SAME for all
    })
    
    # Fighter B row
    all_rows.append({
        'precomp_elo': fb_last['postcomp_elo'],
        'opp_precomp_elo': fa_last['postcomp_elo'],
        # ... all other features ...
        'precomp_rolling_ema': event_ema  # SAME for all
    })

# Predict all fights at once
X_pred = pd.DataFrame(all_rows)
predictions = model.predict_proba(X_pred)[:, 1]
        """)
    else:
        print(f"No historical data before {event_date}")

