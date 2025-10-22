"""
Add rolling_ema feature with proper handling for both training and inference.

This script adds rolling_ema to your dataset in a way that's compatible
with your existing precomp/postcomp framework.
"""

import pandas as pd
import numpy as np

def add_rolling_ema_for_training_and_inference(df):
    """
    Add rolling_ema feature that works for both training and inference.
    
    Key differences from fighter stats:
    - rolling_ema is a GLOBAL temporal indicator (same for all fighters at a given time)
    - It represents the meta-game state, not individual fighter performance
    - For inference, you use the LATEST value, not a fighter-specific value
    
    Returns:
        DataFrame with 'rolling_ema' column added
    """
    df = df.sort_values('DATE').copy()
    
    # Convert win column to numeric
    df['win'] = pd.to_numeric(df['win'], errors='coerce')
    
    # Calculate rolling EMA (365-day window)
    # This is the UFC-wide win rate trend
    df['rolling_ema'] = (
        df.groupby(df['DATE'].dt.date)['win']
        .transform('mean')
        .ewm(span=365, min_periods=50)
        .mean()
        .shift(1)  # Prevent data leakage
    )
    
    # Fill NaN values at the start
    df['rolling_ema'].fillna(0.5, inplace=True)
    
    return df


def get_inference_rolling_ema(df, prediction_date=None):
    """
    Get the rolling_ema value to use for predicting upcoming fights.
    
    Args:
        df: DataFrame with rolling_ema already calculated
        prediction_date: Date of the upcoming fight (optional, defaults to latest)
        
    Returns:
        float: The rolling_ema value to use for prediction
    """
    df_sorted = df.sort_values('DATE').copy()
    
    if prediction_date is None:
        # Use the most recent value
        latest_ema = df_sorted['rolling_ema'].iloc[-1]
        latest_date = df_sorted['DATE'].iloc[-1]
        print(f"Using latest rolling_ema: {latest_ema:.4f} (as of {latest_date})")
        return latest_ema
    else:
        # Get the value closest to but before the prediction date
        prediction_date = pd.to_datetime(prediction_date)
        past_data = df_sorted[df_sorted['DATE'] < prediction_date]
        
        if len(past_data) == 0:
            print(f"Warning: No data before {prediction_date}, using default 0.5")
            return 0.5
        
        latest_ema = past_data['rolling_ema'].iloc[-1]
        latest_date = past_data['DATE'].iloc[-1]
        print(f"Using rolling_ema: {latest_ema:.4f} (as of {latest_date} for prediction on {prediction_date})")
        return latest_ema


def create_inference_example(fighter_1, fighter_2, prediction_date, historical_df):
    """
    Create a prediction example for an upcoming fight.
    
    Args:
        fighter_1: Name of first fighter
        fighter_2: Name of second fighter
        prediction_date: Date of upcoming fight
        historical_df: Historical fight data with rolling_ema
        
    Returns:
        DataFrame: Two rows ready for prediction (one per fighter)
    """
    # Get the latest rolling_ema value
    ema_value = get_inference_rolling_ema(historical_df, prediction_date)
    
    # Get each fighter's most recent precomp stats
    # (This is where you'd get their postcomp stats from their last fight)
    fighter_1_last = historical_df[historical_df['FIGHTER'] == fighter_1].sort_values('DATE').iloc[-1]
    fighter_2_last = historical_df[historical_df['FIGHTER'] == fighter_2].sort_values('DATE').iloc[-1]
    
    # Create prediction rows
    # Fighter 1 row
    row1 = {
        'FIGHTER': fighter_1,
        'OPPONENT': fighter_2,
        'DATE': prediction_date,
        'rolling_ema': ema_value,  # Same for both fighters
        # Add all other features from their last fights' postcomp stats
        # 'precomp_elo': fighter_1_last['postcomp_elo'],
        # ... etc
    }
    
    # Fighter 2 row
    row2 = {
        'FIGHTER': fighter_2,
        'OPPONENT': fighter_1,
        'DATE': prediction_date,
        'rolling_ema': ema_value,  # Same for both fighters
        # Add all other features from their last fights' postcomp stats
        # 'precomp_elo': fighter_2_last['postcomp_elo'],
        # ... etc
    }
    
    return pd.DataFrame([row1, row2])


# ============================================================================
# ALTERNATIVE: If you MUST have precomp/postcomp versions
# ============================================================================

def add_rolling_ema_precomp_postcomp(df):
    """
    Add precomp_rolling_ema and postcomp_rolling_ema for consistency with other features.
    
    Note: This is a bit redundant since rolling_ema is global, but it maintains
    the naming convention for all features.
    
    Interpretation:
    - precomp_rolling_ema: Meta-game state BEFORE this fight (what we use for prediction)
    - postcomp_rolling_ema: Meta-game state AT this fight (becomes next fight's precomp)
    """
    df = df.sort_values('DATE').copy()
    
    # Convert win column to numeric
    df['win'] = pd.to_numeric(df['win'], errors='coerce')
    
    # Calculate base rolling_ema
    base_ema = (
        df.groupby(df['DATE'].dt.date)['win']
        .transform('mean')
        .ewm(span=365, min_periods=50)
        .mean()
    )
    
    # precomp: shifted by 1 (previous value)
    df['precomp_rolling_ema'] = base_ema.shift(1)
    df['precomp_rolling_ema'].fillna(0.5, inplace=True)
    
    # postcomp: current value (becomes next fight's precomp)
    df['postcomp_rolling_ema'] = base_ema
    df['postcomp_rolling_ema'].fillna(0.5, inplace=True)
    
    # Also keep just 'rolling_ema' as an alias for precomp (for compatibility)
    df['rolling_ema'] = df['precomp_rolling_ema']
    
    return df


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/tmp/final.csv', parse_dates=['DATE'], low_memory=False)
    
    print("="*80)
    print("METHOD 1: Simple rolling_ema (RECOMMENDED)")
    print("="*80)
    
    # Add rolling_ema for training
    df_with_ema = add_rolling_ema_for_training_and_inference(df)
    
    # For inference on upcoming fight
    upcoming_ema = get_inference_rolling_ema(df_with_ema, prediction_date='2025-03-15')
    print(f"\nFor upcoming fight on 2025-03-15, use rolling_ema = {upcoming_ema:.4f}")
    print("Apply this same value to BOTH fighters in the bout")
    
    # Save for training
    df_with_ema.to_csv('data/tmp/final_with_rolling_ema.csv', index=False)
    print("\n✅ Saved to: data/tmp/final_with_rolling_ema.csv")
    
    print("\n" + "="*80)
    print("METHOD 2: precomp/postcomp versions (for consistency)")
    print("="*80)
    
    # Add precomp/postcomp versions
    df_with_prepost = add_rolling_ema_precomp_postcomp(df)
    
    print("\nColumns added:")
    print("  - precomp_rolling_ema: Use for training and prediction")
    print("  - postcomp_rolling_ema: Updated after fight, becomes next precomp")
    print("  - rolling_ema: Alias for precomp_rolling_ema")
    
    # For inference
    print("\nFor inference on upcoming fights:")
    print("  1. Get the most recent postcomp_rolling_ema from your dataset")
    print("  2. Use it as precomp_rolling_ema for BOTH fighters")
    
    # Example
    latest_postcomp = df_with_prepost.sort_values('DATE')['postcomp_rolling_ema'].iloc[-1]
    print(f"\n  Latest postcomp_rolling_ema: {latest_postcomp:.4f}")
    print(f"  Use this as precomp_rolling_ema for next fight")
    
    # Save
    df_with_prepost.to_csv('data/tmp/final_with_rolling_ema_prepost.csv', index=False)
    print("\n✅ Saved to: data/tmp/final_with_rolling_ema_prepost.csv")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
For your use case, I recommend METHOD 2 (precomp/postcomp) because:

1. Maintains consistency with your existing feature structure
2. Clear naming convention: all features have precomp/postcomp versions
3. Easy inference: just use postcomp_rolling_ema from last fight as precomp for next

The key difference from fighter stats:
- For fighter stats: each fighter has their own precomp/postcomp values
- For rolling_ema: BOTH fighters use the SAME value (it's a global indicator)

When predicting Fighter A vs Fighter B:
  Fighter A row: precomp_rolling_ema = 0.548 (latest postcomp)
  Fighter B row: precomp_rolling_ema = 0.548 (same value!)
    """)

