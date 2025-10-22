# Converting Previous Fight postcomp_stats to Current Fight precomp_stats

## Overview

When inferencing upcoming fights, we don't have `precomp_stats` available yet. Instead, we need to use the `postcomp_stats` from each fighter's **most recent previous fight** as the `precomp_stats` for their upcoming fight.

---

## The Method

### Step 1: Understand the Data Structure

Each row in the dataset represents **one fighter** in a fight and contains:

**Pre-fight stats (before the fight):**
- `precomp_elo`
- `precomp_strike_elo`
- `precomp_grapple_elo`
- `precomp_boutcount`
- `age` (static, not pre/postcomp)
- ... (all other precomp features)

**Post-fight stats (after the fight):**
- `postcomp_elo`
- `postcomp_strike_elo`
- `postcomp_grapple_elo`
- `postcomp_boutcount`
- ... (all other postcomp features)

**Opponent stats:**
- `opp_FIGHTER` (opponent's name)
- `opp_precomp_elo` (opponent's precomp_elo)
- `opp_precomp_strike_elo`
- ... (opponent's precomp features)

---

### Step 2: The Conversion Process

For a fighter's **next fight**, their `precomp_stats` should equal their **previous fight's** `postcomp_stats`.

**Formula:**
```
Fighter's current fight precomp_stat = Fighter's previous fight postcomp_stat
```

**Example:**

If Charles Oliveira's last fight (Fight A) had:
- `postcomp_elo = 1650`
- `postcomp_strike_elo = 1700`
- `postcomp_boutcount = 42`
- `age = 35.0` (on fight day)

Then his next fight (Fight B, 6 months later) should have:
- `precomp_elo = 1650` (from previous postcomp_elo)
- `precomp_strike_elo = 1700` (from previous postcomp_strike_elo)
- `precomp_boutcount = 42` (from previous postcomp_boutcount)
- `age = 35.5` (35.0 + 0.5 years elapsed)

---

### Step 3: Special Cases

#### A. Elo Decay

If a fighter has been **inactive for more than 274 days** (9 months), apply Elo decay:

```python
from datetime import timedelta

# Get fighter's last fight date
last_fight_date = previous_fight['DATE']
current_fight_date = current_fight['DATE']

# Check if inactive
days_since_last_fight = (current_fight_date - last_fight_date).days

if days_since_last_fight > 274:
    # Apply 2.2% decay to Elo ratings
    precomp_elo = previous_fight['postcomp_elo'] * 0.978
    precomp_strike_elo = previous_fight['postcomp_strike_elo'] * 0.978
    precomp_grapple_elo = previous_fight['postcomp_grapple_elo'] * 0.978
else:
    # No decay
    precomp_elo = previous_fight['postcomp_elo']
    precomp_strike_elo = previous_fight['postcomp_strike_elo']
    precomp_grapple_elo = previous_fight['postcomp_grapple_elo']
```

#### B. Age Updates

Age should be updated based on the time elapsed:

```python
# Calculate age at current fight
days_elapsed = (current_fight_date - last_fight_date).days
years_elapsed = days_elapsed / 365.25

age = previous_fight['age'] + years_elapsed
```

Note: Age is a static column (not precomp/postcomp)

#### C. Bout Count

Bout count increments by 1 after each fight:

```python
precomp_boutcount = previous_fight['postcomp_boutcount']
# (The postcomp_boutcount already includes the previous fight)
```

#### D. Rolling EMA

`rolling_ema` is a **global temporal feature**, not fighter-specific:

```python
# Get the most recent rolling_ema value before the current fight
rolling_ema = df[df['DATE'] < current_fight_date]['rolling_ema'].iloc[-1]
```

---

### Step 4: Complete Algorithm

```python
def get_fighter_precomp_stats(fighter_name, current_fight_date, historical_df):
    """
    Get precomp stats for a fighter based on their previous fight.
    
    Parameters:
    -----------
    fighter_name : str
        Fighter's name
    current_fight_date : datetime
        Date of the current/upcoming fight
    historical_df : DataFrame
        Historical fight data
        
    Returns:
    --------
    dict : precomp stats for the current fight
    """
    # Find fighter's most recent fight before current date
    fighter_history = historical_df[
        (historical_df['FIGHTER'] == fighter_name) & 
        (historical_df['DATE'] < current_fight_date)
    ].sort_values('DATE')
    
    if len(fighter_history) == 0:
        raise ValueError(f"No previous fights found for {fighter_name}")
    
    # Get most recent fight
    last_fight = fighter_history.iloc[-1]
    
    # Calculate days since last fight
    days_since_last = (current_fight_date - last_fight['DATE']).days
    
    # Initialize precomp stats from postcomp stats
    precomp_stats = {}
    
    # Handle Elo with decay
    if days_since_last > 274:
        precomp_stats['precomp_elo'] = last_fight['postcomp_elo'] * 0.978
        precomp_stats['precomp_strike_elo'] = last_fight['postcomp_strike_elo'] * 0.978
        precomp_stats['precomp_grapple_elo'] = last_fight['postcomp_grapple_elo'] * 0.978
    else:
        precomp_stats['precomp_elo'] = last_fight['postcomp_elo']
        precomp_stats['precomp_strike_elo'] = last_fight['postcomp_strike_elo']
        precomp_stats['precomp_grapple_elo'] = last_fight['postcomp_grapple_elo']
    
    # Handle age update
    years_elapsed = days_since_last / 365.25
    precomp_stats['age'] = last_fight['age'] + years_elapsed
    
    # All other stats: simply copy postcomp to precomp
    for col in last_fight.index:
        if col.startswith('postcomp_') and col not in [
            'postcomp_elo', 'postcomp_strike_elo', 'postcomp_grapple_elo'
        ]:
            # Convert postcomp_X to precomp_X
            precomp_col = col.replace('postcomp_', 'precomp_')
            precomp_stats[precomp_col] = last_fight[col]
    
    return precomp_stats
```

---

### Step 5: Building a Complete Fight Prediction

```python
def prepare_fight_for_inference(fighter_a, fighter_b, fight_date, historical_df):
    """
    Prepare a fight for model inference using postcomp -> precomp conversion.
    
    Returns:
    --------
    tuple : (fighter_a_features, fighter_b_features, rolling_ema)
    """
    # Get precomp stats for both fighters
    fighter_a_precomp = get_fighter_precomp_stats(fighter_a, fight_date, historical_df)
    fighter_b_precomp = get_fighter_precomp_stats(fighter_b, fight_date, historical_df)
    
    # Get rolling_ema (global temporal feature)
    past_fights = historical_df[historical_df['DATE'] < fight_date].sort_values('DATE')
    rolling_ema = past_fights['rolling_ema'].iloc[-1] if len(past_fights) > 0 else 0.5
    
    # Build feature vectors for both fighters
    # Fighter A's features include their stats + opponent's stats
    fighter_a_features = {
        **fighter_a_precomp,
        **{f"opp_{k}": v for k, v in fighter_b_precomp.items()},
        'rolling_ema': rolling_ema
    }
    
    # Fighter B's features include their stats + opponent's stats
    fighter_b_features = {
        **fighter_b_precomp,
        **{f"opp_{k}": v for k, v in fighter_a_precomp.items()},
        'rolling_ema': rolling_ema
    }
    
    return fighter_a_features, fighter_b_features, rolling_ema
```

---

## Validation

The method should produce **identical results** to using the dataset's built-in `precomp_stats` when:

1. **Test accuracy** remains the same (69.92-71.05%)
2. **Test log loss** remains the same (~0.5648)
3. **Predictions match** the original predictions

If results differ, there's an error in the conversion logic.

---

## Common Pitfalls

### ❌ Wrong: Using precomp instead of postcomp
```python
# WRONG - This is circular!
precomp_elo = last_fight['precomp_elo']  # ❌
```

### ✅ Correct: Using postcomp from previous fight
```python
# CORRECT - Use postcomp from previous fight
precomp_elo = last_fight['postcomp_elo']  # ✅
```

---

### ❌ Wrong: Not applying Elo decay
```python
# WRONG - Missing decay check
precomp_elo = last_fight['postcomp_elo']  # ❌ (if inactive > 274 days)
```

### ✅ Correct: Checking for inactivity
```python
# CORRECT - Apply decay if needed
if days_since_last > 274:
    precomp_elo = last_fight['postcomp_elo'] * 0.978  # ✅
else:
    precomp_elo = last_fight['postcomp_elo']
```

---

### ❌ Wrong: Using fighter's own rolling_ema
```python
# WRONG - rolling_ema is not fighter-specific
rolling_ema = last_fight['rolling_ema']  # ❌
```

### ✅ Correct: Using global rolling_ema
```python
# CORRECT - Get most recent global value
rolling_ema = df[df['DATE'] < fight_date]['rolling_ema'].iloc[-1]  # ✅
```

---

### ❌ Wrong: Forgetting opponent features
```python
# WRONG - Missing opponent features
features = fighter_a_precomp  # ❌
```

### ✅ Correct: Including opponent as opp_ features
```python
# CORRECT - Include opponent stats with opp_ prefix
features = {
    **fighter_a_precomp,
    **{f"opp_{k}": v for k, v in fighter_b_precomp.items()}
}  # ✅
```

---

## Summary

**The conversion is simple:**
1. Get fighter's most recent fight (before current date)
2. Copy `postcomp_*` stats to `precomp_*` for current fight
3. Apply Elo decay if inactive > 274 days
4. Update age based on time elapsed
5. Get global `rolling_ema` from dataset
6. Build feature vectors including opponent stats

**Validation:**
- Test accuracy should match baseline (69.92-71.05%)
- Test log loss should match baseline (~0.5648)
- This proves the method works correctly

---

**Next Step:** Run validation script to confirm the method works on the test set.

