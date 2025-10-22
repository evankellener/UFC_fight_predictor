# Rolling EMA: precomp vs postcomp Framework

## 🤔 The Question

**How do we handle `rolling_ema` in the precomp/postcomp framework when it's a global temporal feature, not a fighter-specific stat?**

---

## 📊 Understanding the Difference

### Fighter-Level Stats (Traditional precomp/postcomp)

```
Fighter: Jon Jones
├─ Fight 1 (2020-01-15)
│  ├─ precomp_elo: 1500  (before fight)
│  └─ postcomp_elo: 1520 (after fight, updated based on result)
│
├─ Fight 2 (2020-03-20)
│  ├─ precomp_elo: 1520  (= previous postcomp_elo)
│  └─ postcomp_elo: 1495 (after fight, went down because lost)
│
└─ Fight 3 (2020-06-10)
   ├─ precomp_elo: 1495  (= previous postcomp_elo)
   └─ postcomp_elo: 1510 (after fight)
```

**Key**: Each fighter has their OWN values that evolve over time.

### Global Temporal Features (rolling_ema)

```
UFC Meta-Game Timeline:
├─ 2020-01-15: rolling_ema = 0.520 (all fights on this date use this)
├─ 2020-01-20: rolling_ema = 0.523 (updated based on recent UFC outcomes)
├─ 2020-01-25: rolling_ema = 0.528
└─ 2020-02-01: rolling_ema = 0.531

Fight: Jon Jones vs Daniel Cormier (2020-01-20)
├─ Jon Jones row:   rolling_ema = 0.523
└─ Daniel Cormier row: rolling_ema = 0.523  (SAME VALUE)
```

**Key**: rolling_ema is the SAME for both fighters because it represents the UFC-wide meta-game state at that moment.

---

## ✅ Recommended Solution: precomp/postcomp Naming for Consistency

Even though `rolling_ema` is global, we can still use precomp/postcomp naming to maintain consistency with your framework:

### Implementation

```python
def add_rolling_ema_precomp_postcomp(df):
    """
    Add rolling_ema with precomp/postcomp versions.
    
    Interpretation:
    - precomp_rolling_ema: Meta-game state BEFORE this fight
    - postcomp_rolling_ema: Meta-game state AT this fight
    
    The postcomp value becomes the precomp value for subsequent fights.
    """
    df = df.sort_values('DATE').copy()
    
    # Calculate base rolling EMA (UFC-wide win rate trend)
    base_ema = (
        df.groupby(df['DATE'].dt.date)['win']
        .transform('mean')
        .ewm(span=365, min_periods=50)
        .mean()
    )
    
    # precomp: Meta-game state from PREVIOUS fights (shifted by 1)
    df['precomp_rolling_ema'] = base_ema.shift(1)
    df['precomp_rolling_ema'].fillna(0.5, inplace=True)
    
    # postcomp: Current meta-game state (becomes next fight's precomp)
    df['postcomp_rolling_ema'] = base_ema
    df['postcomp_rolling_ema'].fillna(0.5, inplace=True)
    
    return df
```

### Training

```python
# For model training, use precomp_rolling_ema
X_train = df_train[baseline_features + ['precomp_rolling_ema']]
y_train = df_train['win']

model.fit(X_train, y_train)
```

### Inference (Upcoming Fights)

```python
def predict_upcoming_fight(fighter_a, fighter_b, fight_date, historical_df, model):
    """
    Predict an upcoming fight using the precomp/postcomp framework.
    """
    # Get latest meta-game state
    latest_postcomp_ema = historical_df.sort_values('DATE')['postcomp_rolling_ema'].iloc[-1]
    
    # Get each fighter's most recent stats (their last postcomp values)
    fighter_a_last = historical_df[historical_df['FIGHTER'] == fighter_a].sort_values('DATE').iloc[-1]
    fighter_b_last = historical_df[historical_df['FIGHTER'] == fighter_b].sort_values('DATE').iloc[-1]
    
    # Build prediction row for Fighter A
    row_a = {
        'FIGHTER': fighter_a,
        'DATE': fight_date,
        # Fighter A's stats from their last fight
        'precomp_elo': fighter_a_last['postcomp_elo'],
        'precomp_strike_elo': fighter_a_last['postcomp_strike_elo'],
        # ... all other fighter A stats ...
        
        # Opponent (Fighter B) stats
        'opp_precomp_elo': fighter_b_last['postcomp_elo'],
        'opp_precomp_strike_elo': fighter_b_last['postcomp_strike_elo'],
        # ... all other fighter B stats ...
        
        # Meta-game state (SAME for both fighters)
        'precomp_rolling_ema': latest_postcomp_ema
    }
    
    # Build prediction row for Fighter B (reversed)
    row_b = {
        'FIGHTER': fighter_b,
        'DATE': fight_date,
        # Fighter B's stats
        'precomp_elo': fighter_b_last['postcomp_elo'],
        # ... all other fighter B stats ...
        
        # Opponent (Fighter A) stats
        'opp_precomp_elo': fighter_a_last['postcomp_elo'],
        # ... all other fighter A stats ...
        
        # Meta-game state (SAME as Fighter A)
        'precomp_rolling_ema': latest_postcomp_ema
    }
    
    # Predict
    X_pred = pd.DataFrame([row_a, row_b])
    predictions = model.predict_proba(X_pred)[:, 1]
    
    return predictions[0], predictions[1]  # P(A wins), P(B wins)
```

---

## 🔑 Key Insights

### 1. Same Value for Both Fighters

```python
# When predicting Jon Jones vs Stipe Miocic on 2025-03-15

# Both fighters use THE SAME rolling_ema value
jon_jones_row = {
    'precomp_elo': 1650,  # Jon's personal stat
    'opp_precomp_elo': 1580,  # Stipe's personal stat
    'precomp_rolling_ema': 0.548  # Global meta-game (SAME for both)
}

stipe_row = {
    'precomp_elo': 1580,  # Stipe's personal stat
    'opp_precomp_elo': 1650,  # Jon's personal stat
    'precomp_rolling_ema': 0.548  # Global meta-game (SAME for both)
}
```

### 2. Updating After Fights

```python
# After Jon Jones vs Stipe Miocic fight on 2025-03-15
# Result: Jon Jones wins

# Step 1: Update fighter stats
jon_jones_postcomp_elo = update_elo(jon_jones_precomp_elo, won=True)
stipe_postcomp_elo = update_elo(stipe_precomp_elo, won=False)

# Step 2: Update meta-game state
# The fight result (1 = Jon won) gets incorporated into rolling_ema
new_postcomp_ema = update_ema(current_ema=0.548, new_outcome=1)
# This new value will be used as precomp_rolling_ema for the next fight
```

### 3. postcomp becomes precomp

```python
# Timeline view:

# Fight on 2025-03-15
postcomp_rolling_ema = 0.548

# Fight on 2025-03-20 (5 days later)
precomp_rolling_ema = 0.548  # (= previous postcomp)
postcomp_rolling_ema = 0.551  # (updated with recent outcomes)

# Fight on 2025-03-25
precomp_rolling_ema = 0.551  # (= previous postcomp)
postcomp_rolling_ema = 0.549  # (updated with recent outcomes)
```

---

## 📝 Complete Working Example

```python
import pandas as pd
import numpy as np

# Load historical data
df = pd.read_csv('data/tmp/final.csv', parse_dates=['DATE'])
df = df.sort_values('DATE')

# Calculate rolling_ema with precomp/postcomp
base_ema = (
    df.groupby(df['DATE'].dt.date)['win']
    .transform('mean')
    .ewm(span=365, min_periods=50)
    .mean()
)

df['precomp_rolling_ema'] = base_ema.shift(1).fillna(0.5)
df['postcomp_rolling_ema'] = base_ema.fillna(0.5)

# Save
df.to_csv('data/tmp/final_with_rolling_ema_prepost.csv', index=False)

# ============================================================================
# TRAINING
# ============================================================================

features = [
    'precomp_elo', 'precomp_strike_elo', 'precomp_grapple_elo',
    # ... all your other features ...
    'precomp_rolling_ema'  # Add the temporal feature
]

X_train = df_train[features]
y_train = df_train['win']

model = XGBClassifier(**hyperparams)
model.fit(X_train, y_train)

# ============================================================================
# INFERENCE
# ============================================================================

# Upcoming fight: Israel Adesanya vs Sean Strickland on 2025-04-01
upcoming_date = '2025-04-01'

# Get latest meta-game state
latest_ema = df['postcomp_rolling_ema'].iloc[-1]
print(f"Latest meta-game rolling_ema: {latest_ema:.4f}")

# Get each fighter's most recent postcomp stats
izzy_last = df[df['FIGHTER'] == 'Israel Adesanya'].iloc[-1]
sean_last = df[df['FIGHTER'] == 'Sean Strickland'].iloc[-1]

# Build feature vectors
izzy_features = {
    'precomp_elo': izzy_last['postcomp_elo'],
    'precomp_strike_elo': izzy_last['postcomp_strike_elo'],
    # ... all features ...
    'precomp_rolling_ema': latest_ema  # Same for both
}

sean_features = {
    'precomp_elo': sean_last['postcomp_elo'],
    'precomp_strike_elo': sean_last['postcomp_strike_elo'],
    # ... all features ...
    'precomp_rolling_ema': latest_ema  # Same for both
}

# Predict
X_pred = pd.DataFrame([izzy_features, sean_features])
probs = model.predict_proba(X_pred)[:, 1]

print(f"P(Izzy wins): {probs[0]:.1%}")
print(f"P(Sean wins): {probs[1]:.1%}")
```

---

## 🎯 Summary

| Aspect | Fighter Stats (elo, etc.) | rolling_ema |
|--------|--------------------------|-------------|
| **Scope** | Fighter-specific | Global (UFC-wide) |
| **Values** | Different for each fighter | Same for both fighters in a bout |
| **precomp** | Fighter's stat before fight | Meta-game state before fight |
| **postcomp** | Fighter's stat after fight | Meta-game state at fight |
| **Inference** | Use each fighter's last postcomp | Use latest postcomp (same for both) |

### For Your Implementation:

1. ✅ **Use precomp/postcomp naming** for consistency with your framework
2. ✅ **precomp_rolling_ema** = meta-game state (for training and prediction)
3. ✅ **postcomp_rolling_ema** = updated meta-game state (becomes next precomp)
4. ✅ **Both fighters get the SAME value** (it's a global temporal indicator)
5. ✅ **For inference**: Use the latest postcomp_rolling_ema from your dataset

This maintains your existing framework while correctly handling the global nature of the temporal feature! 🚀

