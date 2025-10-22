# Batch UFC Event Prediction Guide

## ✅ YES! You Can Predict Entire Events at Once

**And it's actually MORE consistent and EASIER than one-by-one prediction!**

---

## 🎯 The Key Insight

All fights in a UFC event happen on the **same date**, so they all use the **same `rolling_ema` value**.

```
UFC 300 on April 13, 2024:
├─ Fight 1: Pereira vs Hill       → rolling_ema = 0.4985
├─ Fight 2: Zhang vs Yan          → rolling_ema = 0.4985 (SAME)
├─ Fight 3: Gaethje vs Holloway   → rolling_ema = 0.4985 (SAME)
├─ Fight 4: Oliveira vs Tsarukyan → rolling_ema = 0.4985 (SAME)
└─ Fight 5: Nickal vs Brundage    → rolling_ema = 0.4985 (SAME)

All 10 fighters (2 per fight × 5 fights) use rolling_ema = 0.4985
```

---

## 💡 Why This Works Perfectly

### ✅ Consistency Guaranteed

Since all fights happen on the same date:
1. You pull the **latest `postcomp_rolling_ema`** ONCE before the event
2. Apply it to **ALL fighters** in **ALL fights**
3. No possibility of inconsistency between fights

### ✅ Batch Processing

You can create prediction rows for all fights and predict them in one batch:
- Faster computation
- Easier code
- Same results as one-by-one

### ✅ Each Fighter Still Unique

Even though all fighters share the same `rolling_ema`, each fighter still has their own:
- `precomp_elo` (their personal Elo)
- `precomp_strike_elo` (their striking Elo)
- All other personal stats

---

## 📋 Implementation

### Option 1: Quick Method (Recommended)

```python
import pandas as pd

# Load data
df = pd.read_csv('data/tmp/final_with_rolling_ema_prepost.csv', parse_dates=['DATE'])

# Define the event
event_date = pd.to_datetime("2024-04-13")
fights = [
    ("Alex Pereira", "Jamahal Hill"),
    ("Zhang Weili", "Yan Xiaonan"),
    ("Justin Gaethje", "Max Holloway"),
    # ... more fights
]

# Step 1: Get the event's rolling_ema (ONCE for all fights)
past_data = df[df['DATE'] < event_date].sort_values('DATE')
event_rolling_ema = past_data['postcomp_rolling_ema'].iloc[-1]
print(f"Event rolling_ema: {event_rolling_ema:.4f}")

# Step 2: Build prediction rows for ALL fights
all_rows = []
for fighter_a, fighter_b in fights:
    # Get each fighter's most recent stats
    fa_last = df[df['FIGHTER'] == fighter_a].sort_values('DATE').iloc[-1]
    fb_last = df[df['FIGHTER'] == fighter_b].sort_values('DATE').iloc[-1]
    
    # Fighter A row
    row_a = {
        # Fighter A's personal stats (from their last fight)
        'precomp_elo': fa_last['postcomp_elo'],
        'precomp_strike_elo': fa_last['postcomp_strike_elo'],
        'precomp_grapple_elo': fa_last['postcomp_grapple_elo'],
        # ... all other Fighter A features ...
        
        # Opponent (Fighter B) stats
        'opp_precomp_elo': fb_last['postcomp_elo'],
        'opp_precomp_strike_elo': fb_last['postcomp_strike_elo'],
        # ... all other Fighter B features ...
        
        # Meta-game (SAME for all fighters)
        'precomp_rolling_ema': event_rolling_ema
    }
    all_rows.append(row_a)
    
    # Fighter B row (reversed)
    row_b = {
        # Fighter B's personal stats
        'precomp_elo': fb_last['postcomp_elo'],
        'precomp_strike_elo': fb_last['postcomp_strike_elo'],
        # ... all other Fighter B features ...
        
        # Opponent (Fighter A) stats
        'opp_precomp_elo': fa_last['postcomp_elo'],
        'opp_precomp_strike_elo': fa_last['postcomp_strike_elo'],
        # ... all other Fighter A features ...
        
        # Meta-game (SAME for all fighters)
        'precomp_rolling_ema': event_rolling_ema
    }
    all_rows.append(row_b)

# Step 3: Predict ALL fights at once
X_pred = pd.DataFrame(all_rows)
predictions = model.predict_proba(X_pred[features])[:, 1]

# Step 4: Format results
results = []
for i, (fa, fb) in enumerate(fights):
    prob_a = predictions[i * 2]
    prob_b = predictions[i * 2 + 1]
    
    # Normalize
    total = prob_a + prob_b
    results.append({
        'fight': f"{fa} vs {fb}",
        'prob_a_wins': prob_a / total,
        'prob_b_wins': prob_b / total,
        'favorite': fa if prob_a > prob_b else fb
    })

results_df = pd.DataFrame(results)
print(results_df)
```

### Option 2: Using the Helper Function

```python
from predict_ufc_event import predict_ufc_event

# Train your model first
from src.ensemble_model_best import FightOutcomeModel
fight_model = FightOutcomeModel('data/tmp/final_with_rolling_ema_prepost.csv')
model, acc = fight_model.tune_xgboost_with_rolling_ema()

# Define the event
fights = [
    ("Alex Pereira", "Jamahal Hill"),
    ("Zhang Weili", "Yan Xiaonan"),
    # ... more fights
]

# Predict the entire event at once
results = predict_ufc_event(
    event_name="UFC 300: Pereira vs. Hill",
    event_date="2024-04-13",
    fights_list=fights,
    historical_df=fight_model.df,
    model=fight_model.best_model,
    features=fight_model.importance_columns + ['precomp_rolling_ema']
)

print(results)
```

---

## 🎬 Complete Working Example

```python
import pandas as pd
from src.ensemble_model_best import FightOutcomeModel

# 1. Load and train model
print("Training model...")
fight_model = FightOutcomeModel('data/tmp/final_with_rolling_ema_prepost.csv')
model, acc = fight_model.tune_xgboost_with_rolling_ema()
print(f"Model accuracy: {acc:.2%}")

# 2. Define UFC event
event_name = "UFC 300: Pereira vs. Hill"
event_date = pd.to_datetime("2024-04-13")

fights = [
    ("Alex Pereira", "Jamahal Hill"),       # Main event
    ("Zhang Weili", "Yan Xiaonan"),         # Co-main
    ("Justin Gaethje", "Max Holloway"),     # Title fight
    ("Charles Oliveira", "Arman Tsarukyan"),
    ("Bo Nickal", "Cody Brundage"),
]

print(f"\n{'='*80}")
print(f"Predicting {event_name}")
print(f"Date: {event_date.strftime('%Y-%m-%d')}")
print(f"Fights: {len(fights)}")
print(f"{'='*80}\n")

# 3. Get event's rolling_ema (ONCE for all fights)
df = fight_model.df.sort_values('DATE')
past_data = df[df['DATE'] < event_date]
event_ema = past_data['postcomp_rolling_ema'].iloc[-1]

print(f"Event rolling_ema: {event_ema:.4f}")
print(f"This value is used for ALL {len(fights)*2} fighters\n")

# 4. Predict each fight
# (In production, you'd batch all predictions together)
for i, (fa, fb) in enumerate(fights, 1):
    # Get each fighter's stats
    fa_data = df[df['FIGHTER'] == fa].sort_values('DATE').iloc[-1]
    fb_data = df[df['FIGHTER'] == fb].sort_values('DATE').iloc[-1]
    
    print(f"\nFight {i}: {fa} vs {fb}")
    print(f"  {fa}:")
    print(f"    precomp_elo: {fa_data.get('postcomp_elo', 'N/A')}")
    print(f"    precomp_rolling_ema: {event_ema:.4f}")
    print(f"  {fb}:")
    print(f"    precomp_elo: {fb_data.get('postcomp_elo', 'N/A')}")
    print(f"    precomp_rolling_ema: {event_ema:.4f} (SAME as {fa})")
    
    # In production: add to batch and predict all at once
    # predictions = model.predict_proba(X_all_fights)
```

---

## 📊 Comparison: One-by-One vs Batch

| Aspect | One-by-One | Batch (Entire Event) |
|--------|-----------|---------------------|
| **rolling_ema lookup** | 5 times (once per fight) | 1 time (once for event) ✅ |
| **Consistency** | Risk of variation if not careful | Guaranteed same value ✅ |
| **Speed** | 5 separate predict calls | 1 batch predict call ✅ |
| **Code complexity** | Loop over fights | Single batch operation ✅ |
| **Accuracy** | Same | Same ✅ |

**Winner: Batch prediction is better in every way!** ✅

---

## ⚠️ Important Notes

### 1. Same Date = Same rolling_ema

```python
# Correct ✅
event_date = "2024-04-13"
event_ema = get_latest_ema_before(event_date)  # Get ONCE

for fight in all_fights_on_2024_04_13:
    both_fighters_use(event_ema)  # Apply to ALL
```

### 2. Different Events = Different rolling_ema

```python
# If you're predicting multiple events on different dates:
for event_date, fights in events.items():
    event_ema = get_latest_ema_before(event_date)  # Different per event
    predict_all_fights(fights, event_ema)
```

### 3. Each Fighter Still Unique

```python
# Even though rolling_ema is the same:
fighter_a_row = {
    'precomp_elo': 1650,  # Different for each fighter
    'precomp_rolling_ema': 0.4985  # SAME for all
}

fighter_b_row = {
    'precomp_elo': 1580,  # Different for each fighter
    'precomp_rolling_ema': 0.4985  # SAME for all
}
```

---

## 🎯 Summary

**YES, you can absolutely predict entire UFC events at once!**

✅ **More consistent**: All fights use the same `rolling_ema` (as they should)  
✅ **More efficient**: Pull `rolling_ema` once, apply to all fighters  
✅ **Simpler code**: Batch prediction instead of loops  
✅ **Same accuracy**: No difference from one-by-one  
✅ **Faster**: Single model call for all fights  

**The key**: All fights on the same date → same `rolling_ema` → perfect for batch prediction!

---

## 📁 Files

- ✅ `predict_ufc_event.py` - Helper functions for event prediction
- ✅ `BATCH_EVENT_PREDICTION_GUIDE.md` - This guide
- ✅ `data/tmp/final_with_rolling_ema_prepost.csv` - Data with rolling_ema

**You're all set to predict entire events efficiently and consistently!** 🚀

