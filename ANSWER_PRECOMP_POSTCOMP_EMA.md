# Answer: How to Handle rolling_ema with precomp/postcomp Framework

## 🎯 Your Question

> "How can we add the new feature in a way that allows us to have a post_comp and pre_comp version? All features have precomp and postversion (precomp_elo and postcomp_elo). The postcomp_elo is used for inference when predicting upcoming fights. However, how would we do precomp_ema and postcomp_ema?"

## ✅ The Solution

**I've created both versions for you!**

### Files Created:

1. ✅ **`data/tmp/final_with_rolling_ema_prepost.csv`** - With precomp/postcomp versions
2. ✅ **`add_rolling_ema_with_inference.py`** - Script to generate the features
3. ✅ **`ROLLING_EMA_PRECOMP_POSTCOMP_GUIDE.md`** - Complete documentation

---

## 🔑 Key Difference: Global vs Fighter-Specific

### Fighter Stats (elo, strike accuracy, etc.)
```
Fighter A:
├─ Fight 1: precomp_elo=1500 → postcomp_elo=1520
└─ Fight 2: precomp_elo=1520 (different from Fighter B)

Fighter B:
├─ Fight 1: precomp_elo=1400 → postcomp_elo=1390
└─ Fight 2: precomp_elo=1390 (different from Fighter A)
```
**Each fighter has DIFFERENT values**

### rolling_ema (Global Meta-Game)
```
Fight on 2025-03-15:
├─ Fighter A: precomp_rolling_ema = 0.548
└─ Fighter B: precomp_rolling_ema = 0.548 (SAME!)

Both fighters use the SAME value because it's a global UFC meta-game indicator
```
**Both fighters have the SAME value**

---

## 📊 How It Works

### 1. In the Dataset

Your new dataset has three columns:

| Column | Description | Use |
|--------|-------------|-----|
| `precomp_rolling_ema` | Meta-game state BEFORE fight | For training and prediction |
| `postcomp_rolling_ema` | Meta-game state AT fight | Becomes next fight's precomp |
| `rolling_ema` | Alias for precomp_rolling_ema | For compatibility |

### 2. For Training

```python
# Use precomp_rolling_ema just like any other feature
features = [
    'precomp_elo', 
    'precomp_strike_elo',
    # ... all other features ...
    'precomp_rolling_ema'  # Add the temporal feature
]

X_train = df[features]
y_train = df['win']

model.fit(X_train, y_train)
```

### 3. For Inference (Upcoming Fights)

```python
# Load historical data
df = pd.read_csv('data/tmp/final_with_rolling_ema_prepost.csv')

# Get latest meta-game state
latest_ema = df.sort_values('DATE')['postcomp_rolling_ema'].iloc[-1]
print(f"Latest meta-game state: {latest_ema:.4f}")

# Predict Jones vs Miocic
jones_last = df[df['FIGHTER'] == 'Jon Jones'].iloc[-1]
miocic_last = df[df['FIGHTER'] == 'Stipe Miocic'].iloc[-1]

# Build feature vectors
jones_features = {
    'precomp_elo': jones_last['postcomp_elo'],  # Jon's personal stat
    'precomp_strike_elo': jones_last['postcomp_strike_elo'],
    # ... all other Jon features ...
    'precomp_rolling_ema': latest_ema  # Global meta-game (SAME for both)
}

miocic_features = {
    'precomp_elo': miocic_last['postcomp_elo'],  # Stipe's personal stat
    'precomp_strike_elo': miocic_last['postcomp_strike_elo'],
    # ... all other Stipe features ...
    'precomp_rolling_ema': latest_ema  # Global meta-game (SAME as Jon's)
}

# Predict
X_pred = pd.DataFrame([jones_features, miocic_features])
probs = model.predict_proba(X_pred)[:, 1]
```

---

## 💡 The Pattern

### For Fighter-Specific Stats:
```python
# Each fighter uses their own last postcomp value
fighter_a_precomp_elo = fighter_a_last_fight['postcomp_elo']
fighter_b_precomp_elo = fighter_b_last_fight['postcomp_elo']
# ↑ DIFFERENT values
```

### For rolling_ema:
```python
# Both fighters use the SAME latest value
latest_ema = df['postcomp_rolling_ema'].iloc[-1]

fighter_a_precomp_rolling_ema = latest_ema
fighter_b_precomp_rolling_ema = latest_ema
# ↑ SAME value for both
```

---

## 🎯 Summary Table

| Aspect | Fighter Stats | rolling_ema |
|--------|--------------|-------------|
| **Scope** | Fighter-specific | Global (UFC-wide) |
| **precomp source** | That fighter's last postcomp | Latest postcomp from dataset |
| **Values in same fight** | Different for each fighter | Same for both fighters |
| **Example** | Jon: 1650, Stipe: 1580 | Both: 0.548 |

---

## ✅ What You Need to Do

### Option 1: Use the Generated File (RECOMMENDED)

```python
# Just load the file I created for you!
from src.ensemble_model_best import FightOutcomeModel

# Load data with precomp/postcomp rolling_ema
fight_model = FightOutcomeModel('data/tmp/final_with_rolling_ema_prepost.csv')

# Train with rolling_ema (it's already in the dataset)
model, acc = fight_model.tune_xgboost_with_rolling_ema()

# The model will use precomp_rolling_ema automatically
```

### Option 2: Regenerate Yourself

```bash
# Run the script I created
python add_rolling_ema_with_inference.py

# This creates:
# - data/tmp/final_with_rolling_ema.csv (simple version)
# - data/tmp/final_with_rolling_ema_prepost.csv (precomp/postcomp version)
```

---

## 📋 Complete Example: Prediction Workflow

```python
import pandas as pd
from src.ensemble_model_best import FightOutcomeModel

# 1. Load data with precomp/postcomp rolling_ema
df = pd.read_csv('data/tmp/final_with_rolling_ema_prepost.csv', parse_dates=['DATE'])

# 2. Train model
fight_model = FightOutcomeModel('data/tmp/final_with_rolling_ema_prepost.csv')
model, acc = fight_model.tune_xgboost_with_rolling_ema()
print(f"Accuracy: {acc:.2%}")

# 3. For inference on upcoming fight: Israel Adesanya vs Sean Strickland
#    Date: 2025-04-01

# Get latest meta-game state (same for both fighters)
latest_ema = df.sort_values('DATE')['postcomp_rolling_ema'].iloc[-1]
print(f"Current meta-game rolling_ema: {latest_ema:.4f}")

# Get each fighter's most recent stats
izzy = df[df['FIGHTER'] == 'Israel Adesanya'].sort_values('DATE').iloc[-1]
sean = df[df['FIGHTER'] == 'Sean Strickland'].sort_values('DATE').iloc[-1]

# Build prediction rows
prediction_data = pd.DataFrame([
    {
        'FIGHTER': 'Israel Adesanya',
        'DATE': '2025-04-01',
        # Izzy's stats from his last fight
        'precomp_elo': izzy['postcomp_elo'],
        'precomp_strike_elo': izzy['postcomp_strike_elo'],
        # ... all other Izzy features ...
        
        # Opponent (Sean) stats
        'opp_precomp_elo': sean['postcomp_elo'],
        'opp_precomp_strike_elo': sean['postcomp_strike_elo'],
        # ... all other Sean features ...
        
        # Meta-game (SAME for both)
        'precomp_rolling_ema': latest_ema
    },
    {
        'FIGHTER': 'Sean Strickland',
        'DATE': '2025-04-01',
        # Sean's stats from his last fight
        'precomp_elo': sean['postcomp_elo'],
        'precomp_strike_elo': sean['postcomp_strike_elo'],
        # ... all other Sean features ...
        
        # Opponent (Izzy) stats
        'opp_precomp_elo': izzy['postcomp_elo'],
        'opp_precomp_strike_elo': izzy['postcomp_strike_elo'],
        # ... all other Izzy features ...
        
        # Meta-game (SAME as Izzy's)
        'precomp_rolling_ema': latest_ema
    }
])

# Predict
probs = model.predict_proba(prediction_data)[:, 1]
print(f"P(Izzy wins): {probs[0]:.1%}")
print(f"P(Sean wins): {probs[1]:.1%}")
```

---

## 🎉 Bottom Line

**Yes, you CAN use precomp/postcomp for rolling_ema!**

The key differences:
1. ✅ **Naming**: Same pattern as other features (precomp_rolling_ema, postcomp_rolling_ema)
2. ✅ **Training**: Use precomp_rolling_ema just like precomp_elo
3. ✅ **Inference**: Both fighters get the SAME latest postcomp value
4. ✅ **Files ready**: `data/tmp/final_with_rolling_ema_prepost.csv` is ready to use

**The file is already created and ready for you to use in your models!** 🚀

See `ROLLING_EMA_PRECOMP_POSTCOMP_GUIDE.md` for more detailed examples and explanations.

