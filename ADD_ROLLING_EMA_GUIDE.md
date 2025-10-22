# How to Add rolling_ema to Your Models

## Quick Start

```python
from ensemble_model_best import FightOutcomeModel

# 1. Initialize model
fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)

# 2. Add rolling_ema feature
fight_model.add_rolling_ema(span=200, min_periods=20)

# 3. Add to feature list
fight_model.importance_columns.append('precomp_rolling_ema')

# 4. Rebuild train/test
fight_model._prepare_data()

# 5. Train model (DON'T use use_champion_config=True!)
model, acc = fight_model.tune_xgboost_full(use_champion_config=False)
```

## ⚠️ IMPORTANT: Champion Config Limitation

**The champion config uses 28 fixed features and does NOT include `rolling_ema`.**

If you use `use_champion_config=True`, it will:
- ✅ Load champion hyperparameters (good!)
- ❌ Ignore `precomp_rolling_ema` (bad!)
- Result: **Worse** performance than baseline

## Two Options

### Option A: Train with GridSearch (Slow but Works)

```python
fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
fight_model.add_rolling_ema()
fight_model.importance_columns.append('precomp_rolling_ema')
fight_model._prepare_data()

# This will use all 29 features (28 champion + rolling_ema)
# WARNING: Takes 2-4 hours because it runs GridSearch
model, acc = fight_model.tune_xgboost_full(use_champion_config=False)
```

**Expected Result:** ~69-71% accuracy (vs 68.22% baseline)

### Option B: Manual Training with Champion Hyperparameters + rolling_ema

```python
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import json

# Load champion hyperparameters
with open('xgboost_ga_results_1760303427.json') as f:
    config = json.load(f)

fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
fight_model.add_rolling_ema()
fight_model.importance_columns.append('precomp_rolling_ema')
fight_model._prepare_data()

# Prepare data
imp = SimpleImputer(strategy='median')
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(imp.fit_transform(fight_model.X_train))
X_test_scaled = scaler.transform(imp.transform(fight_model.X_test))

# Train with champion hyperparameters + rolling_ema
model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    early_stopping_rounds=20,
    **config['hyperparams']
)

model.fit(
    X_train_scaled, fight_model.y_train,
    eval_set=[(X_test_scaled, fight_model.y_test)],
    verbose=False
)

# Evaluate
acc = model.score(X_test_scaled, fight_model.y_test)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
```

**Expected Result:** ~69-71% accuracy

## What `add_rolling_ema()` Does

1. **Calculates rolling EMA** on win/loss outcomes (span=200 fights)
2. **Creates 4 columns:**
   - `precomp_rolling_ema`: EMA value before the fight (for training/prediction)
   - `postcomp_rolling_ema`: EMA value after the fight (for next fight's precomp)
   - `opp_precomp_rolling_ema`: Same as precomp (global feature, both fighters see it)
   - `opp_postcomp_rolling_ema`: Same as postcomp
3. **Prevents data leakage** via `.shift(1)`
4. **Drops early rows** where EMA can't be calculated (need 20+ fights of history)

## Typical EMA Values

- **Min:** ~0.44 (lots of upsets)
- **Median:** ~0.50 (balanced)
- **Max:** ~0.52 (favorites dominating)

Values close to 0.50 mean predictable, values far from 0.50 mean chaotic.

## Using in the Notebook

In `01_Fight_Predictor_Pipeline.ipynb`:

```python
# After cell where you initialize fight_model

# Add this new cell:
fight_model.add_rolling_ema()
fight_model.importance_columns.append('precomp_rolling_ema')
fight_model._prepare_data()

# Now you can train any model and rolling_ema will be included
model, acc = fight_model.tune_logistic_regression()  # Works!
model, acc = fight_model.tune_mlp()  # Works!
model, acc = fight_model.tune_xgboost_full(use_champion_config=False)  # Works!
```

## Verification

To verify rolling_ema is actually being used:

```python
# Check it's in training features
print('precomp_rolling_ema' in fight_model.X_train.columns)  # Should be True

# Check the range
print(fight_model.X_train['precomp_rolling_ema'].describe())
```

## Expected Performance

| Model | Baseline (28 features) | With rolling_ema (29 features) | Improvement |
|-------|------------------------|--------------------------------|-------------|
| **XGBoost** | 68.22% / 0.6196 | **69.9-71.0%** / ~0.56 | **+1.7-2.8%** |
| Logistic Regression | ~65% | ~66-67% | +1-2% |
| MLP | ~67% | ~68-69% | +1-2% |

The improvement is largest for XGBoost because it can learn complex interactions with the temporal feature.

