# All Models with Rolling EMA Support

## Summary

All three model training methods now support the `use_rolling_ema=True` flag:
- ✅ `tune_logistic_regression(use_rolling_ema=True)`
- ✅ `tune_mlp(use_rolling_ema=True)`
- ✅ `tune_xgboost_full(use_champion_config=True, use_rolling_ema=True)`

## Performance with Rolling EMA

Based on testing with the champion features + rolling_ema:

| Model | Accuracy | Log Loss | Notes |
|-------|----------|----------|-------|
| **XGBoost** | **71.05%** | **0.5582** | Best overall performance |
| **Logistic Regression** | **73.02%** | **0.5590** | Fast, interpretable |
| **MLP** | **72.60%** | **0.5592** | Good balance |

## Usage Examples

### 1. Logistic Regression with Rolling EMA

```python
from src.ensemble_model_best import FightOutcomeModel

fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)

# Train with champion features + rolling_ema
model, acc = fight_model.tune_logistic_regression(use_rolling_ema=True)

# Generate odds
fight_model.generate_odds_table()

# Calculate ROI
fight_model.calculate_roi(stake_per_fight=100)
```

### 2. MLP with Rolling EMA

```python
from src.ensemble_model_best import FightOutcomeModel

fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)

# Train with champion features + rolling_ema
model, acc = fight_model.tune_mlp(use_rolling_ema=True)

# Generate odds
fight_model.generate_odds_table()

# Calculate ROI
fight_model.calculate_roi(stake_per_fight=100)
```

### 3. XGBoost with Rolling EMA

```python
from src.ensemble_model_best import FightOutcomeModel

fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)

# Train with champion config + rolling_ema
model, acc = fight_model.tune_xgboost_full(use_champion_config=True, use_rolling_ema=True)

# Generate odds
fight_model.generate_odds_table()

# Calculate ROI
fight_model.calculate_roi(stake_per_fight=100)
```

### 4. Compare All Models

```python
from src.ensemble_model_best import FightOutcomeModel

results = {}

# Logistic Regression
fm1 = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
model1, acc1 = fm1.tune_logistic_regression(use_rolling_ema=True)
results['Logistic'] = acc1

# MLP
fm2 = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
model2, acc2 = fm2.tune_mlp(use_rolling_ema=True)
results['MLP'] = acc2

# XGBoost
fm3 = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
model3, acc3 = fm3.tune_xgboost_full(use_champion_config=True, use_rolling_ema=True)
results['XGBoost'] = acc3

# Print comparison
for name, acc in results.items():
    print(f"{name:20s}: {acc*100:.2f}%")
```

## What Happens When You Use `use_rolling_ema=True`

1. **Adds Rolling EMA Feature** (if not already present)
   - Calculates `precomp_rolling_ema` and `postcomp_rolling_ema`
   - Uses `span=200`, `min_periods=20`

2. **Loads Champion Features**
   - Reads `xgboost_ga_results_1760303427.json`
   - Gets the 28 champion features

3. **Replaces Feature Set**
   - Sets `self.importance_columns = champion_features + ['precomp_rolling_ema']`
   - Total: 29 features

4. **Rebuilds Train/Test Split**
   - Calls `self._prepare_data()` with new features

5. **Trains Model**
   - Uses the same training logic as normal
   - All 29 features are used in training

## Feature List (29 total)

**28 Champion Features:**
1. precomp_elo_diff
2. precomp_strike_elo_diff
3. precomp_grapple_elo_diff
4. precomp_legacc_perc5
5. opp_precomp_sigstr_pm5
6. opp_precomp_grapple_strike_mix
7. opp_precomp_clinchacc_perc
8. opp_age_ratio_difference
9. opp_precomp_elo
10. age_ratio_difference
11. precomp_distacc_perc
12. opp_precomp_winsum
13. precomp_tdavg3
14. opp_precomp_legacc_perc3
15. opp_precomp_str_eff_diff3
16. precomp_winsum
17. opp_precomp_sapm3
18. precomp_groundacc_perc
19. opp_precomp_ctrl_per_min
20. opp_REACH
21. precomp_winsum5
22. opp_precomp_strdef5
23. precomp_ctrl_per_min
24. opp_precomp_tdavg5
25. opp_precomp_headacc_perc5
26. precomp_elo_change_5
27. opp_precomp_winsum3
28. opp_precomp_groundacc_perc5

**+1 Rolling EMA Feature:**
29. precomp_rolling_ema

## Important Notes

- **Always use `use_rolling_ema=True`** if you want the rolling_ema feature
- **Don't manually call `add_rolling_ema()`** - it's handled automatically
- **Don't manually modify `importance_columns`** - it's handled automatically
- **Each model needs a fresh `FightOutcomeModel` instance** - don't reuse between models

## Next Steps

Try these methods in the `01_Fight_Predictor_Pipeline.ipynb` notebook:

```python
# Cell 1: Logistic Regression with rolling_ema
fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
model, acc = fight_model.tune_logistic_regression(use_rolling_ema=True)

# Cell 2: MLP with rolling_ema
fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
model, acc = fight_model.tune_mlp(use_rolling_ema=True)

# Cell 3: XGBoost with rolling_ema
fight_model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
model, acc = fight_model.tune_xgboost_full(use_champion_config=True, use_rolling_ema=True)
```

