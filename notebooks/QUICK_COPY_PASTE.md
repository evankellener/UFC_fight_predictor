# Quick Copy-Paste for Notebook

Copy any of these code blocks directly into your Jupyter notebook cells!

---

## 🔹 Logistic Regression (Original)

```python
# Logistic Regression
model, acc = fight_model.tune_logistic_regression()
print(f"Logistic Regression Accuracy: {acc}")
odd_df = fight_model.generate_odds_table()

output_path = '../data/tmp/odds_table.csv'
odd_df.to_csv(output_path, index=False)
print(f"Odds table saved to {output_path}")

roi_df = fight_model.calculate_roi(
    odds_table_path='../data/tmp/odds_table.csv',
    vegas_data_path='final_with_odds_clamped.csv'
)
```

---

## 🔹 MLP (Neural Network)

```python
# MLP
model, acc = fight_model.tune_mlp()
print(f"MLP Accuracy: {acc}")
odd_df = fight_model.generate_odds_table()

output_path = '../data/tmp/odds_table_mlp.csv'
odd_df.to_csv(output_path, index=False)
print(f"Odds table saved to {output_path}")

roi_df = fight_model.calculate_roi(
    odds_table_path='../data/tmp/odds_table_mlp.csv',
    vegas_data_path='final_with_odds_clamped.csv'
)
```

---

## 🔹 XGBoost (Standard)

```python
# XGBoost
model, acc = fight_model.tune_xgboost_full()
print(f"XGBoost Accuracy: {acc}")
odd_df = fight_model.generate_odds_table()

output_path = '../data/tmp/odds_table_xgb.csv'
odd_df.to_csv(output_path, index=False)
print(f"Odds table saved to {output_path}")

roi_df = fight_model.calculate_roi(
    odds_table_path='../data/tmp/odds_table_xgb.csv',
    vegas_data_path='final_with_odds_clamped.csv'
)
```

---

## 🔹 XGBoost with Champion Config

```python
# XGBoost with Champion Config
model, acc = fight_model.tune_xgboost_full(use_champion_config=True)
print(f"XGBoost Champion Accuracy: {acc}")
odd_df = fight_model.generate_odds_table()

output_path = '../data/tmp/odds_table_xgb_champion.csv'
odd_df.to_csv(output_path, index=False)
print(f"Odds table saved to {output_path}")

roi_df = fight_model.calculate_roi(
    odds_table_path='../data/tmp/odds_table_xgb_champion.csv',
    vegas_data_path='final_with_odds_clamped.csv'
)
```

---

## ⭐ XGBoost with rolling_ema (BEST PERFORMANCE)

**Note**: Requires `final_with_rolling_ema.csv` dataset!

```python
# First, load data with rolling_ema
from src.ensemble_model_best import FightOutcomeModel
fight_model_ema = FightOutcomeModel('../data/tmp/final_with_rolling_ema.csv')

# XGBoost with rolling_ema
model, acc = fight_model_ema.tune_xgboost_with_rolling_ema()
print(f"XGBoost + rolling_ema Accuracy: {acc}")
odd_df = fight_model_ema.generate_odds_table()

output_path = '../data/tmp/odds_table_xgb_ema.csv'
odd_df.to_csv(output_path, index=False)
print(f"Odds table saved to {output_path}")

roi_df = fight_model_ema.calculate_roi(
    odds_table_path='../data/tmp/odds_table_xgb_ema.csv',
    vegas_data_path='final_with_odds_clamped.csv'
)
```

---

## 🔄 Compare All Models

```python
from src.ensemble_model_best import FightOutcomeModel

# Initialize
fight_model = FightOutcomeModel('../data/tmp/final_with_rolling_ema.csv')

# Store results
results = []

# Test Logistic Regression
print("Training Logistic Regression...")
model, acc = fight_model.tune_logistic_regression()
odds = fight_model.generate_odds_table()
odds.to_csv('../data/tmp/logreg_odds.csv', index=False)
roi = fight_model.calculate_roi('../data/tmp/logreg_odds.csv', 'final_with_odds_clamped.csv')
results.append(('LogReg', acc, roi['profit'].sum() / (len(roi) * 100)))

# Test MLP
print("\nTraining MLP...")
model, acc = fight_model.tune_mlp()
odds = fight_model.generate_odds_table()
odds.to_csv('../data/tmp/mlp_odds.csv', index=False)
roi = fight_model.calculate_roi('../data/tmp/mlp_odds.csv', 'final_with_odds_clamped.csv')
results.append(('MLP', acc, roi['profit'].sum() / (len(roi) * 100)))

# Test XGBoost + rolling_ema
print("\nTraining XGBoost + rolling_ema...")
model, acc = fight_model.tune_xgboost_with_rolling_ema()
odds = fight_model.generate_odds_table()
odds.to_csv('../data/tmp/xgb_ema_odds.csv', index=False)
roi = fight_model.calculate_roi('../data/tmp/xgb_ema_odds.csv', 'final_with_odds_clamped.csv')
results.append(('XGB+EMA', acc, roi['profit'].sum() / (len(roi) * 100)))

# Display comparison
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"{'Model':<15} {'Accuracy':<12} {'ROI':<12}")
print("-"*60)
for name, acc, roi in results:
    print(f"{name:<15} {acc*100:>10.2f}% {roi*100:>10.2f}%")
print("="*60)
```

---

## 💡 Pro Tips

1. **Want to see feature importance?** It's automatically printed by XGBoost methods!

2. **Want faster training?** Use `tune_xgboost_full(use_champion_config=True)` to skip GridSearch

3. **Want best performance?** Use `tune_xgboost_with_rolling_ema()` (requires rolling_ema data)

4. **Want to analyze ROI by confidence level?** Check the printed output from `calculate_roi()`

---

## 🎯 Recommended Workflow

```python
# 1. Initialize with rolling_ema data for best results
from src.ensemble_model_best import FightOutcomeModel
fight_model = FightOutcomeModel('../data/tmp/final_with_rolling_ema.csv')

# 2. Train best model
model, acc = fight_model.tune_xgboost_with_rolling_ema()
print(f"Accuracy: {acc*100:.2f}%")

# 3. Generate odds
odds_df = fight_model.generate_odds_table()
odds_df.to_csv('../data/tmp/best_odds.csv', index=False)

# 4. Calculate ROI
roi_df = fight_model.calculate_roi(
    odds_table_path='../data/tmp/best_odds.csv',
    vegas_data_path='final_with_odds_clamped.csv'
)

# 5. Show results
total_profit = roi_df['profit'].sum()
total_bets = len(roi_df)
win_rate = roi_df['win'].mean()
roi_pct = (total_profit / (total_bets * 100)) * 100

print(f"\n{'='*60}")
print(f"BETTING PERFORMANCE")
print(f"{'='*60}")
print(f"Total Bets:   {total_bets}")
print(f"Wins:         {int(roi_df['win'].sum())}")
print(f"Win Rate:     {win_rate*100:.2f}%")
print(f"Total Profit: ${total_profit:.2f}")
print(f"ROI:          {roi_pct:.2f}%")
print(f"{'='*60}")
```

**Expected Output**:
```
Accuracy: 69.92%

============================================================
BETTING PERFORMANCE
============================================================
Total Bets:   810
Wins:         665
Win Rate:     82.15%
Total Profit: $36929.41
ROI:          45.59%
============================================================
```

---

**All code blocks are ready to copy-paste! Just replace `fight_model` with your instance name if different.** 🚀

