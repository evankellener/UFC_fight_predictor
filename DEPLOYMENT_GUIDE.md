# UFC Fight Predictor - Deployment Guide

## 🚀 Production Deployment Strategy

Based on comprehensive ROI analysis showing:
- **+27-49% ROI** on straight picks (all high-confidence)
- **+22.6% ROI** on underdog strategy (more selective)
- **No statistical degradation** over time
- **82-91% profitable events**

---

## 📋 Pre-Fight Workflow (3-7 Days Before Event)

### Step 1: Update Data
```bash
# Update your final.csv with latest fight data
# Ensure it includes:
# - All fighters' recent stats
# - Elo ratings up to date
# - Latest fight dates
```

### Step 2: Train/Retrain Model
```python
from src.ensemble_model_best import FightOutcomeModel

# Option A: Train on ALL available data (recommended)
fight_model = FightOutcomeModel(
    'data/tmp/final.csv',
    random_seed=42
)

model, accuracy = fight_model.tune_xgboost_full(
    use_champion_config=True,
    use_rolling_ema=True
)

print(f"Model trained - Accuracy: {accuracy:.2%}")
```

**When to Retrain**:
- ✅ **Monthly**: After every ~8-10 events
- ✅ **Quarterly**: If you want less maintenance
- ⚠️ **After major rule changes**: New weight classes, rule updates
- ⚠️ **If ROI drops below 10%**: For 3+ consecutive events

### Step 3: Generate Predictions for Upcoming Event
```python
# Load upcoming fight card data
# Format: Same as final.csv but with upcoming fights
upcoming_fights = pd.read_csv('data/tmp/upcoming_event.csv')

# Generate predictions
predictions = fight_model.predict_fight_card(upcoming_fights)

# predictions will contain:
# - FIGHTER names
# - model_prob (win probability)
# - opponent info
```

### Step 4: Fetch Current Vegas Odds
```python
# Fetch from your odds provider (e.g., DraftKings, FanDuel)
# You'll need to implement odds fetching based on your source
vegas_odds = fetch_current_odds(event_name="UFC XXX")

# Merge with predictions
picks_with_odds = predictions.merge(vegas_odds, on='FIGHTER')
```

### Step 5: Apply Betting Strategy

#### **Strategy A: High-Confidence Straight Picks** (Higher ROI, More Bets)
```python
# Bet on higher probability fighter in each bout
# Expected: +27-49% ROI, 79-81% win rate

recommendations = []

for bout in picks_with_odds.groupby('BOUT'):
    bout_data = bout[1]
    
    # Pick fighter with higher model probability
    pick = bout_data.loc[bout_data['model_prob'].idxmax()]
    
    recommendations.append({
        'fighter': pick['FIGHTER'],
        'opponent': pick['OPPONENT'],
        'model_prob': pick['model_prob'],
        'vegas_odds': pick['odds'],
        'confidence': 'HIGH' if pick['model_prob'] > 0.70 else 'MEDIUM',
        'bet_amount': calculate_bet_size(pick['model_prob'], pick['odds'])
    })
```

#### **Strategy B: Underdog + Disagreement** (Lower ROI, Fewer Bets, More Stable)
```python
# Only bet underdogs where model disagrees with Vegas by >8%
# Expected: +22.6% ROI, 71.9% win rate

def american_to_implied_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

recommendations = []

for _, row in picks_with_odds.iterrows():
    if row['odds'] > 0:  # Underdog (positive odds)
        implied_prob = american_to_implied_prob(row['odds'])
        disagreement = abs(row['model_prob'] - implied_prob)
        
        if disagreement > 0.08 and row['model_prob'] > 0.50:
            recommendations.append({
                'fighter': row['FIGHTER'],
                'model_prob': row['model_prob'],
                'vegas_implied': implied_prob,
                'disagreement': disagreement,
                'odds': row['odds'],
                'bet_amount': calculate_bet_size(row['model_prob'], row['odds'])
            })
```

### Step 6: Bet Sizing (Kelly Criterion)
```python
def calculate_bet_size(model_prob, vegas_odds, bankroll=10000, kelly_fraction=0.25):
    """
    Calculate bet size using fractional Kelly
    
    kelly_fraction:
      - 0.25 (1/4 Kelly): Conservative, recommended
      - 0.50 (1/2 Kelly): Moderate
      - 1.00 (Full Kelly): Aggressive, high variance
    """
    # Convert to decimal odds
    if vegas_odds > 0:
        decimal_odds = 1 + (vegas_odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(vegas_odds))
    
    # Calculate implied probability from odds
    implied_prob = 1 / decimal_odds
    
    # Calculate edge
    edge = model_prob - implied_prob
    
    # Kelly formula: (edge * decimal_odds - 1) / (decimal_odds - 1)
    if edge > 0:
        kelly = edge / (decimal_odds - 1)
        bet_size = bankroll * kelly * kelly_fraction
        
        # Cap at 5% of bankroll for safety
        bet_size = min(bet_size, bankroll * 0.05)
        
        return max(bet_size, 0)
    else:
        return 0  # No bet if no edge

# Example usage
bankroll = 10000
for pick in recommendations:
    pick['bet_amount'] = calculate_bet_size(
        pick['model_prob'],
        pick['vegas_odds'],
        bankroll=bankroll,
        kelly_fraction=0.25  # Conservative
    )
```

---

## 📊 Post-Fight Workflow

### Step 1: Record Results
```python
# After fights complete, record outcomes
results = {
    'date': event_date,
    'event': event_name,
    'bets': len(recommendations),
    'wins': count_wins,
    'profit': calculate_profit(recommendations, outcomes),
    'roi': calculate_roi(recommendations, outcomes)
}

# Save to tracking spreadsheet
results_df = pd.DataFrame([results])
results_df.to_csv('data/tracking/performance_log.csv', mode='a', header=False)
```

### Step 2: Monitor Performance
```python
# Check rolling performance
recent_results = pd.read_csv('data/tracking/performance_log.csv')
last_10_events = recent_results.tail(10)

roi_last_10 = last_10_events['roi'].mean()
win_rate_last_10 = last_10_events['wins'].sum() / last_10_events['bets'].sum()

print(f"Last 10 events - ROI: {roi_last_10:.2%}, Win Rate: {win_rate_last_10:.2%}")

# Alert if performance drops
if roi_last_10 < 0.10:  # Below 10% for 10 events
    print("⚠️ Performance degrading - consider retraining")
```

---

## 🎯 Recommended Setup

### **For Maximum Profit (Aggressive)**
- **Strategy**: High-confidence straight picks (Strategy A)
- **Training**: Monthly retraining
- **Bet Sizing**: 1/4 Kelly
- **Expected**: +30-40% annual ROI
- **Volume**: ~8 bets per event

### **For Stability (Conservative)**
- **Strategy**: Underdog + disagreement (Strategy B)
- **Training**: Quarterly retraining
- **Bet Sizing**: 1/4 Kelly
- **Expected**: +20-25% annual ROI
- **Volume**: ~1-2 bets per event

### **For Balanced Approach** ⭐ **RECOMMENDED**
- **Strategy**: Combine both
  - Strategy A for 70-80%+ confidence picks
  - Strategy B for selective underdog opportunities
- **Training**: Monthly retraining
- **Bet Sizing**: 1/4 Kelly
- **Expected**: +25-35% annual ROI
- **Volume**: ~5-6 bets per event

---

## ⚠️ Risk Management

### Bankroll Management
```python
INITIAL_BANKROLL = 10000
KELLY_FRACTION = 0.25  # 1/4 Kelly (conservative)
MAX_BET_SIZE = INITIAL_BANKROLL * 0.05  # Never bet >5% on single fight
MIN_BET_SIZE = 10  # Minimum bet

# Update bankroll after each event
current_bankroll = INITIAL_BANKROLL + cumulative_profit
```

### Stop-Loss Rules
- ❌ **Stop betting** if bankroll drops 30%
- ⚠️ **Reduce bet size** if on 5+ bet losing streak
- ⚠️ **Retrain immediately** if 3 consecutive events < 0% ROI

### When to Retrain
✅ **Scheduled**:
- Monthly (after 8-10 events)
- Or quarterly if low maintenance

⚠️ **Emergency Retraining**:
- 3+ consecutive events with negative ROI
- Win rate drops below 60% for 10+ fights
- Major UFC rule changes or meta shifts

---

## 📝 Checklist for Each Event

### Pre-Event (3-7 days before)
- [ ] Update final.csv with latest fighter data
- [ ] Retrain model (if monthly cycle)
- [ ] Generate predictions for upcoming card
- [ ] Fetch current Vegas odds
- [ ] Apply betting strategy
- [ ] Calculate bet sizes (Kelly)
- [ ] Review recommendations
- [ ] Place bets

### Post-Event (after fights)
- [ ] Record actual outcomes
- [ ] Calculate profit/loss
- [ ] Update performance tracking
- [ ] Check rolling metrics
- [ ] Assess if retraining needed

---

## 💻 Complete Prediction Script

```python
# predict_upcoming_event.py
import sys
sys.path.insert(0, 'src')

import pandas as pd
from ensemble_model_best import FightOutcomeModel

def predict_event(event_csv, bankroll=10000, strategy='balanced'):
    """
    Generate predictions and betting recommendations for upcoming event
    
    Args:
        event_csv: Path to CSV with upcoming fights
        bankroll: Current bankroll
        strategy: 'aggressive', 'conservative', or 'balanced'
    """
    
    # 1. Train model on all available data
    print("Training model...")
    model = FightOutcomeModel('data/tmp/final.csv', random_seed=42)
    trained_model, acc = model.tune_xgboost_full(
        use_champion_config=True, 
        use_rolling_ema=True
    )
    print(f"✅ Model trained - Accuracy: {acc:.2%}")
    
    # 2. Load upcoming fights
    upcoming = pd.read_csv(event_csv)
    
    # 3. Generate predictions
    # (You'll need to implement predict_fight_card method)
    predictions = model.predict(upcoming)
    
    # 4. Apply strategy and generate recommendations
    recommendations = apply_strategy(predictions, strategy, bankroll)
    
    # 5. Output recommendations
    print(f"\n{'='*80}")
    print(f"BETTING RECOMMENDATIONS - {strategy.upper()} STRATEGY")
    print(f"{'='*80}\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['fighter']} vs {rec['opponent']}")
        print(f"   Model: {rec['model_prob']:.1%} | Odds: {rec['odds']:+d}")
        print(f"   Recommended Bet: ${rec['bet_amount']:.2f}")
        print()
    
    total_stake = sum(r['bet_amount'] for r in recommendations)
    print(f"Total Stake: ${total_stake:.2f} ({total_stake/bankroll*100:.1f}% of bankroll)")
    
    return recommendations

if __name__ == "__main__":
    # Example usage
    recommendations = predict_event(
        event_csv='data/upcoming/ufc_308.csv',
        bankroll=10000,
        strategy='balanced'
    )
```

---

## 🎓 Key Success Factors

1. **Consistency**: Stick to your strategy, don't chase losses
2. **Discipline**: Only bet on model recommendations
3. **Bankroll Management**: Never exceed 5% per bet
4. **Record Keeping**: Track every bet for performance monitoring
5. **Adaptation**: Retrain when needed, but not too frequently
6. **Patience**: ROI compounds over time (10+ events minimum)

---

## 📈 Expected Long-Term Performance

### Year 1 (Conservative 1/4 Kelly, Balanced Strategy)
- **Bets per month**: ~20-24 (5-6 per event × 4 events)
- **Expected ROI**: 25-30%
- **Starting bankroll**: $10,000
- **Expected end balance**: $12,500-13,000
- **Profit**: $2,500-3,000

### Year 2 (Compounding)
- **Starting bankroll**: $12,750 (avg)
- **Expected ROI**: 25-30%
- **Expected end balance**: $15,950-16,575
- **Profit**: $3,200-3,825

### Year 3 (Compounding)
- **Starting bankroll**: $16,263 (avg)
- **Expected ROI**: 25-30%
- **Expected end balance**: $20,330-21,140
- **Profit**: $4,067-4,877

**5-Year Projection**: $10,000 → $30,500-35,000 (25-30% CAGR)

---

## ⚠️ Important Disclaimers

1. **Past performance ≠ future results**
2. **Variance is high** - expect 20-30% swings month-to-month
3. **Market efficiency may increase** - edge may compress over time
4. **Responsible gambling** - never bet more than you can afford to lose
5. **Legal compliance** - ensure sports betting is legal in your jurisdiction

---

## 🚀 Getting Started Checklist

- [ ] Set up data pipeline for updating fighter stats
- [ ] Create odds fetching system (API or manual)
- [ ] Implement prediction script
- [ ] Set up performance tracking spreadsheet
- [ ] Establish bankroll and risk limits
- [ ] Paper trade for 2-3 events first
- [ ] Start with small bets ($10-20)
- [ ] Scale up after 10+ successful events

---

**Good luck and bet responsibly!** 🍀
