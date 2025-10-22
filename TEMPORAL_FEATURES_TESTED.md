# Temporal Features Tested - Complete Results

This document catalogs all temporal features tested for the UFC fight predictor, including those that didn't provide significant improvements but might be valuable with different feature sets or hyperparameters.

## Test Configuration
- **Baseline**: 68.22% accuracy, 0.6196 log loss (seed 42)
- **Model**: XGBoost with champion hyperparameters
- **Test Set**: Last 365 days (time-series split)
- **Data**: Post-2009, male fights only

---

## 🏆 SUCCESSFUL FEATURES (Beat Baseline on Both Metrics)

### 1. **rolling_ema** ⭐ CHAMPION
- **Type**: Exponential Moving Average of win outcomes
- **Parameters**: `span=200, min_periods=20, shift=1`
- **Results (Seed 42)**: 
  - Accuracy: 69.92% (+1.69%)
  - Log Loss: 0.5648 (-0.0547)
- **Multi-seed**: 10/10 seeds improved (100% success rate)
- **Feature Importance**: Rank 1/29 (0.1524)
- **Statistical Significance**: p < 0.001 (bootstrap test)
- **Why it works**: Exponentially weighted, recent fights have more influence
- **Status**: ✅ **PRODUCTION READY**

### 2. **rolling_wr_100**
- **Type**: Rolling window average of win outcomes
- **Parameters**: `window=100, min_periods=10, shift=1`
- **Results (Seed 42)**:
  - Accuracy: 68.36% (+0.14%)
  - Log Loss: 0.5925 (-0.0271)
- **Why it works**: Shorter window adapts faster to meta-game changes
- **Status**: ✅ Viable alternative, smaller improvement than rolling_ema

---

## ⚠️ MARGINAL FEATURES (Mixed Results or Small Improvements)

### 3. **ufc_era**
- **Type**: Discrete era indicators
- **Parameters**: Pre-2006=0, 2006-2013=0.33, 2014-2020=0.66, 2021+=1.0
- **Results (Seed 42)**:
  - Accuracy: 66.24% (-1.98%)
  - Log Loss: 0.6175 (-0.0021)
- **Analysis**: Slight log loss improvement, accuracy drops
- **Potential**: Might work better with different binning or as interaction term
- **Status**: ⚠️ Not recommended alone, could try with feature engineering

### 4. **days_since_ufc_start**
- **Type**: Normalized days from first UFC event
- **Parameters**: `(date - min_date).days / 10000.0`
- **Results (Seed 42)**:
  - Accuracy: 67.51% (-0.71%)
  - Log Loss: 0.6203 (+0.0007)
- **Analysis**: Slight degradation on both metrics
- **Potential**: Linear temporal encoding might be too simplistic
- **Status**: ⚠️ Could work as interaction term with other features

### 5. **fight_index**
- **Type**: Normalized position in dataset
- **Parameters**: `np.arange(len(df)) / len(df)`
- **Results (Seed 42)**:
  - Accuracy: 67.51% (-0.71%)
  - Log Loss: 0.6223 (+0.0027)
- **Analysis**: Similar to days_since_ufc_start
- **Status**: ⚠️ Not recommended

---

## ❌ UNSUCCESSFUL FEATURES (Hurt Performance)

### 6. **rolling_wr_250**
- **Type**: Rolling window average (medium window)
- **Parameters**: `window=250, min_periods=20, shift=1`
- **Results (Seed 42)**:
  - Accuracy: 65.25% (-2.97%)
  - Log Loss: 0.6100 (-0.0096)
- **Analysis**: Log loss improves but accuracy drops significantly
- **Why it failed**: Window too large, not adaptive enough
- **Status**: ❌ Not recommended

### 7. **rolling_wr_625**
- **Type**: Rolling window average (large window)
- **Parameters**: `window=625, min_periods=20, shift=1`
- **Results (Seed 42)**:
  - Accuracy: 66.53% (-1.69%)
  - Log Loss: 0.6113 (-0.0083)
- **Analysis**: Initially seemed promising but hurt accuracy when done correctly
- **Why it failed**: Window too large, over-smooths temporal patterns
- **Status**: ❌ Not recommended

### 8. **rolling_volatility**
- **Type**: Rolling standard deviation of outcomes
- **Parameters**: `window=250, min_periods=20, shift=1`
- **Results (Seed 42)**:
  - Accuracy: 64.41% (-3.81%)
  - Log Loss: 0.6211 (+0.0015)
- **Analysis**: Hurts both metrics
- **Why it failed**: Volatility of binary outcomes not informative
- **Status**: ❌ Not recommended

### 9. **year_norm**
- **Type**: Normalized year value
- **Parameters**: `(year - 2000) / 100.0`
- **Results (Seed 42)**:
  - Accuracy: 64.83% (-3.39%)
  - Log Loss: 0.6274 (+0.0078)
- **Analysis**: Significantly hurts both metrics
- **Why it failed**: Too simplistic, doesn't capture non-linear temporal patterns
- **Status**: ❌ Not recommended

### 10. **normalized_ufc_timeline**
- **Type**: Linear normalization of fight chronology
- **Parameters**: Simple 0-1 scaling by order
- **Results**: Initial tests showed ~2% accuracy drop
- **Why it failed**: Linear assumption doesn't match UFC evolution
- **Status**: ❌ Not recommended

---

## 🧪 FEATURES NOT YET TESTED (Future Exploration)

These features showed promise in initial analysis but weren't fully tested:

### 1. **Seasonal/Cyclical Features**
- **Concept**: Month/quarter as sine/cosine encoding
- **Rationale**: UFC schedule patterns, fighter preparation cycles
- **Priority**: Medium
- **Potential Issues**: Small effect size, needs large dataset

### 2. **Event-Type Indicators**
- **Concept**: PPV vs Fight Night temporal encoding
- **Rationale**: Different competition levels at different event types
- **Priority**: Low
- **Potential Issues**: Correlation with existing Elo features

### 3. **Rule Change Indicators**
- **Concept**: Binary flags for major UFC rule changes
- **Dates**: 2009 (unified rules), 2017 (new weight classes), etc.
- **Rationale**: Rules affect fight outcomes
- **Priority**: Medium
- **Potential Issues**: Few change points, limited data

### 4. **Rolling Meta-Game Features**
- **Concepts Tested Previously**:
  - `rolling_finish_rate`: Average finish rate over time
  - `rolling_ko_rate`: KO rate trends
  - `rolling_sub_rate`: Submission rate trends
  - `rolling_meta_tdavg`: Takedown trends
- **Results**: Not properly tested due to data leakage issues in initial implementation
- **Priority**: High (worth revisiting with proper leak prevention)
- **Status**: Needs reimplementation

### 5. **Temporal Interaction Features**
- **Concept**: Multiply fighter stats by temporal value
- **Example**: `precomp_tdavg * rolling_ema`
- **Rationale**: Style effectiveness changes over time
- **Priority**: High
- **Status**: Initial tests showed promise but needed more tuning

### 6. **Adaptive Window Rolling Features**
- **Concept**: Rolling features with dynamically adjusted windows
- **Example**: Smaller windows in recent years (faster meta changes)
- **Priority**: Medium
- **Status**: Complex to implement correctly

### 7. **Regime Change Detection**
- **Concept**: Automatic detection of UFC meta-game regime changes
- **Method**: Changepoint detection algorithms
- **Priority**: Low (complex)
- **Status**: Research phase

---

## 📊 Summary Statistics

### Success Rate by Category
| Category | Count | Success Rate | Best Improvement |
|----------|-------|--------------|------------------|
| EMA-based | 1 | 100% | +1.69% acc, -0.0547 LL |
| Simple rolling | 3 | 33% | +0.14% acc, -0.0271 LL |
| Linear encoding | 3 | 0% | -0.71% acc |
| Discrete bins | 1 | 0% | -1.98% acc |
| Volatility | 1 | 0% | -3.81% acc |

### Key Insights

1. **Exponential weighting > Simple averaging**
   - EMA (69.92%) significantly outperforms rolling average (68.36%)
   - Recent data should have more influence

2. **Window size matters**
   - Too small (100): Works but limited improvement
   - Too large (625): Over-smooths, hurts accuracy
   - Optimal (200 for EMA): Balances responsiveness and stability

3. **Binary outcome volatility is not predictive**
   - Rolling standard deviation of wins/losses has no signal
   - Outcomes are already binary, variance doesn't add information

4. **Linear time encoding fails**
   - UFC evolution is non-linear
   - Simple timeline features don't capture meta-game shifts

5. **Adaptive methods win**
   - EMA adapts to recent changes
   - Fixed windows lag behind meta-game evolution

---

## 🔬 Recommendations for Future Work

### High Priority
1. **Test rolling_ema variants**
   - Different spans (150, 250, 300)
   - Different weighting schemes
   - Combine with rolling_wr_100

2. **Proper implementation of meta-game rolling features**
   - Finish rate, KO rate, submission rate
   - With correct leak prevention (shift=1)
   - Test as standalone and with rolling_ema

3. **Temporal interaction terms**
   - `fighter_stat * rolling_ema`
   - Focus on style-dependent stats (tdavg, subavg, sigstr_pm)

### Medium Priority
4. **Rule change indicators**
   - Binary flags for major changes
   - Test with interaction terms

5. **Seasonal encoding**
   - Sine/cosine of month
   - Quarter indicators

6. **Event-type temporal patterns**
   - PPV vs Fight Night trends over time

### Low Priority
7. **Regime change detection**
   - Automatic changepoint detection
   - Complex but could be powerful

8. **Adaptive windows**
   - Dynamically adjust rolling window size
   - Engineering effort vs potential gain unclear

---

## 💡 Lessons Learned

### What Works
- ✅ Exponential weighting (recent bias)
- ✅ Moderate window sizes (100-200 fights)
- ✅ Proper data leakage prevention (shift=1)
- ✅ Features that capture meta-game trends

### What Doesn't Work
- ❌ Linear time encoding
- ❌ Very large windows (over-smoothing)
- ❌ Volatility of binary outcomes
- ❌ Overly simplistic temporal binning

### Critical Success Factors
1. **Leak prevention**: Always use `.shift(1)` on rolling calculations
2. **Proper data ordering**: Sort by date, then restore original order
3. **Validation**: Test on proper time-series splits
4. **Statistical testing**: Bootstrap for significance
5. **Multi-seed testing**: Ensure robustness across random seeds

---

## 📁 Related Files

- **Production Dataset**: `data/tmp/final_with_rolling_ema.csv`
- **Validation Script**: `comprehensive_validation.py`
- **Leakage Test**: `final_leakage_test.py`
- **Documentation**: 
  - `TEMPORAL_FEATURE_SUCCESS.md` (rolling_ema details)
  - `ROLLING_EMA_QUICKSTART.md` (usage guide)
  - `TEMPORAL_FEATURES_TESTED.md` (this file)

---

## 🎯 Bottom Line

**Current Champion: `rolling_ema`**
- Proven performance: +1.69% accuracy, -0.0547 log loss
- Statistically significant: p < 0.001
- Robust: 10/10 seeds improved
- Production ready: Zero data leakage

**Future Potential:**
Several features showed marginal improvements and could be valuable with:
- Different hyperparameters
- Feature combinations
- Interaction terms
- Proper meta-game rolling features

The temporal feature exploration was successful in finding a significant improvement. Continue iterating on the lessons learned above for further gains.

