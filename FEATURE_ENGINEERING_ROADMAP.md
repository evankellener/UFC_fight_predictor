# UFC Fight Predictor - Feature Engineering Roadmap

## Overview
This document outlines advanced feature engineering and statistical manipulation strategies to enhance model performance. Features are organized by pipeline layer for systematic implementation.

---

## Pipeline Architecture & Implementation Layers

```
Layer 1: SQL Data Processing (test.sql, ufc_sql_weightclass.sql)
    ↓
Layer 2: ELO Systems (gpt_elo.py, striking_elo.py, grappling_elo.py)
    ↓
Layer 3: Feature Engineering (ensemble_model_best.py - data preprocessing)
    ↓
Layer 4: Model Input & Selection (ensemble_model_best.py - feature selection)
    ↓
Layer 5: Model Training & Ensemble
```

---

## LAYER 1: SQL Data Processing Layer
**Files:** `src/sql_scripts/test.sql`, `src/sql_scripts/ufc_sql_weightclass.sql`

### 1.1 Weight Class Normalization
**Purpose:** Standardize stats across weight classes (HW vs Flyweight have different natural rates)

```sql
-- Calculate division medians for normalization
CREATE TEMP TABLE division_medians AS
SELECT 
    weight_class,
    MEDIAN(sigstr_pm) as median_sigstr_pm,
    MEDIAN(tdavg) as median_tdavg,
    MEDIAN(sapm) as median_sapm,
    MEDIAN(finish_rate) as median_finish_rate
FROM fighter_stats
GROUP BY weight_class;

-- Add normalized stats
SELECT 
    f.*,
    f.sigstr_pm / NULLIF(dm.median_sigstr_pm, 0) as sigstr_pm_normalized,
    f.tdavg / NULLIF(dm.median_tdavg, 0) as tdavg_normalized,
    f.sapm / NULLIF(dm.median_sapm, 0) as sapm_normalized,
    f.finish_rate / NULLIF(dm.median_finish_rate, 0) as finish_rate_normalized
FROM fighter_stats f
LEFT JOIN division_medians dm ON f.weight_class = dm.weight_class;
```

**New Features:**
- `precomp_sigstr_pm_norm`, `precomp_tdavg_norm`, `precomp_sapm_norm`
- `precomp_finish_rate_norm`
- All with `_norm3` and `_norm5` variants for rolling windows

### 1.2 Advanced Efficiency Metrics
**Purpose:** Capture combat effectiveness beyond raw volume

```sql
-- Strike Differential Rate (more stable than raw differential)
CASE 
    WHEN IFNULL(SUM(totalatt) OVER wu0, 0) = 0 THEN 0.0
    ELSE (IFNULL(SUM(sigstracc) OVER wu0, 0.0) - IFNULL(SUM(sigstrabs) OVER wu0, 0.0)) 
        / IFNULL(SUM(totalatt) OVER wu0, 0.0)
END AS precomp_strike_diff_rate,

-- TD Efficiency Score (comprehensive grappling metric)
CASE 
    WHEN IFNULL(SUM(1) OVER wu0, 0) = 0 THEN 0.0
    ELSE (AVG(tdacc_perc) OVER wu0 * AVG(CAST(tdavg AS FLOAT)) OVER wu0) / 
         NULLIF((AVG(opp_tdacc_perc) OVER wu0 * AVG(CAST(opp_tdavg AS FLOAT)) OVER wu0), 0)
END AS precomp_td_efficiency_score,

-- Control Efficiency (productivity in top position)
CASE 
    WHEN (IFNULL(SUM(CAST(tdacc AS FLOAT)) OVER wu0, 0.0) + 
          IFNULL(SUM(CAST(clinchatt AS FLOAT)) OVER wu0, 0.0)) = 0 THEN 0.0
    ELSE IFNULL(SUM(CAST(ctrl AS FLOAT)) OVER wu0, 0.0) / 
         (IFNULL(SUM(CAST(tdacc AS FLOAT)) OVER wu0, 0.0) + 
          IFNULL(SUM(CAST(clinchatt AS FLOAT)) OVER wu0, 0.0))
END AS precomp_control_efficiency,

-- Damage Per Significant Strike
CASE 
    WHEN IFNULL(SUM(sigstracc) OVER wu0, 0) = 0 THEN 0.0
    ELSE IFNULL(SUM(CAST(kd AS FLOAT)) OVER wu0, 0.0) / 
         IFNULL(SUM(sigstracc) OVER wu0, 0.0)
END AS precomp_damage_per_strike,

-- Durability Score
CASE 
    WHEN IFNULL(SUM(sigstrabs) OVER wu0, 0) = 0 THEN 1.0
    ELSE 1.0 - (IFNULL(SUM(CAST(kdabs AS FLOAT)) OVER wu0, 0.0) / 
                IFNULL(SUM(sigstrabs) OVER wu0, 0.0))
END AS precomp_durability_score
```

**New Features:**
- `precomp_strike_diff_rate`, `precomp_strike_diff_rate3`, `precomp_strike_diff_rate5`
- `precomp_td_efficiency_score`, `precomp_td_efficiency_score3`, `precomp_td_efficiency_score5`
- `precomp_control_efficiency`, `precomp_control_efficiency3`, `precomp_control_efficiency5`
- `precomp_damage_per_strike`, `precomp_damage_per_strike3`, `precomp_damage_per_strike5`
- `precomp_durability_score`, `precomp_durability_score3`, `precomp_durability_score5`

### 1.3 Activity & Layoff Metrics
**Purpose:** Capture ring rust and fight frequency effects

```sql
-- Time since last fight (days)
LAG(DATE) OVER (PARTITION BY FIGHTER ORDER BY DATE) as last_fight_date,
julianday(DATE) - julianday(LAG(DATE) OVER (PARTITION BY FIGHTER ORDER BY DATE)) as days_since_last_fight,

-- Fight frequency (fights in last 12/24 months)
SUM(CASE WHEN julianday(DATE) - julianday(DATE) <= 365 THEN 1 ELSE 0 END) 
    OVER (PARTITION BY FIGHTER ORDER BY DATE ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) as fights_last_12mo,
    
SUM(CASE WHEN julianday(DATE) - julianday(DATE) <= 730 THEN 1 ELSE 0 END) 
    OVER (PARTITION BY FIGHTER ORDER BY DATE ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as fights_last_24mo
```

**New Features:**
- `days_since_last_fight`
- `fights_last_12mo`, `fights_last_24mo`

### 1.4 Strength of Schedule
**Purpose:** Adjust for quality of opposition

```sql
-- Average opponent ELO over last 3, 5 fights
AVG(opp_precomp_elo) OVER (PARTITION BY FIGHTER ORDER BY DATE ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) as precomp_avg_opp_elo3,
AVG(opp_precomp_elo) OVER (PARTITION BY FIGHTER ORDER BY DATE ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as precomp_avg_opp_elo5,

-- Average opponent finish rate faced
AVG(opp_precomp_finish_rate) OVER (PARTITION BY FIGHTER ORDER BY DATE ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) as precomp_avg_opp_finish_rate3,

-- Step up/down in competition
opp_precomp_elo - AVG(opp_precomp_elo) OVER (PARTITION BY FIGHTER ORDER BY DATE ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) as precomp_competition_step
```

**New Features:**
- `precomp_avg_opp_elo3`, `precomp_avg_opp_elo5`
- `precomp_avg_opp_finish_rate3`
- `precomp_competition_step`

---

## LAYER 2: ELO Systems Enhancement
**Files:** `src/gpt_elo.py`, `src/striking_elo.py`, `src/grappling_elo.py`

### 2.1 ELO Derivatives (Time-Series Features)
**Purpose:** Capture momentum and trajectory beyond point-in-time ratings

Add to each ELO system's `process_fights()` method:

```python
def calculate_elo_derivatives(self, df):
    """Calculate velocity and acceleration of ELO changes"""
    df = df.sort_values(['FIGHTER', 'DATE'])
    
    # ELO Velocity (rate of change over last 3 fights)
    df['precomp_elo_velocity'] = df.groupby('FIGHTER')['precomp_elo'].diff(3) / 3
    
    # ELO Acceleration (second derivative)
    df['precomp_elo_acceleration'] = df.groupby('FIGHTER')['precomp_elo_velocity'].diff()
    
    # Mean reversion signal
    df['precomp_elo_career_avg'] = df.groupby('FIGHTER')['precomp_elo'].expanding().mean().reset_index(level=0, drop=True)
    df['precomp_elo_career_std'] = df.groupby('FIGHTER')['precomp_elo'].expanding().std().reset_index(level=0, drop=True)
    df['precomp_elo_reversion_signal'] = (df['precomp_elo'] - df['precomp_elo_career_avg']) / df['precomp_elo_career_std'].replace(0, 1)
    
    return df
```

**New Features (per ELO system):**
- `precomp_elo_velocity`, `precomp_strike_elo_velocity`, `precomp_grapple_elo_velocity`
- `precomp_elo_acceleration`, `precomp_strike_elo_acceleration`, `precomp_grapple_elo_acceleration`
- `precomp_elo_reversion_signal`, `precomp_strike_elo_reversion_signal`, `precomp_grapple_elo_reversion_signal`

### 2.2 Performance Volatility
**Purpose:** Capture consistency vs unpredictability

```python
def calculate_performance_volatility(self, df):
    """Calculate volatility metrics for key performance indicators"""
    df = df.sort_values(['FIGHTER', 'DATE'])
    
    # Strike differential volatility (last 3-5 fights)
    df['precomp_str_diff_volatility3'] = df.groupby('FIGHTER')['precomp_str_eff_diff'].rolling(3, min_periods=2).std().reset_index(level=0, drop=True)
    df['precomp_str_diff_volatility5'] = df.groupby('FIGHTER')['precomp_str_eff_diff'].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    
    # ELO volatility
    df['precomp_elo_volatility3'] = df.groupby('FIGHTER')['precomp_elo'].rolling(3, min_periods=2).std().reset_index(level=0, drop=True)
    df['precomp_elo_volatility5'] = df.groupby('FIGHTER')['precomp_elo'].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    
    return df
```

**New Features:**
- `precomp_str_diff_volatility3`, `precomp_str_diff_volatility5`
- `precomp_elo_volatility3`, `precomp_elo_volatility5`

---

## LAYER 3: Feature Engineering Layer
**File:** `src/ensemble_model_best.py` (in `__init__` or new `engineer_features()` method)

### 3.1 Matchup-Specific Interaction Features
**Purpose:** Capture style clashes and tactical advantages

```python
def create_matchup_interactions(self, df):
    """Create interaction features that capture stylistic matchups"""
    
    # Striker vs Grappler advantage
    df['style_clash_advantage'] = (
        (df['precomp_strike_elo'] - df['precomp_grapple_elo']) * 
        (df['opp_precomp_grapple_elo'] - df['opp_precomp_strike_elo'])
    )
    
    # Range mismatch (reach advantage amplified by distance striking)
    df['reach_differential'] = df['REACH'] - df['opp_REACH']
    df['range_mismatch_advantage'] = (
        df['reach_differential'] * df['precomp_distacc_perc']
    )
    
    # Size differential impact (weight advantage amplified by skill gap)
    df['weight_differential'] = df['WEIGHT'] - df['opp_WEIGHT']
    df['size_skill_advantage'] = (
        df['weight_differential'] * np.abs(df['precomp_elo_diff'])
    )
    
    # Defensive efficiency vs offensive output clash
    df['defense_offense_matchup'] = (
        df['precomp_strdef'] * df['opp_precomp_sigstr_pm']
    )
    
    # Grappling defense vs opponent's grappling offense
    df['grappling_clash'] = (
        df['precomp_tddef'] * df['opp_precomp_tdavg']
    )
    
    # Submission threat vs opponent's submission defense
    df['submission_matchup'] = (
        df['precomp_subavg'] * (100 - df['opp_precomp_tddef'])  # Using tddef as proxy for sub defense
    )
    
    # Pressure fighter score (volume + aggression despite poor defense)
    df['pressure_score'] = (
        (df['precomp_sigstr_pm'] + df['precomp_tdavg']) * 
        (1 / (df['precomp_strdef'].replace(0, 1)))
    )
    df['opp_pressure_score'] = (
        (df['opp_precomp_sigstr_pm'] + df['opp_precomp_tdavg']) * 
        (1 / (df['opp_precomp_strdef'].replace(0, 1)))
    )
    
    return df
```

**New Features:**
- `style_clash_advantage`
- `reach_differential`, `range_mismatch_advantage`
- `weight_differential`, `size_skill_advantage`
- `defense_offense_matchup`, `grappling_clash`, `submission_matchup`
- `pressure_score`, `opp_pressure_score`

### 3.2 Higher-Order Fighter Profile Features
**Purpose:** Capture fighter archetypes and well-roundedness

```python
def create_fighter_profiles(self, df):
    """Create composite features describing fighter archetypes"""
    
    # Well-rounded score (inverse of variance across skills)
    skill_variance = df[[
        'precomp_strike_elo', 
        'precomp_grapple_elo', 
        'precomp_finish_rate'
    ]].std(axis=1)
    df['well_rounded_score'] = 1 / (skill_variance.replace(0, 1))
    df['opp_well_rounded_score'] = 1 / (df[[
        'opp_precomp_strike_elo', 
        'opp_precomp_grapple_elo', 
        'opp_precomp_finish_rate'
    ]].std(axis=1).replace(0, 1))
    
    # Experience x Youth (prime indicator)
    df['prime_indicator'] = (
        np.log1p(df['precomp_winsum'] + df['precomp_losssum']) * 
        (1 / df['age'])
    )
    df['opp_prime_indicator'] = (
        np.log1p(df['opp_precomp_winsum'] + df['opp_precomp_losssum']) * 
        (1 / df['opp_age'])
    )
    
    # Finisher profile (aggressive finishing ability)
    df['finisher_profile'] = (
        df['precomp_finish_rate'] * 
        (df['precomp_sigstr_pm'] + df['precomp_subavg']) * 
        df['precomp_damage_per_strike']  # Requires Layer 1 implementation
    )
    df['opp_finisher_profile'] = (
        df['opp_precomp_finish_rate'] * 
        (df['opp_precomp_sigstr_pm'] + df['opp_precomp_subavg']) * 
        df['opp_precomp_damage_per_strike']
    )
    
    # Gatekeeper score (consistently competitive but doesn't finish)
    df['gatekeeper_score'] = (
        (df['precomp_winsum'] + df['precomp_losssum']) / 
        ((df['precomp_finish_rate'].replace(0, 0.1) + df['precomp_finish_rate3'].replace(0, 0.1)) / 2)
    )
    
    return df
```

**New Features:**
- `well_rounded_score`, `opp_well_rounded_score`
- `prime_indicator`, `opp_prime_indicator`
- `finisher_profile`, `opp_finisher_profile`
- `gatekeeper_score`

### 3.3 Non-Linear Transformations of Existing Features
**Purpose:** Capture non-linear relationships in key differentials

```python
def create_nonlinear_transformations(self, df):
    """Apply non-linear transformations to capture complex relationships"""
    
    # Polynomial ELO differential (advantage is non-linear)
    df['precomp_elo_diff_squared'] = df['precomp_elo_diff'] ** 2
    df['precomp_elo_diff_cubed'] = df['precomp_elo_diff'] ** 3
    df['precomp_strike_elo_diff_squared'] = df['precomp_strike_elo_diff'] ** 2
    df['precomp_grapple_elo_diff_squared'] = df['precomp_grapple_elo_diff'] ** 2
    
    # Log transformations for skewed distributions
    df['log_fight_experience'] = np.log1p(df['precomp_winsum'] + df['precomp_losssum'])
    df['opp_log_fight_experience'] = np.log1p(df['opp_precomp_winsum'] + df['opp_precomp_losssum'])
    
    # Square root for count data
    df['sqrt_sig_strikes_pm'] = np.sqrt(df['precomp_sigstr_pm'])
    df['opp_sqrt_sig_strikes_pm'] = np.sqrt(df['opp_precomp_sigstr_pm'])
    
    # Sigmoid for probabilities/percentages (bound to 0-1)
    df['sigmoid_strike_accuracy'] = 1 / (1 + np.exp(-df['precomp_sigstr_perc'] / 10))
    df['opp_sigmoid_strike_accuracy'] = 1 / (1 + np.exp(-df['opp_precomp_sigstr_perc'] / 10))
    
    # Binned ELO differential with style interactions
    df['elo_tier'] = pd.cut(df['precomp_elo_diff'], 
                              bins=[-np.inf, -100, -50, 50, 100, np.inf],
                              labels=['large_underdog', 'underdog', 'even', 'favorite', 'large_favorite'])
    
    # Interaction: ELO tier x Style advantage
    for tier in ['large_underdog', 'underdog', 'even', 'favorite', 'large_favorite']:
        df[f'elo_tier_{tier}'] = (df['elo_tier'] == tier).astype(int)
        df[f'elo_tier_{tier}_x_style'] = df[f'elo_tier_{tier}'] * df['style_clash_advantage']
    
    return df
```

**New Features:**
- `precomp_elo_diff_squared`, `precomp_elo_diff_cubed`
- `precomp_strike_elo_diff_squared`, `precomp_grapple_elo_diff_squared`
- `log_fight_experience`, `opp_log_fight_experience`
- `sqrt_sig_strikes_pm`, `opp_sqrt_sig_strikes_pm`
- `sigmoid_strike_accuracy`, `opp_sigmoid_strike_accuracy`
- `elo_tier_*` binned features
- `elo_tier_*_x_style` interaction features

### 3.4 Contextual & Situational Features
**Purpose:** Capture psychological and situational factors

```python
def create_contextual_features(self, df):
    """Create features based on fight context and fighter situation"""
    
    # Ring rust (non-linear layoff penalty)
    df['ring_rust_penalty'] = np.log1p(df['days_since_last_fight'])
    
    # Layoff x Age interaction (older fighters suffer more)
    df['layoff_age_penalty'] = df['ring_rust_penalty'] * df['age']
    df['opp_layoff_age_penalty'] = np.log1p(df.get('opp_days_since_last_fight', 0)) * df['opp_age']
    
    # Win requirement (losing streak desperation)
    df['current_losing_streak'] = df.groupby('FIGHTER').apply(
        lambda x: x['result'].rolling(window=5, min_periods=1).apply(
            lambda y: len(y) - y.iloc[-1] - sum(y.iloc[:-1]) if len(y) > 1 else 0
        )
    ).reset_index(level=0, drop=True)
    df['desperation_factor'] = df['current_losing_streak'] * 0.5  # Weight
    
    # Win streak momentum
    df['current_win_streak'] = df.groupby('FIGHTER').apply(
        lambda x: x['result'].rolling(window=5, min_periods=1).apply(
            lambda y: sum(y) if y.iloc[-1] == 1 else 0
        )
    ).reset_index(level=0, drop=True)
    
    # Hot hand effect (3+ wins in 12 months with high finish rate)
    df['hot_hand'] = (
        (df['current_win_streak'] >= 3) & 
        (df['fights_last_12mo'] >= 3) & 
        (df['precomp_finish_rate3'] > 0.6)
    ).astype(int)
    
    # Strength of schedule differential
    df['sos_differential'] = df['precomp_avg_opp_elo3'] - df['opp_precomp_avg_opp_elo3']
    
    # Competition step * ELO differential (step-up fighters as underdogs are dangerous)
    df['step_up_underdog'] = df['precomp_competition_step'] * df['precomp_elo_diff']
    
    return df
```

**New Features:**
- `ring_rust_penalty`
- `layoff_age_penalty`, `opp_layoff_age_penalty`
- `current_losing_streak`, `desperation_factor`
- `current_win_streak`, `hot_hand`
- `sos_differential`
- `step_up_underdog`

### 3.5 Bayesian Smoothing for Low Sample Sizes
**Purpose:** Regularize stats for inexperienced fighters

```python
def apply_bayesian_smoothing(self, df, k=3):
    """Apply Bayesian smoothing to fighter stats based on experience"""
    
    # Calculate division averages
    division_stats = df.groupby('weight_class').agg({
        'precomp_sigstr_pm': 'median',
        'precomp_tdavg': 'median',
        'precomp_finish_rate': 'median',
        'precomp_strdef': 'median'
    }).to_dict('index')
    
    # Number of fights for each fighter
    df['total_fights'] = df['precomp_winsum'] + df['precomp_losssum']
    
    # Apply smoothing: smoothed = (n * stat + k * division_avg) / (n + k)
    for stat in ['precomp_sigstr_pm', 'precomp_tdavg', 'precomp_finish_rate', 'precomp_strdef']:
        df[f'{stat}_smoothed'] = df.apply(
            lambda row: (
                (row['total_fights'] * row[stat] + 
                 k * division_stats.get(row['weight_class'], {}).get(stat, row[stat])) /
                (row['total_fights'] + k)
            ) if row['total_fights'] < 5 else row[stat],
            axis=1
        )
    
    return df
```

**New Features:**
- `precomp_sigstr_pm_smoothed`, `precomp_tdavg_smoothed`
- `precomp_finish_rate_smoothed`, `precomp_strdef_smoothed`

---

## LAYER 4: Model Input & Feature Selection
**File:** `src/ensemble_model_best.py` (update `importance_columns`)

### 4.1 Meta-Learning Features
**Purpose:** Capture model uncertainty and disagreement

```python
def create_meta_features(self, df, probs_dict):
    """Create features from model predictions and disagreements"""
    
    # Requires predictions from sub-models (strike_elo only, grapple_elo only, etc.)
    # This would be implemented during prediction phase
    
    # ELO system disagreement
    df['elo_disagreement_score'] = np.abs(
        df['precomp_strike_elo_diff'] - df['precomp_grapple_elo_diff']
    )
    
    # Variance in ELO systems (high variance = toss-up)
    df['elo_systems_variance'] = df[[
        'precomp_elo_diff',
        'precomp_strike_elo_diff', 
        'precomp_grapple_elo_diff'
    ]].var(axis=1)
    
    # Confidence interval based on historical performance variance
    df['prediction_confidence'] = 1 / (df['precomp_elo_volatility5'].replace(0, 1))
    
    return df
```

**New Features:**
- `elo_disagreement_score`
- `elo_systems_variance`
- `prediction_confidence`

### 4.2 Feature Selection Strategy

After implementing all features, use genetic algorithm or greedy forward search to identify optimal subset:

**Priority Groups for Testing:**

1. **Tier 1 - High Priority (Test First):**
   - Matchup interactions: `style_clash_advantage`, `range_mismatch_advantage`, `defense_offense_matchup`
   - Non-linear ELO: `precomp_elo_diff_squared`, `precomp_strike_elo_diff_squared`
   - Strength of schedule: `precomp_avg_opp_elo3`, `sos_differential`
   - Performance volatility: `precomp_str_diff_volatility3`, `precomp_elo_volatility5`

2. **Tier 2 - Medium Priority:**
   - Fighter profiles: `well_rounded_score`, `prime_indicator`, `finisher_profile`
   - Contextual: `hot_hand`, `layoff_age_penalty`, `step_up_underdog`
   - Advanced efficiency: `precomp_strike_diff_rate`, `precomp_td_efficiency_score`

3. **Tier 3 - Experimental:**
   - Meta-features: `elo_disagreement_score`, `elo_systems_variance`
   - Bayesian smoothing: `*_smoothed` features
   - Weight normalization: `*_norm` features

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Implement Layer 1 SQL features
  - Weight class normalization
  - Advanced efficiency metrics
  - Activity/layoff metrics
  - Strength of schedule

### Phase 2: ELO Enhancements (Week 2)
- [ ] Add ELO derivatives to all ELO systems
- [ ] Implement performance volatility calculations
- [ ] Test individual ELO system improvements

### Phase 3: Core Feature Engineering (Week 3)
- [ ] Implement Tier 1 matchup interactions
- [ ] Add non-linear transformations
- [ ] Create fighter profile features
- [ ] Test Tier 1 features with greedy search

### Phase 4: Advanced Features (Week 4)
- [ ] Implement Tier 2 contextual features
- [ ] Add Bayesian smoothing
- [ ] Create meta-learning features
- [ ] Run comprehensive feature selection (GA)

### Phase 5: Optimization (Week 5)
- [ ] A/B test feature combinations
- [ ] Optimize hyperparameters with new features
- [ ] Validate on holdout set
- [ ] Production deployment

---

## Testing & Validation Protocol

### For Each New Feature Set:

1. **Baseline Comparison:**
   ```python
   # Test with current GA-optimized 28 features
   baseline_accuracy = model.test_accuracy()
   baseline_log_loss = model.test_log_loss()
   baseline_roi = model.calculate_roi()
   ```

2. **Single Feature Addition:**
   ```python
   # Test each new feature individually
   for new_feature in new_features:
       test_features = current_features + [new_feature]
       improvement = model.test_with_features(test_features)
       if improvement > threshold:
           approved_features.append(new_feature)
   ```

3. **Greedy Forward Search:**
   ```python
   # Use existing greedy_forward_search method
   model.greedy_forward_search(
       initial_features=approved_features,
       convergence_threshold=0.001,
       max_iterations=50
   )
   ```

4. **Genetic Algorithm:**
   ```python
   # Final optimization with all approved features
   # Use existing genetic_long_run.py
   python genetic_long_run.py --features approved_features.json
   ```

### Success Metrics:

- **Primary:** Log Loss improvement > 1%
- **Secondary:** Accuracy improvement > 0.5%
- **Tertiary:** ROI improvement > 5%
- **Validation:** Performance holds on 2024-2025 test set

---

## Feature Dependencies Map

```
Layer 1 (SQL) → Creates base stats
    ↓
Layer 2 (ELO) → Uses base stats to calculate ratings
    ↓
Layer 3 (Feature Engineering) → Uses base stats + ELO
    ↓
Layer 4 (Model Input) → Uses all engineered features
    ↓
Layer 5 (Model) → Selects optimal subset
```

**Critical Dependencies:**

- Matchup interactions → Requires Layer 1 & 2 complete
- Fighter profiles → Requires Layer 1 efficiency metrics
- Meta features → Requires all ELO systems from Layer 2
- Bayesian smoothing → Requires weight class normalization from Layer 1

---

## Code Organization Recommendations

### New Files to Create:

1. `src/feature_engineering.py` - Central feature engineering module
   ```python
   class FeatureEngineer:
       def __init__(self, df):
           self.df = df
       
       def create_all_features(self):
           self.create_matchup_interactions()
           self.create_fighter_profiles()
           self.create_nonlinear_transformations()
           self.create_contextual_features()
           return self.df
   ```

2. `src/sql_scripts/advanced_features.sql` - New SQL features

3. `tests/test_new_features.py` - Unit tests for new features
   ```python
   def test_style_clash_advantage():
       # Striker vs grappler should have positive advantage
       assert feature > 0
   
   def test_weight_normalization():
       # Normalized stats should be close to 1.0 on average
       assert 0.8 < feature.mean() < 1.2
   ```

### Integration Points:

Update `ensemble_model_best.py`:
```python
from src.feature_engineering import FeatureEngineer

class FightOutcomeModel:
    def __init__(self, file_path, ...):
        # ... existing code ...
        
        # Add feature engineering step
        engineer = FeatureEngineer(self.df)
        self.df = engineer.create_all_features()
        
        # Update importance_columns with new features
        self.importance_columns = self.load_feature_config()
```

---

## Expected Performance Gains

Based on similar feature engineering in sports analytics:

- **Matchup interactions:** 2-4% log loss improvement
- **Non-linear transformations:** 1-2% improvement  
- **Strength of schedule:** 1-3% improvement
- **Performance volatility:** 1-2% improvement
- **Combined optimally:** 5-10% total improvement possible

**Realistic Target:** 5-7% log loss improvement with proper feature selection

---

## Notes & Warnings

⚠️ **Overfitting Risk:** Each new feature increases overfitting potential. Use cross-validation religiously.

⚠️ **Data Leakage:** Ensure all features use only "precomp" (pre-fight) data, never "postcomp".

⚠️ **Multicollinearity:** Many interaction features will be correlated. Use regularization (L1/L2) or recursive feature elimination.

⚠️ **Computational Cost:** Some features (especially in SQL layer) may slow down data processing. Profile and optimize.

💡 **Pro Tip:** Implement features in batches, test thoroughly, and only promote to production if validated on holdout set.

💡 **Version Control:** Tag each feature set version in git for reproducibility.

---

## References & Inspiration

- **ELO momentum features:** Inspired by FiveThirtyEight's NFL ELO system
- **Matchup interactions:** Common in chess rating systems
- **Bayesian smoothing:** Standard in MLB/NBA player projections
- **Non-linear advantages:** Game theory and Nash equilibrium applications

---

**Last Updated:** 2025-10-10  
**Version:** 1.0  
**Author:** Feature Engineering Roadmap for UFC Predictor

