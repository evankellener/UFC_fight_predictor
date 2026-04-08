# How to Recreate the Model Results

## Current Best Results (April 7, 2026)

On 408 matched test fights (2024-05 to 2025-11):
- **Accuracy: 69.9%** | **Log Loss: 0.5976** | **Brier: 0.2033** | **AUC: 0.7419**
- His (leaky): 70.6% acc / 0.5964 LL / 0.7297 AUC
- No data leakage. Elo is #1 feature. Islam #2 in Elo rankings.

## Pipeline Overview

```
SQLite DB (local only)
    |
    v
mma_ai_pipeline.py (Poisson-Gamma/Beta-Binomial smoothing, AdjPerf z-scores)
    |
    v
Pre-computed CSVs (checked into git, used by app)
    |
    v
predict_mmaai.py (loads CSVs, trains LR+CB ensemble, serves predictions)
    |
    v
app.py (Flask web app)
```

## Step-by-Step Recreation

### Prerequisites
- Python 3.10+
- SQLite database at `data/sqlite_db/sqlite_scrapper.db` with tables:
  `ufc_winlossko`, `ufc_fighter_match_stats_round_smooth`, `ufc_fight_results`,
  `ufc_event_details`, `ufc_fighter_tott`

### Step 1: Set clean taus
Taus are in `data/tmp/tau_optimized.json`. These were optimized via walk-forward CV
(`src/optimize_taus.py`) on 5 folds (2020-2024-05). The test set (2024-05+) was never
seen during optimization.

### Step 2: Build MMA-AI features
```python
import sys, json
sys.path.insert(0, 'src')
from mma_ai_config import PG_TAU_GLOBAL, BB_TAU_GLOBAL
from mma_ai_pipeline import get_fighter_stats_lookup

with open('data/tmp/tau_optimized.json') as f:
    opt = json.load(f)
PG_TAU_GLOBAL.update(opt['pg_tau'])
BB_TAU_GLOBAL.update(opt['bb_tau'])

result = get_fighter_stats_lookup('jan26')
df = result['df']              # 4,923 fights x 199 columns
fighter_stats = result['fighter_stats']  # 2,615 fighters
feature_cols = result['feature_cols']    # 188 individual stat columns
```

### Step 3: Build Elo bouts (from ufc_winlossko ONLY, no joins)
```python
import sqlite3, pandas as pd
conn = sqlite3.connect('data/sqlite_db/sqlite_scrapper.db')
raw = pd.read_sql('SELECT DATE, jevent, jbout, jfighter, win, ko, subw FROM ufc_winlossko ORDER BY DATE', conn)
conn.close()
# ... group by (jevent, jbout), build f1/f2/winner/method, save to CSV
# IMPORTANT: Do NOT join with final_features_fast or ufc_fight_results.
# Those tables have corrupted data that breaks Elo rankings.
```
Saved as `data/tmp/elo_bouts.csv` (8,376 bouts).

### Step 4: Add Elo features
Elo params (from `src/predict_event.py`):
- K=48, KO_MULT=1.80, SUB_MULT=1.20, DECAY=0.923
- Sigmoid decay: max=0.25, midpoint=730, steepness=80
- Logistic scale=449.205
- opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2

Merge on (jbout, DATE) and flip signs when jfighter != Elo's f1 (alphabetical).

### Step 5: Build style matchup features
From each fighter's individual stats, compute:
- K-means clustering (K=4) on 15 normalized style stats
- 6 interaction features:
  - `striking_matchup`: f1 head strikes/min x f2 head defense - vice versa
  - `grappling_matchup`: f1 TD attempts x (1 - f2 TD def) - vice versa
  - `power_matchup`: f1 KD/min x f2 head defense - vice versa
  - `sub_matchup`: f1 sub attempts x (1 - f2 ground acc) - vice versa
  - `wrestling_matchup`: (same as grappling but different formula)
  - `style_distance`: Euclidean distance in normalized style space

Style config saved in `data/tmp/style_config.json`.

### Step 6: Save pre-computed CSVs
```
data/tmp/mmaai_features.csv      - 4,923 fights x 211 columns (diffs + Elo + style)
data/tmp/mmaai_fighter_stats.csv - 2,615 fighters x 192 columns (individual stats)
data/tmp/mmaai_feature_cols.json - 188 individual feature column names
data/tmp/model_feat_cols.json    - 200 model feature names (diffs + Elo + style)
data/tmp/elo_bouts.csv           - 8,376 bouts for Elo computation
data/tmp/style_config.json       - KMeans scaler/centers for style features
data/tmp/tau_optimized.json      - Clean taus + LR params
```

### Step 7: Train model
```python
from predict_mmaai import build_training_data, train_ensemble
data = build_training_data()   # loads from CSVs
models = train_ensemble(data)  # trains LR + CatBoost
```

Training config:
- **Era**: >= 2018-01-01 (drops noisy old fights)
- **Recency lambda**: 0.10
- **LR**: C=0.1, ElasticNet l1_ratio=0.4, saga solver
- **CatBoost**: depth=5, lr=0.05, l2_reg=3, early stopping
- **Ensemble**: LR*0.8 + CB*0.2

### Step 8: Evaluate on test set
Test fights are in `test_predictions (2).csv` (MMA-AI's 411 predictions).
Match on (event_date, normalized fighter name). 408 of 411 match.

## Key Design Decisions

1. **No test-set leakage**: Taus optimized via walk-forward CV, not on test set.
2. **Elo bouts from ufc_winlossko only**: Joining with final_features_fast/ufc_fight_results
   introduced corrupted weightindex/finish_round data that broke rankings.
3. **Style matchup features**: Capture non-transitive MMA dynamics (striker vs grappler)
   that no single-number rating system can encode. `striking_matchup` and
   `grappling_matchup` are top-10 features.
4. **ElasticNet handles feature selection**: 200 features, ~140 survive L1 penalty.
   No manual feature selection needed.
5. **Pre-computed CSVs for deploy**: The 419MB SQLite DB can't be pushed to git.
   All features are pre-built and saved as CSVs (~22MB total).

## Files

| File | Purpose |
|------|---------|
| `src/predict_mmaai.py` | Main prediction module (loads CSVs, trains, predicts) |
| `src/mma_ai_pipeline.py` | MMA-AI feature engineering (PG/BB smoothing, AdjPerf) |
| `src/mma_ai_config.py` | Tau globals for PG/BB smoothing |
| `src/elo_feature.py` | Elo computation engine |
| `src/combined_features.py` | Elo + market feature merging |
| `src/optimize_taus.py` | Walk-forward tau optimization (clean) |
| `src/app.py` | Flask web app |
| `data/tmp/*.csv/json` | Pre-computed artifacts |
