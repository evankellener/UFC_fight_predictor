# MMA-AI.net Model Specification — Complete Reproduction Reference

Every value below is citable to one of three sources: the MMA-AI.net PDF (pages cited
inline), the implementation at [src/mma_ai_config.py](src/mma_ai_config.py), or the
memory file `mma_ai_full_spec.md`. Where sources disagree, both values are listed and
the tie-breaker noted.

Target performance (v7, uncalibrated / calibrated): **70.32% acc, 0.5985 log loss, 0.2057 Brier** (official).
Post-calibration test: 0.710 acc / 0.598 ll / 0.206 Brier. Raw (uncalibrated) test: 0.7008 acc / 0.5949 ll (PDF p.38).

---

## 0. Pipeline Order (CRITICAL — order matters for leakage)

Source: PDF p.26 + memory. Must run in this exact order.

1. Base fight stats → `fight_stats_derived` table
2. **Beta-Binomial** smoothing (binary families + control share) — runs FIRST so `sub_att` is still raw when PG needs it as a count
3. **Poisson-Gamma** smoothing (count families: `*_land`, `*_att`, `kd`, `rev`, plus `_rd1`)
4. Temporarily keep `*_raw`; compute derived stats (totals, accuracy, defense, ratios, per-minute) on SMOOTHED values
5. Drop `*_raw`
6. Feature-family tables (striking, grappling, position)
7. Opponent aggregation with time decay
8. Weight-class priors (`_wc_mean`, `_wc_mad`, `_minimum_mad`)
9. **AdjPerf** (opponent-aware, reliability-weighted z-scores, clipped ±7)
10. Three feature layers per key stat (see §7)

---

## 1. Time Decay

- **λ = 0.13** (~5 year half-life) — v7 default
- Weight: `w = exp(-λ × years_ago)`
- Kish effective n under decay: `n_eff = (Σw)² / Σ(w²)`
- When `n_eff < 1`, fall back to WC-only baseline
- **jan26 alt preset**: λ = 0.347 (~2 year half-life), start_date='2016-01-01'

Exposure:
- Round-1 columns: `t = min(time_sec_rd1, 300) / 60.0`
- All others: `t = time_sec / 60.0`
- Filter `WHERE time_sec > 0` before computing WC priors to avoid div-by-zero

---

## 2. Poisson-Gamma Smoothing (Counts)

**Scope**: every column ending `_land` or `_att`, plus `kd`, `rev`, and their `_rd1` variants. Exclude static fields and binary/duration families.

**Formula** (PDF p.27):

```
wc_rate(c)     = SUM(c) / NULLIF(SUM(t), 0)     -- division-level prior
global_rate(c) = SUM(c) / NULLIF(SUM(t), 0)     -- global fallback
λ_post  = (wc_rate × τ + X) / (τ + t)
X_smooth = t × λ_post
```

`X` = observed count for this fight, `t` = exposure minutes, `τ` = pseudo-minutes.

### Pseudo-minutes τ — Global Defaults

| Stat | τ | Stat | τ |
|---|---|---|---|
| `sig_str_land` | 0.7 | `dist_land` | 0.7 |
| `sig_str_att` | 0.7 | `dist_att` | 0.7 |
| `head_land` | 0.8 | `clinch_land` | 2.5 |
| `head_att` | 0.8 | `clinch_att` | 2.5 |
| `body_land` | 2.5 | `ground_land` | 2.5 |
| `body_att` | 2.5 | `ground_att` | 2.5 |
| `leg_land` | 2.1 | `td_land` | 7.0 |
| `leg_att` | 2.1 | `td_att` | 7.0 |
| `sub_att` | 12.0 | `kd` | 20.0 |
| `rev` | 42.0 | | |

### Pseudo-minutes τ — Round 1

| Stat | τ | Stat | τ |
|---|---|---|---|
| `sig_str_land_rd1` | 0.7 | `td_land_rd1` | 9.0 |
| `sig_str_att_rd1` | 0.7 | `td_att_rd1` | 9.0 |
| `head_land_rd1` | 0.7 | `sub_att_rd1` | 15.0 |
| `head_att_rd1` | 0.7 | `kd_rd1` | 12.0 |
| `body_land_rd1` | 2.5 | `rev_rd1` | 60.0 |
| `body_att_rd1` | 2.5 | | |
| `leg_land_rd1` | 1.7 | | |
| `leg_att_rd1` | 1.7 | | |

### Per-Weight-Class Overrides — Poisson-Gamma

Source: PDF p.28, confirmed in [src/mma_ai_config.py:65-70](src/mma_ai_config.py:65).

| WC index | WC name | Stat | τ override (vs global) |
|---|---|---|---|
| 1 | Flyweight | `rev` | **22.0** (vs 42.0) |
| 7 | Light Heavyweight | `head_rd1` (both `land` and `att`) | **0.5** (vs 0.7 / 0.8) |
| 8 | Heavyweight | `td_land` | **5.0** (vs 7.0) |
| 8 | Heavyweight | `td_att` | **5.0** (vs 7.0) |
| 8 | Heavyweight | `td_land_rd1` | **4.0** (vs 9.0) |
| 8 | Heavyweight | `td_att_rd1` | **4.0** (vs 9.0) |

---

## 3. Beta-Binomial Smoothing (Binary + Control Share)

**Scope** (PDF p.28):
- `ko`, `win`, `decision`: successes = 0/1, attempts = 1 per fight
- `sub_land`: successes = `sub_land`, attempts = `sub_att`
- `ctrl`, `ctrl_rd1` (duration as share): attempts = seconds (`min(rd1_sec, 300)` for rd1), output smoothed_seconds = `p_post × attempts`

**Formula**:

```
p_post = (rate_prior × τ + successes) / (τ + attempts)
if attempts == 0: p_post = rate_prior
```

Zero-attempt guard: return the WC/global prior rate, do NOT fabricate a fraction.

### Pseudo-counts τ — Global Defaults

| Stat | τ |
|---|---|
| `ko` | 23 |
| `win` | 25 |
| `decision` | 20 |
| `sub_land` | 9 |
| `ctrl` | 2 |

### Pseudo-counts τ — Round 1

| Stat | τ |
|---|---|
| `ko_rd1` | 17 |
| `win_rd1` | 15 |
| `decision_rd1` | 16 |
| `sub_land_rd1` | 7 |
| `ctrl_rd1` | 1 |

### Per-Weight-Class Overrides — Beta-Binomial

| WC index | WC name | Stat | τ override (vs global) |
|---|---|---|---|
| 4 | Featherweight* | `sub_land` | **3** (vs 9) |
| 7 | Light Heavyweight | `ctrl` | **1.5** (vs 2) |
| 8 | Heavyweight | `ctrl` | **1.5** (vs 2) |

\* Note: memory says "Featherweight" (WC 4 in the project's mapping — see §9).

---

## 4. Opponent Aggregation

Source: PDF p.26-28.

For a target column `c` (e.g., `head_acc`), gather all past fights where other fighters faced the current opponent. Compute:

```
opp_mean_pers(c) = mean of what others achieved against this opponent
                   (time-weighted: w = exp(-λ · years_ago))

opp_mad_pers(c)  = MAD via two-step median:
                      median_c = PERCENTILE_CONT(0.5) over val
                      opp_mad_pers = PERCENTILE_CONT(0.5) over |val - median_c|

n                = Kish effective n under decay: (Σw)² / Σ(w²)
```

Strict time ordering: join only `event_date < current_fight_event_date`, tie-break by `(event_id, fight_id)`.

**Recommended hardening**: compute **per-column** effective n and set weight to 0 if that column has no history for that opponent, rather than coalescing to 0.

---

## 5. Weight-Class Priors

Per feature table, per column, precompute:

- `c_wc_mean` — weight-class mean
- `c_wc_mad`  — weight-class MAD (two-step median)
- `c_mad_floor` — per-column minimum MAD floor (5th percentile of per-opponent MADs; prevents insane z-scores on tiny samples — see Feb 8 2025 bug fix, PDF p.70)

If a weight class lacks priors, fall back to a **global** row (do NOT coalesce to 0).

---

## 6. Adjusted Performance (AdjPerf)

Source: PDF p.3-4, p.29.

### Formula

```
w_mean = n / (n + K_mean)
w_mad  = n / (n + K_mad)

μ      = w_mean × opp_mean_pers + (1 − w_mean) × wc_mean
σ      = max(w_mad × opp_mad_pers + (1 − w_mad) × wc_mad, mad_floor)

adjperf = clip((observed − μ) / σ, −7, +7)
```

`observed` is the **already-smoothed** feature value from the feature-specific table (not raw). AdjPerf is applied AFTER smoothing.

### K Values (reliability shrinkage)

| Family | Stats | K_mean | K_mad |
|---|---|---|---|
| Default | (fallback) | 4.0 | 4.0 |
| **Striking** | `sig_str`, `head`, `body`, `leg`, `dist`, `clinch`, `ground`, `total_str` | **8.0** | **12.0** |
| **Grappling** | `td`, `sub`, `ctrl`, `rev` | **5.0** | **8.0** |
| **Rare** | `kd`, `ko` | 4.0 | 4.0 |

Guideline from PDF: `K_mad ≥ K_mean` is usually safer for stability with small samples.

### Clipping

Hard clip: `[−7.0, +7.0]`. This is the "prevent one insane fight from dominating training" guard. PDF p.4 confirms.

---

## 7. Three Feature Layers

After AdjPerf, for each base stat `c` (e.g., `head_land_pm`, `sig_str_acc`, `td_def`) produce three decayed-average columns, then difference them fighter1 vs fighter2:

1. **`c_adjperf_dec_avg_diff`** — opponent-adjusted z-score, time-decayed average, differenced
2. **`c_opp_dec_avg_diff`** — what opponents have done against this fighter (decayed avg), differenced
3. **`c_dec_avg_diff`** — fighter's own raw (post-smoothing) decayed average, differenced

Added in v5.2 (Feb 22 2025, PDF p.68): the third layer (fighter's own `c_dec_avg`) was the final addition.

---

## 8. Feature List — Nov 10 2025 Importance List (39 features)

From memory `mma_ai_full_spec.md` (derived from article_posts.md Nov 10 2025):

`age`, `reach_ratio`, `sub_att`, `td_acc`, `head_land`, `head_def`, `ufc_age`,
`head_land_ratio_adjperf`, `distance_acc_adjperf`, `body_acc_adjperf`,
`sig_str_land_ratio`, `ko`, `body_def`, `ground_def_adjperf`, `leg_land_pm_opp`,
`ctrl_r1`, `win_ratio`, `clinch_land_pm`, `days_since_last_fight`, `td_def`,
`ctrl_r1_pm_opp`, `head_acc_adjperf`, `rev_r1_ratio_opp`, `distance_def_adjperf`,
`sub_att_pm_opp`, `strikes_landed_r1_adjperf`, `rev_adjperf`,
`distance_land_ratio_adjperf`, `kd_opp`, `ko_per_sig_str_land_adjperf`,
`win_adjperf`, `weightclass_encoded`, `ground_land_per_ctrl`,
`ground_land_ratio_adjperf`, `distance_per_sig_str_land`,
`td_per_sig_str_att_adjperf`, `leg_def_adjperf`, `sub_def_adjperf`,
`td_land_per_ctrl_adjperf`, `clinch_land_ratio_adjperf`, `rev_ratio_adjperf`,
`clinch_acc_adjperf`, `sig_str_acc_adjperf`,
`days_since_last_fight` (non-diffed copy).

Static non-stat features (v5.2+):
- `age`, `ufc_age`, `reach_ratio`, `WEIGHT`, `days_since_last_fight`, `age_ratio`, `weightclass_encoded`, `scheduled_rounds`

NO odds as features. NO elo/ranking as features in base MMA-AI.

---

## 9. Weight-Class Index Map

From [src/mma_ai_config.py:134-149](src/mma_ai_config.py:134). **The BB/PG overrides above reference THESE indices — verify before implementing.**

| Index | Weight Class |
|---|---|
| 0 | Catchweight |
| 1 | Flyweight |
| 2 | Bantamweight |
| 3 | Featherweight |
| 4 | Lightweight |
| 5 | Welterweight |
| 6 | Middleweight |
| 7 | Light Heavyweight |
| 8 | Heavyweight |
| 9 | W_Strawweight |
| 10 | W_Flyweight |
| 11 | W_Bantamweight |
| 12 | W_Featherweight |

⚠ Note: [src/mma_ai_config.py:91](src/mma_ai_config.py:91) applies Featherweight `sub_land=3` override to WC index **4** (which is Lightweight in this map). Per memory/PDF the override is for **Featherweight**, which is WC index **3**. **Fix before implementing.**

---

## 10. Model Training (AutoGluon)

Source: PDF p.37 (verbatim v7 config block).

```python
train_size        = 0.75
val_size          = 0.15
test_size         = 0.10
n_splits          = 4
num_stack_levels  = 2
use_recency_weights = True
use_bag_holdout   = True        # required if using tuning_data (val split)
num_bag_sets      = 2
decay_rate        = 0.13
shuffle           = True        # v7 preset; jan26 updated to False for temporal CV
start_date        = '2014-04-01'
calibrate         = True
```

### AutoGluon Hyperparameters

From [src/mma_ai_config.py:119-132](src/mma_ai_config.py:119):

```python
AG_HYPERPARAMETERS = {
    "CAT":      {},
    "GBM":      [{"extra_trees": True}, {}],
    "NN_TORCH": {},
}
AG_SETTINGS = {
    "num_stack_levels": 2,
    "num_bag_folds":    4,
    "num_bag_sets":     2,
    "use_bag_holdout":  True,
    "time_limit":       900,    # seconds
    "keep_only_best":   False,
}
```

### Final Ensemble Composition (WeightedEnsemble_L2)

From memory `mma_ai_full_spec.md`:

| Model | Weight |
|---|---|
| CatBoost | 0.74 |
| TabM | 0.17 |
| TabICL | 0.04 |
| LightGBM | 0.04 |

### Calibration

Post-hoc **Platt scaling** (sigmoid), NOT isotonic (PDF p.45-47). Isotonic overfits with ~2,400 fights. Use `sklearn.calibration.CalibratedClassifierCV(method='sigmoid', cv='prefit', ensemble=False)` on a held-out calibration split.

Calibration lift observed (PDF p.47): log loss 0.5948 → 0.5841 (Δ = **0.0107**), Brier 0.0043 improvement, ECE 0.0174 improvement.

### Training Data Rules

- **UNBALANCED** training: keep natural ~59/41 red corner bias. Do NOT rebalance to 50/50 — PDF p.53 shows balancing destroys ROI (corner assignment carries real predictive info).
- Fights filtered by `y_true ∈ {0, 1}` (drop draws, DQ, overturned, majority decisions — PDF p.45 `filter_fights` snippet).
- Both fighters must have ≥2 previous fights (first-fight AdjPerf is undefined — PDF p.69).

---

## 11. Two Preset Configs

From [src/mma_ai_config.py:13-33](src/mma_ai_config.py:13):

| Parameter | **v7** (Sep 2025) | **jan26** (Jan 2026) |
|---|---|---|
| `decay_lambda` | **0.13** (~5yr half-life) | **0.347** (~2yr half-life) |
| `start_date` | '2014-04-01' | '2016-01-01' |
| `shuffle` | True | False (temporal CV) |
| `calibrate` | True | True |
| `train_size` | 0.75 | 0.75 |
| `val_size` | 0.15 | 0.15 |
| `test_size` | 0.10 | 0.10 |

Memory flags that "half-life optimized to 2 years" in Jan 13 2026 update — hence jan26 is the newer preset. Leakage reference §1 confirms `shuffle=False` is the current enforcement.

---

## 12. Target Metrics (what to hit)

Source: PDF p.48 (v7 performance comparison), memory `mma_ai_full_spec.md`.

| Metric | Vegas (same period) | MMA-AI v7 uncalibrated | MMA-AI v7 calibrated |
|---|---|---|---|
| Accuracy | 0.690 | 0.710 | 0.710 |
| Precision | 0.725 | 0.697 | — |
| Recall | 0.725 | 0.859 | — |
| F1 | 0.725 | 0.770 | — |
| Log Loss | 0.587 | 0.603 | **0.598** |
| Brier | 0.201 | 0.208 | **0.206** |

Official published v7 number (memory): **70.32% / 0.5985 ll / 0.2057 Brier** on 411 fights, 2024-05-04 to 2025-11-08.

Training-time per-split (PDF p.38, v7 run):

```
Training  acc = 0.7511  ll = -0.5082
Val       acc = 0.7072  ll = -0.5918
Test      acc = 0.7008  ll = -0.5949
```

(Val and test in the same ballpark = healthy; training acc ~4pp higher = mild healthy overfit, not p-hacked.)

---

## 13. Known Historical Bugs (do NOT re-introduce)

1. **Minimum stddev explosion** (PDF p.70, Feb 8 2025): Postgres double-precision rounding created σ = 0.00000002 in edge cases, producing z-scores of ~11,839,902. **Fix: floor σ to 5th percentile of per-WC stddevs.** (This is the `c_mad_floor` in §5.)
2. **AutoGluon `best_quality` leakage** (Dec 5 2025, memory): presets mixed future data into training. Fix dropped accuracy 70% → 64% ("more honest"). Do NOT re-enable without verifying fold-aware splitter.
3. **Shuffled CV** (Jan 13 2026, memory): was leaking across folds. Switched to temporal (no shuffle).
4. **Published leaky number**: "70.6% acc / 0.5964 ll / 0.7297 AUC" — noted as leaky by user's own measurements. Clean target is ~71% / 0.602. Do NOT calibrate to the leaky figure.
5. **First-fight & two-prior-fight fighters** (PDF p.69): AdjPerf is undefined when opponent has <2 priors. Either drop those rows or use first-time-fighter pooled stats as a fallback.

---

## 14. Implementation Roadmap (reproducing from scratch)

Source: PDF p.35-36 synthesized.

1. Base feature extraction → `fight_stats_derived`
2. Beta-Binomial smoothing (§3)
3. Poisson-Gamma smoothing (§2)
4. Temporary raw preservation (keep `*_raw`)
5. Derived feature computation on smoothed values (totals, accuracy, defense rates, ratios, per-minute)
6. Drop `*_raw`
7. Per-minute and ratio features
8. Feature-family tables (striking / grappling / position)
9. Opponent aggregation with time decay (§4)
10. Weight-class priors (§5)
11. AdjPerf (§6)
12. Three-layer feature materialization (§7)
13. Build diff features (fighter1 - fighter2)
14. Split 75/15/10 with `recency_weights=True`, `decay_rate=0.13`, calibration hold-out
15. AutoGluon fit (§10); post-hoc Platt calibration
16. Evaluate on held-out test window; target §12 metrics

---

## 15. Leakage Hygiene (enforce every run)

From [LEAKAGE_REFERENCE.md](LEAKAGE_REFERENCE.md) + PDF p.28-30, p.35:

- Time-ordered join only: `event_date < current_fight_event_date`, tie-break `(event_id, fight_id)`
- Always `WHERE time_sec > 0` on rate priors (avoid div/0)
- Global fallback rows in every WC prior CTE so sparse classes never collapse to zero
- Scalers/imputers/encoders **fit on train slice only**, `transform` on val/test
- Every EMA/rolling/expanding on within-fighter history must be followed by `.shift(1)`
- No Vegas odds as training features (§evaluation-only)
- Hyperparameter tuning runs only on CV folds inside training — test set never observed during optimization

---

## 16. What the spec does NOT specify (and that's okay)

- Exact number of striking/grappling base stats — pipeline enumerates them from DB schema
- Exact AutoGluon time_limit (PDF says "experimental" preset, memory: 900s)
- Platt calibration `max_iter` (PDF p.47: 100, random_state=42, solver='lbfgs')
- Whether to include `total_str` derived family in addition to sig_str (implementation detail)

These are implementation choices that won't change published metrics by more than noise.

---

## Appendix A — Minimal SQL Pseudocode (PDF p.29)

```sql
-- Opponent history for column c, with optional decay
rows_c AS (
  SELECT cur.fight_id, cur.fighter_id, hist_opp.c AS val,
         CASE WHEN :decay THEN EXP(-lambda * age_years) ELSE 1.0 END AS w
  FROM features.<table> cur
  JOIN fight_mapping fm_cur ON cur.fight_id = fm_cur.fight_id
  JOIN event_mapping em_cur ON fm_cur.event_id = em_cur.event_id
  -- figure out opponent id for current fight
  -- join to all past fights where others faced that opponent
  -- restrict to strictly earlier fights (event_date, event_id, fight_id)
),
med_c AS (
  SELECT fight_id, fighter_id,
         PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY val) AS med
  FROM rows_c WHERE val IS NOT NULL GROUP BY fight_id, fighter_id
),
stats_c AS (
  SELECT fight_id, fighter_id,
         SUM(w*val)/NULLIF(SUM(CASE WHEN val IS NOT NULL THEN w END),0) AS c_opp_mean_pers
  FROM rows_c GROUP BY fight_id, fighter_id
),
mad_c AS (
  SELECT fight_id, fighter_id,
         PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(val - med)) AS c_opp_mad_pers
  FROM rows_c JOIN med_c USING(fight_id, fighter_id)
  GROUP BY fight_id, fighter_id
),
n_hist AS (
  SELECT fight_id, fighter_id,
         CASE WHEN :decay THEN POWER(SUM(w),2)/NULLIF(SUM(POWER(w,2)),0)
              ELSE COUNT(*) END AS n
  FROM rows_c GROUP BY fight_id, fighter_id
);

-- AdjPerf scoring
w_mean = n / (n + K_mean)
w_mad  = n / (n + K_mad)
mu     = w_mean * c_opp_mean_pers + (1 - w_mean) * c_wc_mean
sigma  = GREATEST(w_mad * c_opp_mad_pers + (1 - w_mad) * c_wc_mad, c_mad_floor)
score  = GREATEST(LEAST((observed - mu) / sigma, 7.0), -7.0)
```

---

## Appendix B — Decimal-Exact Numerical Reference

All values in one table for spot-checking an implementation.

| Param | Value | Source |
|---|---|---|
| `decay_lambda` (v7) | 0.13 | PDF p.37, config |
| `decay_lambda` (jan26) | 0.347 | config |
| `start_date` (v7) | 2014-04-01 | PDF p.37 |
| `start_date` (jan26) | 2016-01-01 | config |
| AdjPerf clip | ±7.0 | PDF p.4 |
| K_mean (default) | 4.0 | PDF p.27 |
| K_mad (default) | 4.0 | PDF p.27 |
| K_mean (striking) | 8.0 | PDF p.32 |
| K_mad (striking) | 12.0 | PDF p.32 |
| K_mean (grappling) | 5.0 | PDF p.32 |
| K_mad (grappling) | 8.0 | PDF p.32 |
| RD1 exposure cap | 300 sec | PDF p.27 |
| MAD floor | 5th percentile | PDF p.70 |
| `train_size` | 0.75 | PDF p.37 |
| `val_size` | 0.15 | PDF p.37 |
| `test_size` | 0.10 | PDF p.37 |
| `n_splits` | 4 | PDF p.37 |
| `num_stack_levels` | 2 | PDF p.37 |
| `num_bag_sets` | 2 | PDF p.37 |
| `use_bag_holdout` | True | PDF p.37 |
| `use_recency_weights` | True | PDF p.37 |
| `calibrate` | True | PDF p.37 |
| CatBoost weight | 0.74 | memory |
| TabM weight | 0.17 | memory |
| TabICL weight | 0.04 | memory |
| LightGBM weight | 0.04 | memory |
| Target accuracy | 0.7032 | memory |
| Target log loss | 0.5985 | memory |
| Target Brier | 0.2057 | memory |
| Target AUC | 0.7297 | memory |
| Platt calibration lift (ll) | 0.0107 | PDF p.47 |
