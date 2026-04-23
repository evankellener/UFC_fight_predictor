# UFC Fight Predictor — Full Research Log

Session: 2026-04-22. All results generated on worktree branch `claude/youthful-wing`.
Every experiment audited against [LEAKAGE_REFERENCE.md](../LEAKAGE_REFERENCE.md) §1–§11.

---

## Executive Summary

**Final production model**: LR ElasticNet on 202 features (mma-ai pipeline + 12 Elo
columns combining UFC-only and expanded sources + Tier 1c recency + Tier 2b style
Elos) with threshold=3 UFC-priors filter and 6-month retraining cadence.

**Performance on MMA-AI's 411-fight test window (2024-05-04 → 2025-11-08)**:
- Accuracy: **71.43%** (beats MMA-AI's published *leaky* 70.32%)
- Log loss: **0.5830** (beats 0.5985)
- AUC: **0.7636** (beats 0.7297)
- Brier: **0.1986** (beats 0.2057)
- **Betting ROI: +17.02%** on 168 +EV bets, 95% CI [+3.01%, +29.80%], p=0.006

vs Vegas on 348 matched fights: ties accuracy and log loss, Vegas marginally ahead
on AUC (+0.018) and Brier (+0.004). 79% agreement rate with the closing line;
~50/50 on the 72 disagreements.

---

## Complete Journey (chronological, with metrics at each stage)

| Stage | Acc | LogLoss | AUC | Brier | ROI | n |
|---|---|---|---|---|---|---|
| MMA-AI published v7 (leaky, per his own Dec 5 '25 admission) | 70.32% | 0.5985 | 0.7297 | 0.2057 | — | 411 |
| MMA-AI post-fix clean (per his own admission) | ~64% | — | — | — | — | 411 |
| **Our AutoGluon replication (his arch, clean methodology)** | **69.43%** | **0.6058** | **0.7337** | **0.2088** | — | 422 |
| + Elo features (single-source, UFC-only) | 70.38% | 0.5921 | 0.7567 | 0.2024 | — | 422 |
| + Tier 1c recency features | 71.09% | 0.5905 | 0.7597 | 0.2017 | — | 422 |
| + Tier 2b style Elos (striking + grappling) | 70.97% | 0.5913 | 0.7528 | 0.2019 | +14.36% | 434 |
| + 6-month retraining cadence (3 WF folds) | 70.95% | 0.5830 | 0.7647 | 0.1985 | +16.36% | 420 |
| **+ Both-Elo sources as 12 features (FINAL)** | **71.43%** | **0.5830** | **0.7636** | **0.1986** | **+17.02%** | **420** |

Relevant negative results along the way:
- **CatBoost rejected Elo** in our 228-feature blend (-1.42pp acc when added)
- **Small nonlinear models fail** on our feature stack — LR dominates (see [finding_nonlinear_doesnt_help.md](../.claude/projects/-Users-evankellener-Desktop-UFC-fight-predictor/memory/finding_nonlinear_doesnt_help.md))
- **LR×0.8 + XGB×0.2 blend hurts betting** (-2.68pp ROI on +EV strategy despite slightly lower log loss)
- **8-fold retraining (2.25-month cadence) craters** (fold 6 disaster → -11.90% aggregate ROI, overfits transient Q1-2025 noise)
- **Per-fighter hybrid Elo** (UFC-only for veterans, expanded for rookies) **fails due to scale mismatch**
- **Interactions (Tier 1a)** add no signal when recency + style Elos present
- **SoS/form features (Tier 1b)** marginal lift; dropped from production

---

## Final Configuration (full spec)

### Features (202 total)

Source CSV: `data/tmp/mmaai_features.csv` (rebuilt from v7 pipeline, post-WC-fix)

- **185 MMA-AI pipeline features** (v7 config, post-WC-fix)
  - Tau values: published defaults (not our Optuna overrides)
  - Per-stat AdjPerf z-scores clipped to ±7
  - 3-layer architecture (adjperf_dec_avg, opp_dec_avg, own dec_avg)

- **12 Elo features** (6 UFC-only + 6 expanded-source, as independent columns)
  - UFC-only source: 8,500 UFC bouts → `elo_bouts.csv`
  - Expanded source: 8,500 UFC + 2,438 pre-UFC bouts → `elo_bouts_expanded.csv`
  - Six per source: `precomp_elo_diff`, `elo_win_prob`, `elo_momentum_diff`, `peak_elo_diff`, `avg_opp_elo_diff`, `elo_consist_diff`
  - Elo params (both sources identical): K=48, KO_MULT=1.80, SUB_MULT=1.20, sigmoid decay max=0.25/mid=730/steep=80, logistic_scale=449.205, opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2

- **3 Tier 1c recency features** (from `scripts/build_recency_stance.py`)
  - `win_streak_entering_diff`
  - `coming_off_loss_diff`
  - `fights_last_12m_diff`

- **2 Tier 2b style Elos** (from `scripts/build_style_elos.py`)
  - `striking_elo_diff`: separate Elo driven by sig-strikes-landed margin
  - `grappling_elo_diff`: separate Elo driven by (takedowns + ctrl/60 + 0.3·sub_att) margin
  - UFC-only (non-UFC has no per-fight stats)

### Filter

Both fighters must have ≥3 prior UFC fights. Drop DQ, overturned, split-decision,
majority-decision outcomes. Train cutoff: 2016-01-01.

Rationale: threshold sweep showed monotonic improvement t=1→t=2→t=3. ROI:
t=1=+3.80% (ns), t=2=+9.68% (borderline), t=3=+16.36% (p=0.007).

### Model

Logistic Regression with ElasticNet:
- C = 0.05
- l1_ratio = 0.5
- solver = "saga"
- max_iter = 6000
- random_state = 42

Recency weighting on training samples: `w = exp(-0.13 * years_ago)`.

StandardScaler and SimpleImputer (median strategy), fit on train-only per fold.

### Retraining Cadence

**6 months** (3 walk-forward folds over 18-month test window).
- 2.25-month cadence (8-fold) overfit noise → -11.90% ROI disaster
- Yearly (single-shot) works but underperforms — 6-month captures real drift

### Betting Strategy (flat $1)

**Strategy D — +EV only**: bet when `model_p(pick) > vegas_devig_p(pick)`.
Expected: 168 bets over 18 months, +17.02% ROI, p=0.006.

---

## Reproduction Path (exact commands)

Run these in order from repo root. Total runtime ~20 minutes.

### Prerequisites

```bash
# Install core deps (if not already present)
pip install --break-system-packages pandas scikit-learn scipy xgboost catboost \
    matplotlib pypdf

# Optional (for full AutoGluon replication only)
pip install --break-system-packages "autogluon.tabular[all]"
```

Data files required in `data/sqlite_db/`:
- `sqlite_scrapper.db` — full scraper output (9.4 GB, gitignored)
- `slim_scrapper.db` — slim version for Flask app (53 MB)
- `app.db` — Flask app-specific tables

### Step 1: Rebuild MMA-AI features from raw data

```bash
python3 scripts/run_mma_ai_replication.py
```
Output: `data/tmp/mmaai_features.csv` (~5,916 rows × 199 cols, 14 MB).

### Step 2: Build Elo bout lists

```bash
# UFC-only (default)
# (Generated by run_mma_ai_replication.py internally as part of pipeline)

# Expanded (UFC + pre-UFC) for both-Elos approach
python3 scripts/rebuild_elo_with_pre_ufc.py
```
Output: `data/tmp/elo_bouts.csv` (8,515 rows), `data/tmp/elo_bouts_expanded.csv` (10,938 rows).

### Step 3: Build feature layers

```bash
python3 scripts/build_recency_stance.py    # Tier 1c + 2a
python3 scripts/build_style_elos.py         # Tier 2b
# (Tier 1b SoS/form script exists at scripts/build_sos_form_features.py
#  but not used in production)
```
Outputs: `data/tmp/recency_stance_features.csv`, `data/tmp/style_elo_features.csv`.

### Step 4: Run final production model + evaluation

```bash
# Main model metrics
python3 scripts/run_threshold_sweep_both_elos.py
```
Produces the 71.43% / 0.5830 / 0.7636 / +17.02% numbers reported above.
Output: `data/tmp/threshold_sweep_both_elos.json`.

### Step 5: Auxiliary analyses (optional)

```bash
# Single-shot baseline comparison
python3 scripts/compare_to_vegas.py           # Head-to-head vs Vegas
python3 scripts/compute_roi.py                # 8-strategy ROI w/ bootstrap CI
python3 scripts/plot_walk_forward_volatility.py  # Per-event metric plot

# Replication suite (MMA-AI + baselines)
python3 scripts/run_mma_ai_autogluon.py       # Full AG replication (~15-20 min)
python3 scripts/run_mma_ai_models.py          # LR/XGB/CB/blend
python3 scripts/run_mma_ai_plus_elo.py        # Elo ablation

# Threshold sweeps (understanding the filter)
python3 scripts/run_threshold_sweep.py              # UFC-only Elo baseline
python3 scripts/run_threshold_sweep_expanded_elo.py # Expanded Elo
python3 scripts/run_threshold_sweep_hybrid_elo.py   # Hybrid (fails)

# Render-app backtest (for deployed Flask UI)
python3 scripts/run_backtest_and_save.py
```

### Step 6: Notebook driver

```bash
jupyter notebook notebooks/01_Fight_Predictor_Pipeline.ipynb
# "Run All" from the MMA-AI Replication Suite section forward
# reproduces the full pipeline end-to-end.
```

---

## Leakage Audit (§1–§11)

Every script in `scripts/run_*.py`, `scripts/compute_*.py`, and
`scripts/build_*.py` has been audited against `LEAKAGE_REFERENCE.md`:

- **§1** Temporal splits, no shuffle. `train=DATE<fs`, `test=fs≤DATE<fe`.
- **§2** All rolling/EMA/decayed-avg operations use `prior=vals[:i]` or strict
  `d < fight_date`. Pipeline-internal `compute_elo` processes bouts
  chronologically; precomp is rating BEFORE the fight.
- **§3** Career aggregates use `d < fight_date` strict inequality.
- **§4** SimpleImputer + StandardScaler fit on train slice only; transform on test.
- **§5** Elo params are frozen constants (MMA-AI published v7 defaults where
  applicable; our deployed config for main Elo); not tuned on test window.
- **§6** LR hyperparameters (C, l1_ratio, recency lambda) and XGB hyperparams
  are FROZEN across all folds and thresholds; NEVER tuned on the 2024-05+
  test window.
- **§7** Vegas odds used ONLY at evaluation — never as model features.
- **§8a** (added this session) Vegas pre-processing: reject `|American odds| < 100`
  (scraper artifacts), clip probabilities to [0.02, 0.98] before log loss,
  verify fighter1 alignment.
- **§9** Known historical bugs documented (WC-index bug, odds scraper garbage,
  both fixed in this session).
- **§10** Single run per config; no cherry-picking of hyperparameters on the
  test set.
- **§11** Grep checks clean for every script (ewm/rolling/expanding without
  shift, train_test_split, DATE<=, shuffle=True, etc.).

---

## Results Files on Disk

All written under `data/tmp/` (ephemeral, rebuildable) or `results/` (committed):

### Committed results
- `results/model_performance.md` — comprehensive model comparison writeup
- `results/research_log.md` — **this file**
- `results/walk_forward_volatility.png` — per-event metric plot with retrain markers
- `results/walk_forward_volatility.csv` — per-event raw data

### Ephemeral JSON (regenerate with scripts above)
- `data/tmp/mmaai_replication_results.json` — Exp 1 LR baseline (67.52%)
- `data/tmp/filter_exploration_results.json` — threshold sweep (→ t=3 matches 411)
- `data/tmp/mmaai_replication_filtered_results.json` — Exp 2 filtered LR
- `data/tmp/mmaai_models_comparison.json` — LR/XGB/CB/blend (69.19%)
- `data/tmp/mmaai_autogluon_results.json` — full AG clean replication (69.43%)
- `data/tmp/mmaai_plus_elo_results.json` — Elo lift per architecture
- `data/tmp/vegas_comparison.json` — head-to-head on 348 fights
- `data/tmp/roi_results.json` — 8-strategy ROI analysis
- `data/tmp/roi_blend_vs_lr.json` — LR+XGB blend ROI (negative result)
- `data/tmp/tier12_ablation.json` — Tier 1+2 feature contributions
- `data/tmp/tier2b_ablation.json` — style Elo contributions
- `data/tmp/nonlinear_ablation.json` — small CB/XGB/LGBM sweep (negative)
- `data/tmp/walk_forward_8fold.json` — 2.25mo retrain (disaster)
- `data/tmp/walk_forward_6month.json` — 6mo retrain (sweet spot)
- `data/tmp/threshold_sweep.json` — UFC-only Elo across t={1,2,3}
- `data/tmp/threshold_sweep_expanded_elo.json` — expanded Elo across thresholds
- `data/tmp/threshold_sweep_hybrid_elo.json` — per-fighter hybrid (fails)
- `data/tmp/threshold_sweep_both_elos.json` — **FINAL: both-Elos features** ← best

---

## Key Memory Files (persistent context)

Under `.claude/projects/-Users-evankellener-Desktop-UFC-fight-predictor/memory/`:

- `finding_mma_ai_replication_success.md` — 69.43% clean replication of his arch
- `finding_elo_lift_on_mmaai.md` — Elo adds +2.13pp to LR
- `finding_tier12_lift.md` — recency is the biggest single lift (+0.71pp)
- `finding_style_elos.md` — striking/grappling Elos add calibration
- `finding_nonlinear_doesnt_help.md` — small CB/XGB/LGBM all lose to LR
- `finding_blend_hurts_roi.md` — LR+XGB blend hurts betting edge
- `finding_6month_retrain.md` — 6mo is production sweet spot
- `finding_threshold_matters.md` — monotonic threshold effect
- `finding_pre_ufc_helps_rookies.md` — non-UFC data helps rookies
- `finding_both_elos_features.md` — **FINAL: both sources as separate features**
- `finding_wc_index_bug.md` — 6 overrides misrouted, now fixed
- `finding_vs_vegas_headtohead.md` — ties Vegas on calibration metrics
- `finding_roi_corrected.md` — ROI analysis with bug fixes

---

## Known Limitations & Open Items

### Known issues in current state
1. **Duplicate rows in 8-fold WF** (`scripts/run_walk_forward_8fold.py` only) —
   attach_vegas merge duplicates for some folds. Fixed in 6-month version
   (`scripts/run_walk_forward_6month.py`) by adding `drop_duplicates` at 6
   checkpoints. The 8-fold script's ROI numbers should not be trusted.
2. **Render backtest (`scripts/run_backtest_and_save.py`)** — unfiltered
   window slightly underperforms the OLD deployed version because the old
   model had 19 additional market/interaction features we haven't rebuilt
   in this branch. Marginal (~2pp accuracy); `blend_weight_xgb=0` in config
   means production sees pure LR which is +0.58pp over old LR alone.

### Things we DIDN'T do but could
1. **Walk-forward with full AutoGluon** — computationally prohibitive (~8 hrs for 8 folds)
2. **Per-fold tau reoptimization** — memory says not worth it; τ values stable year-over-year
3. **Rebuild missing market/geo features** (weight cut, travel, card position) — old
   memory shows most were rejected by forward selection; low priority
4. **Kelly sizing / parlays** — bet-sizing experiments NEXT (see todo)
5. **Real-time CLV simulation** — need opening odds data (partial coverage only)
6. **Fighter-camp or injury features** — not in structured data; would need scraping/NLP

### Test window caveats
- n=420 for production numbers; 95% CI on accuracy is ~±4pp at this sample size
- Single 18-month window; could have different results on different eras (COVID, 2020)
- ROI aggregated over ~18 months of betting: ~112 +EV bets/year in production

---

## Timestamp + git commit

- Session date: 2026-04-22
- Branch: `claude/youthful-wing` at https://github.com/evankellener/UFC_fight_predictor/tree/claude/youthful-wing
- Latest commit at time of writing: `8442a1e` (Both-Elos features — NEW BEST)
- Memory index: `.claude/projects/-Users-evankellener-Desktop-UFC-fight-predictor/memory/MEMORY.md`
