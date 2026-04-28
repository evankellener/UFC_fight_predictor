# Leakage Audit: `scripts/train_test_split_2016_2024.py`

## Script under audit

- **Path**: `scripts/train_test_split_2016_2024.py`
- **Purpose**: Single train/test split for an Elastic Net model.
  - Train: 2016-01-01 → 2024-10-01 (~8.75yr)
  - Test:  2024-10-01 → 2026-04-01 (~18 months)
- **Reads**: SQLite via `mma_ai_pipeline.load_base_data()`, `elo_bouts.csv`
- **Writes**: `results/train_test_2016_2024.json`, predictions parquet
- **Date of audit**: 2026-04-27
- **Commit hash**: filled at commit time

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Strictly chronological split | ✅ pass | `df["DATE"] >= TRAIN_START & < TRAIN_END` for train, `>= TRAIN_END & < TEST_END` for test |
| `assert train_max < test_min` | ✅ pass | Hard assert in code |
| Zero overlap between train/test bouts | ✅ pass | Hard assert via set intersection |
| Hyperparam search not on test | ✅ N/A | No hyperparam search; C=0.05, l1=0.5 fixed (production values) |

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| `.shift(1)` in rolling stats | ✅ pass | Inherited from `compute_decayed_averages` (per-fight EMA on prior fights only) |
| EMA excludes current fight outcome | ✅ pass | EMA over fights `< current_date` |

## §3 — Career / history aggregates

| Check | Status | Enforcement |
|---|---|---|
| Each fight's pre-fight feature uses ONLY prior fights | ✅ pass | Step 1-6 features per-fight clean (verified by `test_leakage_per_fold.py`) |
| Strict prior-fight count threshold | ✅ pass | `apply_threshold(base, 3)` |
| `n_eff`, `MAD`, population statistics use ONLY fights ≤ cutoff | ✅ pass | `compute_wc_priors(df_full[df_full["DATE"] < TRAIN_END])` — frozen priors |

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only | ✅ pass | Fit on doubled train, transform test |
| `StandardScaler` fit on train only | ✅ pass | Same |
| Calibrator fit on train predictions only | ✅ pass | `temp_cal()` on undoubled train predictions |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially | ✅ pass | `compute_elo()` strictly sequential (used by `load_base_both_elos`) |
| Recency weight λ anchored | ✅ pass | `np.exp(-1.20 * (TRAIN_END - train_d["DATE"]).dt.days / 365.25)` |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick params | ✅ pass | No tuning |
| τ values | ⚠️ documented | Frozen at `tau_optimized.json`; same as baseline |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions | ✅ N/A | This script doesn't compute ROI; pure prediction metrics only |

## §8 — Features named in memory

Same set as baseline. No new features. Style matchup / KMeans / market features
are NOT used.

## §9 — Historical leakage bugs

| Past bug | Mitigation |
|---|---|
| AutoGluon presets | N/A — Elastic Net only |
| Shuffled CV | N/A — single chronological split |
| WC-index encoding | inherits post-fix `mma_ai_config.py` |
| `ufc_fight_odds` invalid rows | N/A — no odds in this script |
| **MAD computed on full dataset** | ✅ FIXED — per-train_end priors |
| **load_base_data duplicate rows** | ✅ FIXED — drop_duplicates added |
| **BB/PG global-rate fallbacks** | ✅ FIXED — strictly-prior 3-tier fallback |
| **reach_ratio global median** | ✅ FIXED — per-row expanding median |

## §10 — Repo-level missing tests

| Test | Pass? | Notes |
|---|---|---|
| `<` not `<=` on date | ✅ verified | All filters use `<` for train_end / test_end |
| `.shift(1)` in EMA paths | ✅ inherited from pipeline |

## §11 — Grep checklist

| Pattern | Match | Notes |
|---|---|---|
| `.shuffle(` | 0 | not used |
| `KFold(` | 0 | not used |
| `train_test_split` | 0 | not used (we do explicit date-based split) |
| `>= TRAIN_END` (training side) | 0 | training uses `< TRAIN_END` |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Empirical leakage test passed for the per-fold workflow
  (`scripts/test_leakage_per_fold.py`: 0 leaky columns, 5,399 fights compared)
- [x] Code references match implementation

**Author**: claude  
**Audit committed alongside code**: yes
