# Leakage Audit: `walk_forward_clean_mad.py`

## Script under audit

- **Path**: `scripts/walk_forward_clean_mad.py`
- **Purpose**: Surgical fix of the §3 leak found in `parlay_lambda120_8fold_4yr.py`
  audit. Recomputes WC priors / MAD per fold using ONLY fights with
  `DATE < train_end`. Per-fight smoothed/decayed/opponent-history values
  reused from the global Steps 1-6 build (already per-fight clean).
- **Reads**: same DB tables + `elo_bouts.csv` as baseline; produces
  per-fold feature CSVs at `data/tmp/mmaai_features_clean_mad_fold_N.csv`.
- **Writes**: `results/walk_forward_clean_mad.json`, fold-specific CSVs,
  and temporarily swaps `data/tmp/mmaai_features.csv` per fold (with backup).
- **Date of audit**: 2026-04-27 (PRE-run)
- **Commit hash**: TBD (filled at commit time)

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological | ✅ pass | `walk_forward_clean_mad.py:run_fold loop`, `df["DATE"] >= train_start & < train_end` for train; `>= test_start & < test_end` for test |
| Test bouts have zero overlap with training bouts | ✅ pass | Disjoint date ranges by construction |
| Hyperparameter search NEVER reads test fold | ✅ N/A | No hyperparameter search in this script |
| Inner-validation inside outer training | ✅ N/A | No inner validation |

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| `.shift(1)` where rolling stats appear in training | ✅ pass | Inherits from `compute_decayed_averages` (per-fight EMA) |
| EMA / expanding aggregates exclude current fight | ✅ pass | EMA constructed with prior-only values |
| Decay (λ) windows verified | ✅ pass | λ=0.13 from `mma_ai_config.py` |

## §3 — Career / history aggregates ★ THE FIX

| Check | Status | Enforcement |
|---|---|---|
| Each fight's pre-fight feature uses ONLY prior fights | ✅ pass | Per-fight features (Elo, dec_avg, smoothed BB/PG, opp_history) all per-fight clean |
| Strict prior-fight count threshold applied | ✅ pass | `apply_threshold(base, 3)` per fold |
| `n_eff`, `MAD`, population statistics use ONLY fights ≤ cutoff | ✅ **FIX** | `per_fold_features()` line: `train_only = df_full[df_full["DATE"] < train_end]`; `priors = compute_wc_priors(train_only, stat_cols)`. Test fights scored with priors that NEVER saw them or anything after train_end. |

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only | ✅ pass | `fit_one()`: imputer fit on doubled train, transformed on test |
| `StandardScaler` fit on train only | ✅ pass | Same |
| Calibrator fit on train predictions only | ✅ pass | `temp_cal()` on undoubled train predictions |
| Re-fitted PER FOLD | ✅ pass | Per-fold loop refits all 4 |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially | ✅ pass | `compute_elo()` is strictly sequential |
| Recency-weight λ anchored at train_end | ✅ pass | `train_anchor = fold["train_end"]` |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ✅ pass | No tuning in this script |
| τ values fixed globally OR re-optimized per fold from inner training | ⚠️ documented | τs frozen at `tau_optimized.json` values (same as baseline). Those τs were optimized on a separate walk-forward CV; this script uses them unchanged. NOT a hard leak — τs are 11 numbers, not data — but worth noting for completeness. |
| Edge / EV thresholds pre-registered | ⚠️ documented | edge≥5pp + edge≥10pp inherited from prior backtests. Same data-snooping concern as baseline. Documented in baseline audit. |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions | ✅ pass | `attach_vegas_rich()` called after p_test computed |
| Devig'd Vegas probs only for edge/EV, not training | ✅ pass | LR features have no Vegas info |
| User-provided odds validated | ✅ N/A | reads from `odds_table.csv` (validated upstream) |

## §8 — Features named in memory

Same set as baseline (`parlay_lambda120_8fold_4yr.md` audit). The §3 fix
in this script does not introduce new features; it only changes the
denominator of AdjPerf z-scores.

## §8a — Vegas odds pre-processing

Same as baseline: `american_to_decimal` rejects `|o|<100`, probabilities
clipped before log loss.

## §9 — Historical leakage bugs

| Past bug | Mitigation in this script |
|---|---|
| AutoGluon presets | N/A — using LR |
| Shuffled CV | N/A — temporal |
| MMA-AI 70.6% leaky figure | N/A |
| WC-index encoding | inherits post-fix mma_ai_config.py |
| `ufc_fight_odds` invalid rows | uses validated `odds_table.csv` |
| **MAD computed on full dataset** | ✅ **FIXED** — this is the entire point of this script |

## §10 — Repo-level missing tests

| Test | Pass? | Notes |
|---|---|---|
| Feature monotone-non-decreasing | not run | (no test exists) |
| Permutation collapses metric | not run | (no test exists) |
| Removing feature gives stable metrics | partial | not per-feature |
| `<` not `<=` on date | ✅ verified | `df_full["DATE"] < train_end` (strict less-than) |
| `.shift(1)` in EMA training paths | ✅ verified upstream | inherited from pipeline |

## §11 — Grep checklist

| Pattern | Match | Notes |
|---|---|---|
| `\.shuffle\(` | 0 in training path | bootstrap CI uses shuffle but that's metric-level not training |
| `KFold\(` | 0 | not used |
| `train_test_split` | 0 | not used |
| `df\['DATE'\] >= TEST_FIRST` (training side) | 0 | training filters use `< train_end` |
| `for.*in.*iterrows.*outcome` | 0 | not detected |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] No "N/A" without explanation
- [x] Code references match implementation (verified by reading the fns it calls)
- [x] If §1-§11 checks fail, script does NOT run — checked, all pass

**Author**: claude  
**Audit committed alongside code**: yes (same commit)

---

## Expected outcome

The user has predicted (correctly per pattern history) that fixing this leak
will drop the headline ROI numbers. We agree this is the likely outcome.

Quoting the new metrics is honest only if they're worse than the baseline.
If they come out higher than the leaky baseline, that's a red flag —
investigate before publishing.
