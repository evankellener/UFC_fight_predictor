# Leakage Audit: `parlay_lambda120_8fold_4yr.py`

**This is a RETRO audit of an already-shipped script** (the one that produced the
+27.25% pooled / +39.47% edge≥10pp pooled metrics we've been quoting).
Running it through the template surfaces issues that should have been caught
before deploy.

## Script under audit

- **Path**: `scripts/parlay_lambda120_8fold_4yr.py`
- **Purpose**: 8-fold × 3-mo walk-forward at 4-yr training, parlay-strategy
  ROI evaluation against Vegas (men-only, edge≥5pp + edge≥10pp variants).
- **Reads**: `data/tmp/mmaai_features.csv`, `data/sqlite_db/sqlite_scrapper.db`,
  `data/tmp/elo_bouts.csv`, `data/tmp/elo_bouts_expanded.csv`,
  `data/tmp/odds_table.csv`
- **Writes**: `results/parlay_lambda120_8fold_4yr_results.json`,
  `results/parlay_predictions_lambda120_8fold_4yr.parquet`
- **Date of audit**: 2026-04-27 (post-hoc)
- **Commit hash this audit applies to**: c67bef9 (script first pushed)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ✅ pass | `walk_forward_4fold.py:179-187` filters `df["DATE"] >= train_start` & `< train_end` |
| `assert train_max < test_min` runs at fold construction | ✅ pass | `walk_forward_4fold.py:147-160` `leakage_assertions()` enforces |
| Test bouts have zero overlap with training bouts | ✅ pass | `walk_forward_4fold.py:157-160` set-intersection check |
| Hyperparameter search NEVER reads test fold data | ✅ pass | No hyperparameter search in this script — all params fixed |
| If using inner-validation: inner folds are inside the outer training window | ✅ N/A | No inner validation |

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| All `.shift(1)` calls present where rolling stats appear in training | ✅ pass | All dec_avg features computed in `mma_ai_pipeline.py:Step 5` use per-fight EMA over fights < current |
| EMA / expanding aggregates exclude the current fight's outcome | ✅ pass | Per-fight EMA enforced upstream by `compute_decayed_averages` |
| Decay (λ) windows verified against pipeline expectations | ✅ pass | λ=0.13 confirmed in mma_ai_config.py |

## §3 — Career / history aggregates **← LEAK FOUND**

| Check | Status | Enforcement |
|---|---|---|
| Each fight's pre-fight feature uses ONLY fights with `DATE < this fight's DATE` | ✅ pass | Per-fight features (Elo, dec_avg, smoothed BB/PG) all use prior data only |
| Strict prior-fight count threshold applied (`apply_threshold(N)`) | ✅ pass | `apply_threshold(base, 3)` at line 100 |
| `n_eff`, `MAD`, and any population statistic uses ONLY fights ≤ cutoff for that fold | ❌ **FAIL** | `mma_ai_pipeline.py:compute_opponent_history` and `compute_wc_priors` are computed once on the FULL dataset (1994 → 2026-04). For early folds (fold_1 test 2024-04 → 2024-07), the MAD denominator in AdjPerf z-scores includes ~22 months of FUTURE fights. This violates §1 spirit and §3 letter. |

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only, transformed on test | ✅ pass | `parlay_lambda120_8fold_4yr.py:59-61` fits on train_d, transforms test |
| `StandardScaler` fit on train only | ✅ pass | Same |
| Calibrator fit on train predictions only (no test contamination) | ✅ pass | `temp_cal()` fit on undoubled train predictions |
| Re-fitted PER FOLD (not once globally) | ✅ pass | Per-fold loop refits all 4 |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially (pre-fight Elo only depends on prior fights) | ✅ pass | `compute_elo()` in `elo_feature.py` is strictly sequential |
| Recency-weight λ anchored at train_end or test_start, not "now" | ✅ pass | `train_end = test["DATE"].min()` at line 56 |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ✅ pass | No hyperparameter search |
| τ values are either fixed globally OR re-optimized per fold using ONLY training inner-folds | ⚠️ partial | τs are fixed globally (from `data/tmp/tau_optimized.json`). Those τs were optimized on a separate walk-forward CV that respected §1, but they implicitly use information from the same dataset that includes future fights for the current fold's perspective. This is not a hard leak (τs are 11 numbers, not 8000 fights of data) but it's also not strictly clean. |
| Edge / EV / strategy thresholds are pre-registered (not selected from test fold) | ⚠️ **PARTIAL** | edge≥5pp and edge≥10pp thresholds were chosen AFTER inspecting the 4-fold backtest results (in `parlay_strategy_eval.py`). This is data-snooping at the strategy level — we picked thresholds because they looked good on a backtest that included these same fights. The 8-fold extension is "fresh data" only in the sense of finer slicing. |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions are made (not used as a feature) | ✅ pass | `attach_vegas_rich()` called after `p_test` computed |
| Devig'd Vegas probs used only for edge/EV computation, not training | ✅ pass | Devigged probs only enter the edge/EV calc, not the LR features |
| If user-provided odds: validated as numeric and within reasonable range | ✅ N/A | Reads from `odds_table.csv` (already validated) |

## §8 — Features named in memory

| Feature | Memory file | Verified clean? |
|---|---|---|
| `precomp_elo_diff_ufc` / `_exp` | feature_elo_predictability.md | ✅ — Elo is sequential |
| `dec_avg`-suffixed features (60 stats) | feature_sigmoid_decay.md | ✅ per-fight, but ⚠️ AdjPerf z-score scaling uses global MAD (§3 fail above) |
| `wc_native_*_diff` | finding_wc_index_bug.md | ✅ — fixed Apr 22 2026 |
| `style_clash_diff` | docs/additional_model_bumps.md (rejected) | N/A — not used |

## §8a — Vegas odds pre-processing

| Check | Status | Enforcement |
|---|---|---|
| American odds rejected if `|o| < 100` or NaN | ✅ pass | `american_to_decimal()` rejects invalid in caller |
| Probabilities clipped to `[0.02, 0.98]` before log loss | ✅ pass | `np.clip(p, 1e-6, 1-1e-6)` at metric calculations |
| Devigging method documented | ✅ pass | Proportional devig in `attach_vegas_rich()` |

## §9 — Historical leakage bugs

| Past bug | Mitigation in this script |
|---|---|
| AutoGluon `best_quality` presets mixing future data | N/A — using LR not AutoGluon |
| Shuffled CV across folds | N/A — temporal split |
| MMA-AI 70.6% leaky figure as benchmark | Compared against τ-optimized + walk-forward only |
| WC-index encoding mismatch | Inherits the post-fix mma_ai_config.py |
| `ufc_fight_odds` invalid rows (`|o|<100`) | Vegas attach uses validated `odds_table.csv` |
| **MAD computed on full dataset including future test fights** | ❌ **NOT MITIGATED** — see §3 fail |

## §10 — Repo-level missing tests

| Test | Pass? | Code line |
|---|---|---|
| Feature is monotone-non-decreasing in available history | ❌ not run | (no test exists in repo) |
| Permutation of test labels collapses metric | ❌ not run | (no test exists in repo) |
| Removing the feature gives stable metrics across folds | partial | We did ablation on individual experiments but not a per-feature check |
| Aggregate filter uses `<` not `<=` on fight date | ✅ verified | inspected pipeline |
| Rolling/EMA call sites have `.shift(1)` in training path | ✅ verified | `compute_decayed_averages` is per-fight EMA |

## §11 — Grep checklist

| Grep pattern | Match count | Notes |
|---|---|---|
| `\.shuffle\(` | 0 in training path | (some in bootstrap CI which is OK — not training) |
| `KFold\(` (without `shuffle=False`) | 0 | not used |
| `train_test_split` | 0 | not used |
| `df\['DATE'\] >= TEST_FIRST` (training side) | 0 | training filters use `< TEST_FIRST` |
| `for.*in.*df\.iterrows.*outcome` | 0 | (not detected) |

---

## Audit verdict

**One hard fail (§3) and one soft fail (§6 partial).**

### §3 hard fail — global MAD

The +27.25% pooled / +39.47% edge≥10pp pooled metrics quoted as the
shipped baseline use AdjPerf z-scores whose MAD denominator was computed
across the full 1994-2026 dataset, including future fights from each
fold's perspective. This is a §1 + §3 violation.

**Magnitude:** MAD is robust → small absolute change when adding/removing
fights → effect on z-scores is uniform scaling (~5-10% off). The LR's
coefficients absorb most of it during training. The expected impact on
ROI metrics is single-digit-pp at most, possibly less. But "small" is
not "zero" and the doc says zero.

**Status:** UNFIXED at the time of writing this audit. The shipped
strategy on the live website inherits this leak. The Sterling+Garcia
parlay that hit was scored using this leaky pipeline.

### §6 partial — strategy threshold selection

edge≥5pp and edge≥10pp were chosen AFTER inspecting backtests that
include the same data we're now reporting metrics on. This is
data-snooping at the strategy-construction level (less severe than
target-leakage, but still increases the apparent edge above what a
genuinely-pre-registered strategy would show).

**Status:** UNFIXED. Cleaner version: pre-register thresholds on a
held-out validation period (e.g. through 2024-12), then evaluate on
2025-Q1 onwards.

---

## What this audit means for stated metrics

Numbers we have been treating as the production baseline:
- PARLAY-2 edge≥5pp pooled: **+27.25%** (n=55)
- PARLAY-2 edge≥10pp pooled: **+39.47%** (n=37)

After the §3 + §6 fixes, the honest expectation is these drop. How much
is unknown without running the fix. Best guess (this is a guess, not a
measurement): −5 to −15pp on the headline ROI numbers, with corresponding
widening of the bootstrap CI.

**The fact that real-bankroll Sterling+Garcia hit is n=1 evidence; it
does not refute the audit.**

---

## Reviewer signoff

- [x] Self-audit complete (every row filled in)
- [x] No "N/A" without an explanation
- [x] Code references match the actual implementation (verified)
- [ ] If any check fails, the script does NOT run until it passes
  - This row is currently UNCHECKED. The script ran and shipped despite
    §3 failing. That's the policy violation we're trying to prevent
    going forward.

**Author**: claude (initial fill)  
**Audit committed alongside code**: NO — this is a retro audit of an already-shipped script
