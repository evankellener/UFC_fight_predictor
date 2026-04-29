# Leakage Audit — `scripts/roi_by_month.py`

## Script under audit

- **Path**: `scripts/roi_by_month.py`
- **Purpose**: Post-hoc analysis of test-set ROI broken down by calendar month across four betting strategies (+EV, edge≥5pp, edge≥10pp, all-picks).
- **Reads which data sources?**: `results/train_test_2016_2024_predictions.parquet` (pre-computed predictions from a separate script); Vegas odds via `attach_vegas_rich` (CSV-preferred, DB-fallback).
- **Writes what?**: `results/roi_by_month.json`, `results/roi_by_month.png`. No model artifacts, no modified CSVs.
- **Date of audit**: 2026-04-28
- **Commit hash this audit applies to**: (filled at commit time)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ☑ N/A | No training in this script — reads pre-computed predictions only |
| `assert train_max < test_min` runs at fold construction | ☑ N/A | No folds constructed; inherited from train_test_split_2016_2024.py |
| Test bouts have zero overlap with training bouts | ☑ N/A | No training in this script |
| Hyperparameter search NEVER reads test fold data | ☑ N/A | No hyperparameter search |
| If using inner-validation: inner folds are inside the outer training window | ☑ N/A | No inner validation |

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| All `.shift(1)` calls present where rolling stats appear in training | ☑ N/A | No features computed; reads finished predictions |
| EMA / expanding aggregates exclude the current fight's outcome | ☑ N/A | No aggregates computed |
| Decay (λ) windows verified against pipeline expectations | ☑ N/A | No decay in this script |

## §3 — Career / history aggregates

| Check | Status | Enforcement |
|---|---|---|
| Each fight's pre-fight feature uses ONLY fights with `DATE < this fight's DATE` | ☑ N/A | No career aggregates computed |
| Strict prior-fight count threshold applied (`apply_threshold(N)`) | ☑ N/A | Threshold applied upstream in prediction script |
| `n_eff`, `MAD`, and any population statistic uses ONLY fights ≤ cutoff for that fold | ☑ N/A | No such statistics computed here |

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only, transformed on test | ☑ N/A | No imputer or scaler used |
| `StandardScaler` fit on train only | ☑ N/A | No scaler used |
| Calibrator fit on train predictions only (no test contamination) | ☑ N/A | No calibrator used |
| Re-fitted PER FOLD (not once globally) | ☑ N/A | No folds |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially (pre-fight Elo only depends on prior fights) | ☑ N/A | No Elo computation |
| Recency-weight λ anchored at train_end or test_start, not "now" | ☑ N/A | No recency weighting |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ☑ N/A | No model tuning; post-hoc analysis only |
| τ values are either fixed globally OR re-optimized per fold using ONLY training inner-folds | ☑ N/A | No τ optimization |
| Edge / EV / strategy thresholds are pre-registered (not selected from test fold) | ☑ pass | Strategies (ev>0, edge≥5pp, edge≥10pp) are all reported, none selected from test metrics |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions are made (not used as a feature) | ☑ pass | Predictions loaded from parquet; Vegas attached via `attach_vegas_rich` afterward |
| Devigged Vegas probs used only for edge/EV computation, not training | ☑ pass | No training; p_vegas used only for edge_pp and EV calculation |
| If user-provided odds: validated as numeric and within reasonable range | ☑ N/A | Odds come from DB/CSV, not user input |

## §8 — Features named in memory

No features computed in this script. Reads finished predictions only.

| Feature | Memory file | Verified clean? |
|---|---|---|
| p_pred (model probability) | Inherited from train_test_split_2016_2024.py | Yes — predictions generated before Vegas lookup |

## §8a — Vegas odds pre-processing

| Check | Status | Enforcement |
|---|---|---|
| American odds rejected if `|o| < 100` or NaN | ☑ pass | Handled in `attach_vegas_rich` (inherited logic) |
| Probabilities clipped to `[0.02, 0.98]` before log loss | ☑ N/A | No log loss computed in this script |
| Devigging method documented (proportional / power / shin) | ☑ pass | Inherited from `attach_vegas_rich`; proportional devig |

## §9 — Historical leakage bugs

| Past bug | Mitigation in this script |
|---|---|
| AutoGluon `best_quality` presets mixing future data | N/A — no AutoGluon |
| Shuffled CV across folds | N/A — no CV |
| MMA-AI 70.6% leaky figure as benchmark | N/A — no benchmark comparison |
| WC-index encoding mismatch | N/A — no WC encoding |
| `ufc_fight_odds` invalid rows (`|o|<100`) | Handled in `attach_vegas_rich` |
| MAD computed on full dataset including future test fights | N/A — no MAD computed |

## §10 — Repo-level missing tests

| Test | Pass? | Code line |
|---|---|---|
| Feature is monotone-non-decreasing in available history | ☑ N/A | No features computed |
| Permutation of test labels collapses metric | ☑ N/A | Post-hoc analysis, not model evaluation |
| Removing the feature gives stable metrics across folds | ☑ N/A | No features, no folds |
| Aggregate filter uses `<` not `<=` on fight date | ☑ N/A | No date filtering on aggregates |
| Rolling/EMA call sites have `.shift(1)` in training path | ☑ N/A | No rolling/EMA |

## §11 — Grep checklist

| Grep pattern | Match count | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | |
| `KFold\(` (without `shuffle=False`) | 0 | |
| `train_test_split` | 0 | |
| `df\['DATE'\] >= TEST_FIRST` (training side) | 0 | |
| `for.*in.*df\.iterrows.*outcome` | 0 | |

---

## Reviewer signoff (filled at commit time)

- [x] Self-audit complete (every row above filled in)
- [x] No "N/A" without an explanation in the row
- [x] Code references match what the code actually does (re-checked after writing)
- [x] If any check fails, the script does NOT run until it passes

**Author**: claude
**Audit committed alongside code**: yes
