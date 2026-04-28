# Leakage Audit Template

**Required before writing or modifying any walk-forward / backtest / metric script.**
**No exceptions. No "I'll do it after."**

This template forces explicit enforcement of every section of
[LEAKAGE_REFERENCE.md](../LEAKAGE_REFERENCE.md). Copy this file to
`docs/audits/<script_name>.md`, fill in every row before writing the
script, then commit the audit alongside the code in the same commit.

If a row says "doesn't apply," explain WHY in one sentence. "N/A" without
an explanation is a leak waiting to happen.

---

## Script under audit

- **Path**: `scripts/<script_name>.py`
- **Purpose** (one sentence):
- **Reads which data sources?** (DB tables, CSVs, JSON caches):
- **Writes what?** (model artifacts, results JSON, modifies shared CSVs):
- **Date of audit**:
- **Commit hash this audit applies to** (filled in at commit time):

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ☐ pass / ☐ N/A | code line: |
| `assert train_max < test_min` runs at fold construction | ☐ pass / ☐ N/A | code line: |
| Test bouts have zero overlap with training bouts | ☐ pass / ☐ N/A | code line: |
| Hyperparameter search NEVER reads test fold data | ☐ pass / ☐ N/A | code line: |
| If using inner-validation: inner folds are inside the outer training window | ☐ pass / ☐ N/A | code line: |

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| All `.shift(1)` calls present where rolling stats appear in training | ☐ pass / ☐ N/A | code line: |
| EMA / expanding aggregates exclude the current fight's outcome | ☐ pass / ☐ N/A | code line: |
| Decay (λ) windows verified against pipeline expectations | ☐ pass / ☐ N/A | code line: |

## §3 — Career / history aggregates

| Check | Status | Enforcement |
|---|---|---|
| Each fight's pre-fight feature uses ONLY fights with `DATE < this fight's DATE` | ☐ pass / ☐ N/A | code line: |
| Strict prior-fight count threshold applied (`apply_threshold(N)`) | ☐ pass / ☐ N/A | code line: |
| `n_eff`, `MAD`, and any population statistic uses ONLY fights ≤ cutoff for that fold | ☐ pass / ☐ N/A | code line: |

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only, transformed on test | ☐ pass / ☐ N/A | code line: |
| `StandardScaler` fit on train only | ☐ pass / ☐ N/A | code line: |
| Calibrator fit on train predictions only (no test contamination) | ☐ pass / ☐ N/A | code line: |
| Re-fitted PER FOLD (not once globally) | ☐ pass / ☐ N/A | code line: |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially (pre-fight Elo only depends on prior fights) | ☐ pass / ☐ N/A | code line: |
| Recency-weight λ anchored at train_end or test_start, not "now" | ☐ pass / ☐ N/A | code line: |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ☐ pass / ☐ N/A | code line: |
| τ values are either fixed globally OR re-optimized per fold using ONLY training inner-folds | ☐ pass / ☐ N/A | code line: |
| Edge / EV / strategy thresholds are pre-registered (not selected from test fold) | ☐ pass / ☐ N/A | code line: |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions are made (not used as a feature) | ☐ pass / ☐ N/A | code line: |
| Devig'd Vegas probs used only for edge/EV computation, not training | ☐ pass / ☐ N/A | code line: |
| If user-provided odds: validated as numeric and within reasonable range | ☐ pass / ☐ N/A | code line: |

## §8 — Features named in memory

For any feature reused from prior experiments, cross-reference its memory
file. List the features used here:

| Feature | Memory file | Verified clean? |
|---|---|---|
|  |  |  |

## §8a — Vegas odds pre-processing

| Check | Status | Enforcement |
|---|---|---|
| American odds rejected if `|o| < 100` or NaN | ☐ pass / ☐ N/A | code line: |
| Probabilities clipped to `[0.02, 0.98]` before log loss | ☐ pass / ☐ N/A | code line: |
| Devigging method documented (proportional / power / shin) | ☐ pass / ☐ N/A | code line: |

## §9 — Historical leakage bugs

Confirm we are NOT re-introducing any of:

| Past bug | Mitigation in this script |
|---|---|
| AutoGluon `best_quality` presets mixing future data | |
| Shuffled CV across folds | |
| MMA-AI 70.6% leaky figure as benchmark | |
| WC-index encoding mismatch | |
| `ufc_fight_odds` invalid rows (`|o|<100`) | |
| MAD computed on full dataset including future test fights | |

## §10 — Repo-level missing tests

Run at minimum (cite where in the script):

| Test | Pass? | Code line |
|---|---|---|
| Feature is monotone-non-decreasing in available history | ☐ | |
| Permutation of test labels collapses metric | ☐ | |
| Removing the feature gives stable metrics across folds | ☐ | |
| Aggregate filter uses `<` not `<=` on fight date | ☐ | |
| Rolling/EMA call sites have `.shift(1)` in training path | ☐ | |

## §11 — Grep checklist

Run these greps against the script and confirm zero matches (or
explain each match):

| Grep pattern | Match count | Notes |
|---|---|---|
| `\.shuffle\(` | | |
| `KFold\(` (without `shuffle=False`) | | |
| `train_test_split` | | |
| `df\['DATE'\] >= TEST_FIRST` (training side) | | |
| `for.*in.*df\.iterrows.*outcome` | | |

---

## Reviewer signoff (filled at commit time)

- [ ] Self-audit complete (every row above filled in)
- [ ] No "N/A" without an explanation in the row
- [ ] Code references match what the code actually does (re-checked after writing)
- [ ] If any check fails, the script does NOT run until it passes

**Author**: (claude / human / both)
**Audit committed alongside code**: yes / no
