# Leakage audit — `scripts/strategy_confidence_tests.py`

## Script under audit

- **Path**: `scripts/strategy_confidence_tests.py`
- **Purpose** (one sentence): Post-hoc statistical confidence tests
  (permutation test on ROI, pre-registered held-out evaluation, calibration
  plot) on the already-frozen Elastic Net test predictions.
- **Reads which data sources?**:
  - `results/train_test_2016_2024_predictions.parquet`
    (391 frozen predictions from `scripts/train_test_split_2016_2024.py`)
  - Vegas odds via `attach_vegas_rich` from
    `scripts/build_walkforward_vegas_multi_threshold.py`
    (CSV-preferred, DB fallback — applied AFTER predictions; never as features)
- **Writes what?**:
  - `results/strategy_confidence.json`
  - `results/calibration.png`
- **Date of audit**: 2026-04-28
- **Commit hash this audit applies to**: (filled at commit)

This script does **NO training**. It does NOT fit any model, scaler,
imputer, or calibrator. It only consumes already-frozen predictions and
runs statistical tests + plotting. Most leakage rows are therefore
"N/A — no training" with the reason stated once.

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ☑ pass | inherits from `train_test_split_2016_2024.py`; this script does not split. |
| `assert train_max < test_min` runs at fold construction | ☑ N/A | no folds — only post-hoc analysis on existing test set. |
| Test bouts have zero overlap with training bouts | ☑ pass | inherits frozen predictions; we never read training data. |
| Hyperparameter search NEVER reads test fold data | ☑ pass | no hyperparameters tuned here. |
| If using inner-validation: inner folds are inside outer training window | ☑ N/A | no model training. |

**Held-out subdivision (Test 2 — pre-registered)**: the test set
(2024-10 → 2026-04) is split chronologically at **2025-10-01** into:
  - Validation slice (2024-10 → 2025-09): used to pick the single best
    strategy via ROI ranking
  - Held-out slice (2025-10 → 2026-04): evaluated **exactly once** on
    the strategy chosen above. Cutoff is fixed in code BEFORE looking at
    held-out data; `assert validation_end < heldout_start` runs at split
    time.

## §2 — Rolling / EMA / expanding windows

N/A — no rolling features computed in this script. Predictions are
already frozen.

## §3 — Career / history aggregates

N/A — no aggregates computed.

## §4 — Scalers / imputers / encoders

N/A — no fit calls. The temperature calibrator from
`train_test_split_2016_2024.py` is already baked into `p_pred`.

## §5 — Elo + time-aware decay

N/A — no Elo computed.

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ☑ pass | no hyperparams here |
| τ values fixed | ☑ N/A | no τ |
| Edge / EV / strategy thresholds are pre-registered | ☑ pass | strategy thresholds {ev_positive, edge_5pp, edge_10pp} are fixed in code BEFORE permutation test runs. The held-out test in §Test 2 picks ONE strategy on validation slice via ROI ranking, then evaluates ONCE on held-out — single-shot, no peek-back. |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions | ☑ pass | `attach_vegas_rich` called on already-frozen `p_pred` |
| Devig'd Vegas probs used only for edge/EV computation, not training | ☑ pass | no training |
| If user-provided odds: numeric & in range | ☑ pass | inherits `attach_vegas_rich` validation |

## §8 — Features named in memory

N/A — no features used. Only `p_pred` (frozen), `win` (label), Vegas
devigged probs, decimal odds.

## §8a — Vegas odds pre-processing

| Check | Status | Enforcement |
|---|---|---|
| American odds rejected if `|o| < 100` or NaN | ☑ pass | enforced inside `attach_vegas_rich` (LEAKAGE_REFERENCE.md §8a) |
| Probabilities clipped to `[0.02, 0.98]` before log loss | ☑ pass | calibration plot clips for display only; not a training input |
| Devigging method documented | ☑ pass | proportional devigging, inside `attach_vegas_rich` |

## §9 — Historical leakage bugs

| Past bug | Mitigation in this script |
|---|---|
| AutoGluon `best_quality` mixing future data | not used |
| Shuffled CV across folds | not used |
| MMA-AI 70.6% leaky figure as benchmark | not used as benchmark; baseline is 69.57%/0.5951 (zero-leakage) |
| WC-index encoding mismatch | no WC encoding here |
| `ufc_fight_odds` invalid rows (`|o|<100`) | `attach_vegas_rich` rejects them |
| MAD computed on full dataset including future test fights | no MAD computed |

## §10 — Repo-level missing tests

| Test | Pass? | Code line |
|---|---|---|
| Permutation of test labels collapses metric | ☑ | this IS the permutation test (Test 1) |
| Aggregate filter uses `<` not `<=` on fight date | ☑ | held-out cutoff: `DATE < HELDOUT_START` for validation, `DATE >= HELDOUT_START` for held-out — see code |

## §11 — Grep checklist

| Pattern | Match | Notes |
|---|---|---|
| `\.shuffle\(` | 1 | `np.random.default_rng(seed).permutation(...)` for the permutation test — INTENDED. Not applied to features or splits. |
| `KFold\(` | 0 | not used |
| `train_test_split` | 0 | not used (chronological split done manually with date filter) |
| `df\['DATE'\] >= TEST_FIRST` (training side) | 0 | no training side |
| `for.*in.*df\.iterrows.*outcome` | 0 | not used |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] No "N/A" without an reason
- [x] Code references match what the code does (re-checked after writing)
- [x] If any check fails, the script does NOT run until it passes

**Author**: claude
**Audit committed alongside code**: yes
