# Leakage audit — `scripts/error_analysis_test_set.py`

## Script under audit

- **Path**: `scripts/error_analysis_test_set.py`
- **Purpose**: Descriptive error analysis on the held-out 2024-10 → 2026-04
  test set using the beta-CV-OOF calibrated predictions. Slice accuracy
  by Vegas confidence, weight class, experience, recency, and date.
  Identify where the model bleeds.
- **Reads**: `results/train_calib_compare_v2_predictions.parquet`,
  Vegas via `attach_vegas_rich`, base data via `load_base_both_elos`
  (only for joining context columns; not for training).
- **Writes**: `results/error_analysis.json`, `results/error_analysis.png`.
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

This script does **NO training** and **NO model selection**. It is
purely descriptive — it characterizes errors on data already evaluated.
No multiple-comparison risk in the standard betting-strategy sense
because we are not picking new strategies; we are describing where the
fixed model is currently right or wrong.

---

## §1 — Temporal splits

Inherits the test set boundary (2024-10 → 2026-04) from the parent
`train_calib_compare_v2.py` audit. ☑ pass.

## §2-§5 — Rolling / aggregates / scalers / Elo

N/A — no fitting in this script.

## §6 — Model selection / hyperparameter tuning

| Check | Status | Notes |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ☑ pass | no hyperparams |
| Slice definitions pre-registered | ☑ pass | Vegas-quartile bins, WC bins, experience tiers, recency tiers, and date quarters all defined as constants at the top of the script BEFORE looking at slice-level metrics. |
| Caveat: post-hoc hypothesis generation | ⚠ acknowledged | If we find e.g. "model is bad at heavyweight fights" we are NOT permitted to declare "we should bet only at non-heavyweight" — that would be selection on the test set. Findings here are inputs to feature work, not deployment decisions. |

## §7 — Market / odds

Vegas attached AFTER predictions; never used as feature. ☑ pass.

## §8 — Features named in memory

Context columns merged from base data (read-only):
- `weightindex` — weight-class id, 1-12 per `feedback_wc_index_encoding`
- `f1_priors`, `f2_priors` — individual fighter prior-fight counts
- `ufc_age_diff` — age delta
- `days_since_last_fight_f1` — fighter 1 recency
- `coming_off_loss_diff` — momentum

These are pre-existing features; we do not modify them.

## §11 — Grep checklist

| Pattern | Match | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | none |
| `KFold\(` | 0 | none |
| `train_test_split` | 0 | none |
| `\.fit\(` | 0 | no model fitting |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Multiple-comparison risk explicitly disclosed (§6)
- [x] Code refs match what the code does
- [x] Findings will be treated as inputs to feature work, NOT as test-set strategy selection

**Author**: claude
