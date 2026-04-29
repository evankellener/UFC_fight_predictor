# Leakage audit — `scripts/walkforward_priors_features.py`

## Script under audit

- **Path**: `scripts/walkforward_priors_features.py`
- **Purpose**: Test the hypothesis (motivated by error_analysis_test_set
  finding that the model is −7 pp vs Vegas on n=113 fights with 3-4
  priors) that adding UFC-priors-derived features closes the rookie gap.
  Same train (2016-01 → 2024-10) / test (2024-10 → 2026-04) split as
  the honest baseline; identical pipeline; the ONLY change is the
  feature set.
- **Reads**: SQLite DB; Vegas via `attach_vegas_rich`.
- **Writes**:
  - `results/walkforward_priors_features.json`
  - `results/walkforward_priors_features_predictions.parquet`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

## Pre-registered hypothesis

If priors-derived features close the rookie weakness:
- Pooled accuracy/log loss/Brier should improve OR be unchanged
- **The rookie 3-4 slice gap (currently model 73.45% vs Vegas 80.53%, Δ −7.08 pp) should narrow**
- Other slices should not significantly degrade

If pooled metrics get worse, we reject the hypothesis. If pooled is
stable but the rookie slice doesn't improve, we conclude these features
don't address the gap.

## New features (all derivable from existing `f1_priors`, `f2_priors`)

`f1_priors`, `f2_priors` already exist in `load_base_both_elos`. They are
computed strictly as count of UFC fights with `DATE < this fight's DATE`
— no leakage. (Verified by reading
`scripts/run_threshold_sweep_both_elos.py:392-397`.)

| feature | formula | rationale |
|---|---|---|
| `priors_diff` | f1_priors − f2_priors | raw experience-gap signal |
| `min_priors_x` | min(f1_priors, f2_priors) | "rookie-ness" of the matchup |
| `max_priors_x` | max(f1_priors, f2_priors) | "veteran-ness" of the matchup |
| `priors_log_ratio` | log(f1_priors+1) − log(f2_priors+1) | scale-invariant gap |
| `rookie_diff` | int(f1_priors≤5) − int(f2_priors≤5) | binary asymmetry |
| `both_rookie` | int((f1_priors≤5) & (f2_priors≤5)) | both inexperienced |

`min_priors_x` / `max_priors_x` are not "diff" features — they are
absolute matchup-level features. They violate the strict
"diff-only" feature schema but are appropriate here (rookie-ness is a
property of the matchup, not a fighter-level asymmetry).

---

## §1 — Temporal splits

| Check | Status |
|---|---|
| Train/test strictly chronological | ☑ pass (inherits parent script) |
| `assert train_max < test_min` | ☑ pass |
| Test bouts have zero overlap | ☑ pass |
| Hyperparameter search NEVER reads test fold data | ☑ pass — no HP search; EN(C=0.05, l1=0.5) fixed |
| CV-OOF folds (Strategy B) inside train window | ☑ pass — KFold(5, shuffle=False) on sorted train rows |

## §2 — Rolling / EMA / expanding windows

N/A — new features are pre-fight counts only, no rolling. ☑ pass.

## §3 — Career / history aggregates

| Check | Status |
|---|---|
| Pre-fight feature uses ONLY fights with `DATE <` this fight | ☑ pass — `f1_priors`/`f2_priors` constructed via `(hist_dates < this_date).sum()` |
| Strict prior-fight count threshold | ☑ pass — `apply_threshold(3)` |

## §4 — Scalers / imputers / encoders

| Check | Status |
|---|---|
| Imputer fit on train only | ☑ pass |
| Scaler fit on train only | ☑ pass |
| Beta calibrator (Strategy B) fit on CV-OOF predictions of TRAIN rows only | ☑ pass — KFold uses `shuffle=False`; never sees test |

## §5 — Elo + time-aware decay

Inherits sequential Elo. ☑ pass.

## §6 — Model selection / hyperparameter tuning

| Check | Status |
|---|---|
| Test fold metrics never used to pick HP | ☑ pass |
| Pre-registered hypothesis stated above | ☑ pass |
| Edge / EV / strategy thresholds pre-registered | ☑ pass — same {ev>0, edge≥5pp, edge≥10pp} as baseline |

## §7 — Market / odds

Vegas attached AFTER predictions. ☑ pass.

## §8 — Features named in memory

| feature | provenance |
|---|---|
| existing `*_diff`, `*_ufc`, `*_exp` features | identical to baseline `train_test_split_2016_2024.py` |
| new `priors_diff`, `min_priors_x`, ... | this script (formula above) |

## §9 — Historical leakage bugs

None re-introduced. New features are additions only.

## §10 — Repo-level missing tests

| Test | Status |
|---|---|
| Aggregate filter uses `<` not `<=` on fight date | ☑ pass — `< TRAIN_END`, `< TEST_END` |
| Rolling/EMA `.shift(1)` | ☑ pass — inherited |

## §11 — Grep checklist

| Pattern | Match | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | none |
| `KFold\(.*shuffle=True` | 0 | KFold uses `shuffle=False` |
| `train_test_split` | 0 | not the sklearn function |
| `df\['DATE'\] >= TEST_FIRST` (training side) | 0 | training mask uses `<` |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Pre-registered hypothesis stated explicitly
- [x] Code refs match what the code does
- [x] If any check fails, the script does NOT run

**Author**: claude
