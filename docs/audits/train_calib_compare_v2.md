# Leakage audit — `scripts/train_calib_compare_v2.py`

## Script under audit

- **Path**: `scripts/train_calib_compare_v2.py`
- **Purpose** (one sentence): Compare two calibration strategies on the
  full 2016→2024-10 train baseline — (A) in-sample tail (last 6mo of
  train) vs (B) 5-fold CV-OOF on full train — across 7 calibrator types,
  evaluated on the held-out 2024-10 → 2026-04 test set.
- **Reads**: SQLite DB; Vegas via `attach_vegas_rich` (post-prediction).
- **Writes**: `results/calib_compare_v2.json`,
  `results/calib_compare_v2.png`,
  `results/train_calib_compare_v2_predictions.parquet`.
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological | ☑ pass | `df.DATE` masks; no shuffle |
| `assert train_max < test_min` | ☑ pass | code `# §1 hard-asserts` |
| Test bouts have zero overlap with training bouts | ☑ pass | `assert not (train_keys & test_keys)` |
| Hyperparameter search NEVER reads test fold data | ☑ pass | calibrators fit on train-only data; test only consumed at evaluation |
| If using inner-validation: inner folds inside outer training window | ☑ pass | both calibrator-fit slices (in-sample tail; CV-OOF folds) sit entirely inside `[TRAIN_START, TRAIN_END)` |

Boundaries:
  - `TRAIN_START = 2016-01-01`
  - `TRAIN_END   = 2024-10-01` (= `TEST_START`)
  - `TEST_END    = 2026-04-01`
  - `INSAMPLE_TAIL_START = 2024-04-01` (last 6 months of train)

CV-OOF folds (Strategy B): `KFold(n_splits=5, shuffle=False)` on the
**chronologically sorted** train rows. Each fold is a contiguous
chronological block. No row appears in both train_minus_k and fold_k.

## §2 — Rolling / EMA / expanding windows

Inherits the per-fight clean Steps 1-6 from `mma_ai_pipeline`. WC priors
frozen at TRAIN_END=2024-10-01. ☑ pass.

## §3 — Career / history aggregates

Pre-fight features use only `DATE <` this fight. ☑ pass.

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only | ☑ pass | for Strategy A: imp fit once on full doubled train. For Strategy B: imp re-fit per fold on train_minus_k. |
| `StandardScaler` fit on train only | ☑ pass | same pattern as imputer |
| Calibrator fit on train-only data (no test contamination) | ☑ pass | Strategy A fits on tail of train; Strategy B fits on OOF predictions of train rows; neither sees test. |
| Re-fitted PER FOLD (Strategy B) | ☑ pass | imp/sc re-fit per of 5 CV folds |

**Caveat acknowledged for Strategy A**: the in-sample tail predictions
come from a model that was trained on those rows. The (p, y) pairs are
slightly optimistic (model has memorized some of the residual). Because
EN at C=0.05 has 38/207 active features (heavy regularization), the
optimism is small — but Strategy A is reported as **comparison only**;
Strategy B is the unbiased reference.

## §5 — Elo + time-aware decay

Inherits sequential Elo. λ=1.20 anchored at TRAIN_END. ☑ pass.

## §6 — Model selection / hyperparameter tuning

Test set never used to pick calibrator. The 7 calibrators are fixed;
no calibrator selection on test metrics. ☑ pass.

## §7 — Market / odds / contextual features

Vegas attached AFTER predictions, never as feature. ☑ pass.

## §8 / §8a — Features / Vegas pre-processing

Inherits `train_test_split_2016_2024.py`. ☑ pass.

## §9 — Historical leakage bugs

| Past bug | Mitigation |
|---|---|
| AutoGluon mixing future data | not used |
| Shuffled CV across folds | `shuffle=False` enforced in KFold; sorted by DATE first |
| MMA-AI 70.6% leaky figure as benchmark | not used |
| WC-index encoding mismatch | inherits verified path |
| `ufc_fight_odds` invalid rows | rejected by `attach_vegas_rich` |
| MAD on full dataset | not computed here |

## §10 — Repo-level missing tests

| Test | Pass | Code line |
|---|---|---|
| Aggregate filter uses `<` not `<=` on fight date | ☑ | `< TRAIN_END`, `< TEST_END` |
| Rolling/EMA `.shift(1)` | ☑ | inherited |

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
- [x] No "N/A" without a reason
- [x] Code refs match what the code does
- [x] If any check fails, the script does NOT run

**Author**: claude
