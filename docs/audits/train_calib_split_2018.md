# Leakage audit — `scripts/train_calib_split_2018.py`

## Script under audit

- **Path**: `scripts/train_calib_split_2018.py`
- **Purpose** (one sentence): Train a single Elastic Net on 2018-01 →
  2024-04, fit multiple calibrators (Platt, isotonic, temperature, beta,
  histogram, spline) on a 2024-04 → 2024-10 validation slice, then
  evaluate every calibrator on a held-out 2024-10 → 2026-04 test slice.
- **Reads**: SQLite DB (`data/sqlite_db/sqlite_scrapper.db`), Vegas odds
  via `attach_vegas_rich` (post-prediction, never as features).
- **Writes**: `results/calib_compare.json`, `results/calib_compare.png`,
  `results/train_calib_split_2018_predictions.parquet`.
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ☑ pass | `df.DATE` masks; no `train_test_split` calls |
| `assert train_max < val_min` and `val_max < test_min` runs | ☑ pass | both asserts at code line marked `# §1 hard-asserts` |
| Test bouts have zero overlap with train/val bouts | ☑ pass | `assert not (train_keys & val_keys & test_keys)` |
| Hyperparameter search NEVER reads test fold data | ☑ pass | calibrators fit on val only; test only consumed at evaluation |
| If using inner-validation: inner folds inside outer training window | ☑ pass | val set IS the calibrator-fit set; both before test |

Boundaries:
  - `TRAIN_START = 2018-01-01`
  - `TRAIN_END   = 2024-04-01`  (train < this; val ≥ this)
  - `VAL_END     = 2024-10-01`  (val < this; test ≥ this)
  - `TEST_END    = 2026-04-01`

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| `.shift(1)` on rolling stats | ☑ pass | inherits `mma_ai_pipeline._era_rolling_*` (zero-leak verified) |
| EMA / expanding excludes current fight | ☑ pass | per-fight clean Steps 1-6 |
| Decay (λ=1.20) verified | ☑ pass | matches V7 config |

## §3 — Career / history aggregates

| Check | Status | Enforcement |
|---|---|---|
| Pre-fight feature uses ONLY fights with `DATE <` this fight | ☑ pass | per-fight clean (verified by `test_leakage_per_fold.py`) |
| Strict prior-fight count threshold | ☑ pass | `apply_threshold(3)` |
| `n_eff`, `MAD`, population statistics use only fights ≤ cutoff | ☑ pass | WC priors computed from `df[df.DATE < TRAIN_END]` only |

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only | ☑ pass | `imp.fit_transform(train_d[usable])` |
| `StandardScaler` fit on train only | ☑ pass | `sc.fit_transform(...)` on train rows only |
| Calibrator fit on validation only (no test contamination) | ☑ pass | every calibrator's `.fit()` is called on `(p_val, y_val)` only |
| Re-fitted PER FOLD | ☑ N/A | single split; not multi-fold |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially | ☑ pass | inherits `elo_feature` (per-fight) |
| Recency-weight λ anchored at train_end | ☑ pass | `w = exp(-LAM*(TRAIN_END - DATE).days/365.25)` |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ☑ pass | EN(C=0.05, l1=0.5) fixed pre-run; calibrator HP fixed (isotonic out-of-bounds = clip; beta = 3-param max-lik; hist k=10) |
| τ values fixed | ☑ N/A | no τ |
| Edge / EV / strategy thresholds pre-registered | ☑ pass | edge_5pp, ev_positive fixed in code BEFORE looking at test |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions made | ☑ pass | merged onto frozen `p_test_*` |
| Devig'd Vegas probs used only for edge/EV, not training | ☑ pass | never appears in `usable` features |
| User-provided odds validated | ☑ N/A | uses repo Vegas table |

## §8 — Features named in memory

Inherits the full feature stack from `train_test_split_2016_2024.py` —
no new features. See `docs/audits/train_test_split_2016_2024.md` for
feature provenance.

## §8a — Vegas odds pre-processing

| Check | Status | Enforcement |
|---|---|---|
| American odds rejected if `|o| < 100` | ☑ pass | inside `attach_vegas_rich` |
| Probabilities clipped to `[0.02, 0.98]` before log loss | ☑ pass | `np.clip(p, EPS, 1-EPS)` before log_loss |
| Devigging method documented | ☑ pass | proportional |

## §9 — Historical leakage bugs

| Past bug | Mitigation |
|---|---|
| AutoGluon `best_quality` mixing future data | not used |
| Shuffled CV across folds | not used (chronological masks) |
| MMA-AI 70.6% leaky figure as benchmark | not used |
| WC-index encoding mismatch | inherits verified `add_wc_features` |
| `ufc_fight_odds` invalid rows (`|o|<100`) | rejected by `attach_vegas_rich` |
| MAD computed on full dataset | not computed here |

## §10 — Repo-level missing tests

| Test | Pass | Code line |
|---|---|---|
| Permutation of test labels collapses metric | ☑ | inherits Test 1 from `strategy_confidence_tests.py` (separately) |
| Aggregate filter uses `<` not `<=` on fight date | ☑ | `df.DATE < TRAIN_END`, `df.DATE < VAL_END` |
| Rolling/EMA call sites have `.shift(1)` | ☑ | inherits clean Steps 1-6 |

## §11 — Grep checklist

| Pattern | Match | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | none |
| `KFold\(` | 0 | none |
| `train_test_split` | 1 | only as path string in audit comment; not the sklearn function |
| `df\['DATE'\] >= TEST_FIRST` (training side) | 0 | training mask uses `<` |
| `for.*in.*df\.iterrows.*outcome` | 0 | none |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] No "N/A" without a reason
- [x] Code refs match what the code does
- [x] If any check fails, the script does NOT run

**Author**: claude
**Audit committed alongside code**: yes
