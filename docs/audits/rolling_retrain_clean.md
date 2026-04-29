# Leakage Audit — `scripts/rolling_retrain_clean.py`

## Script under audit

- **Path**: `scripts/rolling_retrain_clean.py`
- **Purpose**: Quarterly expanding-window walk-forward on the clean market-features pipeline. Each quarter gets its own model trained on all data through the quarter's start date. Tests whether retraining closer to Q1-2026 recovers the ROI lost by the 15-month-stale static model.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv`; Vegas via `attach_vegas_rich`.
- **Writes**: `results/rolling_retrain_clean.json`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test strictly chronological | ☑ pass | each fold: `train = df[DATE < fold_end]`, `test = df[DATE >= fold_end & DATE < next_fold_end]` |
| `assert train_max < test_min` | ☑ pass | explicit assertion per fold before fit |
| Test bouts zero overlap with train | ☑ pass | explicit assertion per fold |
| HP search never reads test fold | ☑ pass — C=0.05, l1=0.5, λ=1.2 pre-specified from prior experiments | no search |
| Inner folds inside training window | N/A — no inner CV; single EN fit per fold |

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| All `.shift(1)` present for rolling stats | ☑ pass — inherited from `build_through_step6()` |
| EMA aggregates exclude current fight | ☑ pass — inherited |
| Decay λ windows verified | ☑ pass — λ=0.13 for features, λ=1.2 for sample weights |

Critical: `build_through_step6()` is called ONCE on the full dataset. The rolling/EMA features are computed using only fights prior to each row's date (verified in walkforward_market_features audit). The fold split only affects which rows are train vs test — the per-row feature values are unchanged.

## §3 — Career / history aggregates

| Check | Status | Enforcement |
|---|---|---|
| Pre-fight features use DATE < this fight | ☑ pass — inherited |
| Threshold applied | ☑ pass — `apply_threshold(3)` |
| WC priors frozen at fold train_end | ☑ pass — `compute_wc_priors(df[DATE < fold_train_end])` per fold |

Critical: WC priors recomputed per fold using only `df[DATE < fold_train_end]`. For Q1-2026 fold, priors are frozen at 2026-01-01. No future data leaks into priors.

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only | ☑ pass — re-instantiated and re-fit per fold |
| `StandardScaler` fit on train only | ☑ pass — re-instantiated per fold |
| Calibrator | N/A — no calibrator; raw EN probabilities for clean comparison |
| Re-fitted per fold | ☑ pass — all preprocessing re-fit inside fold loop |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo sequential | ☑ pass — inherited; computed once on full dataset in date order |
| Recency weight anchored at fold train_end | ☑ pass — `w = exp(-λ × (fold_train_end − date).days / 365.25)` per fold |

## §6 — Model selection / hyperparameter tuning

C=0.05, l1=0.5 inherited from en_hyperparam_sweep (pre-specified, not selected from any fold's test metrics). λ=1.2 inherited from lambda_sweep_clean. Betting strategies (all, +EV, p≥0.65) pre-registered. ☑ pass.

## §7 — Market / odds

Vegas attached after predictions per fold. ☑ pass.

## §8 — Features

Same feature set as `walkforward_market_features.py`. `card_position_norm_career_diff` uses `d < dt` guard (empirically verified). Novel market features merged after `select_features()` to prevent ordering bug. ☑ pass.

## §8a — Vegas odds

Inherited. ☑ pass.

## §9 — Historical leakage bugs

| Past bug | Mitigation |
|---|---|
| Global `compute_wc_priors` leaking future | Per-fold priors frozen at fold_train_end |
| `compute_adjperf` called globally | Called per fold with fold-specific priors |
| Shuffled CV | No CV; expanding window only |
| `mmaai_features.csv` overwritten between folds | Backup/restore per fold; modules reloaded per fold |

## §10 — Repo-level tests

| Test | Pass? |
|---|---|
| `assert train_max < test_min` | ☑ per fold |
| `assert no (DATE,jbout) overlap` | ☑ per fold |
| Aggregate filter `<` not `<=` | ☑ inherited |

## §11 — Grep checklist

| Pattern | Count | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | |
| `KFold\(` | 0 | |
| `train_test_split` | 0 | |
| training-side `>= fold_end` | 0 | train mask uses strict `<` |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Per-fold WC priors verified — no future leakage
- [x] Recency weight anchored at fold_train_end per fold
- [x] All preprocessing re-fit per fold

**Author**: claude
