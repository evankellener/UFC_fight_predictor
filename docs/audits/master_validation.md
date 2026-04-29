# Leakage Audit — `scripts/master_validation.py`

## Script under audit

- **Path**: `scripts/master_validation.py`
- **Purpose**: Gold-standard validation framework for any model configuration.
  3-fold 6-month expanding walk-forward + permutation ROI test.
  Produces a single PASS/FAIL verdict and p-value for every tactic tested.
  Intended to be run on any new configuration before it is adopted.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv`; Vegas via `attach_vegas_rich`.
- **Writes**: `results/master_validation.json`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

---

## ⚠ Pre-registration requirement

This script is only meaningful if the configuration under test is fixed
BEFORE running. Post-hoc adjustment of parameters after seeing results
invalidates the permutation p-value. The config block at the top of the
script is the pre-registered specification.

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test strictly chronological | ☑ pass | expanding window; each fold `train = df[DATE < fold_start]` |
| `assert train_max < test_min` | ☑ pass | explicit assertion per fold |
| Test bouts zero overlap with train | ☑ pass | explicit assertion per fold |
| HP search never reads test fold | ☑ pass — all hyperparams pre-specified in config block | no search loop |
| Permutation shuffles WITHIN fold, not across | ☑ pass — `won_pick` shuffled per-fold independently |

## §2 — Rolling / EMA / windows

`build_through_step6()` runs once on full dataset. Rolling/EMA features use only prior-fight data per row (inherited, verified in walkforward_market_features audit). ☑ pass.

## §3 — Career / history aggregates

WC priors recomputed per fold using only `df[DATE < fold_train_end]`. ☑ pass.

## §4 — Scalers / imputers

Re-instantiated and re-fit per fold on train only. ☑ pass.

## §5 — Elo

Inherited; computed sequentially before any fold split. Recency weights anchored at fold train_end. ☑ pass.

## §6 — Model selection

All hyperparameters pre-specified. Betting filter (men's, no-BW, +EV) pre-specified. No test-set selection. ☑ pass.

## §7 — Market / odds

Vegas attached after predictions per fold. ☑ pass.

## §8 — Features

Identical to `walkforward_market_features.py`. `select_features()` called before market merge. ☑ pass.

## §8a–§11

Inherited from walkforward_market_features and rolling_retrain_clean audits. ☑ pass.

---

## Permutation test design

- 1000 permutations per run
- Labels (`won_pick`) shuffled WITHIN each fold's filtered bet frame independently
  (restricted permutation — preserves win count per fold)
- Null distribution = pooled ROI across 3 folds under label shuffle
- p-value = fraction of null permutations with pooled ROI ≥ observed ROI (one-tailed)
- Folds shuffled independently to preserve temporal structure

## Verdict gates

| Gate | Threshold | Rationale |
|---|---|---|
| Consistency | ≥ 2/3 folds positive | Real edge is consistent, not driven by one lucky period |
| Significance | p < 0.10 | One-tailed; small n means strict p<0.05 has low power |
| Strong | 3/3 positive AND p < 0.05 | Reserve for deployment decisions |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Pre-registration requirement documented
- [x] Permutation design correct (within-fold restricted shuffle)

**Author**: claude
