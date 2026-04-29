# Leakage Audit — `scripts/en_hyperparam_sweep.py`

## Script under audit

- **Path**: `scripts/en_hyperparam_sweep.py`
- **Purpose**: Sweep EN hyperparameters C ∈ {0.01,0.05,0.1,0.2,0.5,1.0} ×
  l1_ratio ∈ {0.3,0.5,0.7} on the clean market-features pipeline to find
  the regularization setting that lets the most signal through without
  overfitting, given that C=0.05 zeros 176/212 features.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv`; Vegas via `attach_vegas_rich`.
- **Writes**: `results/en_hyperparam_sweep.json`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

## ⚠ Multiple-comparison disclosure (§6)

C and l1_ratio are swept on the **same held-out test set** used by all
prior experiments. Selecting the best cell = test-set selection. Results
are **exploratory only**. The chosen (C, l1_ratio) must be pre-registered
and validated on future data before deployment.

---

## §1 — Temporal splits

| Check | Status |
|---|---|
| Train/test strictly chronological | ☑ pass — `DATE < TRAIN_END` / `DATE >= TRAIN_END` |
| `assert train_max < test_min` | ☑ pass |
| Test bouts zero overlap | ☑ pass |
| HP search never reads test fold | ⚠ disclosed — exploratory sweep on test |

## §2 — Rolling / EMA / windows

N/A — C/l1_ratio only affect the EN fit weights, not feature values. ☑ pass.

## §3 — Career / history aggregates

Identical to `walkforward_market_features.py`. ☑ pass.

## §4 — Scalers / imputers

Imputer and scaler re-instantiated per cell, fit on train only. ☑ pass.

## §5 — Elo

Inherited. ☑ pass.

## §6 — Model selection

⚠ Disclosed above. λ=1.2 fixed throughout sweep to isolate C/l1_ratio effect.

## §7 — Market / odds

Vegas attached after predictions. ☑ pass.

## §8 — Features

Same as `walkforward_market_features.py` (empirically verified zero-leakage).

## §9–§11

No new leakage risks. Grep: 0 shuffle, 0 KFold, 0 train_test_split.

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Multiple-comparison risk disclosed
- [x] Exploratory only — no deployment decision

**Author**: claude
