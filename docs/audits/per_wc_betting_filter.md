# Leakage Audit — `scripts/per_wc_betting_filter.py`

## Script under audit

- **Path**: `scripts/per_wc_betting_filter.py`
- **Purpose**: Break down ROI by men's weight class on the clean pipeline test set, identify which individual WCs are profitable, and evaluate whether filtering to positive-ROI WCs improves overall betting performance.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv`; Vegas via `attach_vegas_rich`.
- **Writes**: `results/per_wc_betting_filter.json`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

---

## ⚠ Multiple-comparison disclosure (§6)

WC filter selections are made by observing test-set ROI per WC. This is test-set selection — selecting which WCs to bet based on the same data used to evaluate. Results are **exploratory only**. Any WC filter identified here must be pre-registered and validated on future data before deployment.

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test strictly chronological | ☑ pass | same DATE guards as all prior scripts |
| `assert train_max < test_min` | ☑ pass | explicit assertion |
| Test bouts zero overlap | ☑ pass | explicit assertion |
| HP search never reads test fold | ☑ pass — C=0.05, l1=0.5, λ=1.2 pre-specified | N/A |
| Inner folds | N/A — single split |

## §2–§5

All inherited from `walkforward_market_features.py`. ☑ pass.

## §6 — Model selection

⚠ Disclosed above. WC filter selected from test-set ROI — exploratory only.
EN hyperparameters pre-specified. ☑ pass.

## §7 — Market / odds

Vegas attached after predictions. ☑ pass.

## §8 — Features

Identical to `walkforward_market_features.py`. ☑ pass.

## §9–§11

No new leakage risks. 0 shuffle, 0 KFold, 0 train_test_split.

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Multiple-comparison risk disclosed
- [x] Exploratory only — no deployment decision

**Author**: claude
