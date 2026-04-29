# Leakage Audit — `scripts/rolling_retrain_6month.py`

## Script under audit

- **Path**: `scripts/rolling_retrain_6month.py`
- **Purpose**: Same expanding-window walk-forward as `rolling_retrain_clean.py` but with 6-month fold windows instead of quarterly (3 folds vs 6). Tests whether longer fold windows reduce the instability observed in the quarterly version while still retraining close enough to benefit Q1-2026.
- **Reads / Writes**: identical to `rolling_retrain_clean.md`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

All leakage checks are identical to `docs/audits/rolling_retrain_clean.md`. The only change is FOLDS: 3 × 6-month windows instead of 6 × 3-month windows. Per-fold WC priors frozen at fold train_end, imputer/scaler re-fit per fold, recency weights anchored at fold train_end. ☑ pass.

⚠ Same exploratory caveat: fold count/size not selected from test metrics — 6-month window is pre-specified based on prior session finding (finding_6month_retrain.md).

**Author**: claude
