# Era-Rolling Weight-Class Baselines

**Status:** shipped 2026-04-24 in commit `3c118ae`.
**Motivation:** drift. The model's accuracy on recent fights (fold_4: 65.08%) was 11pp lower than on older fights (fold_1: 76.03%) despite training on a rolling 7-year window per fold. λ recency weighting helped but couldn't close the gap.

## The change

Replaced the all-time per-weight-class shrinkage prior in `src/mma_ai_pipeline.py` with a per-row 2-year rolling window.

**Before** (both `beta_binomial_smooth` and `poisson_gamma_smooth`):
```python
wc_rates = df.groupby("weightindex")[stat].mean().to_dict()  # all-time, leaky
```

**After** (via two new helpers):
```python
rolling_rate = _era_rolling_mean(df, stat)   # for binary rates
rolling_rate = _era_rolling_ratio(df, numer, denom)  # for counts/time
```

Both helpers use `groupby("weightindex").rolling("730D", on="DATE", closed="left")` so the current row and any same-date fights are excluded.

## Why it matters

The UFC has drifted significantly over 9 years:

| Stat | 2017–18 | 2026 | Change |
|---|---|---|---|
| sig_str_land_per_min | 2.59 | 3.28 | +27% |
| total_str_land_per_min | 3.87 | 4.57 | +18% |
| ko_dec_avg | 0.88 | 1.16 | +33% |
| head_acc_dec_avg | 0.277 | 0.323 | +17% |

The old all-time baseline meant a 2026 fighter at 4.0 sig_str_pm was smoothed toward ~3.2 (all-time mean) and looked "elite." In reality they're average-for-2026. The feature encoded era inflation as skill.

With 2-year rolling baselines, each fighter is compared to their era's peers. A 2026 fighter at 4.0 sig_str_pm is compared to ~3.2 (2024-2026 mean) → shrinks less → more information content preserved.

## Leakage guarantees (LEAKAGE_REFERENCE.md)

1. **§1 Temporal:** `closed='left'` in the rolling means `[d - 730d, d)` — strictly less than current row's date.
2. **§3 Strict prior:** same-date same-WC fights are excluded by the left-closure.
3. **§4 Scaler fit:** unchanged, still train-only.
4. **Window choice:** 2-year is manually set as a hyperparameter (not tuned on test).

Verified on 140 synthetic rows at ship time — every row's baseline matched the hand-computed `mean(stat over priors in [d - 730d, d) with same wc)`.

## Results (walk-forward, t=3, 4 folds × 6mo test, calibrated, n=522)

| Metric | Before | After | Δ |
|---|---|---|---|
| fold_1 accuracy | 67.77% | 70.25% | +2.48pp |
| fold_2 accuracy | 72.87% | 72.09% | −0.78pp |
| fold_3 accuracy | 69.63% | 72.59% | +2.96pp |
| fold_4 accuracy | 68.25% | **71.43%** | **+3.18pp** |
| Pooled accuracy | 69.67% | **71.62%** | +1.95pp |
| Pooled +EV ROI (vs Vegas) | +7.47% | **+13.99%** | +6.52pp |
| Cross-fold accuracy std | 3.48pp | **1.47pp** | variance halved |
| Pooled log loss | 0.6026 | 0.6004 | −0.0022 |
| Pooled Brier | ~0.207 | 0.2063 | flat |

## What didn't improve

- **Calibration (LL, Brier):** only marginal improvement. Era-rolling sharpens *features*, but the LR's elastic-net regularization still produces probabilities that hedge toward 0.5. Vegas still beats us by ~0.033 on log loss across every fold — that gap is structural calibration, not drift.
- **fold_2 accuracy dipped 0.78pp** — within noise; fold_2 +EV ROI also dropped (15.78 → 9.68). One fold's regression is acceptable given the global win.

## What this supersedes

- `finding_threshold_matters.md` — the monotonic t=1 < t=2 < t=3 ROI pattern was an artifact of single-shot methodology + incomplete Vegas coverage. Post-era-rolling walk-forward shows t=1 and t=3 tie on +EV ROI.
- Urgency on quarterly retrains — drift is now meaningfully reduced. Still good practice; less critical.

## Future work

- **Sweep window sizes** (1yr, 2yr, 3yr, 4yr) to find the optimum. Currently picked 2yr by intuition.
- **Era-relative z-score features** as an additive signal family. Marginal now that rolling baselines exist, but possibly +0.5-1pp accuracy.
- **Close the LL/Brier gap to Vegas** — requires either regularization changes (raise C from 0.05) or a new calibrator family (isotonic failed earlier, Platt/Beta tied on ECE).
