# Leakage audit — `scripts/strategy_confidence_beta_cvoof.py`

## Script under audit

- **Path**: `scripts/strategy_confidence_beta_cvoof.py`
- **Purpose** (one sentence): Re-run Tier 1 strategy-confidence tests
  (Vegas Monte-Carlo, label-shuffle, pre-registered held-out) on the
  beta-calibrated CV-OOF predictions from
  `results/train_calib_compare_v2_predictions.parquet`, to determine
  whether the +11.44% ROI on edge_5pp survives a multiple-comparison-aware
  null test.
- **Reads**: `results/train_calib_compare_v2_predictions.parquet`
  (column `p_B_beta`); Vegas via `attach_vegas_rich`.
- **Writes**: `results/strategy_confidence_beta_cvoof.json`.
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

This script does **NO training** and **NO calibration**. It consumes
already-frozen predictions and runs statistical tests. Most leakage rows
are inherited from the parent audit
(`docs/audits/train_calib_compare_v2.md`) and are not duplicated here.
The added-risk surface is "did we accidentally pick up future info via
the prediction column we're testing?" — answered below.

## §1 — Temporal splits

The test set (2024-10 → 2026-04) is the same one used by the parent
calibration audit. Beta calibrator was fit on CV-OOF predictions of the
2016-01 → 2024-10 train rows ONLY; the calibrator never saw any test row.
☑ pass.

## §6 — Pre-registration & multiple comparisons

This is the central concern for THIS specific script.

| Check | Status | Notes |
|---|---|---|
| Strategy thresholds (ev>0, edge≥5pp, edge≥10pp) pre-registered | ☑ pass | identical fixed thresholds as the original `strategy_confidence_tests.py` |
| Calibrator (beta-CV-OOF) chosen on which basis? | ⚠ disclosed | beta was selected by ROI from a 14-cell sweep (7 calibrators × 2 strategies) on this same test set. The MC p-value reported here therefore overstates significance for any single hypothesis test. The honest interpretation is: "if beta-CV-OOF clears p<0.05/14 ≈ p<0.0036 (Bonferroni) we have a real signal; otherwise it's plausibly multiple-testing noise." Both raw p and Bonferroni-adjusted thresholds are reported. |
| Held-out (2025-10 → 2026-04) cutoff fixed BEFORE running | ☑ pass | same cutoff as parent Tier 1 audit |

## §7 — Vegas odds

Attached AFTER predictions; never as feature. ☑ pass.

## §11 — Grep checklist

Same as parent. ☑ pass.

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Multiple-comparison risk explicitly disclosed (§6)
- [x] Code refs match what the code does

**Author**: claude
