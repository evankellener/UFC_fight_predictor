---
name: feedback_leakage_audit
description: ALWAYS do a thorough leakage audit of the ENTIRE pipeline when user asks — don't just check Elo/model training
type: feedback
---

When the user asks to "ensure there's no data leakage" or "make sure it's clean," audit the FULL pipeline — not just the model training split or Elo computation. The leakage has repeatedly been in the feature engineering layer (smoothing priors, weight-class baselines, AdjPerf z-scores, global medians) where statistics are computed on ALL data including future fights. Check every .mean(), .median(), .std(), groupby aggregate, and prior computation to verify it only uses data available at each fight's date.

**Why:** User has been burned multiple times by being told "it's clean" when it wasn't. The leakage was always in the deepest layer of the pipeline (PG/BB smoothing priors, WC priors for AdjPerf). This destroyed trust and wasted weeks of work.

**How to apply:** Before claiming anything is leakage-free, trace every global statistic back to its source data and verify temporal correctness.
