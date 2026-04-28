# Claude instructions for UFC Fight Predictor

This file is auto-loaded into every Claude session in this repo. The rules
below are non-negotiable — violating them has cost time, money, and trust
in past sessions.

---

## Hard rules

### Leakage discipline

1. **Before writing any walk-forward / backtest / metric script:**
   - Copy `docs/LEAKAGE_AUDIT_TEMPLATE.md` → `docs/audits/<script_name>.md`
   - Fill in every row of §1-§11. "N/A" requires a one-sentence reason.
   - Commit the audit in the SAME commit as the script.
   - If you can't fill in every row honestly, do not write the script.

2. **Before claiming "zero leakage" or quoting any model metric:**
   - Run `scripts/test_leakage_per_fold.py` with matching `train_end`
   - Confirm `0 columns differed`
   - If anything differs, fix or document the bound on impact

3. **Never use these to derive features** (they aggregate over the full df,
   leaking future data):
   - `df[col].mean()` / `.median()` / `.sum()` / `.std()` / `np.percentile`
   - `df.groupby([wc])[col].mean()` (the all-time variant)
   - Use `_era_rolling_mean` / `_all_prior_rolling_mean` / `_era_rolling_ratio`
     from `src/mma_ai_pipeline.py` instead.

4. **Never use `mma_ai_pipeline.build_features()` output directly for
   walk-forward.** It runs `compute_wc_priors` + `compute_adjperf` globally,
   which leaks 58 `*_adjperf_dec_avg_diff` columns. Use the per-fold pattern
   from `scripts/notebook_clean_walkforward.py` (or
   `scripts/train_test_split_2016_2024.py`).

### Reporting discipline

5. **Don't dress up small samples as validation.** n<30 outcomes is
   uninformative regardless of which side they fall on.

6. **Don't promise time estimates without measuring.** Profile one
   iteration, multiply. If you can't profile cheaply, don't estimate —
   say "I don't know" with a wide honest range.

7. **When metrics drop after a leakage fix, REPORT the drop honestly.**
   Don't spin "small leak, probably fine" — that phrase has caused real
   damage. If a leak exists, fix it or document the impact bound.

### Trust discipline

8. The user has been through multiple cycles in this project of:
   "model claims +X% ROI" → audit reveals leak → metrics collapse.
   Words like "rigorous" and "ZERO leakage" must be backed by procedure
   (filled audit + passing empirical test) BEFORE they're spoken.

---

## Honest baseline (post-2026-04-27 leakage fix)

The clean baseline metric for any new model to beat:

```
Train: 2016-01 → 2024-10  (1,907 fights, ~8.75yr)
Test:  2024-10 → 2026-04  (391 fights, ~18mo)
Single Elastic Net (C=0.05, l1_ratio=0.5), per-fold WC priors

  Accuracy:  69.57%
  Log loss:  0.5951
  Brier:     0.2041
  AUC:       0.7532
```

Verified zero leakage by `scripts/test_leakage_per_fold.py` at
`train_end=2024-10-01` and `train_end=2025-04-01`. Result: 0 columns
differed when future data was added/removed.

**Do NOT compare new models to the leaky baselines from earlier sessions
(+14.08%, +27.25%, +39.47% ROI, 71.62% accuracy).** Those came from
contaminated pipelines and are not honest targets.

---

## Architecture quick reference

| Path | What it does |
|---|---|
| `src/mma_ai_pipeline.py` | MMA-AI Steps 1-6 are per-fight clean. Steps 7-9 (compute_wc_priors, compute_adjperf, assemble_features) leak when called globally — must be invoked per-fold. |
| `scripts/notebook_clean_walkforward.py` | Reference per-fold workflow; the architecture template for new walk-forward scripts |
| `scripts/train_test_split_2016_2024.py` | Single train/test split, Elastic Net only, verified clean |
| `scripts/test_leakage_per_fold.py` | The empirical leakage test. Run before quoting metrics. |
| `scripts/test_leakage_empirical.py` | Tests static build_features() pipeline (still has 58 known leaks; use to detect regressions only) |
| `LEAKAGE_REFERENCE.md` | The leakage doc. Read §0 (mandatory audit policy). |
| `docs/LEAKAGE_AUDIT_TEMPLATE.md` | The template every walk-forward script must fill |
| `docs/audits/` | Filled audits — one per script |

## Pre-commit hook

A pre-commit hook lives at `scripts/git-hooks/pre-commit`. Install once:
```
cp scripts/git-hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```
It blocks commits that add walk-forward / training / leakage-test scripts
without a matching audit doc.
