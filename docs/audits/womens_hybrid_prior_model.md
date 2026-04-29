# Leakage Audit — `scripts/womens_hybrid_prior_model.py`

## Script under audit

- **Path**: `scripts/womens_hybrid_prior_model.py`
- **Purpose**: Test a hierarchical (gender+WC) prior for women's divisions. Men's WC priors are unchanged. For women's WCs (1–4), each per-WC prior is shrunk toward a pooled "all-women" prior using sample-size-weighted blending (α = n_wc / (n_wc + κ)). Compare accuracy and ROI to the mixed baseline.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv`; Vegas via `attach_vegas_rich`.
- **Writes**: `results/womens_hybrid_prior_model.json`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test strictly chronological | ☑ pass | same DATE guards as all prior scripts |
| `assert train_max < test_min` | ☑ pass | explicit assertion |
| Test bouts zero overlap | ☑ pass | explicit assertion |
| HP search never reads test fold | ☑ pass — C=0.05, l1=0.5, λ=1.2, κ=50 all pre-specified | N/A |
| Inner folds | N/A — single split |

## §2 — Rolling / EMA

N/A — hybrid prior only changes WC normalization in adjperf, not rolling windows. ☑ pass.

## §3 — Career / history aggregates

Critical: the hybrid pooled-women prior is computed on `train_only` (df[DATE < TRAIN_END]) — same data as the per-WC priors. ☑ pass.

## §4 — Scalers / imputers

Fit on train only. ☑ pass.

## §5 — Elo

Inherited. ☑ pass.

## §6 — Model selection

κ=50 is pre-specified based on domain reasoning (shrink heavily when n_wc < 50 fighter-bouts), not selected from test metrics. C/l1/λ inherited. ☑ pass.

⚠ Exploratory — comparing hybrid vs baseline on same test set is test-set selection.

## §7 — Market / odds

Vegas attached after predictions. ☑ pass.

## §8 — Features

Same feature set as `walkforward_market_features.py`. The only change is the adjperf z-score normalization for women's WC rows.

| Feature | Memory file | Verified clean? |
|---|---|---|
| All features | walkforward_market_features audit | ☑ yes |

## §8a — Vegas odds

Inherited. ☑ pass.

## §9 — Historical leakage bugs

| Past bug | Mitigation |
|---|---|
| WC-index encoding mismatch | Women's: 1–4, men's: 5–12 verified |
| Global prior leaking future | Pooled prior computed on train_only only |
| Shuffled CV | No CV |

## §10 — Repo-level tests

Explicit train_max < TRAIN_END assertion. ☑ pass.

## §11 — Grep checklist

| Pattern | Count |
|---|---|
| `\.shuffle\(` | 0 |
| `KFold\(` | 0 |
| `train_test_split` | 0 |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Pooled prior computed on train_only only — no leakage
- [x] κ=50 pre-specified, not tuned on test

**Author**: claude
