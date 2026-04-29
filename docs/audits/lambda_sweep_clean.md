# Leakage Audit — `scripts/lambda_sweep_clean.py`

## Script under audit

- **Path**: `scripts/lambda_sweep_clean.py`
- **Purpose**: Sweep training recency-weight λ ∈ {0.13, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8} on the clean market-features pipeline to diagnose the Q1-2026 accuracy drift and identify whether a higher λ reduces it.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv`; Vegas via `attach_vegas_rich`.
- **Writes**: `results/lambda_sweep_clean.json`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

## ⚠ Multiple-comparison disclosure (§6)

This script evaluates 8 λ values on the **same held-out test set** used by
every prior experiment in this session. Selecting the λ with the best test-set
ROI would be test-set selection — a form of leakage. Results must be treated
as **exploratory / hypothesis-generating only**. The best λ identified here
must be pre-registered and validated on future data before deployment.

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test strictly chronological | ☑ pass | `train = df[DATE < TRAIN_END]`; `test = df[DATE >= TRAIN_END]` |
| `assert train_max < test_min` | ☑ pass | explicit assertion before each λ fit |
| Test bouts zero overlap with train | ☑ pass | explicit assertion |
| HP search never reads test fold | ☑ pass — λ is the variable being swept, not selected by test metrics | λ values are pre-specified by user; no search loop reads test outcome |
| Inner folds inside training window | N/A — no inner CV; λ sweep is exploratory on test |

## §2 — Rolling / EMA / expanding windows

All feature decay is inherited from `load_base_both_elos()` (λ_feat=0.13,
verified clean). The λ swept here is the **sample weight** in the EN fit
only — it does not affect feature values. ☑ pass.

## §3 — Career / history aggregates

Identical to `walkforward_market_features.py`. WC priors frozen at
`TRAIN_END=2024-10-01`; `card_position_norm_career_diff` uses strict `d < dt`
guard (empirically verified zero-leakage). ☑ pass.

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| Imputer fit on train only | ☑ pass | `imp.fit_transform(train[usable])` |
| Scaler fit on train only | ☑ pass | `sc.fit_transform(...)` on train |
| No calibrator | N/A — uncalibrated predictions used for λ sweep comparability |
| Re-fitted per λ | ☑ pass — imputer/scaler re-instantiated inside λ loop |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo sequential | ☑ pass — inherited |
| Recency weight anchored at `TRAIN_END` | ☑ pass — `w = exp(-λ × (TRAIN_END − date).days / 365.25)` |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics used to pick λ | ⚠ DISCLOSED — this is exploratory; see header note | Results are diagnostic; no deployment decision made here |
| EN C=0.05, l1=0.5 fixed | ☑ pass — only λ varies |
| Strategy thresholds pre-registered | ☑ pass — same {all_picks, +EV, edge≥5pp, edge≥10pp, p≥0.65} as baseline |

## §7 — Market / odds

Vegas attached after predictions. ☑ pass.

## §8 — Features

Same feature set as `walkforward_market_features.py` (verified clean).
`card_position_norm_career_diff` empirically verified zero-leakage.

## §9 — Historical leakage bugs

None re-introduced. λ sweep only changes sample weights in EN fit.

## §10 — Repo-level missing tests

| Test | Pass? |
|---|---|
| Aggregate filter `<` not `<=` | ☑ pass |
| Train/test assertions before each λ | ☑ pass |

## §11 — Grep checklist

| Pattern | Match | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | none |
| `KFold\(` | 0 | no inner CV |
| `train_test_split` | 0 | not used |
| `df['DATE'] >= TEST` on training side | 0 | training mask uses `<` |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Multiple-comparison risk explicitly disclosed (§6 header)
- [x] Results treated as exploratory — no deployment decision
- [x] λ values pre-specified by user; not selected from test metrics

**Author**: claude
