# Leakage Audit — `scripts/mens_only_model.py`

## Script under audit

- **Path**: `scripts/mens_only_model.py`
- **Purpose**: Train a men's-only Elastic Net model (weightclass_encoded ∈ {5..12}) and compare its ROI/metrics to the mixed model's performance on the same men's test fights, to test whether removing women's fights from training and evaluation improves betting ROI.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv`; Vegas via `attach_vegas_rich`.
- **Writes**: `results/mens_only_model.json`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ☑ pass | `train = df[DATE>=TRAIN_START & DATE<TRAIN_END]`; `test = df[DATE>=TRAIN_END & DATE<TEST_END]` |
| `assert train_max < test_min` | ☑ pass | explicit assertion before fit |
| Test bouts zero overlap with train | ☑ pass | explicit assertion on (DATE,jbout) pairs |
| HP search never reads test fold | ☑ pass — C=0.05, l1=0.5, λ=1.2 pre-specified; no search | N/A |
| Inner folds inside training window | N/A — single split |

## §2 — Rolling / EMA / expanding windows

N/A — men's filter applied after feature construction. Feature values unchanged. ☑ pass.

## §3 — Career / history aggregates

Identical to `walkforward_market_features.py`. WC priors frozen at TRAIN_END. ☑ pass.

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only | ☑ pass | fit on men's train subset |
| `StandardScaler` fit on train only | ☑ pass | fit on men's train subset |
| Calibrator | N/A — no calibrator; comparing raw EN probabilities |
| Re-fitted per fold | N/A — single split |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially | ☑ pass — inherited; computed across ALL fights before subsetting |
| Recency-weight λ anchored at TRAIN_END | ☑ pass — `w = exp(-λ × (TRAIN_END − date).days / 365.25)`, λ=1.2 |

## §6 — Model selection

C=0.05, l1=0.5, λ=1.2 inherited from en_hyperparam_sweep / lambda_sweep_clean. Not re-selected from men's test set. ☑ pass.

## §7 — Market / odds

Vegas attached after predictions. ☑ pass.

## §8 — Features

All features identical to `walkforward_market_features.py` (verified clean). No new features.

| Feature | Memory file | Verified clean? |
|---|---|---|
| EN baseline + market features | walkforward_market_features audit | ☑ yes |

## §8a — Vegas odds pre-processing

Inherited from `attach_vegas_rich`. ☑ pass.

## §9 — Historical leakage bugs

| Past bug | Mitigation |
|---|---|
| AutoGluon future mixing | Not used |
| Shuffled CV | No CV |
| MMA-AI 70.6% leaky benchmark | Not referenced |
| WC-index encoding mismatch | Men's WCs: 5–12; verified via DB encoding |
| `ufc_fight_odds` invalid rows | Inherited |
| MAD on full dataset | Not used |

## §10 — Repo-level tests

| Test | Pass? |
|---|---|
| Aggregate filter `<` not `<=` | ☑ inherited |
| Assertions before fit | ☑ explicit |

## §11 — Grep checklist

| Pattern | Count | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | |
| `KFold\(` | 0 | |
| `train_test_split` | 0 | |
| training-side `>= TEST` date | 0 | |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Men's WC encoding verified (5–12)
- [x] Exploratory only — men's-only ROI not pre-registered

**Author**: claude
