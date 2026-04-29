# Leakage Audit — `scripts/womens_division_model.py`

## Script under audit

- **Path**: `scripts/womens_division_model.py`
- **Purpose**: Train a women's-only Elastic Net model (weightclass_encoded ∈ {1,2,3,4}) and compare its metrics to the mixed model's performance on the same women's fights, to test whether a division-specific model outperforms the shared one.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv`; Vegas via `attach_vegas_rich`.
- **Writes**: `results/womens_division_model.json`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ☑ pass | `train = df[DATE>=TRAIN_START & DATE<TRAIN_END]`; `test = df[DATE>=TRAIN_END & DATE<TEST_END]` |
| `assert train_max < test_min` runs at fold construction | ☑ pass | explicit assertion before fit |
| Test bouts have zero overlap with training bouts | ☑ pass | explicit assertion on (DATE,jbout) pairs |
| Hyperparameter search NEVER reads test fold data | ☑ pass — C=0.05, l1=0.5, λ=1.2 all pre-specified (inherited from en_hyperparam_sweep) | N/A — no HP search |
| If using inner-validation: inner folds inside outer training window | N/A — single train/test split, no inner CV |

## §2 — Rolling / EMA / expanding windows

N/A — women's filter applied AFTER feature construction. All rolling/EMA features are built by the inherited pipeline (load_base_both_elos → add_wc_features). Feature values unchanged; only the model-fitting subset is restricted. ☑ pass.

## §3 — Career / history aggregates

Identical to `walkforward_market_features.py`. WC priors frozen at `TRAIN_END=2024-10-01`. `card_position_norm_career_diff` uses strict `d < dt` guard (empirically verified zero-leakage in walkforward_market_features audit). ☑ pass.

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only, transformed on test | ☑ pass | `imp.fit_transform(train[usable])` |
| `StandardScaler` fit on train only | ☑ pass | `sc.fit_transform(...)` on train subset only |
| Calibrator fit on train predictions only | N/A — no calibrator; comparing raw EN probabilities for apples-to-apples vs mixed model |
| Re-fitted PER FOLD | N/A — single split; imputer/scaler fit once on women's train |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially | ☑ pass — inherited; Elo is computed across ALL fights chronologically before subsetting |
| Recency-weight λ anchored at train_end | ☑ pass — `w = exp(-λ × (TRAIN_END − date).days / 365.25)`, λ=1.2 |

Note on Elo subset: Elo is computed on ALL fights before filtering to women's. This is correct — a fighter's Elo history legitimately includes cross-divisional context if they ever fought in multiple weight classes, and the Elo computation is sequential regardless.

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ☑ pass — C=0.05, l1=0.5, λ=1.2 inherited from prior experiment; not selected from women's test set | pre-specified constants |
| τ values fixed globally | ☑ pass — inherited |
| Edge / EV / strategy thresholds pre-registered | ☑ pass — same {all_picks, +EV, p≥0.65} as baseline |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions | ☑ pass — `attach_vegas_rich` called post-predict |
| Devig'd Vegas probs for edge/EV only | ☑ pass |
| User-provided odds validated | N/A — Vegas odds from `attach_vegas_rich`, same source as all prior scripts |

## §8 — Features named in memory

All features identical to `walkforward_market_features.py` (verified clean). No new features introduced.

| Feature | Memory file | Verified clean? |
|---|---|---|
| EN baseline features (select_features output) | finding_zero_leakage_baseline.md | ☑ yes |
| home_advantage_diff | feature_market_edge.md | ☑ yes |
| card_position_norm_career_diff | walkforward_market_features audit | ☑ yes (empirical 0 diff) |

## §8a — Vegas odds pre-processing

Inherited from `attach_vegas_rich`. ☑ pass (same as all prior scripts).

## §9 — Historical leakage bugs

| Past bug | Mitigation in this script |
|---|---|
| AutoGluon `best_quality` mixing future data | Not used |
| Shuffled CV across folds | No CV; `sort_values("DATE")` only |
| MMA-AI 70.6% leaky figure as benchmark | Not referenced |
| WC-index encoding mismatch | Women's WCs: 1=W.Straw, 2=W.Fly, 3=W.Bantam, 4=W.Feather — verified via DB encoding |
| `ufc_fight_odds` invalid rows | Inherited from `attach_vegas_rich` |
| MAD computed on full dataset | Not used |

## §10 — Repo-level missing tests

| Test | Pass? | Code line |
|---|---|---|
| Feature is monotone-non-decreasing in available history | ☑ inherited | inherited pipeline |
| Permutation of test labels collapses metric | N/A — diagnostic comparison, not a new feature test | |
| Removing feature gives stable metrics | N/A | |
| Aggregate filter uses `<` not `<=` on fight date | ☑ pass | inherited |
| Rolling/EMA call sites have `.shift(1)` | ☑ pass | inherited |

Explicit assertion: `assert train["DATE"].max() < TRAIN_END` and `assert test["DATE"].min() >= TRAIN_END`.

## §11 — Grep checklist

| Grep pattern | Match count | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | none |
| `KFold\(` | 0 | no inner CV |
| `train_test_split` | 0 | not used |
| `df['DATE'] >= TEST_FIRST` on training side | 0 | training mask uses `<` |
| `for.*in.*df\.iterrows.*outcome` | 0 | none |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] No "N/A" without explanation
- [x] Code references match actual code
- [x] Women's WC encoding verified (1–4)

**Author**: claude
**Audit committed alongside code**: yes
