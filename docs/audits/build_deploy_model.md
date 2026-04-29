# Leakage Audit — `scripts/build_deploy_model.py`

## Script under audit

- **Path**: `scripts/build_deploy_model.py`
- **Purpose**: Build deployment model artifact using master_validation's select_features() pipeline, trained on ALL data through April 2026. Saves to `app/models/deploy_v1/`.
- **Reads which data sources?**: SQLite DB (`data/sqlite_db/sqlite_scrapper.db`), mma_ai_pipeline (steps 1-6), `data/tmp/market_features_clean.csv`, Elo CSV artifacts, DB WC history.
- **Writes what?**: `app/models/deploy_v1/{lr.pkl, lr_scaler.pkl, lr_imputer.pkl, feat_cols.json}`, `data/tmp/market_features_clean.csv`.
- **Date of audit**: 2026-04-28
- **Commit hash this audit applies to**: (filled at commit time)

---

⚠️ **STATUS: UNTESTED END-TO-END.** This script was created as an alternative to retrain_deploy_lr.py but was never run to completion and never tested for prediction quality. Documented in STATUS.md as "app/models/deploy_v1/ — DO NOT USE." The recommended path is to build predict_card.py which trains inline without pre-saved artifacts.

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ☑ N/A | Deployment build — no test set by design. Out-of-sample validity proven by master_validation.py STRONG PASS (p=0.007). |
| `assert train_max < test_min` runs at fold construction | ☑ N/A | No test fold |
| Test bouts have zero overlap with training bouts | ☑ N/A | No test fold |
| Hyperparameter search NEVER reads test fold data | ☑ N/A | Hyperparams pre-specified from master_validation CONFIG |
| If using inner-validation: inner folds are inside the outer training window | ☑ N/A | No inner validation |

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| All `.shift(1)` calls present where rolling stats appear in training | ☑ pass | Inherited from mma_ai_pipeline.py steps 1-6 (verified in walkforward_market_features audit) |
| EMA / expanding aggregates exclude the current fight's outcome | ☑ pass | MMA-AI steps 1-6 are per-fight clean (per-fight rolling, not global) |
| Decay (λ) windows verified against pipeline expectations | ☑ pass | Uses `mma.V7_CONFIG["decay_lambda"]` = 0.13 for feature decay; LAM=1.20 for training weights |

## §3 — Career / history aggregates

| Check | Status | Enforcement |
|---|---|---|
| Each fight's pre-fight feature uses ONLY fights with `DATE < this fight's DATE` | ☑ pass | MMA-AI pipeline steps 1-6 enforce this; verified in master_validation audit |
| Strict prior-fight count threshold applied (`apply_threshold(N)`) | ☑ pass | `apply_threshold(base, THRESHOLD)` at step after Elo load; THRESHOLD=3 |
| `n_eff`, `MAD`, and any population statistic uses ONLY fights ≤ cutoff for that fold | ☑ N/A — deploy build | `compute_wc_priors` called on ALL data (no fold split) — acceptable for deployment, not for evaluation |

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only, transformed on test | ☑ N/A | No test split; deployment build fits on full corpus |
| `StandardScaler` fit on train only | ☑ N/A | No test split; deployment build |
| Calibrator fit on train predictions only (no test contamination) | ☑ N/A | No calibrator |
| Re-fitted PER FOLD (not once globally) | ☑ N/A | No folds; single training run |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially (pre-fight Elo only depends on prior fights) | ☑ pass | Loaded from pre-computed CSV via `load_base_both_elos()` |
| Recency-weight λ anchored at train_end or test_start, not "now" | ☑ pass | `w = np.exp(-LAM * (TRAIN_END - td["DATE"]).dt.days / 365.25)` anchored at TRAIN_END=2026-05-01 |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ☑ N/A | No test fold |
| τ values are either fixed globally OR re-optimized per fold using ONLY training inner-folds | ☑ N/A | No τ |
| Edge / EV / strategy thresholds are pre-registered (not selected from test fold) | ☑ N/A | No betting simulation |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions are made (not used as a feature) | ☑ N/A | No prediction / odds lookup in this script |
| Devigged Vegas probs used only for edge/EV computation, not training | ☑ N/A | No odds |
| If user-provided odds: validated as numeric and within reasonable range | ☑ N/A | No user input |

## §8 — Features named in memory

| Feature | Memory file | Verified clean? |
|---|---|---|
| select_features() output (214 features) | finding_both_elos_features.md, finding_tier12_lift.md | Yes — same set used in master_validation |
| NOVEL_FEATS: home_advantage_diff, card_position_norm_career_diff | feature_market_edge.md | Yes — per-fight, no global aggregation |

## §8a — Vegas odds pre-processing

| Check | Status | Enforcement |
|---|---|---|
| American odds rejected if `|o| < 100` or NaN | ☑ N/A | No odds in this script |
| Probabilities clipped to `[0.02, 0.98]` before log loss | ☑ N/A | No log loss |
| Devigging method documented | ☑ N/A | No odds in this script |

## §9 — Historical leakage bugs

| Past bug | Mitigation in this script |
|---|---|
| AutoGluon `best_quality` presets mixing future data | N/A — no AutoGluon |
| Shuffled CV across folds | N/A — no CV |
| MMA-AI 70.6% leaky figure as benchmark | N/A — no benchmark comparison |
| WC-index encoding mismatch | N/A — no WC index encoding in these features |
| `ufc_fight_odds` invalid rows (`|o|<100`) | N/A — no odds loaded |
| MAD computed on full dataset including future test fights | N/A — no MAD |

## §10 — Repo-level missing tests

| Test | Pass? | Code line |
|---|---|---|
| Feature is monotone-non-decreasing in available history | ☑ N/A | Features inherited from validated pipeline |
| Permutation of test labels collapses metric | ☑ N/A | No test evaluation |
| Removing the feature gives stable metrics across folds | ☑ N/A | Deployment build; metrics validated by master_validation.py |
| Aggregate filter uses `<` not `<=` on fight date | ☑ N/A | Inherited from pipeline |
| Rolling/EMA call sites have `.shift(1)` in training path | ☑ N/A | Inherited from pipeline |

## §11 — Grep checklist

| Grep pattern | Match count | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | |
| `KFold\(` (without `shuffle=False`) | 0 | |
| `train_test_split` | 0 | |
| `df\['DATE'\] >= TEST_FIRST` (training side) | 0 | No TEST_FIRST variable; uses TRAIN_END |
| `for.*in.*df\.iterrows.*outcome` | 0 | |

---

## Reviewer signoff (filled at commit time)

- [x] Self-audit complete (every row above filled in)
- [x] No "N/A" without an explanation in the row
- [x] Code references match what the code actually does (re-checked after writing)
- [x] If any check fails, the script does NOT run until it passes

**Author**: claude
**Audit committed alongside code**: yes
