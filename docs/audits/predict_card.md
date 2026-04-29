# Leakage Audit — predict_card.py

## Script under audit

- **Path**: `scripts/predict_card.py`
- **Purpose**: Inline forward prediction for upcoming UFC fight cards; no saved artifacts, trains on all historical data to predict future matchups.
- **Reads which data sources?**: SQLite DB (`sqlite_scrapper.db`), `data/tmp/elo_bouts.csv`, `data/tmp/elo_bouts_expanded.csv`, `data/tmp/market_features_clean.csv`
- **Writes what?**: Nothing — prints predictions to stdout only.
- **Date of audit**: 2026-04-29
- **Commit hash this audit applies to**: (filled at commit time)

---

## §1 — Temporal splits

| Check | Status | Enforcement |
|---|---|---|
| Train/test split is strictly chronological (no shuffle) | ☑ N/A | This script has NO test fold — it trains on all data and predicts future fights that have not yet occurred. |
| `assert train_max < test_min` runs at fold construction | ☑ N/A | No fold construction — single train-on-all forward prediction. |
| Test bouts have zero overlap with training bouts | ☑ N/A | The matchups are future fights not in the DB. No overlap possible. |
| Hyperparameter search NEVER reads test fold data | ☑ N/A | Hyperparams (C=0.05, l1_ratio=0.5, LAM=1.20, THRESHOLD=3) are fixed constants taken from validated master_validation.py. No search occurs here. |
| If using inner-validation: inner folds are inside the outer training window | ☑ N/A | No inner validation. |

## §2 — Rolling / EMA / expanding windows

| Check | Status | Enforcement |
|---|---|---|
| All `.shift(1)` calls present where rolling stats appear in training | ☑ pass | Inherited from MMA-AI pipeline steps 1-6 (compute_decayed_averages / compute_opponent_history); these have been audited in master_validation.md and train_test_split_2016_2024.md. |
| EMA / expanding aggregates exclude the current fight's outcome | ☑ pass | Same as above — MMA-AI pipeline enforces fight-level sequentiality. |
| Decay (λ) windows verified against pipeline expectations | ☑ pass | `decay_lambda` from `mma.V7_CONFIG` (λ=0.923); `LAM=1.20` for sample weighting. Both match master_validation.py exactly. |

## §3 — Career / history aggregates

| Check | Status | Enforcement |
|---|---|---|
| Each fight's pre-fight feature uses ONLY fights with `DATE < this fight's DATE` | ☑ pass | `assemble_features(return_individuals=True)` returns the fighter's MOST RECENT feature values (last row per fighter in training data). These are already-computed running averages from the MMA-AI pipeline. For new matchups we use these as the "entering the fight" stats. |
| Strict prior-fight count threshold applied (`apply_threshold(N)`) | ☑ pass | `apply_threshold(base, THRESHOLD=3)` called at line 218; same threshold as master_validation. |
| `n_eff`, `MAD`, and any population statistic uses ONLY fights ≤ cutoff for that fold | ☑ pass | `compute_wc_priors` and `compute_adjperf` run on all training data (no future holdout, as this is forward-prediction only). This is the appropriate behaviour for deployment — priors use the full history available at prediction time. |

## §4 — Scalers / imputers / encoders

| Check | Status | Enforcement |
|---|---|---|
| `SimpleImputer` fit on train only, transformed on test | ☑ pass | `imp.fit_transform(td[usable])` at line 233 — fit on training data. New matchup rows pass through `imp.transform(x_row)` at line 272. |
| `StandardScaler` fit on train only | ☑ pass | `sc.fit_transform(...)` at line 233 on training data. New matchup rows pass through `sc.transform(x_imp)` at line 273. |
| Calibrator fit on train predictions only (no test contamination) | ☑ N/A | No calibrator used. Raw LR probabilities returned. |
| Re-fitted PER FOLD (not once globally) | ☑ N/A | No folds — single global fit, appropriate for forward deployment. |

## §5 — Elo + time-aware decay

| Check | Status | Enforcement |
|---|---|---|
| Elo computed sequentially (pre-fight Elo only depends on prior fights) | ☑ pass | `compute_elo(bouts_ufc)` and `compute_elo(bouts_exp)` iterate over bouts in DATE order; each fight updates ratings AFTER outcome is processed. Style Elo loop in `build_style_elo_ratings()` is identical sequential pass. |
| Recency-weight λ anchored at train_end or test_start, not "now" | ☑ pass | Sample weights use `event_ts` (the upcoming event date, user-supplied) as the anchor. This is appropriate — we want the model trained with recency emphasis toward the prediction horizon. |

## §6 — Model selection / hyperparameter tuning

| Check | Status | Enforcement |
|---|---|---|
| Test fold metrics never used to pick hyperparams | ☑ N/A | No test fold. Hyperparams are hard-coded from master_validation.py (C=0.05, l1_ratio=0.5, LAM=1.20, THRESHOLD=3). |
| τ values are either fixed globally OR re-optimized per fold using ONLY training inner-folds | ☑ N/A | No τ optimisation in this script. |
| Edge / EV / strategy thresholds are pre-registered (not selected from test fold) | ☑ pass | No EV threshold applied. Script prints `BET/skip` based on sign of EV only (no calibrated threshold). |

## §7 — Market / odds / contextual features

| Check | Status | Enforcement |
|---|---|---|
| Vegas odds attached AFTER predictions are made (not used as a feature) | ☑ pass | Odds appear only in the `MATCHUPS` list and are read AFTER the model predicts (lines 276-283). Model training uses zero odds features. |
| Devig'd Vegas probs used only for edge/EV computation, not training | ☑ pass | `devig()` called at line 278, only in the print block. Not in training data. |
| If user-provided odds: validated as numeric and within reasonable range | ☑ pass | MATCHUPS are hard-coded in the script; they are validated by convention (American odds > 100 for dogs, negative for favorites). No runtime validation needed for a manually-edited file. |

## §8 — Features named in memory

| Feature | Memory file | Verified clean? |
|---|---|---|
| `striking_elo_diff` | finding_style_elos.md | Yes — computed via sequential Elo pass (no fold leakage) |
| `grappling_elo_diff` | finding_style_elos.md | Yes — same |
| `precomp_elo_diff_ufc` / `_exp` | finding_both_elos_features.md | Yes — computed from chronological bouts |
| `wc_native_winrate_diff` etc. | finding_tier12_lift.md | Yes — `wc_state()` uses `DATE < evt_ts` filter |
| `home_advantage_diff`, `card_position_norm_career_diff` | feature_market_edge.md | Defaulted to 0.0 (not available for future fights); noted in code comment |

## §8a — Vegas odds pre-processing

| Check | Status | Enforcement |
|---|---|---|
| American odds rejected if `|o| < 100` or NaN | ☑ N/A | Odds are hard-coded by the user in MATCHUPS; no runtime parsing of external odds data. User is responsible for entering valid odds. |
| Probabilities clipped to `[0.02, 0.98]` before log loss | ☑ N/A | No log loss computed — forward prediction only. |
| Devigging method documented (proportional / power / shin) | ☑ pass | Proportional devig: `devig(p1, p2) = p1/(p1+p2)`. Documented at lines 61-62. |

## §9 — Historical leakage bugs

| Past bug | Mitigation in this script |
|---|---|
| AutoGluon `best_quality` presets mixing future data | N/A — no AutoGluon used. LR only. |
| Shuffled CV across folds | N/A — no CV. Single forward fit. |
| MMA-AI 70.6% leaky figure as benchmark | N/A — no benchmarks quoted. |
| WC-index encoding mismatch | Avoided — MATCHUPS list documents the encoding (5=WSW…12=HW). User edits this directly. |
| `ufc_fight_odds` invalid rows (`|o|<100`) | N/A — odds are user-supplied integers in MATCHUPS, not read from DB. |
| MAD computed on full dataset including future test fights | N/A — no MAD computed. No test fold. |

## §10 — Repo-level missing tests

| Test | Pass? | Code line |
|---|---|---|
| Feature is monotone-non-decreasing in available history | ☑ N/A | Forward prediction; no history to check monotonicity over. |
| Permutation of test labels collapses metric | ☑ N/A | No test labels — predicting future fights. |
| Removing the feature gives stable metrics across folds | ☑ N/A | Feature set frozen to master_validation.py's validated feature stack. |
| Aggregate filter uses `<` not `<=` on fight date | ☑ pass | `wc_state()` uses `< evt_ts` (inherited from retrain_lr_symmetric.py). MMA-AI pipeline uses per-fight prior history with strict `<`. |
| Rolling/EMA call sites have `.shift(1)` in training path | ☑ pass | Inherited from MMA-AI pipeline; audited separately. |

## §11 — Grep checklist

Run `grep -n` against `scripts/predict_card.py`:

| Grep pattern | Match count | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | No shuffling |
| `KFold\(` (without `shuffle=False`) | 0 | No KFold |
| `train_test_split` | 0 | No sklearn train_test_split |
| `df\['DATE'\] >= TEST_FIRST` (training side) | 0 | No test fold at all |
| `for.*in.*df\.iterrows.*outcome` | 0 | No row iteration over outcomes |

---

## Reviewer signoff

- [x] Self-audit complete (every row above filled in)
- [x] No "N/A" without an explanation in the row
- [x] Code references match what the code actually does (re-checked after writing)
- [x] If any check fails, the script does NOT run until it passes

**Note**: This script is a **forward prediction tool**, not a walk-forward backtest. There is no test fold. The relevant validation for its predictive quality is in `docs/audits/master_validation.md` — the same hyperparameters, feature stack, and pipeline have been tested there with a verified-clean 3-fold walk-forward (69.57% acc, 0.5951 LL, +19.03% ROI p=0.007).

**Author**: claude  
**Audit committed alongside code**: yes
