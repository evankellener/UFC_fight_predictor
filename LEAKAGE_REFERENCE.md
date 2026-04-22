# Data Leakage Reference

Checklist + index of every leakage guardrail in this repo. Any agent running a new experiment must verify each item below still holds for the code path they touch. Two historical models in this codebase were inflated by leakage (see §9) — treat these as load-bearing, not decorative.

## 1. Temporal splits (never shuffle, never touch test)

- Train/test split is **chronological by DATE** — train on older fights, test on recent. [src/model.py:135](src/model.py:135)
- CV must be `TimeSeriesSplit` or explicit walk-forward folds. `shuffle=False` is enforced in [src/mma_ai_config.py:28](src/mma_ai_config.py:28).
- Tau / hyperparameter optimization runs only on walk-forward CV folds; the held-out test set (2024-05+) is never read during optimization. [src/optimize_taus.py:4,42-49](src/optimize_taus.py:4)
- Walk-forward in notebook: 8 folds × ~1.5 months, training window ends strictly before each fold's start date. [notebooks/01_Fight_Predictor_Pipeline.ipynb](notebooks/01_Fight_Predictor_Pipeline.ipynb) (cell 23)
- Backtest pattern: `tr = df[DATE < fs]; te = df[fs <= DATE < fe]`. [scripts/run_backtest_and_save.py:106](scripts/run_backtest_and_save.py:106)
- **Red flag**: any `train_test_split(...)` without `shuffle=False`, or any CV fold where train overlaps test dates.

## 2. Rolling / EMA / expanding windows must be shifted

Rule: **every `ewm()`, `rolling()`, `expanding()`, `cumsum()` on within-fighter history must be followed by `.shift(1)`** so the current row's outcome never enters its own feature.

Confirmed-safe call sites:
- [src/elo_feature.py:443](src/elo_feature.py:443) — per-WC event EMA, `.shift(1)`
- [src/predict_event.py:198,212,228](src/predict_event.py:198) — event-level EMAs, shifted
- [src/ensemble_model_best.py:368,373](src/ensemble_model_best.py:368) — precomp rolling EMA
- [add_rolling_ema_with_inference.py:35](add_rolling_ema_with_inference.py:35) — `.shift(1)` comment
- SQL: LAG(..., 1, 0.0) on all EMA-z accumulators — [src/sql_scripts/ufc_sql_new_adjperf2.sql:648-678](src/sql_scripts/ufc_sql_new_adjperf2.sql:648), [ufc_sql_new_adjperf3.sql:317-338](src/sql_scripts/ufc_sql_new_adjperf3.sql:317)
- SQL opponent rolling mean uses `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` — [ufc_sql_new_adjperf2.sql:186](src/sql_scripts/ufc_sql_new_adjperf2.sql:186)
- Inference-time EMA in [src/elo_feature.py:457](src/elo_feature.py:457) is intentionally unshifted — it's a snapshot *before* the upcoming fight, not a training row. Do not copy this pattern into training code.

Quickstart doc: [ROLLING_EMA_QUICKSTART.md](ROLLING_EMA_QUICKSTART.md) lines 60, 93, 120, 125, 173.

**⚠ Known suspect**: [src/elo_feature_enhancer.py:231-262](src/elo_feature_enhancer.py:231) uses `.shift(0)` before `.rolling().sum()` on 8 features (`precomp_elo_change_3/5`, `postcomp_elo_change_3/5`, and `opp_` variants). The inputs (`precomp_elo_prev`, `postcomp_elo_prev`) are already lagged, so the leakage may be zero, but nothing verifies this. Any experiment that relies on these features should prove it explicitly.

## 3. Career / history aggregates must exclude the current fight

Rule: **every career-level feature must filter `d < fight_date`** (strict inequality) before aggregating.

Confirmed-safe:
- [scripts/build_market_features.py:74-77](scripts/build_market_features.py:74) — `career_pos_before()` uses `d < dt`
- [scripts/build_sos_form_features.py:45-49](scripts/build_sos_form_features.py:45) — `stats_before()` uses `d < dt`
- [src/combined_features.py:163-169,244-250](src/combined_features.py:163) — missed-weight history & card position use `d < fight_date`
- Prior-streak / prior-12m features sorted by DATE first, then walked forward — [scripts/build_market_features.py:84-110](scripts/build_market_features.py:84)

**Red flag**: `d <= dt` or `d == dt` filters, or aggregates built without an explicit date cutoff.

## 4. Scalers / imputers / encoders fit on train only

Rule: **`fit()` or `fit_transform()` runs only on the training slice; test data gets `transform()`**.

Confirmed-safe:
- [src/model.py:172-174](src/model.py:172) — `StandardScaler.fit_transform(train_X)` then `transform(test_X)`
- [src/optimize_taus.py:128-129](src/optimize_taus.py:128) — `SimpleImputer` + `RobustScaler` fit on training fold only
- All walk-forward scripts follow `sc.fit_transform(tr[cols]); sc.transform(te[cols])`

**⚠ Known suspect**: [src/ensemble_model_best.py:283](src/ensemble_model_best.py:283) fits `SimpleImputer` on the full `self.df` before train/test filtering. Median estimates therefore see test data. Any experiment touching this model should either move the imputer inside the split or verify the impact is negligible.

## 5. Elo + time-aware decay

- Elo is computed only from raw bout records in `ufc_winlossko` — never joined against result-dependent tables. [src/RECREATE.md:65-67,125-127](src/RECREATE.md:65)
- `get_fighter_elo(event_date)` has explicit as-of semantics with inactivity decay — [src/elo_feature.py:569,585](src/elo_feature.py:569)
- Elo params (K, KO_MULT, SUB_MULT, DECAY, sigmoid) come from Bayesian optimization, not test-set fitting — [src/predict_event.py:15-34](src/predict_event.py:15)

## 6. Model selection / hyperparameter tuning

- Tuning uses a validation split inside the training set, never the final test set — [src/model.py:237,246,307](src/model.py:237)
- `GridSearchCV` uses `TimeSeriesSplit(n_splits=5)` everywhere in [src/ensemble_model_best.py](src/ensemble_model_best.py) (lines 563, 626, 742, 966, 2156, 2244, 2956, 3249, 4424)

## 7. Market / odds / contextual features

Vegas lines themselves are result-correlated (sharp closing lines near-optimal), so:
- Any feature built from odds must use **pre-fight** (opening or early) quotes, not closing lines from after-the-fact sources.
- "Zero vegas leakage" is claimed for current contextual features in [scripts/update_notebook.py:34,75,167,445](scripts/update_notebook.py:34) — these use weight cut, home advantage, card position, strength of schedule, not odds.
- The only place the model should meet Vegas odds is **at evaluation** (CLV, ROI, edge-vs-market) — not as a training feature.

## 8. Features named in memory (cross-reference before reusing)

- `feature_elo_predictability.md` — event-level EMA span=15, shifted 1 event
- `feature_sigmoid_decay.md` — time-aware Elo decay; no direct leakage risk, but decay params were tuned
- `feature_age_prime.md` — deterministic Gaussian on age; leakage-safe
- `feature_market_edge.md` — prior events only, `d < dt`

## 8a. Vegas odds pre-processing (added 2026-04-22)

Vegas odds join is **evaluation-only** (§7), but the *mechanics* of joining and
scoring have their own gotchas:

- **Reject scraper-artifact American odds**: `|avg_odds| < 100` is not a valid
  American odds value. Examples found in `ufc_fight_odds`: `-1, -2, -40, 34,
  49`. Converting these to decimal via the standard formula gives absurd payouts
  (e.g., `-1` → decimal 101, "bet $1 win $100"), which corrupt ROI and log-loss
  arithmetic. Enforced in [scripts/compute_roi.py](scripts/compute_roi.py) and
  [scripts/compare_to_vegas.py](scripts/compare_to_vegas.py). ~291 rows in the
  current odds table fail this check.

- **Clip probabilities before log loss**: sklearn's `log_loss` default
  `eps=1e-15` is pathologically sensitive to probabilities near 0 or 1. A single
  bad data point (e.g., `p=1.000` for a fighter who lost) adds `-log(1e-15)≈34.5`
  to the sum. Always clip to `[0.02, 0.98]` or similar before computing log
  loss when comparing model outputs to market odds (or to any source producing
  extreme probabilities). Brier is robust; AUC is ranking-only. Applied
  uniformly to all models being compared.

- **Fighter1 alignment**: our features' `jfighter` is the red-corner fighter
  (parsed from `BOUT` string). The odds table's `jfighter` may not match. Flip
  the Vegas probability when the two disagree — never assume ordering aligns.

## 9. Historical leakage bugs (do not re-introduce)

From [memory/mma_ai_full_spec.md](../../../../.claude/projects/-Users-evankellener-Desktop-UFC-fight-predictor/memory/mma_ai_full_spec.md) and [memory/handoff_prompt.md](../../../../.claude/projects/-Users-evankellener-Desktop-UFC-fight-predictor/memory/handoff_prompt.md):

- **Dec 5 2025**: AutoGluon presets were mixing future data into training. Fix dropped accuracy 70% → 64% ("more honest"). Do not re-enable `best_quality` presets without verifying the fold-aware splitter.
- **Jan 13 2026**: Shuffled CV was leaking across folds. Switched to temporal (no shuffle) + per-stat-per-WC τ.
- Reference model MMA-AI published "70.6% acc / 0.5964 LL / 0.7297 AUC" — noted as **leaky** by our measurements; their clean number is ~71% / 0.602. Do not calibrate target metrics to the leaky figure.
- **Apr 22 2026 (WC-index bug)**: All 6 per-weight-class τ overrides in `src/mma_ai_config.py` were routed against a WRONG weightindex-to-name map. E.g., `BB[4]={"sub_land":3}` labeled "Featherweight" was hitting index 4 = Women's Featherweight (30 fights). Verify any hardcoded WC literal against the DB's actual encoding (Stipe=12=HW, Khabib=8=LW, Pantoja=5=Fly). See `memory/finding_wc_index_bug.md`.
- **Apr 22 2026 (odds scraper garbage)**: `ufc_fight_odds` contains ~291 rows with invalid American odds (|o|<100 or NaN). A single bad row with `avg_odds_f2=0.0` propagated through devig to produce `p_vegas=1.000` for a fighter who lost → added ~0.10 to mean log loss over 349 fights. Reject invalid odds before any comparison; clip probabilities to `[0.02, 0.98]` before log loss.

## 10. Things this repo does NOT currently test

No dedicated leakage test suite exists. Before shipping a new feature, an experiment should at minimum:

1. Assert the feature is **monotone-non-decreasing in available history** — the value for fight N does not change when fight N+1 data is added.
2. Assert a permutation of labels on the test set collapses metric to baseline (catches target leakage).
3. Re-run one walk-forward fold with the feature removed to confirm claimed lift is stable across folds, not a single-fold artifact.
4. If the feature is an aggregate: verify the aggregation filter uses `<` not `<=` on the fight date.
5. If the feature involves rolling/EMA: grep the call site for `.shift(1)` in the training path.

## 11. Quick grep checklist for new experiments

Before merging any new feature branch, run these checks:

```
# Rolling / EMA without shift
grep -nE "\.ewm\(|\.rolling\(|\.expanding\(" <changed files> | grep -v "shift(1)"

# Train/test split without shuffle=False
grep -n "train_test_split" <changed files>

# Scaler/imputer fit on full df
grep -nE "\.fit\(|fit_transform\(" <changed files>

# Date filter using <= instead of <
grep -nE "date\s*<=|DATE\s*<=" <changed files>
```

Every hit is a place to justify or fix. Silence is the goal.
