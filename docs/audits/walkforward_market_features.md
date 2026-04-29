# Leakage audit — `scripts/walkforward_market_features.py`

## Script under audit

- **Path**: `scripts/walkforward_market_features.py`
- **Purpose**: Re-test the hypothesis that market/contextual features
  (home advantage, travel, card position, timezone, stance) improve pooled
  metrics and/or close the rookie gap. These features were tested on the
  *leaky* MMA-AI pipeline (pre-2026-04-27) and mostly failed to survive
  EN-L1 regularisation. The hypothesis here is that they were drowned out
  by 58 `*_adjperf_dec_avg_diff` columns that no longer exist in the clean
  pipeline. With only ~37 active features, they may now contribute
  independent signal.
- **Reads**: SQLite DB; `data/tmp/market_features_clean.csv` (written by
  `scripts/build_market_features.py`); Vegas via `attach_vegas_rich`.
- **Writes**:
  - `results/walkforward_market_features.json`
  - `results/walkforward_market_features_predictions.parquet`
- **Date of audit**: 2026-04-28
- **Commit hash**: (filled at commit)

## Pre-registered hypothesis

If the market/contextual features add genuine signal to the clean pipeline:
- At least one of the 7 new features should survive EN-L1 (non-zero coef)
- Pooled log loss / Brier / AUC should improve (any improvement is positive)
- Pooled accuracy should be stable or improve

If all 7 features are zeroed out by EN-L1, we conclude the features are
still redundant with the existing feature set. If pooled metrics degrade
we reject.

Secondary (not decision-making): check whether the rookie 3-4 slice gap
narrows — the same gap targeted by the prior priors-features experiment.

## New features

All features are pre-computed in `data/tmp/market_features_clean.csv` by
`scripts/build_market_features.py`. Provenance of each:

| feature | source | leakage analysis |
|---|---|---|
| `home_advantage_diff` | fighter nationality (static) vs event country (current fight) | ✅ no leakage — both are known before the fight |
| `travel_distance_diff_km` | haversine(fighter home, event venue) | ✅ no leakage — static home country, event location pre-announced |
| `tz_diff_diff_hr` | abs(fighter home tz − event tz) difference | ✅ same |
| `is_main_event` | from `ufc_card_position` table, property of THIS bout | ✅ cards are publicly announced before the fight; no future info |
| `card_position_norm_career_diff` | mean of (pos/total−1) across PRIOR fights; see §3 | ✅ verified `<` guard below |
| `stance_mismatch` | static fighter attribute from `ufc_fighter_tott` | ✅ pre-fight static |
| `southpaw_advantage_diff` | same | ✅ same |

**Critical ordering note**: `feats_baseline = select_features(df)` is
computed BEFORE the novel market features are merged into `df`. This
prevents `home_advantage_diff` and `card_position_norm_career_diff`
(which end in `_diff`) from accidentally appearing in the baseline. The
comparison is baseline (37 features) vs baseline + 7 market features.

---

## §1 — Temporal splits

| Check | Status |
|---|---|
| Train/test strictly chronological | ☑ pass (inherits parent script) |
| `assert train_max < TRAIN_END` | ☑ pass |
| `assert test_min >= TRAIN_END` | ☑ pass |
| Test bouts have zero overlap with train | ☑ pass |
| HP search never reads test fold | ☑ pass — no HP search; EN(C=0.05, l1=0.5) fixed |
| CV-OOF folds (beta calibrator) inside train window | ☑ pass — KFold(5, shuffle=False) on sorted train rows |

## §2 — Rolling / EMA / expanding windows

N/A — new features are fight-level or career-prior aggregates. No rolling.
☑ pass.

## §3 — Career / history aggregates

| Check | Status |
|---|---|
| `card_position_norm_career_diff` uses only prior fights | ☑ pass — `career_pos_before(jf, dt)` at `build_market_features.py:76` uses `[p for d, p in h if d < dt]` (strict `<`) |
| No same-day fights included in career aggregate | ☑ pass — `d < dt` excludes same-date fights |
| `coming_off_loss_diff`, `win_streak_entering_diff`, `fights_last_12m_diff` in CSV | ✅ NOT added as new features — these are already present and active in the baseline via `load_base_both_elos` + `select_features`. Explicitly excluded from NEW_FEATURES to prevent double-counting. |
| `apply_threshold(3)` | ☑ pass — both fighters must have ≥3 UFC priors |

## §4 — Scalers / imputers / encoders

| Check | Status |
|---|---|
| Imputer fit on train only | ☑ pass |
| Scaler fit on train only | ☑ pass |
| Beta calibrator fit on CV-OOF predictions of TRAIN rows only | ☑ pass — KFold(5, shuffle=False); never sees test |
| Market feature NaN imputation strategy | ☑ pass — median imputation on train; applied to test with train params |

## §5 — Elo + time-aware decay

Inherits sequential Elo. ☑ pass.

## §6 — Model selection / hyperparameter tuning

| Check | Status |
|---|---|
| Test fold metrics never used to pick HP | ☑ pass |
| Pre-registered hypothesis stated above | ☑ pass |
| Decision criterion stated before running | ☑ pass — "at least one feature survives EN-L1 AND pooled ll/Brier/AUC improves" |

## §7 — Market / odds

Vegas attached AFTER predictions, used only for slice reporting. ☑ pass.

## §8 — Features named in memory

| feature | provenance |
|---|---|
| baseline `*_diff`, `*_ufc`, `*_exp` features | identical to `train_test_split_2016_2024.py` |
| `home_advantage_diff` | `build_market_features.py` (static nationality vs event country) |
| `travel_distance_diff_km` | `build_market_features.py` (haversine) |
| `tz_diff_diff_hr` | `build_market_features.py` |
| `is_main_event` | `build_market_features.py` (card position lookup) |
| `card_position_norm_career_diff` | `build_market_features.py` (prior-fight mean) |
| `stance_mismatch` | `build_recency_stance.py` (already in df, excluded from baseline) |
| `southpaw_advantage_diff` | same |

## §9 — Historical leakage bugs

- `build_market_features.py` was written before the leakage audit framework.
  It was not formally audited. However, each feature's source was traced
  above (§3) and all pass the temporal guard. The script writes a CSV
  read-only by this script — it doesn't modify DB or model artifacts.
- `career_pos_before()` strict `<` guard verified at source line 76.

## §10 — Repo-level missing tests

| Test | Status |
|---|---|
| Aggregate filter uses `<` not `<=` | ☑ pass — `< TRAIN_END`, `< TEST_END` in split masks |
| `career_pos_before` uses `d < dt` | ☑ pass — verified at source |

## §11 — Grep checklist

| Pattern | Match | Notes |
|---|---|---|
| `\.shuffle\(` | 0 | none |
| `KFold\(.*shuffle=True` | 0 | KFold uses `shuffle=False` |
| `train_test_split` | 0 | not the sklearn function |
| `df\['DATE'\] >= TEST` (training side) | 0 | training mask uses `<` |
| `select_features` called before market merge | ✅ required — verified in code |

---

## Reviewer signoff

- [x] Self-audit complete
- [x] Pre-registered hypothesis stated explicitly
- [x] Code refs match what the code does
- [x] `select_features` called before market feature merge (critical ordering)
- [x] `coming_off_loss_diff` etc. NOT double-added
- [x] `card_position_norm_career_diff` `<` guard verified at source

**Author**: claude
