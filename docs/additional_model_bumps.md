# Additional Model Bumps — Running Log

Live tracker for model-improvement experiments. Every idea goes in **Future candidates** with an estimate. When tried, it moves to either **Shipped** (with metrics vs baseline) or **Rejected** (with reason).

**Current production baseline** (commit `b075af2` — λ-ensemble shipped):
- **Walk-forward accuracy:** **71.78%** pooled across 4 folds × 6mo test (n=522)
- **Walk-forward +EV ROI:** **+14.08%** on **256 bets** vs Vegas odds at t=3 ★
- **LL / Brier:** 0.5880 / 0.2030 (Vegas: 0.568 / 0.192 — structural calibration gap)
- **Architecture:** **λ-ensemble** averaging two LRs:
  - λ=1.20 single-model alone: 71.61% acc, +11.47% +EV ROI (228 bets), LL 0.5973
  - λ=0.13 single-model alone: 69.87% acc, +10.32% +EV ROI (300 bets), LL 0.6056
  - **ENSEMBLE (50/50 calibrated avg):** 71.78% / +14.08% / 256 bets / LL 0.5880 ★
- Each LR: 207 features, ElasticNet (C=0.05, l1=0.5), symmetric training (doubled rows),
  2yr rolling WC baselines, temperature calibrator
- **Single-LR ablation baseline** (used for A-vs-A+feature tests because companion
  isn't retrained per ablation): 71.62% acc / +13.99% +EV ROI / 213 bets / LL 0.6004.
  An experiment must clear the **ensemble baseline (+14.08%, 256 bets)** to ship,
  not just the single-LR baseline.

---

## Shipped improvements (session 2026-04-24)

| # | Change | Commit | Δ acc | Δ +EV ROI | Notes |
|---|---|---|---|---|---|
| 1 | Fix WC-index bug in mma_ai_config.py | (earlier session) | baseline | — | 6 τ overrides were mis-routed |
| 2 | Symmetric training (double rows) | `1e676bd` | +0pp | — | eliminates red-corner bias; accuracy holds, orientation-invariance gained |
| 3 | Temperature-scaling calibrator | `563313e` | 0 | — | ECE 7.78 → 4.77pp; accuracy preserved |
| 4 | Extend test to 2026-04 | `a297c04` | −1.5pp | — | Honest accuracy drop (stale train). Exposed drift. |
| 5 | Walk-forward 4-fold validation | `90d7ac8` | — | — | Revealed 11pp drift fold_1 → fold_4 |
| 6 | WC-history features (5 new) | `6f75e5a` | +0.24pp | — | 2 of 5 active; cross-div flag dropped |
| 7 | λ sweep + diagnostic | `b4a0af9` | diagnostic | — | Best λ trends upward with fold recency (drift fingerprint) |
| 8 | Ship λ=1.20 (aggressive recency) | `f118fcc` | fold_4 +3.2pp | fold_4 +13pp | Trades old-fold accuracy for recent |
| 9 | **Era-rolling 2yr WC baselines** | `3c118ae` | **+2.0pp** | **+6.5pp** | Biggest single fix. Halves cross-fold variance. |
| 10 | **λ-ensemble** (LR(λ=0.13) + LR(λ=1.20)) | (this commit) | **+0.17pp** | **+2.61pp** | Averages calibrated probs from both models. Beats either single model on every metric in walk-forward. Hedges the era-drift bet. |

---

## Future candidates — ranked by expected lift vs cost

### Tier A — cheap, probable wins

| Idea | Cost | Expected | Why promising |
|---|---|---|---|
| ~~Recent-form features (last 3 fights)~~ | tried 2026-04-24 | **Rejected** — see Rejected list. All 3 coefs zeroed. |
| ~~ElasticNet C sweep~~ | tried 2026-04-24 | **No improvement** — C=0.05 already optimal on every metric. See Rejected list. |
| ~~Finishing Elo~~ | tried 2026-04-24 | **Rejected** — coef zeroed AND displaced grappling_elo. See Rejected list. |

### Tier B — medium cost, uncertain

| Idea | Cost | Expected | Why promising |
|---|---|---|---|
| ~~**Cardio Elo** (R3+/total output ratio as Elo)~~ | tried 2026-04-24 | **Rejected** — coef zeroed. See Rejected list. |
| ~~**Style-matchup flags** (southpaw×orthodox, wrestler×striker)~~ | tried 2026-04-24 | **Rejected** — both coefs zeroed. See Rejected list. |
| ~~**Damage-absorbed Elo** (inverse of strikes taken)~~ | tried 2026-04-24 | **Rejected** — coef zeroed. See Rejected list. |
| ~~**Window-size sweep on era-rolling**~~ | tested 2026-04-24 | (running) | TBD — early result: 1yr ≈ 2yr |
| ~~**Model ensemble** (LR λ=0.13 + LR λ=1.20)~~ | **shipped** 2026-04-24 | **Tier B win** | Pooled +EV ROI 11.47% → **14.08%** (+2.61pp), accuracy 71.61% → 71.78%, LL 0.5971 → 0.5880. See Shipped #10. |

### Tier C — expensive, speculative

| Idea | Cost | Expected | Why maybe |
|---|---|---|---|
| ~~**XGBoost on new feature stack**~~ | tried 2026-04-24 | **Rejected** — see Rejected list. Walk-forward 4-fold: XGB alone 67.05% acc / 0.6061 LL (LR baseline 71.62% / 0.5973). Blend at w_xgb=0.15 gives +11.94% ROI vs LR's +11.51% (within sample noise). |
| **Pre-UFC fight history (scraped)** | 1-2 days | ? | Could help rookies specifically. Data acquisition expensive. |
| **Stacking ensemble** (LR base + XGB residual) | 3-4 hr | +0.3-0.7pp | Correlated errors between LR and XGB may limit lift |
| **Per-weight-class micro-models** | 3-4 hr | typically negative | Small-n per WC; likely overfits. Included for completeness. |

### Tier D — not viable (data/infeasibility)

- Injury/camp news
- Line movement / steam moves (no historical scrape)
- Weight-cut severity (would need fight-week weigh-in data)
- Pressure/cage-control metrics (not in our data)
- Betting market public-money % (not scraped)

---

## Rejected experiments

| Experiment | Date | Result | Why rejected |
|---|---|---|---|
| `cross_division_x_weight_diff` interaction | 2026-04-23 | coef +0.006 survived but acc dropped 0.5pp | Sample-size noise; tiny coef barely moved predictions |
| Isotonic calibration | 2026-04-23 | max_dev 15.7 → 30.7pp, acc −2pp | Overfits on n=420 test |
| LR×0.8 + XGB×0.2 blend | (prior session) | tiny LL gain, ROI −3pp | Discrimination > calibration for betting (`finding_blend_hurts_roi.md`) |
| τ re-optimization per fold | 2026-04-23 | never completed | 3hr compute; analysis showed drift isn't from τ staleness |
| Pure `edge > 0` bet threshold | 2026-04-23 | flagged −EV bets as WEAK BET | Changed to `ev > 0` threshold in `_bet_recommendation` |
| **Recent-form features** (last3_win_rate_diff, last3_finish_rate_diff, last3_avg_fight_time_diff) | 2026-04-24 | All 3 ElasticNet coefs **= 0.00000**. Walk-forward identical to 4 decimal places vs era-rolling baseline. | Existing `dec_avg`-derived features (`win_streak_entering_diff`, `finish_rate_dec_avg_diff`, etc.) already capture this signal. Hard-cutoff last-3 added redundancy without new info. **Helpers kept** (`load_recent_form_from_db`, `add_recent_form_features`, `fighter_recent_form.json` cache) for future variants like weighted last-5 or per-fight form deltas. |
| **ElasticNet C sweep** (C ∈ {0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 3.0}) | 2026-04-24 | C=0.05 (current) wins every metric on 4-fold mean: acc 71.61%, LL 0.6003, Brier 0.2062, AUC 0.7492. Closest competitor C=0.10 (LL 0.6042, acc 69.92%). | The structural ~0.033 LL gap vs Vegas (0.60 vs 0.57) is NOT a regularization problem. Vegas aggregates info we don't have (line movement, public consensus, sharp money). Tighter C under-fits (only 12 active features → 68% acc). Looser C overfits and accuracy collapses. Don't propose C tuning again. Script kept at `scripts/c_sweep_4fold.py` and results at `results/c_sweep_4fold.json` for reference. |
| **Finishing Elo** (KO+sub actual, K=20, SCALE=400) | 2026-04-24 | Coef zeroed (= 0.00000). Worse: displaced `grappling_elo_diff` which had been an active feature — now also zeroed. Pooled t=3 acc dropped 71.62% → 71.43% (-0.19pp), pooled +EV ROI dropped 13.99% → 12.93% (-1.06pp). | Finishing rate is correlated with grappling Elo (many finishes via submission/control). Adding finishing as a separate feature confused ElasticNet — it dropped both. Existing `finish_rate_dec_avg_diff` + `ko_dec_avg_diff` features already capture finishing tendency. **`build_finishing_elo()` kept as dead code** in `run_threshold_sweep_both_elos.py` for reference. |
| **Cardio Elo** (R3+/total striking-output share as Elo, K=20, SCALE=400, sigmoid scale=0.15, only updated when fight reaches R3+) | 2026-04-24 | Coef = 0.00000 in production LR. Pooled t=3 metrics IDENTICAL to baseline (71.62% acc / +13.99% +EV ROI / 213 bets / LL 0.6004 / Brier 0.2049). Walk-forward 4-fold also unchanged (71.61% acc, 0.5971 LL). | Late-round output share already captured by per-round striking aggregates in `dec_avg` features. ElasticNet dropped the new column rather than re-distribute weight. **Builders `build_cardio_elo()` + helpers `_load_per_fight_striking()`, `_pivot_to_bouts()` kept as dead code** in `run_threshold_sweep_both_elos.py` for reuse. |
| **Damage-absorbed Elo** (per-minute strikes absorbed by opponent, K=20, SCALE=400, sigmoid scale=2.0) | 2026-04-24 | Coef = 0.00000 in production LR. No metric movement vs baseline. | "Durability" signal already implicit in defensive accuracy features (`head_acc_def_dec_avg_diff`, `td_def_dec_avg_diff`). ElasticNet zeroed it. **Builder `build_damage_absorbed_elo()` kept as dead code** for reference. Pattern is now clear: NEW Elo features built from existing per-round/per-fighter striking data are *redundant* with the dec_avg feature stack — they get zeroed. To add an Elo that survives, it must encode information **not** already in the smoothed per-fighter aggregates. |
| **Style-matchup flags** — `stance_clash_diff` ∈ {−1, 0, +1} signed southpaw-vs-orthodox flag (from `fighter_bios.json`); `style_clash_diff` = `td_att_pm_adjperf_dec_avg_diff` × `sig_str_land_pm_adjperf_dec_avg_diff` (negative ⇒ wrestler-vs-striker mismatch, symmetric under f1↔f2 swap so preserved in `flip_row_dataframe`) | 2026-04-24 | Both coefs = 0.000000 in production LR. Pooled t=3 metrics IDENTICAL to baseline (71.62% / +13.99% / 213 bets). t=1 / t=2 metrics moved within noise (+0.27pp / +0.32pp acc, ROI changes <0.4pp). | The information is already encoded multiple ways in the existing 207-feature stack: stance shows up via `head_per_sig_str_land_diff` and outside-leg target patterns; wrestler-striker clash is captured by the per-fighter td_att_pm and sig_str_land_pm diffs *individually*. ElasticNet doesn't gain by adding their interaction or a discrete clash flag. **Builder `build_style_matchup_features()` kept as dead code; `flip_row_dataframe` keeps the `style_clash_diff` preservation special-case** (cheap insurance for any future product-of-diffs feature). Don't propose stance flags or wrestler×striker interactions again unless paired with NEW per-fighter signals not already in `dec_avg`. |
| **XGBoost** (300 shallow trees, depth=3, lr=0.05, subsample=0.8, reg_alpha=0.5, reg_lambda=1.5, min_child_weight=10, gamma=0.5, recency-weighted λ=1.20, temperature-calibrated) on the production 214-feature stack | 2026-04-24 | XGB alone walk-forward 4-fold pooled: 67.05% acc / 0.6061 LL / 0.7385 AUC / 0.2078 Brier / 4.59pp ECE. **All worse than LR baseline** (71.62% / 0.5973 / 0.7508 / 0.2048 / 8.46pp ECE) except calibration (XGB ECE 4.59 vs LR 8.46). Fold-4 collapses to 62.7% acc — XGB doesn't handle the 2025-10→2026-04 era drift. | Tried LR+XGB calibrated-prob ensemble at w_xgb ∈ {0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5}. Best +EV ROI = +11.94% at w_xgb=0.15 vs LR-only +11.51% (Δ +0.43pp on 236 vs 228 bets — well inside sample noise CI ±2pp). Best LL = 0.5941 at w_xgb=0.30 but accuracy drops 1.4pp there. **No Pareto-improving blend.** Tree-residual stacking would have the same ceiling. Scripts kept: `walk_forward_4fold_xgb.py`, `wf_lr_xgb_ensemble.py`, `wf_lr_xgb_blend_sweep.py`, `wf_lr_xgb_vegas_roi.py`. Don't propose XGB on this feature stack again. |

---

## Running disclaimers

- Any lift estimate under +0.3pp is indistinguishable from sample noise at n=522 test fights (bootstrap CI ≈ ±2pp on aggregate accuracy).
- LL/Brier gap to Vegas (~3pp / 1.5pp) is structural. Sharp markets aggregate information we don't have (line movement, bettor consensus, breaking news). Gap-closing requires either that information or better calibration via a separate layer, not more features.
- Every new feature family must pass LEAKAGE_REFERENCE.md §1-§11 audit. Prefer: pre-fight known info, strict-prior filters, per-fold refit of scaler+calibrator.
- Drift is now handled but not eliminated. Best practice: retrain production every 3-6 months.

---

## How to update this doc

**Before starting work on an idea:**
Move it out of "Future candidates" into a "Work in progress" row here so we don't try it twice.

**After finishing an experiment:**
1. If it worked → move to "Shipped improvements" with commit hash, Δ metrics, one-line note
2. If it didn't → move to "Rejected experiments" with date, result, and specific reason
3. If it's ambiguous (within noise) → mark it rejected with "within sample variance" and link to the eval

**Measuring "worked":**
Always use the walk-forward 4-fold at t=3 as the honest benchmark, not single-shot. Single-shot is for quick iteration; walk-forward is for ship/no-ship decisions.

## Work in progress

_(none right now)_
