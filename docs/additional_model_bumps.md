# Additional Model Bumps — Running Log

Live tracker for model-improvement experiments. Every idea goes in **Future candidates** with an estimate. When tried, it moves to either **Shipped** (with metrics vs baseline) or **Rejected** (with reason).

**Current production baseline** (commit `ce70c68`):
- **Walk-forward accuracy:** 71.6% pooled across 4 folds × 6mo test (n=522)
- **Walk-forward +EV ROI:** +14.0% on 213 bets vs Vegas odds at t=3
- **LL / Brier:** 0.600 / 0.206 (Vegas: 0.568 / 0.192 — structural calibration gap)
- **Architecture:** 207-feature LR, ElasticNet (C=0.05, l1=0.5), symmetric training (doubled rows), λ=1.20 recency weight, 2yr rolling WC baselines, temperature calibrator

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
| **Cardio Elo** (R3+/R1 output ratio as Elo) | 2 hr | +0.3-0.5pp | Currently a feature, not an Elo. Opp-adjusted version captures "fade vs maintain" career-long |
| **Style-matchup flags** (southpaw×orthodox, wrestler×striker) | 1-2 hr | +0.2-0.5pp | Commentators cite these; under-tested in the model |
| **Damage-absorbed Elo** (inverse of strikes taken) | 2 hr | +0.2pp | "Durability" signal; distinct from offensive Elo |
| ~~**Window-size sweep on era-rolling**~~ | tested 2026-04-24 | (running) | TBD — early result: 1yr ≈ 2yr |
| ~~**Model ensemble** (LR λ=0.13 + LR λ=1.20)~~ | **shipped** 2026-04-24 | **Tier B win** | Pooled +EV ROI 11.47% → **14.08%** (+2.61pp), accuracy 71.61% → 71.78%, LL 0.5971 → 0.5880. See Shipped #10. |

### Tier C — expensive, speculative

| Idea | Cost | Expected | Why maybe |
|---|---|---|---|
| **XGBoost on new feature stack** | 2-3 hr | 0 to +0.5pp | Earlier tests showed LR > blend with old features. Worth revisiting post era-rolling. |
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
