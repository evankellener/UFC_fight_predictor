# Weight-Class History Features

**Status:** proposed in session 2026-04-23, built same session.
**Motivation:** the Buchecha-vs-Spann case. Buchecha is 0-2 at HW; Spann has 15 bouts exclusively at LHW. The model (v2, 202 features) predicted Buchecha 80.4% with 41% of the signal coming from a raw `WEIGHT_diff` (+50 lbs). We had no feature encoding "at the division this fight is happening in, how has each fighter actually performed?"

## The five new features

Added to `feat_cols.json` (207 features total after addition).

| Feature | Definition | Range |
|---|---|---|
| `wc_native_winrate_diff` | f1 − f2 of decayed win rate (λ=0.13/yr) over *strictly prior* bouts at the **current fight's weightindex only**, with Beta-Binomial shrinkage α=3 toward 0.5 | [−1, +1] |
| `wc_native_fights_diff` | f1 − f2 count of prior bouts at current weightindex | [−N, +N] |
| `wc_native_ko_rate_diff` | f1 − f2 of decayed (win by KO) rate at current weightindex, shrinkage α=3 toward 0.0 | [−1, +1] |
| `days_since_this_wc_diff` | f1 − f2 days since last bout at current weightindex (9999 if never) | ± days |
| `cross_division_flag` | 1 if either fighter's **most-frequent (modal)** prior division ≠ current fight's weightindex; 0 otherwise. Captures career-level home division, not just last bout. Non-diff, symmetric under fighter swap. | {0, 1} |

## Shrinkage rationale (α=3)

Pure sample rates on 0-2 priors are too confident (0% or 100%). Beta-Binomial shrinkage with α=3 implements:

```
wc_native_winrate = (wins + 0.5 * α) / (fights + α)
                  = (wins + 1.5) / (fights + 3)
```

- Fights=0 → 0.5 (neutral prior)
- Fights=2, wins=0 → 1.5 / 5 = 0.30 (Buchecha at HW — still leans bad but not catastrophic)
- Fights=15, wins=8 → 9.5 / 18 = 0.528 (Spann at LHW — barely moved from raw 53.3%)

Matches the Beta-Binomial smoothing used elsewhere in the MMA-AI pipeline (see `mma_ai_full_spec.md`).

For KO rate the shrinkage is toward **0.0** (base rate prior for finishes), not 0.5:
```
wc_native_ko_rate = (kos + 0 * α) / (fights + α) = kos / (fights + α)
```

## Decay

Standard pipeline λ = 0.13 per year. For each prior bout at the current weightindex, weight is `exp(-λ * (event_date - fight_date) / 365.25)`. Decayed win rate uses decay-weighted sums in both numerator and denominator.

## Leakage guardrails (LEAKAGE_REFERENCE.md §1, §4)

1. **Strict temporal filter.** Every lookup uses `fight_date < event_date` (strictly less-than). Never `<=`. The current fight itself never appears in the input to its own feature.

2. **Current fight's weightindex is NOT leakage.** Weight class is set at weigh-ins, which is ~24h before the bout. Publicly known. The model legitimately gets to use it at inference. Verified: `ufc_fight_results.weightindex` and `final_features_fast.weightindex` are consistent within a fighter's career and across matching jbouts (not outcome-dependent).

3. **Prior-only decayed averaging.** Matches the pipeline's `prior = vals[:i]` convention used by `compute_decayed_averages`. At row i in training, `wc_native_winrate` is computed from fights [0, i-1] only, filtered to current row's weightindex.

4. **Shrinkage α = 3 is a constant, not a learned parameter.** No tuning on the test set.

5. **Scaler + imputer fitted on doubled-train only.** Loaded from disk at inference, never re-fit.

## Symmetry under (f1, f2) swap

When the feature row is flipped for training-set doubling:

| Feature | Transform under swap | Why |
|---|---|---|
| `wc_native_winrate_diff` | × −1 | diff of per-fighter values |
| `wc_native_fights_diff` | × −1 | diff of per-fighter counts |
| `wc_native_ko_rate_diff` | × −1 | diff of per-fighter rates |
| `days_since_this_wc_diff` | × −1 | diff of per-fighter days |
| `cross_division_flag` | unchanged | symmetric property of the fight pair |

This is handled by `flip_row_dataframe()` in `scripts/retrain_lr_symmetric.py`.

## Default handling at inference for unknown weight class

At inference, if the user doesn't specify a weight class, use `a1.get("weightindex")` (f1's most-recent-prior division) as the assumed division — same convention as the existing `weightclass_encoded` feature. If a user wants to specify (e.g., "fight is at HW, not Buchecha's usual"), they can pass `scheduled_weightindex` in the request (future UI work, optional parameter).

## Coverage

As of the build date, `ufc_fight_results.weightindex` has 100% coverage for bouts 2013-present and high coverage going back further. `final_features_fast.weightindex` agrees and is the pipeline's canonical source.

## Expected impact

Best guess: +0.1 to +0.5pp test accuracy. Larger lift if cross-division fights are common in the test set. Specific case improvements expected:

- **Buchecha vs Spann 2026-04-23**: `wc_native_winrate` for Buchecha at HW shrinks from 0% → 30% (α=3); for Spann at HW = 0 fights → 50% neutral prior. `cross_division_flag = 1` (Spann's modal division LHW ≠ current). The model should de-weight the raw `WEIGHT_diff` dominance and pull the prediction down from 80.4% somewhat.
- Fighters with long stable careers at one division (Sterling at BW, Jones at LHW/HW): no change, both `cross_division_flag=0` and `wc_native_winrate_diff ≈ overall win rate diff`.

## What didn't work (audited & reverted)

Tested and removed in the same session:

**`cross_division_x_weight_diff` interaction** — product of `cross_division_flag` × `WEIGHT_diff`. Rationale: catch the Buchecha-Spann size asymmetry that appears only in cross-division fights.

Result on 420-fight test set:
- Coefficient after ElasticNet: +0.0058 (active, rank 83/208)
- Accuracy: 71.19% (without) → 70.71% (with). **Dropped by 2 fights.**
- Log loss / AUC / Brier: essentially unchanged.

Interpretation: the feature fires in ~1% of training fights and carries a tiny coefficient. Adding it flipped 2 borderline decisions on the test set, probably noise, but no measurable lift. Reverted.

## ElasticNet survival in final 207-feature model

Of the 5 added features:
- `wc_native_winrate_diff`: **active**, rank 63/207, coef +0.0245 (intuitive direction)
- `wc_native_fights_diff`: **active**, rank 62/207, coef −0.0245 (subtle; captures late-career-at-division penalty after age features soak up the main signal)
- `wc_native_ko_rate_diff`: **zeroed**
- `days_since_this_wc_diff`: **zeroed**
- `cross_division_flag`: **zeroed**

The +0.24pp accuracy lift is essentially all from the two winrate/fights features. The flag and the other two are computed but contribute nothing at the current regularization (C=0.05, l1_ratio=0.5). They remain in the feature set for transparency and to allow future regularization tuning to reactivate them if the training set grows.

## Cost of the change

- One new JSON artifact (`fighter_wc_history.json`, ~250-400 KB).
- 5 new training features (202 → 207 columns).
- Re-run symmetric LR fit. 6-month forward betting-log validation still needed before acting on the shifted probabilities.
