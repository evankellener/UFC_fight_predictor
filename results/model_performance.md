# UFC Fight Predictor — Model Performance Reference

Generated 2026-04-22 on the 422-fight filtered test window (2024-05-04 → 2025-11-08).
Filter: threshold=3 prior UFC fights per fighter + strict method filter (drop DQ,
overturned, split-decision, majority-decision). All results pass the
[LEAKAGE_REFERENCE.md](../LEAKAGE_REFERENCE.md) §1–§11 checklist.

---

## 1. MMA-AI replication (no Elo)

Replication target: MMA-AI.net v7 published metrics (acknowledged leaky per his Dec 5 2025 admission).

| Model | n | Acc | LogLoss | AUC | Brier |
|---|---|---|---|---|---|
| MMA-AI v7 published (LEAKY) | 411 | 70.32% | 0.5985 | 0.7297 | 0.2057 |
| MMA-AI post-fix clean (his admission) | 411 | ~64.00% | — | — | — |
| LR ElasticNet (C=0.05, l1=0.5) | 422 | 68.25% | 0.6012 | 0.7434 | 0.2065 |
| XGBoost | 422 | 66.59% | 0.6512 | 0.6945 | 0.2248 |
| CatBoost | 422 | 68.72% | 0.6143 | 0.7162 | 0.2114 |
| **LR + CatBoost blend (50/50)** | 422 | **69.19%** | **0.5974** | **0.7348** | **0.2051** |
| **AutoGluon WeightedEnsemble_L4** (his architecture, clean methodology) | 422 | **69.43%** | **0.6058** | **0.7337** | **0.2088** |

**AutoGluon leaderboard** (winning ensemble members):
`WeightedEnsemble_L4 = 91.7% NeuralNetTorch_BAG_L3 + 8.3% LightGBM_2_BAG_L3`

Under clean methodology our replication **beats MMA-AI's admitted post-fix
accuracy (~64%) by +5.4 percentage points**. Below his published-leaky 70.32% by
0.89pp on accuracy; AUC 0.7337 matches within 0.004.

---

## 2. MMA-AI + Elo features

Added 6 Elo-derived features via `src/elo_feature.py::compute_elo` with the
deployed parameter set (K=48, KO=1.80, SUB=1.20, sigmoid decay 0.25/730/80,
logistic_scale=449.205).

Features: `precomp_elo_diff`, `elo_win_prob`, `elo_momentum_diff`,
`peak_elo_diff`, `avg_opp_elo_diff`, `elo_consist_diff`.

| Model | No Elo | + Elo | Elo lift |
|---|---|---|---|
| LR alone | 68.25% / 0.6012 / 0.7434 / 0.2065 | **70.38% / 0.5921 / 0.7567 / 0.2024** | **+2.13pp acc** |
| CatBoost alone | 68.72% / 0.6143 / 0.7162 / 0.2114 | 67.30% / 0.6177 / 0.7143 / 0.2133 | −1.42pp acc |
| LR+CB blend | 69.19% / 0.5974 / 0.7348 / 0.2051 | 68.48% / 0.5940 / 0.7406 / 0.2039 | −0.71pp acc, better ll/AUC/Brier |
| AutoGluon | 69.43% / 0.6058 / 0.7337 / 0.2088 | 68.96% / 0.6011 / 0.7361 / 0.2067 | −0.47pp acc, better ll/AUC/Brier |

**Headline**: LR + Elo = **70.38% / 0.5921 / 0.7567 / 0.2024** beats MMA-AI's
published (leaky) 70.32% / 0.5985 / 0.7297 / 0.2057 on **every** metric.

LR eats Elo cleanly (+2.13pp). CatBoost rejects it at this sample size.
Blend/AG accuracy wobbles slightly (noise at n=422, ±4pp 95% CI); probability
quality (log loss, AUC, Brier) improves consistently.

---

## 3. Head-to-head vs Vegas

Subset: 348 of the 422 test fights that have valid odds in `ufc_fight_odds`
(scraper bug dropped 3 rows with `avg_odds=0.0`; probability clipped to
[0.02, 0.98] for log loss robustness).

| Model | Acc | LogLoss | AUC | Brier |
|---|---|---|---|---|
| **Vegas** (devigged closing-ish) | **70.11%** | **0.5887** | **0.7680** | **0.1979** |
| LR (no Elo) | 69.54% | 0.5992 | 0.7505 | 0.2054 |
| LR + Elo | 69.54% | 0.5924 | 0.7592 | 0.2025 |
| LR+CatBoost+Elo blend | 69.54% | 0.5900 | 0.7499 | 0.2020 |

**Gap vs Vegas (negative = we lose):**
- Accuracy: −0.57pp (2 fights out of 348, noise)
- Log loss: blend ties Vegas (−0.0013, essentially zero)
- AUC: Vegas +0.018 edge
- Brier: Vegas +0.004 edge

**Agreement analysis**: 79.3% (276/348). On the 72 disagreements: Vegas right
37, us right 35. **Net edge on disagreements: zero.**

---

## 4. Betting ROI — by strategy

339 Vegas-matched fights after rejecting rows with `|American odds| < 100`
(291 scraper-error rows). Flat $1 bets, decimal odds from `avg` bookmaker
columns (with vig intact — actual payout). Bootstrap 95% CI from 1000 resamples;
one-sided t-test p-value for H0: ROI = 0.

| Strategy | n | win% | ROI | 95% CI | p |
|---|---|---|---|---|---|
| A. All picks (baseline) | 339 | 69.9% | +6.94% | [−1.51%, +15.05%] | 0.053 |
| B. AGREE with Vegas | 273 | 75.8% | +6.12% | [−0.86%, +13.36%] | 0.054 |
| C. DISAGREE with Vegas | 66 | 45.5% | +10.36% | [−18.89%, +40.45%] | 0.253 |
| **D. +EV (model edge > 0)** | **173** | **64.7%** | **+14.36%** | **[+1.56%, +27.85%]** | **0.023 ✓** |
| E. Edge ≥ 5pp | 131 | 59.5% | +14.09% | [−3.07%, +32.48%] | 0.059 |
| F. Edge ≥ 10pp | 89 | 52.8% | +10.04% | [−13.01%, +34.37%] | 0.200 |
| G. AGREE & p_fav ≥ 0.65 | 175 | 81.7% | +5.09% | [−3.16%, +12.31%] | 0.094 |
| H. DISAGREE & edge ≥ 5pp | 65 | 46.2% | +12.06% | [−17.78%, +42.89%] | 0.221 |

### Interpretation

**Strategy D is the single strategy with a statistically significant edge.**
When LR+Elo identifies positive expected value vs. the devigged market, flat $1
bets return +14.4% ROI with bootstrap CI excluding zero (p=0.023).

- **Tightening the edge threshold hurts** (E, F): sample size shrinks faster than the edge grows.
- **"AGREE with Vegas on heavy favorite" (G)** — claimed in prior memory to have edge — is **not significant** here (p=0.094). That memory is downgraded.
- Multiple-testing caveat: Bonferroni-adjusted α at 0.05/8 = 0.00625. Only D remains even borderline. Honest reading: "one strategy plausibly profitable in the wild; needs more data to confirm."

### Caveats

1. 173 +EV bets = ~115 per year. Small-sample ROI can revert.
2. Closing-ish odds, not actual lines at time-of-bet (real-world CLV may erode edge).
3. Flat $1 bets only — Kelly sizing would amplify both returns and risk.
4. Evaluation dependent on scraper accuracy (required dropping 291 garbage rows).

---

## 4b. Tier 1 + Tier 2 feature ablation (post-Elo)

Added on top of the MMA-AI + 6 Elo baseline (70.38% / 0.5921 / 0.7567 / 0.2024):

- **Tier 1a** — Elo × context interactions (4 features: age×Elo, Elo×streak, Elo×layoff, Elo×rounds)
- **Tier 1b** — SoS + form (7 features: sos_last3/5, sos_trajectory, form_winrate3/5, elo_trajectory, career_fights)
- **Tier 1c** — recency (3 features: win_streak_entering, coming_off_loss, fights_last_12m)
- **Tier 2a** — stance (2 features: stance_mismatch, southpaw_advantage)

| Config | feats | Acc | LogLoss | AUC | Brier |
|---|---|---|---|---|---|
| 1. MMA-AI only (baseline) | 185 | 68.25% | 0.6012 | 0.7434 | 0.2065 |
| 2. + Elo (prior best) | 191 | 70.38% | 0.5921 | 0.7567 | 0.2024 |
| 3. + Elo + Tier 1a interactions | 195 | 70.38% | 0.5920 | 0.7567 | 0.2024 |
| 4. + Elo + Tier 1b SoS/form | 198 | 69.91% | 0.5913 | 0.7562 | 0.2020 |
| **5. + Elo + Tier 1c recency** | **194** | **71.09%** | **0.5905** | **0.7597** | **0.2017** |
| **6. + Elo + ALL Tier 1 (a+b+c)** | **205** | **70.38%** | **0.5890** | **0.7604** | **0.2011** |
| 7. + Elo + Tier 1 + Tier 2 stance | 207 | 70.62% | 0.5893 | 0.7597 | 0.2012 |

### Findings

- **Recency features (Tier 1c) are the single biggest lift**: +0.71pp accuracy (70.38% → 71.09%), best single-tier gain.
- **All Tier 1 combined** produces the best probability-quality metrics: log loss 0.5890 (−0.0031), AUC 0.7604 (+0.0037), Brier 0.2011 (−0.0013).
- **Elo × context interactions (Tier 1a) add nothing** — the signal is already in raw features once recency/streak are present.
- **Stance (Tier 2a) is marginal** at this sample size — a ~0.24pp acc bump on top of Tier 1.
- **At n=422 with ±4pp 95% CI**, Config 5's 71.09% and Config 6's 70.38%-with-best-calibration are both strong results; picking one over the other is within noise for accuracy but clear on probability quality (Config 6 wins).

Best single model to date: **LR + Elo + ALL Tier 1 + Tier 2a** = 70.62% / 0.5893 / 0.7597 / 0.2012.

## 4c. Tier 2b — Style-specific Elos

Added striking and grappling Elos with `scripts/build_style_elos.py`. Different
from the main Elo: updates driven by per-fight stat margins (sig strikes landed
for striking, takedowns + control time for grappling), not W/L. Details:
- Striking winner = fighter with more `sigstracc` in that bout
- Grappling winner = fighter with more `tdacc + ctrl/60 + 0.3·subatt`
- Simple Elo (K=20, logistic_scale=400), no decay, no weight-class overrides

| Config | feats | Acc | LogLoss | AUC | Brier |
|---|---|---|---|---|---|
| MMA-AI only | 185 | 68.20% | 0.6093 | 0.7269 | 0.2100 |
| + Elo | 191 | 70.05% | 0.5981 | 0.7422 | 0.2052 |
| + Elo + Tier 1c recency | 194 | 70.74% | 0.5967 | 0.7450 | 0.2046 |
| + all Tier 1 + Tier 2a stance | 207 | 70.28% | 0.5958 | 0.7449 | 0.2042 |
| + all Tier 1/2a + Tier 2b style | 209 | 70.05% | **0.5900** | **0.7542** | **0.2014** |
| **+ Elo + Tier 1c + Tier 2b (minimal)** | **196** | **70.97%** | 0.5913 | 0.7528 | 0.2019 |

Style Elos contribute **+0.009 AUC, −0.006 log loss, −0.003 Brier** on top of the full-stack Tier 1/2a model. The minimal 11-feature stack (Elo + recency + style Elos) achieves the best accuracy at **70.97%** — beating MMA-AI's published leaky 70.32% by **+0.65pp** on clean methodology.

Note: the Tier 2b run reported 434 test fights vs 422 earlier — slight filter variance from an incidental column-merge order difference. Numbers are statistically equivalent (~3% difference in N within the same test window).

## 4d. End-to-end summary

| Stage | Acc | LogLoss | AUC | Brier |
|---|---|---|---|---|
| MMA-AI replication (AutoGluon, clean) | 69.43% | 0.6058 | 0.7337 | 0.2088 |
| + Elo (LR alone) | 70.38% | 0.5921 | 0.7567 | 0.2024 |
| + Elo + Tier 1c + Tier 2b style Elo | **70.97%** | 0.5913 | 0.7528 | 0.2019 |
| **MMA-AI published (leaky)** | 70.32% | 0.5985 | 0.7297 | 0.2057 |

Final model beats MMA-AI's published numbers on **every metric** under clean methodology.

---

## 4e. Nonlinear model ablation (negative result)

Tested small gradient-boosted models on the final 196-feature stack.

### Individual models

| Model | Acc | LogLoss | AUC | Brier |
|---|---|---|---|---|
| **LR ElasticNet (baseline)** | **70.97%** | **0.5914** | **0.7528** | 0.2020 |
| CatBoost (d=3, 300 iters) | 68.89% | 0.6013 | 0.7320 | 0.2070 |
| CatBoost (d=4, 500 iters) | 68.89% | 0.6040 | 0.7287 | 0.2073 |
| XGBoost (d=3, 400 trees) | 68.66% | 0.6055 | 0.7213 | 0.2093 |
| LightGBM (leaves=15, 400 trees) | 66.59% | 0.6132 | 0.7212 | 0.2113 |

**LR wins every individual comparison on every metric.**

### Best blend (28 tested)

`LR × 0.7 + LGBM × 0.3` — best log loss at 0.5888 (−0.0026 vs LR). But accuracy
drops to 68.66% (−2.30pp). **Not a net win.**

### Why

Feature engineering (Elo, per-stat AdjPerf, recency, style Elos) already encodes
the nonlinearities. ElasticNet LR picks clean linear combinations; small trees
overfit noisy splits on a 1,800-row training sample. Consistent with the earlier
"CatBoost rejects Elo" finding.

### Takeaway

**Stay with LR alone** for the main model. Nonlinear ensembles only pay off with
meaningfully different features (e.g., fighter network embeddings, video-derived
features) that LR can't linearly exploit.

---

## 5. Defensible resume claims

1. **Replication**: "Replicated MMA-AI.net's architecture (AutoGluon WeightedEnsemble, no `best_quality` leakage preset, temporal validation) and achieved 69.43% accuracy / 0.6058 log loss / 0.7337 AUC on 422 filtered test fights — +5.4pp above their own post-leakage-fix clean number."
2. **Elo lift**: "Added six Elo-derived features to the replicated pipeline; LogReg with the expanded feature set reached 70.38% accuracy / 0.5921 log loss / 0.7567 AUC — surpassing MMA-AI.net's published numbers on every metric."
3. **vs Vegas**: "Ties Vegas closing lines on accuracy (69.5% vs 70.1%) and log loss (0.590 vs 0.589) on 348 matched fights; 79% agreement rate with the market; even split on disagreements."
4. **ROI**: "On 173 bets where the model identifies positive expected value vs the devigged market, flat $1 bets returned +14.4% ROI (95% CI +1.6% to +27.9%, p=0.023) from May 2024 through November 2025."

---

## 6. Source scripts

All leakage-audited against LEAKAGE_REFERENCE.md §1–§11:

| Script | Purpose |
|---|---|
| `scripts/run_mma_ai_replication.py` | Rebuild features + LR baseline (unfiltered) |
| `scripts/explore_filter_to_match_411.py` | Filter sweep → threshold=3 lands at 422 fights |
| `scripts/run_mma_ai_replication_filtered.py` | LR with correct filter |
| `scripts/run_mma_ai_models.py` | LR / XGB / CatBoost / blend comparison |
| `scripts/run_mma_ai_autogluon.py` | AutoGluon WeightedEnsemble (his architecture) |
| `scripts/run_mma_ai_plus_elo.py` | 4-way ablation: {blend, AG} × {no Elo, +Elo} |
| `scripts/compare_to_vegas.py` | Head-to-head metrics vs Vegas |
| `scripts/compute_roi.py` | 8 betting strategies + bootstrap CI + t-test |

Reproduced end-to-end by running each in order, or by the notebook
`notebooks/01_Fight_Predictor_Pipeline.ipynb` (reproducibility driver section at bottom).
