# Project Status — UFC Fight Predictor
Last updated: 2026-04-29

---

## What this project does

Predicts UFC fight outcomes and identifies +EV bets by comparing model win
probabilities against Vegas closing odds. The validated edge: **+19% ROI on
men's non-BW fights where the model shows positive expected value.**

---

## The validated result (the only number that matters)

| Metric | Value |
|---|---|
| Strategy | Men's divisions, no Bantamweight, +EV bets only |
| Pooled ROI | **+19.03%** (97 bets over 18 months) |
| p-value (Vegas-null permutation test) | **0.007** |
| Verdict | **✅ STRONG PASS** (3/3 folds positive) |
| Actual win rate on bets | 62.9% vs 51.1% Vegas implied |

This was produced by `scripts/master_validation.py`. Read that file — the
CONFIG block at the top is the pre-registered specification. Don't adjust
parameters after running.

---

## Model configuration

```python
C          = 0.05        # ElasticNet regularization
L1_RATIO   = 0.5         # ElasticNet mixing
LAM        = 1.20        # Recency decay (training sample weights)
THRESHOLD  = 3           # Min prior UFC fights to include a fighter
TRAIN_START= "2016-01-01"
BET_MENS   = True        # Skip women's divisions
BET_NO_BW  = True        # Skip Bantamweight (too inconsistent)
NOVEL_FEATS= ["home_advantage_diff", "card_position_norm_career_diff"]
```

Training data: 2016-01 → 2024-10 (1,907 fights). Evaluated on 2024-10 →
2026-04 via 3-fold 6-month expanding walk-forward.

**Do not compare to pre-2026-04-27 results.** Earlier sessions had leakage
(+14%, +27% ROI figures). Those are wrong. +19% is the honest number.

---

## What still needs to be built

**A prediction script.** Given a list of upcoming fights and Vegas lines, output
which bets are +EV. This does not exist yet in working form.

Design: one script, ~150 lines, no external artifacts needed:
1. Build feature matrix from the pipeline (same as master_validation)
2. Train model on all data through the current date
3. For each matchup + Vegas line, compute EV = p_model × dec_odds − 1
4. Print bets where EV > 0

The `app/models/` directory is a mess — ignore it and build predictions
inline instead.

---

## Scripts that matter

| Script | Purpose | Status |
|---|---|---|
| `scripts/master_validation.py` | Gold-standard validation framework | ✅ Use this |
| `scripts/test_leakage_per_fold.py` | Empirical leakage check | ✅ Run before quoting metrics |
| `scripts/notebook_clean_walkforward.py` | Reference per-fold pipeline | ✅ Architecture template |
| `scripts/train_test_split_2016_2024.py` | Single-split baseline | ✅ Clean |

## Scripts to ignore (experiments, superseded)

| Script | Why ignore |
|---|---|
| `scripts/rolling_retrain_clean.py` | Quarterly retrain — MARGINAL result, worse than static |
| `scripts/rolling_retrain_6month.py` | 6-month retrain — also MARGINAL overall |
| `scripts/womens_division_model.py` | Tested: women's-only model worse (281 train rows) |
| `scripts/womens_hybrid_prior_model.py` | Tested: no effect |
| `scripts/mens_only_model.py` | Finding baked into master_validation config |
| `scripts/per_wc_betting_filter.py` | Exploratory — WC filter selected from test data |
| `scripts/retrain_deploy_lr.py` | Broken — creates 15-feature model |
| `scripts/build_deploy_model.py` | Incomplete — never tested end-to-end |

---

## Artifacts state

**`app/models/blend_v2/`** — DO NOT USE. Corrupted during 2026-04-29 session.
Multiple conflicting pkl files with mismatched feature counts (195, 202, 207).

**`app/models/deploy_v1/`** — DO NOT USE. Built in the same session, never
validated, predictions untested.

If you need the model for inference, train it inline from the pipeline.
Takes ~1 minute. See master_validation.py's `build_fold_features()` for the
exact recipe.

---

## Key findings (confirmed clean, post-leakage-fix)

- **Bantamweight** consistently negative ROI — excluded from betting
- **Women's +EV bets** slightly negative — excluded from betting
- **Data ceiling reached**: stance, win streak, age features were tested and
  don't improve ROI beyond the baseline. The model extracts near-maximum
  signal from public UFC fight statistics.
- **Rolling retrain** (6-month) helps Q1-2026 but hurts H2 overall. Static
  model trained through Oct 2024 outperforms rolling on the full 18-month test.
- **Bet sizing**: Half Kelly at 2.5% cap recommended (see finding_bet_sizing.md)
- **Mid-edge bets** (5-10pp model vs Vegas edge) historically best slice —
  not yet formally validated through master_validation

---

## Leakage rules (non-negotiable)

1. Never use `df[col].mean()` / `df.groupby(wc)[col].mean()` globally — leaks future data
2. Run `scripts/test_leakage_per_fold.py` before quoting any metric
3. Any new walk-forward script needs a matching audit in `docs/audits/`
4. WC priors must be recomputed inside the fold using only `train` data

See `LEAKAGE_REFERENCE.md` and `CLAUDE.md` for full rules.

---

## Suggested next session

Build `scripts/predict_card.py` — a single self-contained script that:
- Takes upcoming matchups as input (CSV or hardcoded list)
- Trains inline on all available data
- Accepts Vegas lines from the user
- Outputs: win probability, EV, BET/SKIP recommendation for each fight

Keep it under 200 lines. No external artifacts. No predictor_v2.
