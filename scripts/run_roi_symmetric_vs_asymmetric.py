"""ROI comparison: symmetric LR vs asymmetric LR on the same test set.

Trains both models single-shot (no walk-forward) on the same data, the
symmetric one on doubled rows. Then:
  - Attach Vegas odds to the test fights
  - Compute edge = p_model - p_vegas (on the picked side)
  - For each betting strategy, compute ROI, bootstrap CI, t-test p-value

This isolates the impact of training-set symmetrization on betting edge.
"""
import sys, pickle, json, sqlite3, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from run_threshold_sweep_both_elos import (
    load_base_both_elos, apply_threshold, attach_vegas,
    TEST_FIRST, TEST_LAST, LAM,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats
from retrain_lr_symmetric import flip_row_dataframe

RNG = np.random.default_rng(42)
FILTER_THRESHOLD = 3


def train_lr(train_df, feats):
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(train_df[feats])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    y = train_df["win"].astype(int).values
    w = np.exp(-LAM * (TEST_FIRST - train_df["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs, y, sample_weight=w)
    return lr, imp, sc


def predict(test_df, feats, lr, imp, sc):
    X = imp.transform(test_df[feats])
    X = sc.transform(X)
    return lr.predict_proba(X)[:, 1]


def compute_bets(test_df, p_model):
    """Returns a DataFrame of all +EV bets with (p_pick, dec_pick, edge, won)."""
    test = test_df.copy().reset_index(drop=True)
    test["p_model"] = p_model
    # Attach Vegas
    tv = attach_vegas(test[["DATE", "jbout", "jfighter"]].drop_duplicates())
    m = test.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                        "dec_odds_f1", "dec_odds_f2"]],
                   on=["DATE", "jbout", "jfighter"], how="left")
    m = m.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    m = m[m["p_vegas_f1"].notna()].copy()
    pf1 = m["p_model"].values >= 0.5
    y_f1 = m["win"].astype(int).values
    m["p_pick"] = np.where(pf1, m["p_model"], 1 - m["p_model"])
    m["dec_pick"] = np.where(pf1, m["dec_odds_f1"], m["dec_odds_f2"])
    m["p_vegas_pick"] = np.where(pf1, m["p_vegas_f1"], 1 - m["p_vegas_f1"])
    m["edge"] = m["p_pick"] - m["p_vegas_pick"]
    m["won"] = np.where(pf1, y_f1, 1 - y_f1)
    return m


def compute_roi(bets, edge_lo, edge_hi=1.0):
    """Compute ROI, win rate, and stats for all flat $1 bets with
    edge in [edge_lo, edge_hi). Returns dict with summary + p-value."""
    sub = bets[(bets["edge"] >= edge_lo) & (bets["edge"] < edge_hi)].copy()
    if len(sub) == 0:
        return dict(n=0)
    # Flat $1 stake, win pays (dec-1), loss pays -1
    profits = np.where(sub["won"] == 1,
                       (sub["dec_pick"] - 1.0).values,
                       -1.0)
    n = len(sub)
    total_staked = float(n)  # $1 each
    total_profit = float(profits.sum())
    roi = total_profit / total_staked
    # One-sided t-test H0: mean profit per bet <= 0
    if n > 1:
        t_stat, p_two = stats.ttest_1samp(profits, 0)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:
        p_one = 1.0
    # Bootstrap 95% CI on ROI
    boot_roi = []
    for _ in range(500):
        idx = RNG.choice(n, n, replace=True)
        boot_roi.append(profits[idx].mean())
    lo, hi = np.percentile(boot_roi, [2.5, 97.5])
    return dict(n=n, win_rate=sub["won"].mean(),
                avg_edge_pp=sub["edge"].mean() * 100,
                avg_dec_odds=sub["dec_pick"].mean(),
                roi=roi, ci_lo=lo, ci_hi=hi, p_value=p_one)


def report(name, bets):
    print(f"\n{'='*76}")
    print(f"  {name}")
    print(f"{'='*76}")
    strategies = [
        ("Strategy D (all +EV, edge>0)",      0.0,   1.0),
        ("Low-edge (0-2.5pp)",                0.0,   0.025),
        ("Low-mid edge (2.5-5pp)",            0.025, 0.05),
        ("Mid-edge GOLDMINE (5-10pp)",        0.05,  0.10),
        ("Big-edge NOISE (>=10pp)",           0.10,  1.0),
    ]
    print(f"{'strategy':<36s} {'n':>4s} {'edge_pp':>8s} {'win%':>6s} "
          f"{'odds':>6s} {'ROI':>8s} {'95% CI':>18s} {'p':>8s}")
    print("-" * 110)
    for label, lo, hi in strategies:
        r = compute_roi(bets, lo, hi)
        if r["n"] == 0:
            print(f"  {label:<34s} {'-':>4s} {'-':>8s} {'-':>6s} {'-':>6s} {'-':>8s} {'-':>18s} {'-':>8s}")
            continue
        ci = f"[{r['ci_lo']*100:+.1f}%, {r['ci_hi']*100:+.1f}%]"
        print(f"  {label:<34s} {r['n']:>4d} {r['avg_edge_pp']:>7.2f}pp "
              f"{r['win_rate']*100:>5.1f}% {r['avg_dec_odds']:>5.2f} "
              f"{r['roi']*100:>+7.2f}% {ci:>18s} {r['p_value']:>7.3f}")


def main():
    print("="*76)
    print("ROI comparison: ASYMMETRIC vs SYMMETRIC LR (single-shot training)")
    print("="*76)

    base = load_base_both_elos()
    df = apply_threshold(base, FILTER_THRESHOLD)
    train = df[df["DATE"] < TEST_FIRST].copy()
    test  = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()

    # Feature selection (matches production)
    feats = [c for c in df.columns if (c.endswith("_diff") or c.endswith("_ufc")
             or c.endswith("_exp") or c in ("weightclass_encoded", "scheduled_rounds",
                                             "days_since_last_fight_f1"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    feats = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    print(f"Features: {len(feats)}   train: {len(train):,}   test: {len(test):,}")

    # ── Train asymmetric (old methodology) ──────────────────────────────
    print("\nTraining ASYMMETRIC LR (single orientation, pre-fix)...")
    lr_a, imp_a, sc_a = train_lr(train, feats)
    p_a = predict(test, feats, lr_a, imp_a, sc_a)
    bets_a = compute_bets(test, p_a)
    print(f"  test rows with vegas: {len(bets_a)}")
    report("ASYMMETRIC LR (old, single-orient training)", bets_a)

    # ── Train symmetric (new methodology) ───────────────────────────────
    print("\nTraining SYMMETRIC LR (doubled training data)...")
    train_flipped = flip_row_dataframe(train)
    train_doubled = pd.concat([train, train_flipped], ignore_index=True)
    lr_s, imp_s, sc_s = train_lr(train_doubled, feats)
    p_s = predict(test, feats, lr_s, imp_s, sc_s)
    bets_s = compute_bets(test, p_s)
    print(f"  test rows with vegas: {len(bets_s)}")
    report("SYMMETRIC LR (new, doubled-orient training)", bets_s)

    # ── Head-to-head summary ────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("HEAD-TO-HEAD SUMMARY (all +EV, edge>0)")
    print(f"{'='*76}")
    for name, bets in [("ASYMMETRIC", bets_a), ("SYMMETRIC", bets_s)]:
        r = compute_roi(bets, 0.0)
        if r["n"]:
            print(f"  {name:<12s}: n={r['n']:3d}  ROI={r['roi']*100:+6.2f}%  "
                  f"CI=[{r['ci_lo']*100:+.1f}%, {r['ci_hi']*100:+.1f}%]  p={r['p_value']:.3f}")

    print(f"\nMid-edge (5-10pp) ROI comparison:")
    for name, bets in [("ASYMMETRIC", bets_a), ("SYMMETRIC", bets_s)]:
        r = compute_roi(bets, 0.05, 0.10)
        if r["n"]:
            print(f"  {name:<12s}: n={r['n']:3d}  ROI={r['roi']*100:+6.2f}%  "
                  f"CI=[{r['ci_lo']*100:+.1f}%, {r['ci_hi']*100:+.1f}%]  p={r['p_value']:.3f}")


if __name__ == "__main__":
    main()
