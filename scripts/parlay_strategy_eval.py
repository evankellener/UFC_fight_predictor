"""Parlay strategy evaluator on the production ensemble walk-forward.

Re-runs the λ-ensemble walk-forward AND saves per-fight predictions, then tests
multiple parlay strategies:

  S1 — Straight-bet baseline (current production: bet $1 on every +EV pick)
  S2 — 2-leg parlay of ALL +EV picks on the same event (round-robin)
  S3 — 2-leg parlay of the TOP-2 +EV picks on the same event by edge
  S4 — 2-leg parlay of TOP-2 by model_prob (highest-confidence picks)
  S5 — 3-leg parlay of TOP-3 +EV picks on the same event
  S6 — 2-leg parlay only when BOTH legs have edge ≥ 5pp
  S7 — 2-leg parlay only when BOTH legs have model_prob ≥ 0.65 (favorites only)

Each parlay leg payout = product of decimal odds. We bet $1 per parlay.

Critical methodology notes:
  - Parlay EV for independent legs: prod(model_prob_i × dec_odds_i) − 1.
    If each leg is +EV, the parlay is +EV with MULTIPLIED edge.
  - We treat fights on the same card as independent — slight correlation possible
    (judges/venue) but no strong reason to expect systematic correlation.
  - Walk-forward folds — same leakage guards as ensemble script.
  - Vegas odds attached AFTER predictions made (no leak).
"""
import sys, json, time, warnings
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features, flip_row_dataframe
from walk_forward_4fold import FOLDS, select_features, leakage_assertions
from build_walkforward_vegas_multi_threshold import attach_vegas_rich

LAMBDA_A, LAMBDA_B = 0.13, 1.20
EPS = 1e-6


def temp_cal(p_train, y_train):
    p_train = np.clip(p_train, EPS, 1 - EPS)
    logit = np.log(p_train / (1 - p_train))
    def nll(T):
        if T <= 0: return 1e9
        pc = 1 / (1 + np.exp(-logit / T))
        pc = np.clip(pc, EPS, 1 - EPS)
        return -(y_train * np.log(pc) + (1 - y_train) * np.log(1 - pc)).mean()
    return float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)


def apply_temp(p, T):
    p = np.clip(p, EPS, 1 - EPS); logit = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-logit / T))


def fit_and_predict(train, test, feats, lam):
    train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    train_end = test["DATE"].min()
    w = np.exp(-lam * (train_end - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)
    X_tr_orig = sc.transform(imp.transform(train[usable]))
    p_tr = lr.predict_proba(X_tr_orig)[:, 1]
    T = temp_cal(p_tr, train["win"].astype(int).values)
    Xte = sc.transform(imp.transform(test[usable]))
    p_raw = lr.predict_proba(Xte)[:, 1]
    return apply_temp(p_raw, T)


def collect_predictions():
    """Run ensemble walk-forward, return per-fight DataFrame:
    DATE, jevent, jbout, jfighter, p_model, p_vegas, dec_odds_pick, dec_odds_other,
    pick_a (1 if model picked f1), won_pick (1 if pick won), edge."""
    print("Loading...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)
    print(f"  {len(df)} rows @ t=3")

    all_test = []
    for fold in FOLDS:
        train_start = pd.Timestamp(fold["train_start"]); train_end = pd.Timestamp(fold["train_end"])
        test_start  = pd.Timestamp(fold["test_start"]);  test_end  = pd.Timestamp(fold["test_end"])
        train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
        test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
        leakage_assertions(train, test, fold)
        p_a = fit_and_predict(train, test, feats, LAMBDA_A)
        p_b = fit_and_predict(train, test, feats, LAMBDA_B)
        p_ens = (p_a + p_b) / 2
        test = test.copy(); test["p_model"] = p_ens; test["fold"] = fold["name"]
        all_test.append(test)
        print(f"  {fold['name']}: {len(test)} predictions")

    pred = pd.concat(all_test, ignore_index=True)
    keys = pred[["DATE", "jbout", "jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    merged = pred.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                            "dec_odds_f1", "dec_odds_f2"]],
                        on=["DATE", "jbout", "jfighter"], how="left")
    merged = merged.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    # Note: each bout appears TWICE (once with each fighter as f1). We dedupe to
    # ONE row per bout for parlay analysis to avoid double-counting.
    # Pick the row where the model's prob is the higher one (model's pick).
    # Actually simpler: only keep rows where jfighter == winner-or-loser canonical,
    # but easier: keep the row where the model's pick aligns with f1 (p_model>=0.5).
    # Then we always bet on f1's odds dec_odds_f1.
    matched = merged[merged["p_vegas_f1"].notna()].copy()
    print(f"  Matched against Vegas: {len(matched)}")

    # Pick side: model's chosen fighter (p>=0.5 → bet on jfighter, else opponent)
    matched["pick_a"] = (matched["p_model"] >= 0.5).astype(int)
    matched["dec_odds_pick"] = np.where(matched["pick_a"] == 1,
                                         matched["dec_odds_f1"],
                                         matched["dec_odds_f2"])
    matched["p_pick"] = np.where(matched["pick_a"] == 1,
                                  matched["p_model"], 1 - matched["p_model"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"] == 1,
                                        matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"] == 1, y == 1, y == 0).astype(int)
    matched["edge"] = matched["p_pick"] - matched["p_vegas_pick"]
    matched["ev"] = matched["p_pick"] * matched["dec_odds_pick"] - 1.0

    # Dedupe: each bout has 2 rows (one per corner). The model_prob from the
    # other corner = 1 - model_prob. The "pick" side is the same regardless of
    # which corner row we look at. Keep just one row per bout.
    matched = matched.drop_duplicates(subset=["DATE", "jbout"]).reset_index(drop=True)
    print(f"  Deduplicated to one row per bout: {len(matched)}")
    return matched


def straight_bet_roi(df, mask=None):
    if mask is None: mask = df["ev"] > 0
    bets = df[mask]
    if len(bets) == 0: return dict(n=0, roi=0.0, hits=0)
    pnl = np.where(bets["won_pick"] == 1, bets["dec_odds_pick"] - 1.0, -1.0)
    return dict(n=int(len(bets)), roi=float(pnl.mean() * 100),
                hits=int(bets["won_pick"].sum()),
                hit_rate=float(bets["won_pick"].mean() * 100))


def parlay_2leg(df, picker="all_pos_ev", edge_min=None, p_min=None, top_k=None):
    """
    For each event (DATE), select +EV legs by `picker`, then form ALL 2-leg
    parlay combinations from the selected legs. Bet $1 per parlay.

    picker:
      'all_pos_ev' — every +EV pick on the card
      'top_k_edge' — top-k by edge (descending) among +EV picks
      'top_k_prob' — top-k by p_pick (descending) among +EV picks
    edge_min / p_min — additional filters
    top_k — for top-k pickers

    Returns parlay summary.
    """
    pos = df[df["ev"] > 0].copy()
    if edge_min is not None: pos = pos[pos["edge"] >= edge_min]
    if p_min is not None:   pos = pos[pos["p_pick"] >= p_min]

    legs_total, parlays_total, hits, pnl_sum = 0, 0, 0, 0.0
    for date, grp in pos.groupby("DATE"):
        g = grp.copy()
        if picker == "top_k_edge" and top_k:
            g = g.sort_values("edge", ascending=False).head(top_k)
        elif picker == "top_k_prob" and top_k:
            g = g.sort_values("p_pick", ascending=False).head(top_k)
        if len(g) < 2: continue
        for a, b in combinations(g.itertuples(index=False), 2):
            payout = a.dec_odds_pick * b.dec_odds_pick
            won = a.won_pick * b.won_pick  # both must win
            pnl = payout - 1.0 if won else -1.0
            parlays_total += 1
            legs_total += 2
            hits += int(won)
            pnl_sum += pnl

    if parlays_total == 0:
        return dict(n_parlays=0, roi=0.0, hits=0, hit_rate=0.0)
    return dict(n_parlays=parlays_total, roi=float(pnl_sum / parlays_total * 100),
                hits=int(hits), hit_rate=float(hits / parlays_total * 100),
                n_legs=int(legs_total))


def parlay_nleg(df, n_legs, edge_min=None, p_min=None, picker="top_k_edge"):
    """N-leg parlay forming combinations of size n_legs per card."""
    pos = df[df["ev"] > 0].copy()
    if edge_min is not None: pos = pos[pos["edge"] >= edge_min]
    if p_min is not None:    pos = pos[pos["p_pick"] >= p_min]

    parlays_total, hits, pnl_sum = 0, 0, 0.0
    for date, grp in pos.groupby("DATE"):
        g = grp.copy()
        if picker == "top_k_edge":
            g = g.sort_values("edge", ascending=False).head(n_legs)
        elif picker == "top_k_prob":
            g = g.sort_values("p_pick", ascending=False).head(n_legs)
        if len(g) < n_legs: continue
        for combo in combinations(g.itertuples(index=False), n_legs):
            payout = float(np.prod([c.dec_odds_pick for c in combo]))
            won = int(np.prod([c.won_pick for c in combo]))
            pnl = payout - 1.0 if won else -1.0
            parlays_total += 1
            hits += won
            pnl_sum += pnl

    if parlays_total == 0:
        return dict(n_parlays=0, roi=0.0, hits=0, hit_rate=0.0)
    return dict(n_parlays=parlays_total, roi=float(pnl_sum / parlays_total * 100),
                hits=int(hits), hit_rate=float(hits / parlays_total * 100))


def main():
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    cache = out_dir / "parlay_predictions.parquet"
    if cache.exists():
        print(f"Loading cached predictions from {cache}")
        matched = pd.read_parquet(cache)
    else:
        matched = collect_predictions()
        matched.to_parquet(cache)
        print(f"Saved predictions to {cache}")

    print()
    print(f"Total +EV picks (straight): {(matched['ev'] > 0).sum()}")
    print(f"Total events with ≥2 +EV picks: "
          f"{(matched[matched['ev']>0].groupby('DATE').size() >= 2).sum()}")
    print(f"Total events with ≥3 +EV picks: "
          f"{(matched[matched['ev']>0].groupby('DATE').size() >= 3).sum()}")

    print("\n" + "=" * 80)
    print("STRAIGHT-BET BASELINES (one bet per pick, $1 each)")
    print("=" * 80)
    sb_all_pos = straight_bet_roi(matched)
    print(f"  ALL +EV: n={sb_all_pos['n']:>4d}  ROI={sb_all_pos['roi']:>+6.2f}%  "
          f"hit_rate={sb_all_pos['hit_rate']:.1f}%")
    sb_high = straight_bet_roi(matched, mask=(matched["ev"] > 0) & (matched["edge"] >= 0.05))
    print(f"  +EV ∩ edge≥5pp: n={sb_high['n']:>4d}  ROI={sb_high['roi']:>+6.2f}%  "
          f"hit_rate={sb_high['hit_rate']:.1f}%")
    sb_fav = straight_bet_roi(matched, mask=(matched["ev"] > 0) & (matched["p_pick"] >= 0.65))
    print(f"  +EV ∩ p≥0.65 (favorites): n={sb_fav['n']:>4d}  ROI={sb_fav['roi']:>+6.2f}%  "
          f"hit_rate={sb_fav['hit_rate']:.1f}%")

    print("\n" + "=" * 80)
    print("PARLAY STRATEGIES (2-leg, $1 per parlay)")
    print("=" * 80)
    cases = [
        ("S2: ALL +EV pairs on same card",     dict(picker="all_pos_ev")),
        ("S3: TOP-2 by edge (per card)",       dict(picker="top_k_edge", top_k=2)),
        ("S4: TOP-2 by p_pick (per card)",     dict(picker="top_k_prob", top_k=2)),
        ("S6: edge ≥ 5pp",                     dict(picker="all_pos_ev", edge_min=0.05)),
        ("S6b: edge ≥ 5pp, top-2 by edge",     dict(picker="top_k_edge", top_k=2, edge_min=0.05)),
        ("S7: p_pick ≥ 0.65 (favorites)",      dict(picker="all_pos_ev", p_min=0.65)),
        ("S7b: p_pick ≥ 0.65, top-2 by prob",  dict(picker="top_k_prob", top_k=2, p_min=0.65)),
    ]
    parlay_rows = []
    for label, kwargs in cases:
        r = parlay_2leg(matched, **kwargs)
        parlay_rows.append((label, r))
        print(f"  {label:<45s}  n={r['n_parlays']:>4d}  ROI={r['roi']:>+7.2f}%  "
              f"hit_rate={r['hit_rate']:.1f}%")

    print("\n" + "=" * 80)
    print("PARLAY STRATEGIES (3-leg, $1 per parlay)")
    print("=" * 80)
    for n_legs in (3, 4):
        cases3 = [
            (f"S5: top-{n_legs} by edge (+EV)",      dict(n_legs=n_legs, picker="top_k_edge")),
            (f"S5b: top-{n_legs} by p_pick (+EV)",   dict(n_legs=n_legs, picker="top_k_prob")),
            (f"S5c: top-{n_legs} by edge, edge≥5pp", dict(n_legs=n_legs, picker="top_k_edge", edge_min=0.05)),
        ]
        for label, kwargs in cases3:
            r = parlay_nleg(matched, **kwargs)
            print(f"  {label:<45s}  n={r['n_parlays']:>4d}  ROI={r['roi']:>+7.2f}%  "
                  f"hit_rate={r['hit_rate']:.1f}%")

    out_path = out_dir / "parlay_strategy_eval.json"
    out_path.write_text(json.dumps([{"label": l, **r} for l, r in parlay_rows], indent=2))
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
