"""Build per-fold walk-forward model-vs-Vegas comparison JSON.

Runs the 4 folds (matching walk_forward_4fold.py), matches each test fight
with Vegas odds, and computes:

  Per-fold (and pooled):
    n_total, n_matched, vig_mean
    model:  accuracy, log_loss, brier, ROI (flat bets on every pick)
    vegas:  accuracy, log_loss, brier
    edge-strategy ROI: bets only when model probability exceeds devigged
                      Vegas probability on the picked side

Saves: results/walkforward_vegas_comparison.json

The /api/model/vegas_comparison endpoint reads this when available,
otherwise falls back to the legacy blend backtest data.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from run_threshold_sweep_both_elos import (
    load_base_both_elos, apply_threshold, attach_vegas, LAM,
)
from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features, flip_row_dataframe
from walk_forward_4fold import FOLDS, select_features, leakage_assertions

EPS = 1e-6


def run_fold_with_vegas(df, fold, feats):
    train_start = pd.Timestamp(fold["train_start"])
    train_end   = pd.Timestamp(fold["train_end"])
    test_start  = pd.Timestamp(fold["test_start"])
    test_end    = pd.Timestamp(fold["test_end"])

    train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
    test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
    leakage_assertions(train, test, fold)

    # Symmetric train
    train_flipped = flip_row_dataframe(train)
    train_doubled = pd.concat([train, train_flipped], ignore_index=True)
    usable = [c for c in feats if c in train_doubled.columns and train_doubled[c].std() > 1e-8]

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_doubled[usable]))
    ytr = train_doubled["win"].astype(int).values
    w = np.exp(-LAM * (train_end - train_doubled["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    # Predict on full test (matched + unmatched)
    Xte = sc.transform(imp.transform(test[usable]))
    p_model = lr.predict_proba(Xte)[:, 1]
    test = test.copy()
    test["p_model"] = p_model

    # Attach Vegas odds. `attach_vegas` merges on jbout + realigns to the
    # row's jfighter (so p_vegas_f1 is always "from f1's perspective").
    tv = attach_vegas(test[["DATE", "jbout", "jfighter"]].drop_duplicates())
    merged = test.merge(
        tv[["DATE", "jbout", "jfighter", "p_vegas_f1", "dec_odds_f1", "dec_odds_f2"]],
        on=["DATE", "jbout", "jfighter"], how="left",
    )
    merged = merged.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return merged


def fold_metrics(m_matched):
    """Compute all comparison metrics on a set of matched (vegas + model) rows."""
    if len(m_matched) == 0:
        return {"n": 0}
    y = m_matched["win"].astype(int).values
    pm = np.clip(m_matched["p_model"].values, EPS, 1 - EPS)
    pv = np.clip(m_matched["p_vegas_f1"].values, EPS, 1 - EPS)
    dec_a = m_matched["dec_odds_f1"].values
    dec_b = m_matched["dec_odds_f2"].values

    # --- Accuracy / LL / Brier (model vs Vegas on same fights)
    acc_m = float(accuracy_score(y, (pm >= 0.5).astype(int)))
    acc_v = float(accuracy_score(y, (pv >= 0.5).astype(int)))
    ll_m = float(log_loss(y, pm))
    ll_v = float(log_loss(y, pv))
    br_m = float(brier_score_loss(y, pm))
    br_v = float(brier_score_loss(y, pv))

    # --- Vig (market overround)
    # Vegas probs are already devigged in attach_vegas; we re-compute
    # overround from raw decimal odds for display.
    implied_a = 1.0 / dec_a; implied_b = 1.0 / dec_b
    vig = float(np.mean(implied_a + implied_b - 1.0))

    # --- Flat-bet ROI: bet $1 on model's pick at Vegas odds
    pick_a = pm >= 0.5
    dec_pick = np.where(pick_a, dec_a, dec_b)
    won_pick = np.where(pick_a, y == 1, y == 0)
    pnl_flat = np.where(won_pick, dec_pick - 1.0, -1.0)
    roi_flat = float(pnl_flat.mean() * 100)

    # --- +EV strategy: bet only when model edge > 0 on picked side
    vpick = np.where(pick_a, pv, 1 - pv)
    mpick = np.where(pick_a, pm, 1 - pm)
    edge = mpick - vpick
    # Only bets where EV > 0 at the actual vigged odds
    ev = mpick * dec_pick - 1.0
    pos_ev_mask = ev > 0
    n_ev = int(pos_ev_mask.sum())
    roi_ev = float(pnl_flat[pos_ev_mask].mean() * 100) if n_ev > 0 else None
    ev_win_rate = float(won_pick[pos_ev_mask].mean() * 100) if n_ev > 0 else None

    # --- Mid-edge (5-10pp) slice per finding_ev_slice_analysis.md
    mid_edge_mask = (edge >= 0.05) & (edge < 0.10)
    n_mid = int(mid_edge_mask.sum())
    roi_mid = float(pnl_flat[mid_edge_mask].mean() * 100) if n_mid > 0 else None

    # --- Vegas-at-Vegas-odds baseline (unprofitable by construction)
    # Vegas pick flat bet: whoever Vegas thinks wins, bet on at Vegas odds
    vegas_pick_a = pv >= 0.5
    dec_vegas_pick = np.where(vegas_pick_a, dec_a, dec_b)
    won_vegas_pick = np.where(vegas_pick_a, y == 1, y == 0)
    pnl_vegas_flat = np.where(won_vegas_pick, dec_vegas_pick - 1.0, -1.0)
    roi_vegas_flat = float(pnl_vegas_flat.mean() * 100)

    return {
        "n": int(len(m_matched)),
        "vig_mean_pct": round(vig * 100, 3),
        # Model metrics
        "acc_model":  round(acc_m, 4),
        "ll_model":   round(ll_m, 4),
        "brier_model": round(br_m, 4),
        # Vegas metrics
        "acc_vegas":  round(acc_v, 4),
        "ll_vegas":   round(ll_v, 4),
        "brier_vegas": round(br_v, 4),
        # Deltas (positive = model better)
        "acc_delta":  round(acc_m - acc_v, 4),
        "ll_delta":   round(ll_v - ll_m, 4),     # lower is better for LL, so flip
        "brier_delta": round(br_v - br_m, 4),    # same
        # ROI strategies
        "roi_flat_model":       round(roi_flat, 3),        # flat on every model pick
        "roi_flat_vegas_pick":  round(roi_vegas_flat, 3),  # flat on every Vegas pick
        "roi_pos_ev":           round(roi_ev, 3) if roi_ev is not None else None,
        "n_pos_ev":             n_ev,
        "ev_win_rate":          round(ev_win_rate, 2) if ev_win_rate is not None else None,
        "roi_mid_edge":         round(roi_mid, 3) if roi_mid is not None else None,
        "n_mid_edge":           n_mid,
    }


def main():
    print("=" * 76)
    print("Walk-forward 4-fold model-vs-Vegas comparison")
    print("=" * 76)

    print("Loading base features...")
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)

    per_fold_raw = {}
    all_matched = []

    for fold in FOLDS:
        print(f"\n── {fold['name']}  test {fold['test_start']} → {fold['test_end']}")
        merged = run_fold_with_vegas(df, fold, feats)
        matched = merged[merged["p_vegas_f1"].notna()].copy()
        n_total = len(merged)
        n_matched = len(matched)
        pct = 100 * n_matched / n_total if n_total > 0 else 0
        print(f"  matched with Vegas: {n_matched}/{n_total} ({pct:.1f}%)")

        m = fold_metrics(matched)
        m["n_total"] = n_total
        m["n_matched"] = n_matched
        m["pct_matched"] = round(pct, 1)
        m["test_start"] = fold["test_start"]
        m["test_end"] = fold["test_end"]
        per_fold_raw[fold["name"]] = m

        # For pooled metrics, collect matched rows with fold tag
        matched = matched.copy()
        matched["fold"] = fold["name"]
        all_matched.append(matched)

        print(f"  acc model={m['acc_model']*100:.2f}%  vegas={m['acc_vegas']*100:.2f}%  "
              f"(Δ={m['acc_delta']*100:+.2f}pp)")
        print(f"  ll  model={m['ll_model']:.4f}  vegas={m['ll_vegas']:.4f}  "
              f"(Δ={m['ll_delta']:+.4f})")
        print(f"  brier model={m['brier_model']:.4f}  vegas={m['brier_vegas']:.4f}  "
              f"(Δ={m['brier_delta']:+.4f})")
        print(f"  ROI flat-model={m['roi_flat_model']:+.2f}%   flat-vegas={m['roi_flat_vegas_pick']:+.2f}%   "
              f"+EV={m['roi_pos_ev']}% (n={m['n_pos_ev']})")

    # Pooled metrics
    pooled_df = pd.concat(all_matched, ignore_index=True)
    pooled = fold_metrics(pooled_df)
    pooled["n_total"] = sum(per_fold_raw[f]["n_total"] for f in per_fold_raw)
    pooled["n_matched"] = len(pooled_df)
    pooled["pct_matched"] = round(100 * pooled["n_matched"] / pooled["n_total"], 1)

    print("\n" + "=" * 76)
    print("POOLED across 4 folds")
    print("=" * 76)
    print(f"n matched: {pooled['n_matched']}/{pooled['n_total']} ({pooled['pct_matched']}%)")
    print(f"acc model={pooled['acc_model']*100:.2f}%  vegas={pooled['acc_vegas']*100:.2f}%  "
          f"(Δ={pooled['acc_delta']*100:+.2f}pp)")
    print(f"ll  model={pooled['ll_model']:.4f}  vegas={pooled['ll_vegas']:.4f}  "
          f"(Δ={pooled['ll_delta']:+.4f})")
    print(f"brier model={pooled['brier_model']:.4f}  vegas={pooled['brier_vegas']:.4f}  "
          f"(Δ={pooled['brier_delta']:+.4f})")
    print(f"ROI flat-model={pooled['roi_flat_model']:+.2f}%   flat-vegas={pooled['roi_flat_vegas_pick']:+.2f}%")
    print(f"ROI +EV strategy: {pooled['roi_pos_ev']}% on {pooled['n_pos_ev']} bets  "
          f"(win rate: {pooled['ev_win_rate']}%)")
    if pooled["roi_mid_edge"] is not None:
        print(f"ROI mid-edge (5-10pp) slice: {pooled['roi_mid_edge']:+.2f}% on {pooled['n_mid_edge']} bets")

    out = Path("results/walkforward_vegas_comparison.json")
    out.write_text(json.dumps({
        "methodology": "4-fold walk-forward, 7yr train / 6mo test, symmetric LR, "
                       "predictions matched with devig'd multi-book Vegas odds. "
                       "Edge = model_prob - devig_vegas_prob on picked side. "
                       "+EV = bets where model_prob × dec_odds > 1.0 (true positive EV).",
        "folds": [{"name": f["name"], **per_fold_raw[f["name"]]} for f in FOLDS],
        "pooled": pooled,
        "test_window": f"{FOLDS[0]['test_start']} → {FOLDS[-1]['test_end']}",
    }, indent=2, default=str))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
