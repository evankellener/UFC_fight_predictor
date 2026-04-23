"""Event-by-event metrics over the 6-month walk-forward test window.

Shows volatility and the impact of retraining. Each UFC event gets one point
per metric; rolling means smooth the series. Vertical dashed lines mark
retraining boundaries (fold transitions).

Writes: results/walk_forward_volatility.png

Reuses the production 6-month walk-forward logic from
`scripts/run_walk_forward_6month.py`. Same leakage guardrails.
"""
import json, sqlite3, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from elo_feature import compute_elo
from run_walk_forward_6month import (apply_filter, merge_all_layers, attach_vegas,
                                     train_lr, predict, metrics,
                                     TEST_FIRST, TEST_LAST, N_FOLDS, TRAIN_YEARS,
                                     TRAIN_ERA_FLOOR)

DT = Path("data/tmp")
OUT = Path("results/walk_forward_volatility.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("="*70)
    print("Building per-event metrics from 6-month walk-forward")
    print("="*70)
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    df = apply_filter(df)
    df = merge_all_layers(df)

    feats = [c for c in df.columns if (c.endswith("_diff") or c in
             ("weightclass_encoded", "scheduled_rounds", "days_since_last_fight_f1"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]

    # Build folds (same as 6-month script)
    span = (TEST_LAST - TEST_FIRST).days
    folds = []
    for i in range(N_FOLDS):
        fs = TEST_FIRST + pd.Timedelta(days=int(round(i * span / N_FOLDS)))
        fe = TEST_FIRST + pd.Timedelta(days=int(round((i+1) * span / N_FOLDS))) \
             if i < N_FOLDS-1 else TEST_LAST
        folds.append((fs, fe))
    retrain_dates = [fs for fs, _ in folds[1:]]  # retraining happens at fold 2, 3 starts
    print(f"Retraining boundaries: {[str(d.date()) for d in retrain_dates]}")

    # Walk-forward, collect per-row predictions
    all_rows = []
    for i, (fs, fe) in enumerate(folds, 1):
        train_start = max(TRAIN_ERA_FLOOR, fs - pd.DateOffset(years=TRAIN_YEARS))
        tr = df[(df["DATE"] >= train_start) & (df["DATE"] < fs)].copy()
        te = df[(df["DATE"] >= fs) & (df["DATE"] < fe)].copy() if i < N_FOLDS \
             else df[(df["DATE"] >= fs) & (df["DATE"] <= fe)].copy()
        if len(te) == 0: continue
        usable = [c for c in feats if c in tr.columns and tr[c].std() > 1e-8]
        lr, imp, sc = train_lr(tr, usable)
        p_te = predict(te, lr, imp, sc, usable)
        te_c = te.copy(); te_c["p_model"] = p_te; te_c["fold"] = i
        all_rows.append(te_c)

    wf = pd.concat(all_rows, ignore_index=True)
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    # Attach Vegas for ROI calc
    all_test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()
    test_v = attach_vegas(all_test)
    wf = wf.merge(test_v[["DATE", "jbout", "jfighter", "p_vegas_f1",
                          "dec_odds_f1", "dec_odds_f2"]],
                  on=["DATE", "jbout", "jfighter"], how="left")
    wf = wf.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    print(f"Per-fight predictions: {len(wf)}  Vegas-matched: {wf['p_vegas_f1'].notna().sum()}")

    # ── Per-event aggregation ───────────────────────────────────────
    # One UFC event typically = one DATE. Group by DATE and compute:
    #   n, acc, ll, brier, n_ev_bets, ev_profit
    wf["y"] = wf["win"].astype(int)
    p_clip = np.clip(wf["p_model"].values, 0.02, 0.98)
    wf["p_clip"] = p_clip
    wf["pred"] = (p_clip >= 0.5).astype(int)
    wf["correct"] = (wf["pred"] == wf["y"]).astype(int)

    # Strategy D bets + profits (only for Vegas-matched rows)
    wfv = wf[wf["p_vegas_f1"].notna()].copy()
    p_m = wfv["p_model"].values
    p_v = wfv["p_vegas_f1"].values
    pick_f1 = p_m >= 0.5
    edge_on_pick = np.where(pick_f1, p_m - p_v, (1 - p_m) - (1 - p_v))
    wfv["is_ev_bet"] = edge_on_pick > 0
    dec = np.where(pick_f1, wfv["dec_odds_f1"].values, wfv["dec_odds_f2"].values)
    correct_pick = np.where(pick_f1, wfv["y"].values, 1 - wfv["y"].values)
    wfv["bet_profit"] = np.where(wfv["is_ev_bet"],
                                  np.where(correct_pick == 1, dec - 1, -1.0),
                                  0.0)
    # Merge bet info back
    wf = wf.merge(wfv[["DATE", "jbout", "jfighter", "is_ev_bet", "bet_profit"]],
                  on=["DATE", "jbout", "jfighter"], how="left")
    wf["is_ev_bet"] = wf["is_ev_bet"].fillna(False)
    wf["bet_profit"] = wf["bet_profit"].fillna(0.0)

    # Per-event aggregation
    ev = wf.groupby("DATE").agg(
        n=("y", "size"),
        acc=("correct", "mean"),
        ll=("y", lambda s: float(log_loss(s, wf.loc[s.index, "p_clip"].values,
                                           labels=[0, 1])) if len(s.unique()) >= 1 else np.nan),
        brier=("y", lambda s: float(brier_score_loss(s, wf.loc[s.index, "p_clip"].values))),
        n_ev_bets=("is_ev_bet", "sum"),
        ev_profit=("bet_profit", "sum"),
    ).reset_index().sort_values("DATE").reset_index(drop=True)

    # Cumulative ROI
    ev["cum_profit"] = ev["ev_profit"].cumsum()
    ev["cum_bets"] = ev["n_ev_bets"].cumsum()
    ev["cum_roi"] = np.where(ev["cum_bets"] > 0,
                              ev["cum_profit"] / ev["cum_bets"], 0.0)

    print(f"\nPer-event aggregated: {len(ev)} events, "
          f"{ev['n'].sum()} total fights, "
          f"{ev['n_ev_bets'].sum()} +EV bets, "
          f"cumulative ROI at end: {ev['cum_roi'].iloc[-1]*100:+.2f}%")

    # Rolling means (window = 5 events ≈ ~6 weeks of UFC)
    W = 5
    ev["acc_roll"]   = ev["acc"].rolling(W, min_periods=1).mean()
    ev["ll_roll"]    = ev["ll"].rolling(W, min_periods=1).mean()
    ev["brier_roll"] = ev["brier"].rolling(W, min_periods=1).mean()

    # ── Plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("Walk-forward metrics per UFC event "
                 "(6-month retrain cadence — May 2024 → Nov 2025)",
                 fontsize=13, fontweight="bold")

    ax_acc, ax_ll = axes[0]
    ax_br, ax_roi = axes[1]

    date_fmt = mdates.DateFormatter("%b %Y")

    def add_retrain_markers(ax, label_above=None):
        for d in retrain_dates:
            ax.axvline(d, color="#d62728", linestyle="--",
                        alpha=0.55, linewidth=1.2, zorder=0)
        if label_above:
            for d in retrain_dates:
                ax.text(d, label_above, "retrain",
                         color="#d62728", fontsize=8, rotation=90,
                         ha="right", va="top", alpha=0.75)

    # Panel 1: Accuracy
    ax = ax_acc
    ax.scatter(ev["DATE"], ev["acc"]*100, s=np.clip(ev["n"]*3, 5, 60),
               c="#1f77b4", alpha=0.35, label=f"Per-event acc (size ∝ n fights)")
    ax.plot(ev["DATE"], ev["acc_roll"]*100, color="#1f77b4",
             linewidth=2.2, label=f"{W}-event rolling mean")
    ax.axhline(70.95, color="grey", linestyle=":", alpha=0.6,
                label="Pooled mean 70.95%")
    add_retrain_markers(ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per event")
    ax.set_ylim(30, 105)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: Log loss
    ax = ax_ll
    ax.scatter(ev["DATE"], ev["ll"], s=np.clip(ev["n"]*3, 5, 60),
               c="#2ca02c", alpha=0.35, label="Per-event log loss")
    ax.plot(ev["DATE"], ev["ll_roll"], color="#2ca02c",
             linewidth=2.2, label=f"{W}-event rolling mean")
    ax.axhline(0.5830, color="grey", linestyle=":", alpha=0.6,
                label="Pooled mean 0.5830")
    add_retrain_markers(ax)
    ax.set_ylabel("Log loss (lower = better)")
    ax.set_title("Log loss per event")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 3: Brier
    ax = ax_br
    ax.scatter(ev["DATE"], ev["brier"], s=np.clip(ev["n"]*3, 5, 60),
               c="#9467bd", alpha=0.35, label="Per-event Brier")
    ax.plot(ev["DATE"], ev["brier_roll"], color="#9467bd",
             linewidth=2.2, label=f"{W}-event rolling mean")
    ax.axhline(0.1985, color="grey", linestyle=":", alpha=0.6,
                label="Pooled mean 0.1985")
    add_retrain_markers(ax)
    ax.set_ylabel("Brier score (lower = better)")
    ax.set_title("Brier score per event")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 4: Cumulative ROI on +EV bets
    ax = ax_roi
    ax.plot(ev["DATE"], ev["cum_profit"], color="#ff7f0e",
             linewidth=2.4, label="Cumulative $ profit (per $1 bets)")
    ax.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
    # Annotate per-event +EV bets with bar at bottom
    ax2 = ax.twinx()
    ax2.bar(ev["DATE"], ev["n_ev_bets"], width=5, alpha=0.25,
             color="#ff7f0e", label="+EV bets/event")
    ax2.set_ylabel("+EV bets per event", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_ylim(bottom=0)
    add_retrain_markers(ax)
    ax.set_ylabel("Cumulative profit ($)")
    ax.set_title(f"Cumulative ROI on +EV bets  "
                 f"(final: {ev['cum_roi'].iloc[-1]*100:+.2f}% over {int(ev['cum_bets'].iloc[-1])} bets)")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    # Add a text annotation explaining retrain lines
    fig.text(0.01, 0.01,
             "Red dashed lines = model retraining points (every ~6 months).",
             fontsize=9, color="#d62728", style="italic")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nSaved plot to {OUT}")

    # Also save per-event CSV for user inspection
    csv_out = OUT.with_suffix(".csv")
    ev.to_csv(csv_out, index=False)
    print(f"Saved per-event data to {csv_out}")

    # Volatility stats for the summary
    print("\nVolatility summary (per-event metric std-devs):")
    for col, name in [("acc", "Accuracy"), ("ll", "Log loss"), ("brier", "Brier")]:
        print(f"  {name:10s}: std={ev[col].std():.4f}  "
              f"range=[{ev[col].min():.4f}, {ev[col].max():.4f}]")


if __name__ == "__main__":
    main()
