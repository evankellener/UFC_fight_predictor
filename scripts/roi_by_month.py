"""ROI on the test set, broken out by month.

Inputs:
  - results/train_test_2016_2024_predictions.parquet   (predictions from
    train_test_split_2016_2024.py — 391 test fights, 2024-10 → 2026-04)
  - data/tmp/odds_table.csv (or DB)                    (devigged Vegas odds)

For each test fight:
  - model_pick = side with model_p >= 0.5
  - edge_pp    = model_p_pick - vegas_devig_pick
  - ev         = model_p_pick * dec_pick - 1
  - bet_rule:  bet $1 on model_pick when ev > 0 (a.k.a. +EV bet)
  - pnl        = (dec_pick - 1) on win, -1 on loss

Aggregations:
  - month-by-month: n_bets, win rate, ROI, cumulative bankroll
  - additionally: edge≥5pp slice, edge≥10pp slice

Outputs:
  - results/roi_by_month.json — table of per-month metrics
  - results/roi_by_month.png  — chart with cumulative bankroll + monthly ROI bars

LEAKAGE_REFERENCE.md compliance:
  §7  Vegas odds attached AFTER predictions; never used as features
  §8a American odds rejected if |o|<100; probs clipped before LL
  No new training in this script — just post-hoc evaluation.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")

EPS = 1e-6


def main():
    print("=" * 78)
    print("ROI BY MONTH — clean Elastic Net test predictions")
    print("Audit: docs/audits/train_test_split_2016_2024.md")
    print("=" * 78)

    pred_path = Path("results/train_test_2016_2024_predictions.parquet")
    if not pred_path.exists():
        print(f"❌ Missing {pred_path}. Run scripts/train_test_split_2016_2024.py first.")
        sys.exit(1)
    preds = pd.read_parquet(pred_path)
    preds["DATE"] = pd.to_datetime(preds["DATE"])
    print(f"\nLoaded {len(preds)} predictions  "
          f"({preds['DATE'].min().date()} → {preds['DATE'].max().date()})")

    # Attach Vegas odds via the existing helper (CSV-preferred, DB-fallback)
    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    keys = preds[["DATE", "jbout", "jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    merged = preds.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                              "dec_odds_f1", "dec_odds_f2"]],
                         on=["DATE", "jbout", "jfighter"], how="left")
    matched = merged[merged["p_vegas_f1"].notna()].copy()
    print(f"Vegas-matched: {len(matched)} / {len(merged)} "
          f"({100*len(matched)/len(merged):.0f}%)")

    # Resolve which side the model picked, then compute edge / EV
    matched["pick_a"] = (matched["p_pred"] >= 0.5).astype(int)
    matched["dec_pick"]    = np.where(matched["pick_a"]==1,
                                       matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_model_pick"] = np.where(matched["pick_a"]==1,
                                        matched["p_pred"], 1 - matched["p_pred"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"]==1,
                                        matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    matched["edge_pp"] = (matched["p_model_pick"] - matched["p_vegas_pick"]) * 100
    matched["ev"]      = matched["p_model_pick"] * matched["dec_pick"] - 1

    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["pnl"] = np.where(matched["won_pick"]==1,
                               matched["dec_pick"] - 1, -1.0)
    matched["month"] = matched["DATE"].dt.to_period("M")

    # ── Strategies ───────────────────────────────────────────────────────
    strategies = {
        "all_picks":      matched.copy(),                           # bet every model pick
        "ev_positive":    matched[matched["ev"] > 0].copy(),        # bet only +EV
        "edge_5pp":       matched[(matched["ev"] > 0) & (matched["edge_pp"] >= 5)].copy(),
        "edge_10pp":      matched[(matched["ev"] > 0) & (matched["edge_pp"] >= 10)].copy(),
    }

    # ── Pooled metrics ──
    print()
    print(f"{'strategy':<18s}  {'n_bets':>6s}  {'win%':>6s}  {'ROI':>8s}  {'mean_edge':>10s}")
    print("-" * 60)
    pooled = {}
    for name, d in strategies.items():
        if len(d) == 0:
            print(f"  {name:<16s}  (no qualifying bets)"); continue
        n = len(d); win_rate = d["won_pick"].mean() * 100; roi = d["pnl"].mean() * 100
        mean_edge = d["edge_pp"].mean()
        pooled[name] = {"n": int(n), "win_rate": float(win_rate),
                        "roi": float(roi), "mean_edge_pp": float(mean_edge)}
        print(f"  {name:<16s}  {n:>6d}  {win_rate:>5.1f}%  {roi:>+7.2f}%  {mean_edge:>+8.2f}pp")

    # ── Per-month ROI for each strategy ──
    monthly = {}
    for name, d in strategies.items():
        if len(d) == 0: continue
        m = d.groupby("month").agg(n_bets=("pnl", "count"),
                                    n_wins=("won_pick", "sum"),
                                    pnl_total=("pnl", "sum"),
                                    avg_edge=("edge_pp", "mean")).reset_index()
        m["win_rate"]  = m["n_wins"] / m["n_bets"]
        m["roi"]       = m["pnl_total"] / m["n_bets"]   # per-$ ROI for the month
        m["bankroll"]  = m["pnl_total"].cumsum()        # if betting $1 per bet
        monthly[name] = m

    # ── Save JSON summary ──
    out = {"pooled": pooled,
           "monthly": {name: m[["month","n_bets","n_wins","pnl_total","win_rate","roi","bankroll","avg_edge"]]
                              .assign(month=lambda x: x["month"].astype(str))
                              .to_dict("records")
                       for name, m in monthly.items()}}
    Path("results").mkdir(exist_ok=True)
    Path("results/roi_by_month.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✓ Saved results/roi_by_month.json")

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Top: cumulative bankroll for each strategy
    ax1 = axes[0]
    colors = {"all_picks": "#888", "ev_positive": "#2563eb",
              "edge_5pp": "#16a34a", "edge_10pp": "#a855f7"}
    for name, m in monthly.items():
        if name not in colors: continue
        x = m["month"].dt.to_timestamp()
        ax1.plot(x, m["bankroll"], marker="o", linewidth=2, label=name,
                 color=colors[name], markersize=5)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Cumulative profit ($1 per bet)")
    ax1.set_title("Test-set ROI — cumulative bankroll over time\n"
                  "(2024-10 → 2026-04, clean Elastic Net, 391 fights)",
                  fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom: per-month ROI bars (default to ev_positive strategy)
    ax2 = axes[1]
    chosen = "ev_positive"
    if chosen in monthly:
        m = monthly[chosen]
        x = m["month"].dt.to_timestamp()
        bars = ax2.bar(x, m["roi"] * 100, width=20,
                       color=["#16a34a" if r > 0 else "#dc2626" for r in m["roi"]],
                       edgecolor="black", linewidth=0.5)
        # n_bets on top of each bar
        for xi, mi, ni in zip(x, m["roi"]*100, m["n_bets"]):
            ax2.annotate(f"n={ni}", xy=(xi, mi),
                          xytext=(0, 5 if mi >= 0 else -15),
                          textcoords="offset points", ha="center", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel(f"Monthly ROI %  ({chosen})")
    ax2.set_xlabel("Month")
    ax2.grid(alpha=0.3, axis="y")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_png = Path("results/roi_by_month.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"✓ Saved {out_png}")

    # ── Quick monthly table for the chosen strategy ──
    print()
    print(f"── Monthly breakdown ({chosen}) ──")
    if chosen in monthly:
        m = monthly[chosen]
        print(f"{'month':<10s} {'n_bets':>6s} {'win%':>6s} {'roi%':>7s} {'bankroll':>9s} {'avg_edge_pp':>11s}")
        for _, r in m.iterrows():
            print(f"  {str(r['month']):<8s} {r['n_bets']:>6d} {r['win_rate']*100:>5.1f}%"
                  f" {r['roi']*100:>+6.2f}% {r['bankroll']:>+8.2f} {r['avg_edge']:>+10.2f}")


if __name__ == "__main__":
    main()
