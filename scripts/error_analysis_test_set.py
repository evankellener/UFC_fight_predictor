"""Error analysis on the held-out test set (2024-10 → 2026-04).

Uses the beta-CV-OOF calibrated predictions (current best model) and slices
test-set accuracy / log loss / Brier / pnl across:

  1. Vegas-confidence band (favorite prob 50-60, 60-70, 70-80, 80-95)
  2. Model-vs-Vegas agreement (agree / disagree)
  3. Model confidence band (where the calibration plot showed under-prediction)
  4. Weight class (weightindex 1-12)
  5. Experience tier (min(prior_fights) of pick / opponent)
  6. Recency tier (days since last fight, picked side)
  7. Date quarter (drift across 18-month test window)
  8. Underdog-pick analysis (when model picks the Vegas underdog)

PURELY DESCRIPTIVE. No model fitting, no calibrator fitting, no strategy
selection. Findings inform feature work, not bet-deployment decisions.

Audit: docs/audits/error_analysis_test_set.md
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

EPS = 1e-9

# ── Pre-registered slice definitions ─────────────────────────────────────
VEGAS_BINS    = [0.50, 0.60, 0.70, 0.80, 0.95]
MODEL_BINS    = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.95]
EXP_TIERS     = [(3, 5, "rookie 3-4"),
                 (5, 10, "developing 5-9"),
                 (10, 20, "established 10-19"),
                 (20, 999, "veteran 20+")]
RECENCY_TIERS = [(0, 91, "active <3mo"),
                 (91, 183, "moderate 3-6mo"),
                 (183, 365, "stale 6-12mo"),
                 (365, 99999, "long layoff >12mo")]
WC_LABELS = {1: "W.Straw", 2: "W.Fly", 3: "W.Bantam", 4: "W.Feather",
             5: "Fly", 6: "Bantam", 7: "Feather", 8: "Light",
             9: "Welter", 10: "Middle", 11: "L.Heavy", 12: "Heavy"}


# ─── Load data ───────────────────────────────────────────────────────────

def load_test_with_context() -> pd.DataFrame:
    pred_path = Path("results/train_calib_compare_v2_predictions.parquet")
    if not pred_path.exists():
        print(f"❌ Missing {pred_path}. Run train_calib_compare_v2.py first.")
        sys.exit(1)
    preds = pd.read_parquet(pred_path)
    preds["DATE"] = pd.to_datetime(preds["DATE"])
    preds = preds.rename(columns={"p_B_beta": "p_pred"})
    keep = ["DATE", "jevent", "jbout", "jfighter", "opp_jfighter", "win", "p_pred"]
    preds = preds[keep].copy()

    # Base data context: WC, prior fights, age, recency
    from run_threshold_sweep_both_elos import load_base_both_elos
    base = load_base_both_elos()
    ctx = base[["DATE", "jbout", "jfighter", "weightindex",
                "f1_priors", "f2_priors", "ufc_age_diff",
                "days_since_last_fight_f1"]].copy()
    ctx["DATE"] = pd.to_datetime(ctx["DATE"])
    ctx = ctx.drop_duplicates(subset=["DATE", "jbout", "jfighter"])
    preds = preds.merge(ctx, on=["DATE", "jbout", "jfighter"], how="left")

    # Vegas
    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    keys = preds[["DATE", "jbout", "jfighter"]].drop_duplicates()
    tv = attach_vegas_rich(keys)
    preds = preds.merge(
        tv[["DATE", "jbout", "jfighter", "p_vegas_f1", "dec_odds_f1", "dec_odds_f2"]],
        on=["DATE", "jbout", "jfighter"], how="left",
    )
    matched = preds[preds["p_vegas_f1"].notna()].copy()

    # Side-of-pick aliases
    matched["pick_a"]       = (matched["p_pred"] >= 0.5).astype(int)
    matched["dec_pick"]     = np.where(matched["pick_a"]==1,
                                       matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_model_pick"] = np.where(matched["pick_a"]==1,
                                       matched["p_pred"], 1 - matched["p_pred"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"]==1,
                                       matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    matched["p_vegas_fav"]  = np.maximum(matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    matched["edge_pp"] = (matched["p_model_pick"] - matched["p_vegas_pick"]) * 100
    matched["ev"]      = matched["p_model_pick"] * matched["dec_pick"] - 1
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["pnl"] = np.where(matched["won_pick"]==1, matched["dec_pick"] - 1, -1.0)
    # Vegas pick + did vegas's pick win
    matched["vegas_pick_a"] = (matched["p_vegas_f1"] >= 0.5).astype(int)
    matched["vegas_won"]   = np.where(matched["vegas_pick_a"]==1, y == 1, y == 0).astype(int)
    matched["agree"]       = (matched["pick_a"] == matched["vegas_pick_a"]).astype(int)
    # Underdog flag: model picks the Vegas underdog
    matched["picked_dog"] = (matched["p_vegas_pick"] < 0.5).astype(int)
    # Min priors of the two fighters
    matched["min_priors"] = matched[["f1_priors", "f2_priors"]].min(axis=1)
    return matched


# ─── Slice helpers ───────────────────────────────────────────────────────

def slice_metrics(d: pd.DataFrame) -> dict:
    if len(d) == 0: return {"n": 0}
    p = d["p_pred"].values
    y = d["win"].astype(int).values
    pc = np.clip(p, 1e-6, 1-1e-6)
    out = {
        "n":             int(len(d)),
        "model_acc_pct": float((d["won_pick"]).mean() * 100),
        "vegas_acc_pct": float((d["vegas_won"]).mean() * 100),
        "log_loss":      float(-(y*np.log(pc) + (1-y)*np.log(1-pc)).mean()),
        "brier":         float(np.mean((p - y) ** 2)),
        "pnl_all_pct":   float(d["pnl"].mean() * 100),
        "win_rate_pick": float(d["won_pick"].mean() * 100),
    }
    return out


def print_table(title, rows, cols):
    print()
    print(f"── {title} ──")
    head = "  " + "  ".join(f"{c:>10s}" for c in cols)
    print(head); print("  " + "-" * (len(head) - 2))
    for label, r in rows:
        if r["n"] == 0:
            print(f"  {label:<22s}  (0)")
            continue
        cells = [f"{label:<22s}"]
        for c in cols:
            if c == "label":            continue
            v = r.get(c)
            if   v is None:             cells.append(f"{'-':>10s}")
            elif c == "n":              cells.append(f"{v:>10d}")
            elif "pct" in c or "acc" in c or "rate" in c:
                cells.append(f"{v:>+9.2f}%")
            else:                       cells.append(f"{v:>10.4f}")
        print("  " + "  ".join(cells))


# ─── Slicing functions ──────────────────────────────────────────────────

def by_vegas_band(df):
    rows = []
    for lo, hi in zip(VEGAS_BINS[:-1], VEGAS_BINS[1:]):
        sl = df[(df["p_vegas_fav"] >= lo) & (df["p_vegas_fav"] < hi)]
        rows.append((f"vegas_fav [{lo:.2f},{hi:.2f})", slice_metrics(sl)))
    return rows


def by_model_band(df):
    rows = []
    for lo, hi in zip(MODEL_BINS[:-1], MODEL_BINS[1:]):
        # use side-of-pick prob (always >= 0.5)
        sl = df[(df["p_model_pick"] >= lo) & (df["p_model_pick"] < hi)]
        rows.append((f"p_model_pick [{lo:.2f},{hi:.2f})", slice_metrics(sl)))
    return rows


def by_agreement(df):
    return [("agree (model=vegas)", slice_metrics(df[df["agree"]==1])),
            ("disagree (model≠vegas)", slice_metrics(df[df["agree"]==0]))]


def by_underdog(df):
    return [("model picks fav (≥50% Vegas)",  slice_metrics(df[df["picked_dog"]==0])),
            ("model picks dog (<50% Vegas)",  slice_metrics(df[df["picked_dog"]==1]))]


def by_weightclass(df):
    rows = []
    for wci in sorted(df["weightindex"].dropna().unique()):
        wci = int(wci)
        sl = df[df["weightindex"] == wci]
        label = WC_LABELS.get(wci, f"WC{wci}")
        rows.append((f"{label} (idx={wci})", slice_metrics(sl)))
    return rows


def by_experience(df):
    rows = []
    for lo, hi, label in EXP_TIERS:
        sl = df[(df["min_priors"] >= lo) & (df["min_priors"] < hi)]
        rows.append((label, slice_metrics(sl)))
    return rows


def by_recency(df):
    rows = []
    for lo, hi, label in RECENCY_TIERS:
        sl = df[(df["days_since_last_fight_f1"] >= lo) &
                (df["days_since_last_fight_f1"] < hi)]
        rows.append((label, slice_metrics(sl)))
    return rows


def by_quarter(df):
    df = df.copy()
    df["quarter"] = df["DATE"].dt.to_period("Q")
    rows = []
    for q in sorted(df["quarter"].unique()):
        sl = df[df["quarter"] == q]
        rows.append((str(q), slice_metrics(sl)))
    return rows


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("ERROR ANALYSIS — held-out test set, beta-CV-OOF predictions")
    print("Audit: docs/audits/error_analysis_test_set.md")
    print("=" * 78)

    df = load_test_with_context()
    print(f"\nLoaded {len(df)} Vegas-matched test fights "
          f"({df['DATE'].min().date()} → {df['DATE'].max().date()})")
    print(f"Pooled model accuracy: {df['won_pick'].mean()*100:.2f}%")
    print(f"Pooled Vegas-pick accuracy: {df['vegas_won'].mean()*100:.2f}%")

    cols = ["n", "model_acc_pct", "vegas_acc_pct", "log_loss", "brier", "pnl_all_pct"]

    sections = [
        ("Vegas-confidence band",       by_vegas_band),
        ("Model-confidence band",       by_model_band),
        ("Model vs Vegas agreement",    by_agreement),
        ("Underdog vs favorite picks",  by_underdog),
        ("Weight class",                by_weightclass),
        ("Experience tier (min priors)", by_experience),
        ("Recency tier (days since)",   by_recency),
        ("Calendar quarter",            by_quarter),
    ]
    out_json = {"n_test": int(len(df)),
                "pooled_model_acc": float(df["won_pick"].mean()*100),
                "pooled_vegas_acc": float(df["vegas_won"].mean()*100),
                "sections": {}}
    for name, fn in sections:
        rows = fn(df)
        print_table(name, rows, cols)
        out_json["sections"][name] = {label: r for label, r in rows}

    # ── Plot: a 2x2 summary ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Top-left: model vs vegas accuracy by Vegas band
    ax = axes[0, 0]
    rows = by_vegas_band(df)
    labels = [lab for lab, _ in rows]
    ns     = [r["n"]              for _, r in rows]
    macc   = [r.get("model_acc_pct", 0) for _, r in rows]
    vacc   = [r.get("vegas_acc_pct", 0) for _, r in rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, macc, 0.4, label="model", color="#2563eb")
    ax.bar(x + 0.2, vacc, 0.4, label="vegas", color="#dc2626")
    for xi, n in zip(x, ns):
        ax.annotate(f"n={n}", xy=(xi, 5), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("Pick win rate %"); ax.set_title("Accuracy by Vegas-favorite confidence")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Top-right: pnl by Vegas band
    ax = axes[0, 1]
    pnl = [r.get("pnl_all_pct", 0) for _, r in rows]
    colors = ["#16a34a" if p > 0 else "#dc2626" for p in pnl]
    ax.bar(x, pnl, color=colors, edgecolor="black", linewidth=0.5)
    for xi, p in zip(x, pnl): ax.annotate(f"{p:+.1f}%", xy=(xi, p),
                                           xytext=(0, 5 if p>=0 else -12),
                                           textcoords="offset points",
                                           ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("PnL per bet %"); ax.set_title("Pooled PnL by Vegas-favorite confidence")
    ax.grid(alpha=0.3, axis="y")

    # Bottom-left: accuracy by weight class
    ax = axes[1, 0]
    rows = by_weightclass(df)
    labels = [lab.split()[0] for lab, _ in rows]
    ns    = [r["n"]              for _, r in rows]
    macc  = [r.get("model_acc_pct", 0) for _, r in rows]
    vacc  = [r.get("vegas_acc_pct", 0) for _, r in rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, macc, 0.4, label="model", color="#2563eb")
    ax.bar(x + 0.2, vacc, 0.4, label="vegas", color="#dc2626")
    for xi, n in zip(x, ns):
        ax.annotate(f"n={n}", xy=(xi, 3), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("Pick win rate %"); ax.set_title("Accuracy by weight class")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Bottom-right: accuracy by quarter
    ax = axes[1, 1]
    rows = by_quarter(df)
    labels = [lab for lab, _ in rows]
    ns    = [r["n"]              for _, r in rows]
    macc  = [r.get("model_acc_pct", 0) for _, r in rows]
    vacc  = [r.get("vegas_acc_pct", 0) for _, r in rows]
    x = np.arange(len(labels))
    ax.plot(x, macc, "o-", label="model", color="#2563eb", linewidth=2)
    ax.plot(x, vacc, "o-", label="vegas", color="#dc2626", linewidth=2)
    for xi, n in zip(x, ns): ax.annotate(f"n={n}", xy=(xi, macc[xi]),
                                           xytext=(0, 5),
                                           textcoords="offset points",
                                           ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, fontsize=8, ha="right")
    ax.set_ylabel("Pick win rate %"); ax.set_title("Accuracy by calendar quarter (drift?)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_png = Path("results/error_analysis.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\n✓ Saved {out_png}")

    Path("results").mkdir(exist_ok=True)
    Path("results/error_analysis.json").write_text(json.dumps(out_json, indent=2, default=str))
    print(f"✓ Saved results/error_analysis.json")


if __name__ == "__main__":
    main()
