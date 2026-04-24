"""Render a calibration plot (before vs after temperature scaling).

Saves: results/calibration_plot.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, "app"); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from predictor_v2 import PredictorV2
from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, TEST_FIRST, TEST_LAST
from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features

v2 = PredictorV2(verbose=False)
base = load_base_both_elos()
df = apply_threshold(base, 3)
df = add_wc_features(df, load_wc_history_from_db())
test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()
X = v2.imputer.transform(test[v2.feat_cols].values)
X = v2.scaler.transform(X)
p_raw = v2.lr.predict_proba(X)[:, 1]
p_cal = v2._cal_apply(p_raw) if v2._cal_apply else p_raw
y = test["win"].astype(int).values


def bucket(p, y, n_bins=10):
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5) == (y == 1)).astype(int)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if hi == edges[-1] else (conf >= lo) & (conf < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rows.append((float(conf[m].mean()), float(correct[m].mean()), n))
    return rows

raw = bucket(p_raw, y)
cal = bucket(p_cal, y)

# Overall metrics for the annotation boxes
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
def _metrics(p):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5) == (y == 1)).astype(int)
    n_total = len(y)
    ece = 0.0
    for lo in np.arange(0.5, 1.0, 0.05):
        hi = lo + 0.05
        m = (conf >= lo) & (conf <= hi) if hi >= 1.0 else (conf >= lo) & (conf < hi)
        if m.sum() == 0: continue
        ece += m.sum()/n_total * abs(conf[m].mean() - correct[m].mean())
    return {
        "acc": accuracy_score(y, (p >= 0.5).astype(int)),
        "ll":  log_loss(y, pc),
        "brier": brier_score_loss(y, pc),
        "ece": ece * 100,
    }
m_raw = _metrics(p_raw)
m_cal = _metrics(p_cal)
# Date range for the title
date_lo = pd.Timestamp(test["DATE"].min()).strftime("%Y-%m")
date_hi = pd.Timestamp(test["DATE"].max()).strftime("%Y-%m")
N_TEST = len(y)

# ─── render ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#2a2d35",
    "axes.labelcolor": "#1a1d23",
    "xtick.color": "#3a3d45",
    "ytick.color": "#3a3d45",
    "axes.titleweight": "600",
    "axes.titlesize": 13,
    "axes.titlecolor": "#1a1d23",
})
fig, ax = plt.subplots(figsize=(9, 7), dpi=140)
fig.patch.set_facecolor("#fafaf7")
ax.set_facecolor("#fafaf7")

# ±5pp band around the diagonal (shaded)
xs = np.linspace(0.5, 1.0, 100)
ax.fill_between(xs, xs - 0.05, xs + 0.05, color="#c8c8b8", alpha=0.22,
                label="±5pp calibration band", zorder=1)

# Perfect-calibration diagonal
ax.plot([0.5, 1.0], [0.5, 1.0], color="#8a8a80", linestyle=":", linewidth=1.5,
        label="Perfect calibration", zorder=2)

# Uncalibrated — faded
x_r, y_r, n_r = zip(*raw)
ax.plot(x_r, y_r, color="#d49a5c", linewidth=1.8, alpha=0.55,
        label="Raw LR (uncalibrated)", zorder=3, linestyle="--")
ax.scatter(x_r, y_r, s=[n*4 for n in n_r], color="#d49a5c", alpha=0.4,
           edgecolors="#a6773f", linewidths=1, zorder=4)

# Calibrated — bold
x_c, y_c, n_c = zip(*cal)
ax.plot(x_c, y_c, color="#1a8a73", linewidth=2.8,
        label=f"Temperature-calibrated (T={v2.calibrator_method and round(1/np.e**0,4) or ''})",
        zorder=5)
# Fix the label — get actual T from params
import pickle
cal_payload = pickle.load(open("app/models/blend_v2/calibrator.pkl", "rb"))
T = cal_payload["params"].get("T", 1.0)
# Replace the line's label using handles manually later
ax.scatter(x_c, y_c, s=[n*6 for n in n_c], color="#1a8a73", alpha=0.55,
           edgecolors="#0d5848", linewidths=1.3, zorder=6)

# Point labels (n= ...) on the CALIBRATED points
for xi, yi, ni in zip(x_c, y_c, n_c):
    ax.annotate(f"n={ni}", xy=(xi, yi), xytext=(7, -3),
                textcoords="offset points",
                fontsize=8, color="#4a5563", alpha=0.85)

# Styling
ax.set_xlim(0.48, 1.0)
ax.set_ylim(0.48, 1.03)
ax.set_xticks(np.arange(0.5, 1.01, 0.05))
ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.grid(True, which="major", color="#e6e3d8", linewidth=0.7, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Predicted probability (model's confidence in its pick)",
              fontsize=11, labelpad=10)
ax.set_ylabel("Actual win rate (fraction of picks that won)",
              fontsize=11, labelpad=10)
ax.set_title(f"v2 calibration — raw LR vs temperature-scaled\n"
             f"{N_TEST} test fights, {date_lo} → {date_hi}",
             pad=18, loc="left")

# Custom legend labels (with T value now known)
handles, labels = ax.get_legend_handles_labels()
for i, lab in enumerate(labels):
    if "Temperature" in lab:
        labels[i] = f"Temperature-calibrated (T = {T:.3f})"
leg = ax.legend(handles, labels, loc="lower right", frameon=True,
                framealpha=0.95, facecolor="#ffffff", edgecolor="#dcdcd0",
                fontsize=10)
leg.get_frame().set_linewidth(0.5)

# Find the most-divergent raw bucket for the annotation arrow
_biggest = max(raw, key=lambda r: abs(r[1] - r[0]))
_x_big, _y_big, _ = _biggest
ax.annotate(
    f"At predicted {int(_x_big*100)}% raw LR,\n"
    f"actual win rate is {int(_y_big*100)}%.\n"
    f"Temperature scaling (T<1) amplifies\n"
    f"logits — pulling under-confident\n"
    f"probabilities toward the truth.",
    xy=(_x_big, _y_big), xytext=(0.54, 0.93),
    fontsize=9, color="#2a2d35",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffdf4",
              edgecolor="#d8d0b0", linewidth=0.6),
    arrowprops=dict(arrowstyle="->", color="#8a8a80", lw=0.9,
                    connectionstyle="arc3,rad=-0.15"),
)

# ECE callout in lower-right corner of plot
ece_text = (f"ECE:  raw {m_raw['ece']:.2f}pp  →  calibrated {m_cal['ece']:.2f}pp\n"
            f"Log loss:  raw {m_raw['ll']:.4f}  →  {m_cal['ll']:.4f}\n"
            f"Accuracy unchanged ({m_cal['acc']*100:.2f}%)")
ax.text(0.51, 0.52, ece_text, fontsize=9, color="#2a2d35",
        verticalalignment="bottom", horizontalalignment="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff",
                  edgecolor="#dcdcd0", linewidth=0.6))

plt.tight_layout()
out = Path("results/calibration_plot.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=140, facecolor=fig.get_facecolor())
print(f"Saved {out}  ({out.stat().st_size / 1024:.0f} KB)")
