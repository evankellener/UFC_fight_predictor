"""Train + calibration comparison.

Splits:
  Train      2018-01-01 → 2024-04-01   (~6.25yr — base Elastic Net fit here)
  Validation 2024-04-01 → 2024-10-01   (~6mo   — calibrators fit here ONLY)
  Test       2024-10-01 → 2026-04-01   (~18mo  — held-out evaluation ONLY)

WC priors frozen at TRAIN_END=2024-04-01 (val + test treated as future).

Calibrators (all fit on val predictions, evaluated on test):
  - uncalibrated   : raw EN output  (baseline)
  - temperature    : 1-param logit rescale
  - platt          : 2-param logistic on logit (a*logit + b)
  - isotonic       : non-parametric monotone (sklearn.isotonic)
  - beta           : 3-param  sigmoid(a·log p − b·log(1−p) + c)
  - histogram      : 10 equal-width bins, output bin's empirical mean
  - spline         : monotone cubic spline on logits

Each is scored on test for log loss, Brier, ECE, accuracy, AUC, ROI.
ROI computed against attach_vegas_rich devigged odds.

Audit: docs/audits/train_calib_split_2018.md
"""
import sys, json, time, warnings
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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             roc_auc_score)
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import PchipInterpolator

import mma_ai_pipeline as mma

EPS         = 1e-6
LAM         = 1.20
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END   = pd.Timestamp("2024-04-01")    # train < this; val ≥ this
VAL_END     = pd.Timestamp("2024-10-01")    # val < this; test ≥ this
TEST_END    = pd.Timestamp("2026-04-01")
THRESHOLD   = 3
EN_C        = 0.05
EN_L1       = 0.5

# ─── Calibrator implementations ──────────────────────────────────────────

def _logit(p):
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureCalib:
    """1-parameter logit / T."""
    name = "temperature"
    def fit(self, p, y):
        lg = _logit(p)
        def nll(T):
            if T <= 0: return 1e9
            pc = _sigmoid(lg / T)
            pc = np.clip(pc, EPS, 1-EPS)
            return -(y*np.log(pc) + (1-y)*np.log(1-pc)).mean()
        self.T = float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)
        return self
    def predict(self, p):
        return _sigmoid(_logit(p) / self.T)


class PlattCalib:
    """2-parameter sigmoid(a·logit + b) — equivalent to LR on logit."""
    name = "platt"
    def fit(self, p, y):
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(_logit(p).reshape(-1, 1), y)
        self.a = float(lr.coef_[0, 0]); self.b = float(lr.intercept_[0])
        return self
    def predict(self, p):
        return _sigmoid(self.a * _logit(p) + self.b)


class IsotonicCalib:
    """Non-parametric monotone."""
    name = "isotonic"
    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1-EPS)
        self.iso.fit(p, y)
        return self
    def predict(self, p):
        return np.clip(self.iso.predict(p), EPS, 1-EPS)


class BetaCalib:
    """Beta calibration — Kull et al. 2017.
    Maps p → sigmoid(a·log p − b·log(1−p) + c)."""
    name = "beta"
    def fit(self, p, y):
        p = np.clip(p, EPS, 1-EPS)
        s1, s2 = np.log(p), -np.log(1 - p)
        X = np.column_stack([s1, s2])
        # logistic regression with intercept
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(X, y)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.coef_[0, 1])
        self.c = float(lr.intercept_[0])
        return self
    def predict(self, p):
        p = np.clip(p, EPS, 1-EPS)
        z = self.a * np.log(p) + self.b * (-np.log(1 - p)) + self.c
        return _sigmoid(z)


class HistogramCalib:
    """K equal-width bins; output bin's empirical mean."""
    name = "histogram"
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
    def fit(self, p, y):
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges[-1] += 1e-9
        idx = np.digitize(p, edges) - 1
        idx = np.clip(idx, 0, self.n_bins - 1)
        rates = np.full(self.n_bins, np.nan)
        for k in range(self.n_bins):
            sel = idx == k
            if sel.sum() >= 5:
                rates[k] = y[sel].mean()
            else:
                # fallback: bin midpoint
                rates[k] = 0.5 * (edges[k] + edges[k+1])
        # enforce monotonic non-decreasing (post-hoc; matches isotonic spirit)
        for k in range(1, self.n_bins):
            if rates[k] < rates[k-1]:
                rates[k] = rates[k-1]
        self.edges = edges
        self.rates = np.clip(rates, EPS, 1-EPS)
        return self
    def predict(self, p):
        idx = np.digitize(p, self.edges) - 1
        idx = np.clip(idx, 0, self.n_bins - 1)
        return self.rates[idx]


class SplineCalib:
    """PCHIP monotone spline through validation deciles in logit-space."""
    name = "spline"
    def fit(self, p, y, n_anchors=10):
        order = np.argsort(p)
        p_s, y_s = p[order], y[order]
        # split into n_anchors equal-count bins
        edges = np.linspace(0, len(p_s), n_anchors + 1, dtype=int)
        xs, ys = [], []
        for k in range(n_anchors):
            lo, hi = edges[k], edges[k+1]
            if hi - lo < 5: continue
            xs.append(p_s[lo:hi].mean())
            ys.append(y_s[lo:hi].mean())
        xs, ys = np.array(xs), np.array(ys)
        # enforce monotone non-decreasing
        for k in range(1, len(ys)):
            if ys[k] < ys[k-1]: ys[k] = ys[k-1]
        # extend endpoints to [0,1] for safety
        if xs[0]  > 0.0: xs = np.r_[0.0, xs]; ys = np.r_[max(ys[0]-0.05, EPS), ys]
        if xs[-1] < 1.0: xs = np.r_[xs, 1.0]; ys = np.r_[ys, min(ys[-1]+0.05, 1-EPS)]
        # de-duplicate xs
        xs, idx = np.unique(xs, return_index=True); ys = ys[idx]
        self.spline = PchipInterpolator(xs, np.clip(ys, EPS, 1-EPS))
        return self
    def predict(self, p):
        return np.clip(self.spline(np.clip(p, 0.0, 1.0)), EPS, 1-EPS)


CALIBRATORS = [
    ("uncalibrated", None),
    ("temperature",  TemperatureCalib()),
    ("platt",        PlattCalib()),
    ("isotonic",     IsotonicCalib()),
    ("beta",         BetaCalib()),
    ("histogram",    HistogramCalib(n_bins=10)),
    ("spline",       SplineCalib()),
]


# ─── Pipeline build ──────────────────────────────────────────────────────

def build_through_step6():
    print("Building pipeline through Step 6 (per-fight clean)...")
    df = mma.load_base_data()
    df = mma.beta_binomial_smooth(df)
    df = mma.poisson_gamma_smooth(df)
    df = mma.compute_derived_features(df)
    stat_cols = sorted(set(c for c in df.columns if
                 (c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
                  c.endswith("_ratio") or c.endswith("_per_ctrl") or
                  c in ["ko_smooth", "win_smooth", "decision_smooth",
                        "sub_land_smooth", "sub_land_rate", "ctrl_pm",
                        "ko_per_sig_str_land", "td_per_sig_str_att",
                        "ground_per_ctrl", "dist_per_sig_str_land",
                        "head_per_sig_str_land", "rev_per_ctrlopp",
                        "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
                        "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"])
                 and c in df.columns and not c.startswith("opp_") and not c.endswith("_raw")))
    df = mma.compute_decayed_averages(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    df = mma.compute_opponent_history(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    return df, stat_cols


def split_features(df_full, stat_cols, train_end):
    """WC priors frozen at train_end. val + test are 'future' rows scored
    using priors that never saw them."""
    train_only = df_full[df_full["DATE"] < train_end].copy()
    print(f"  WC priors from {len(train_only):,} train-only rows  "
          f"(DATE < {train_end.date()})")
    priors = mma.compute_wc_priors(train_only, stat_cols)
    df_with_adj = mma.compute_adjperf(df_full, stat_cols, priors)
    result = mma.assemble_features(df_with_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    return result


# ─── Metrics ─────────────────────────────────────────────────────────────

def expected_calibration_error(p, y, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1); edges[-1] += 1e-9
    idx = np.digitize(p, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    e = 0.0; n = len(y)
    for k in range(n_bins):
        sel = idx == k
        if sel.sum() == 0: continue
        e += (sel.sum() / n) * abs(p[sel].mean() - y[sel].mean())
    return float(e)


def metrics_block(p, y):
    pc = np.clip(p, EPS, 1-EPS)
    out = {
        "n":        int(len(y)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(y, pc)),
        "brier":    float(brier_score_loss(y, pc)),
        "ece":      expected_calibration_error(p, y),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = None
    return out


def roi_block(test_df, p, eps_thresh=0.0):
    """Compute ROI on a *copy* of test rows with prediction `p`.
    Strategies: all_picks, ev_positive (ev>0), edge_5pp."""
    d = test_df.copy()
    d["p_pred"] = p
    # need Vegas attached upstream — assumed in d already
    matched = d[d["p_vegas_f1"].notna()].copy()
    if len(matched) == 0:
        return {"all_picks": None, "ev_positive": None, "edge_5pp": None}
    matched["pick_a"]       = (matched["p_pred"] >= 0.5).astype(int)
    matched["dec_pick"]     = np.where(matched["pick_a"]==1,
                                       matched["dec_odds_f1"], matched["dec_odds_f2"])
    matched["p_model_pick"] = np.where(matched["pick_a"]==1,
                                       matched["p_pred"], 1 - matched["p_pred"])
    matched["p_vegas_pick"] = np.where(matched["pick_a"]==1,
                                       matched["p_vegas_f1"], 1 - matched["p_vegas_f1"])
    matched["edge_pp"]      = (matched["p_model_pick"] - matched["p_vegas_pick"]) * 100
    matched["ev"]           = matched["p_model_pick"] * matched["dec_pick"] - 1
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["pnl"] = np.where(matched["won_pick"]==1, matched["dec_pick"]-1, -1.0)

    def stats(sl):
        if len(sl) == 0: return None
        return {"n": int(len(sl)),
                "win_pct": float(sl["won_pick"].mean()*100),
                "roi_pct": float(sl["pnl"].mean()*100)}
    return {
        "all_picks":   stats(matched),
        "ev_positive": stats(matched[matched["ev"] > 0]),
        "edge_5pp":    stats(matched[(matched["ev"] > 0) & (matched["edge_pp"] >= 5)]),
    }


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    overall_t0 = time.time()
    print("=" * 78)
    print("TRAIN + CALIBRATION COMPARISON — Elastic Net")
    print(f"  Train      : {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  Validation : {TRAIN_END.date()} → {VAL_END.date()}  (calibrator fit only)")
    print(f"  Test       : {VAL_END.date()} → {TEST_END.date()}  (held-out)")
    print(f"  EN(C={EN_C}, l1={EN_L1})    Threshold: ≥{THRESHOLD} prior UFC fights")
    print(f"  Audit: docs/audits/train_calib_split_2018.md")
    print("=" * 78)

    df_full, stat_cols = build_through_step6()
    print(f"\n✓ Step 1-6 build done ({time.time()-overall_t0:.0f}s)")

    print(f"\nComputing features with priors frozen at train_end={TRAIN_END.date()}...")
    feats_df = split_features(df_full, stat_cols, TRAIN_END)
    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup = Path("data/tmp/mmaai_features.csv.before_calib2018")
    if feats_csv.exists() and not backup.exists():
        import shutil; shutil.copy2(feats_csv, backup)
    feats_df.to_csv(feats_csv, index=False)

    try:
        for mod in list(sys.modules):
            if (mod.startswith("run_threshold_sweep_both_elos")
                or mod == "retrain_lr_symmetric"
                or mod == "walk_forward_4fold"):
                del sys.modules[mod]
        from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
        from retrain_lr_symmetric import (load_wc_history_from_db, add_wc_features,
                                          flip_row_dataframe)
        from walk_forward_4fold import select_features

        base = load_base_both_elos()
        df = apply_threshold(base, THRESHOLD)
        df = add_wc_features(df, load_wc_history_from_db())
        feats = select_features(df)

        train = df[(df["DATE"] >= TRAIN_START) & (df["DATE"] < TRAIN_END)].copy()
        val   = df[(df["DATE"] >= TRAIN_END)   & (df["DATE"] < VAL_END)].copy()
        test  = df[(df["DATE"] >= VAL_END)     & (df["DATE"] < TEST_END)].copy()

        # §1 hard-asserts
        assert train["DATE"].max() < TRAIN_END, "§1: train_max ≥ TRAIN_END"
        assert val["DATE"].max()   < VAL_END,   "§1: val_max ≥ VAL_END"
        assert val["DATE"].min()  >= TRAIN_END, "§1: val_min < TRAIN_END"
        assert test["DATE"].min() >= VAL_END,   "§1: test_min < VAL_END"
        train_keys = set(zip(train["DATE"], train["jbout"]))
        val_keys   = set(zip(val["DATE"],   val["jbout"]))
        test_keys  = set(zip(test["DATE"],  test["jbout"]))
        assert not (train_keys & val_keys), "§1: train/val bout overlap"
        assert not (train_keys & test_keys), "§1: train/test bout overlap"
        assert not (val_keys & test_keys),   "§1: val/test bout overlap"
        print(f"\n✓ Leakage assertions pass")
        print(f"  Train: {len(train):>5,d} fights ({train['DATE'].min().date()} → {train['DATE'].max().date()})")
        print(f"  Val:   {len(val):>5,d} fights ({val['DATE'].min().date()} → {val['DATE'].max().date()})")
        print(f"  Test:  {len(test):>5,d} fights ({test['DATE'].min().date()} → {test['DATE'].max().date()})")

        # Symmetric doubled training
        train_d = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
        usable = [c for c in feats if c in train_d.columns and train_d[c].std() > 1e-8]
        print(f"  Usable features: {len(usable)}")

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
        ytr = train_d["win"].astype(int).values
        w   = np.exp(-LAM * (TRAIN_END - train_d["DATE"]).dt.days.values / 365.25)

        lr = LogisticRegression(C=EN_C, penalty="elasticnet", l1_ratio=EN_L1,
                                solver="saga", max_iter=8000, random_state=42)
        lr.fit(Xtr, ytr, sample_weight=w)
        n_active = int((np.abs(lr.coef_[0]) > 1e-8).sum())
        print(f"\n  Elastic Net fit: {len(usable)} features, {n_active} active")

        # ── Predict val + test (raw, uncalibrated) ──
        Xv = sc.transform(imp.transform(val[usable]))
        Xt = sc.transform(imp.transform(test[usable]))
        p_val_raw = lr.predict_proba(Xv)[:, 1]
        p_test_raw = lr.predict_proba(Xt)[:, 1]
        y_val  = val["win"].astype(int).values
        y_test = test["win"].astype(int).values

        # ── Attach Vegas to test (post-prediction; never as feature) ──
        from build_walkforward_vegas_multi_threshold import attach_vegas_rich
        keys = test[["DATE", "jbout", "jfighter"]].drop_duplicates()
        tv = attach_vegas_rich(keys)
        test_with_vegas = test.merge(
            tv[["DATE", "jbout", "jfighter", "p_vegas_f1", "dec_odds_f1", "dec_odds_f2"]],
            on=["DATE", "jbout", "jfighter"], how="left",
        )
        print(f"  Vegas-matched test fights: "
              f"{test_with_vegas['p_vegas_f1'].notna().sum()} / {len(test_with_vegas)}")

        # ── Fit each calibrator on val, evaluate on test ─────────────────
        print("\n" + "=" * 78)
        print("CALIBRATOR COMPARISON  (val→fit, test→evaluate)")
        print("=" * 78)
        header = f"{'method':<14s}  {'acc':>6s}  {'logloss':>7s}  {'brier':>6s}  {'ECE':>6s}  {'AUC':>6s}    {'ROI(all)':>9s}  {'ROI(+EV)':>9s}  {'ROI(5pp)':>9s}"
        print(header); print("-" * len(header))

        results = {}
        per_test_pred = {}
        for name, cal in CALIBRATORS:
            if cal is None:
                p_test_cal = p_test_raw
                cal_params = None
            else:
                cal.fit(p_val_raw, y_val)
                p_test_cal = cal.predict(p_test_raw)
                cal_params = {k: v for k, v in vars(cal).items()
                              if isinstance(v, (int, float, str, bool))}
            m  = metrics_block(p_test_cal, y_test)
            r  = roi_block(test_with_vegas, p_test_cal)
            results[name] = {"params": cal_params, "metrics": m, "roi": r}
            per_test_pred[name] = p_test_cal

            roi_all = (f"{r['all_picks']['roi_pct']:>+7.2f}%"
                       if r['all_picks'] else "    -   ")
            roi_ev  = (f"{r['ev_positive']['roi_pct']:>+7.2f}%"
                       if r['ev_positive'] else "    -   ")
            roi_5   = (f"{r['edge_5pp']['roi_pct']:>+7.2f}%"
                       if r['edge_5pp']    else "    -   ")
            print(f"  {name:<12s}  {m['accuracy']*100:>5.2f}%  {m['log_loss']:>7.4f}  "
                  f"{m['brier']:>6.4f}  {m['ece']:>6.4f}  "
                  f"{(m['auc'] or 0):>6.4f}    {roi_all}  {roi_ev}  {roi_5}")

        # ── Calibration plot: reliability for each method ────────────────
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
        axes = axes.flatten()
        bins = np.linspace(0.0, 1.0, 11); centers = 0.5 * (bins[:-1] + bins[1:])
        for ax, (name, _) in zip(axes, CALIBRATORS):
            p = per_test_pred[name]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
            idx = np.digitize(p, bins) - 1; idx = np.clip(idx, 0, 9)
            mp, ob, ns = [], [], []
            for k in range(10):
                sel = idx == k
                if sel.sum() == 0: continue
                mp.append(p[sel].mean()); ob.append(y_test[sel].mean()); ns.append(int(sel.sum()))
            ax.plot(mp, ob, "o-", color="#2563eb", markersize=6)
            for x, y_, n in zip(mp, ob, ns):
                ax.annotate(f"{n}", xy=(x, y_), xytext=(4, -10),
                             textcoords="offset points", fontsize=7, color="#444")
            m = results[name]["metrics"]
            ax.set_title(f"{name}\nECE={m['ece']:.3f}  Brier={m['brier']:.3f}", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        if len(CALIBRATORS) < len(axes):
            for k in range(len(CALIBRATORS), len(axes)):
                axes[k].axis("off")
        for ax in axes:
            if ax.has_data():
                ax.set_xlabel("Predicted prob"); ax.set_ylabel("Observed rate")
        fig.suptitle("Calibration comparison — Elastic Net (train 2018-01 → 2024-04, "
                     "calib fit on 2024-04→2024-10, test 2024-10→2026-04)",
                     fontsize=12, y=1.00)
        plt.tight_layout()
        out_png = Path("results/calib_compare.png")
        plt.savefig(out_png, dpi=140, bbox_inches="tight")
        print(f"\n✓ Saved {out_png}")

        # ── Save predictions parquet for downstream analysis ─────────────
        Path("results").mkdir(exist_ok=True)
        out_pred = test[["DATE", "jevent", "jbout", "jfighter", "opp_jfighter", "win"]].copy()
        for name in per_test_pred:
            out_pred[f"p_{name}"] = per_test_pred[name]
        out_pred.to_parquet("results/train_calib_split_2018_predictions.parquet", index=False)

        out = {
            "config": {
                "train_start": str(TRAIN_START.date()),
                "train_end":   str(TRAIN_END.date()),
                "val_end":     str(VAL_END.date()),
                "test_end":    str(TEST_END.date()),
                "threshold": THRESHOLD, "recency_lambda": LAM,
                "model": "ElasticNet", "C": EN_C, "l1_ratio": EN_L1,
                "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
                "n_features": int(len(usable)), "n_active": int(n_active),
            },
            "results": results,
            "audit":   "docs/audits/train_calib_split_2018.md",
            "total_runtime_min": round((time.time() - overall_t0)/60, 1),
        }
        Path("results/calib_compare.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"✓ Saved results/calib_compare.json")
        print(f"\nTotal runtime: {(time.time() - overall_t0)/60:.1f} minutes")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
