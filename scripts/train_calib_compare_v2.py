"""Calibration comparison v2 — keep the original 2016→2024-10 train baseline,
test two strategies for sourcing calibrator-fit data:

  Strategy A — IN-SAMPLE TAIL
      Fit each calibrator on EN's predictions over the LAST 6 months of
      training (2024-04 → 2024-10).  These predictions are in-sample
      (model saw the rows during training), so they are slightly
      optimistic.  Reported as comparison only.

  Strategy B — 5-FOLD CV-OOF  (textbook-correct)
      KFold(5, shuffle=False) on chronologically-sorted train rows.
      For each fold k: train EN on train_minus_k, predict on fold_k →
      gives every train row a prediction made by a model that never saw it.
      Fit each calibrator on those (p, y) pairs.
      Final EN is fit on the FULL train, then test predictions are
      transformed by the calibrator.

Train: 2016-01 → 2024-10  (1,907 fights)
Test:  2024-10 → 2026-04  (391 fights, never touched until eval)

Calibrators (same set as v1):
    uncalibrated, temperature, platt, isotonic, beta, histogram, spline

Audit: docs/audits/train_calib_compare_v2.md
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
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             roc_auc_score)
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator

import mma_ai_pipeline as mma

EPS                  = 1e-6
LAM                  = 1.20
TRAIN_START          = pd.Timestamp("2016-01-01")
TRAIN_END            = pd.Timestamp("2024-10-01")
TEST_END             = pd.Timestamp("2026-04-01")
INSAMPLE_TAIL_START  = pd.Timestamp("2024-04-01")
THRESHOLD            = 3
EN_C                 = 0.05
EN_L1                = 0.5
N_CV_FOLDS           = 5

# ─── Calibrator implementations (same as v1) ─────────────────────────────

def _logit(p):
    p = np.clip(p, EPS, 1-EPS); return np.log(p / (1 - p))

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureCalib:
    name = "temperature"
    def fit(self, p, y):
        lg = _logit(p)
        def nll(T):
            if T <= 0: return 1e9
            pc = _sigmoid(lg / T); pc = np.clip(pc, EPS, 1-EPS)
            return -(y*np.log(pc) + (1-y)*np.log(1-pc)).mean()
        self.T = float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)
        return self
    def predict(self, p): return _sigmoid(_logit(p) / self.T)


class PlattCalib:
    name = "platt"
    def fit(self, p, y):
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(_logit(p).reshape(-1, 1), y)
        self.a = float(lr.coef_[0, 0]); self.b = float(lr.intercept_[0])
        return self
    def predict(self, p): return _sigmoid(self.a * _logit(p) + self.b)


class IsotonicCalib:
    name = "isotonic"
    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1-EPS)
        self.iso.fit(p, y); return self
    def predict(self, p): return np.clip(self.iso.predict(p), EPS, 1-EPS)


class BetaCalib:
    name = "beta"
    def fit(self, p, y):
        p = np.clip(p, EPS, 1-EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(X, y)
        self.a = float(lr.coef_[0, 0]); self.b = float(lr.coef_[0, 1])
        self.c = float(lr.intercept_[0]); return self
    def predict(self, p):
        p = np.clip(p, EPS, 1-EPS)
        z = self.a * np.log(p) + self.b * (-np.log(1 - p)) + self.c
        return _sigmoid(z)


class HistogramCalib:
    name = "histogram"
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
    def fit(self, p, y):
        edges = np.linspace(0.0, 1.0, self.n_bins + 1); edges[-1] += 1e-9
        idx = np.clip(np.digitize(p, edges) - 1, 0, self.n_bins - 1)
        rates = np.full(self.n_bins, np.nan)
        for k in range(self.n_bins):
            sel = idx == k
            rates[k] = (y[sel].mean() if sel.sum() >= 5
                        else 0.5*(edges[k] + edges[k+1]))
        for k in range(1, self.n_bins):
            if rates[k] < rates[k-1]: rates[k] = rates[k-1]
        self.edges = edges; self.rates = np.clip(rates, EPS, 1-EPS)
        return self
    def predict(self, p):
        idx = np.clip(np.digitize(p, self.edges) - 1, 0, self.n_bins - 1)
        return self.rates[idx]


class SplineCalib:
    name = "spline"
    def fit(self, p, y, n_anchors=10):
        order = np.argsort(p); p_s, y_s = p[order], y[order]
        edges = np.linspace(0, len(p_s), n_anchors + 1, dtype=int)
        xs, ys = [], []
        for k in range(n_anchors):
            lo, hi = edges[k], edges[k+1]
            if hi - lo < 5: continue
            xs.append(p_s[lo:hi].mean()); ys.append(y_s[lo:hi].mean())
        xs, ys = np.array(xs), np.array(ys)
        for k in range(1, len(ys)):
            if ys[k] < ys[k-1]: ys[k] = ys[k-1]
        if xs[0]  > 0.0: xs = np.r_[0.0, xs]; ys = np.r_[max(ys[0]-0.05, EPS), ys]
        if xs[-1] < 1.0: xs = np.r_[xs, 1.0]; ys = np.r_[ys, min(ys[-1]+0.05, 1-EPS)]
        xs, idx = np.unique(xs, return_index=True); ys = ys[idx]
        self.spline = PchipInterpolator(xs, np.clip(ys, EPS, 1-EPS)); return self
    def predict(self, p):
        return np.clip(self.spline(np.clip(p, 0.0, 1.0)), EPS, 1-EPS)


def make_calibrators():
    """Fresh instances per strategy."""
    return [
        ("uncalibrated", None),
        ("temperature",  TemperatureCalib()),
        ("platt",        PlattCalib()),
        ("isotonic",     IsotonicCalib()),
        ("beta",         BetaCalib()),
        ("histogram",    HistogramCalib(n_bins=10)),
        ("spline",       SplineCalib()),
    ]


# ─── Pipeline ────────────────────────────────────────────────────────────

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
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0; n = len(y)
    for k in range(n_bins):
        sel = idx == k
        if sel.sum() == 0: continue
        e += (sel.sum() / n) * abs(p[sel].mean() - y[sel].mean())
    return float(e)


def metrics_block(p, y):
    pc = np.clip(p, EPS, 1-EPS)
    out = {"n": int(len(y)),
           "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
           "log_loss": float(log_loss(y, pc)),
           "brier":    float(brier_score_loss(y, pc)),
           "ece":      expected_calibration_error(p, y)}
    try:    out["auc"] = float(roc_auc_score(y, p))
    except: out["auc"] = None
    return out


def roi_block(test_df, p):
    d = test_df.copy(); d["p_pred"] = p
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
    matched["edge_pp"] = (matched["p_model_pick"] - matched["p_vegas_pick"]) * 100
    matched["ev"]      = matched["p_model_pick"] * matched["dec_pick"] - 1
    y = matched["win"].astype(int).values
    matched["won_pick"] = np.where(matched["pick_a"]==1, y == 1, y == 0).astype(int)
    matched["pnl"] = np.where(matched["won_pick"]==1, matched["dec_pick"]-1, -1.0)
    def stats(sl):
        if len(sl) == 0: return None
        return {"n": int(len(sl)),
                "win_pct": float(sl["won_pick"].mean()*100),
                "roi_pct": float(sl["pnl"].mean()*100)}
    return {"all_picks":   stats(matched),
            "ev_positive": stats(matched[matched["ev"] > 0]),
            "edge_5pp":    stats(matched[(matched["ev"] > 0) & (matched["edge_pp"] >= 5)])}


# ─── Single base-EN training (Strategy A; also re-used as final for Strategy B)

def fit_base_en(train_df, usable, lam):
    """Symmetric doubled training, sample-weighted EN. Returns (lr, imp, sc)."""
    from retrain_lr_symmetric import flip_row_dataframe
    train_d = pd.concat([train_df, flip_row_dataframe(train_df)], ignore_index=True)
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_d[usable]))
    ytr = train_d["win"].astype(int).values
    w   = np.exp(-lam * (TRAIN_END - train_d["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=EN_C, penalty="elasticnet", l1_ratio=EN_L1,
                            solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)
    return lr, imp, sc


def predict_base(lr, imp, sc, df, usable):
    return lr.predict_proba(sc.transform(imp.transform(df[usable])))[:, 1]


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    overall_t0 = time.time()
    print("=" * 78)
    print("CALIBRATION COMPARISON v2 — full 2016→2024-10 train")
    print(f"  Train       : {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  Test        : {TRAIN_END.date()} → {TEST_END.date()}")
    print(f"  In-sample tail (Strategy A) : "
          f"{INSAMPLE_TAIL_START.date()} → {TRAIN_END.date()}")
    print(f"  CV-OOF (Strategy B)         : {N_CV_FOLDS}-fold KFold(shuffle=False)")
    print(f"  EN(C={EN_C}, l1={EN_L1})    Threshold: ≥{THRESHOLD} prior fights")
    print(f"  Audit: docs/audits/train_calib_compare_v2.md")
    print("=" * 78)

    df_full, stat_cols = build_through_step6()
    print(f"\n✓ Step 1-6 build done ({time.time()-overall_t0:.0f}s)")

    print(f"\nFreezing WC priors at train_end={TRAIN_END.date()}...")
    feats_df = split_features(df_full, stat_cols, TRAIN_END)
    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup = Path("data/tmp/mmaai_features.csv.before_calibv2")
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
        from retrain_lr_symmetric import (load_wc_history_from_db, add_wc_features)
        from walk_forward_4fold import select_features

        base = load_base_both_elos()
        df = apply_threshold(base, THRESHOLD)
        df = add_wc_features(df, load_wc_history_from_db())
        feats = select_features(df)

        train = df[(df["DATE"] >= TRAIN_START) & (df["DATE"] < TRAIN_END)].copy()
        test  = df[(df["DATE"] >= TRAIN_END)   & (df["DATE"] < TEST_END)].copy()
        train = train.sort_values("DATE").reset_index(drop=True)

        # §1 hard-asserts
        assert train["DATE"].max() < TRAIN_END, "§1: train_max ≥ TRAIN_END"
        assert test["DATE"].min()  >= TRAIN_END, "§1: test_min < TRAIN_END"
        assert not (set(zip(train["DATE"], train["jbout"])) &
                    set(zip(test["DATE"],  test["jbout"]))), "§1: bout overlap"
        print(f"\n✓ Leakage assertions pass")
        print(f"  Train: {len(train):>5,d} fights "
              f"({train['DATE'].min().date()} → {train['DATE'].max().date()})")
        print(f"  Test:  {len(test):>5,d} fights "
              f"({test['DATE'].min().date()} → {test['DATE'].max().date()})")

        usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
        print(f"  Usable features: {len(usable)}")

        # ── Final base EN (used for both strategies' test-time prediction) ──
        print("\n[final base EN] fit on full train (1-shot)...")
        t0 = time.time()
        lr_final, imp_final, sc_final = fit_base_en(train, usable, LAM)
        n_active = int((np.abs(lr_final.coef_[0]) > 1e-8).sum())
        print(f"  done in {time.time()-t0:.0f}s  ({n_active}/{len(usable)} active)")

        # Test predictions (raw)
        p_test_raw = predict_base(lr_final, imp_final, sc_final, test, usable)
        y_test = test["win"].astype(int).values

        # Attach Vegas to test
        from build_walkforward_vegas_multi_threshold import attach_vegas_rich
        keys = test[["DATE", "jbout", "jfighter"]].drop_duplicates()
        tv = attach_vegas_rich(keys)
        test_with_vegas = test.merge(
            tv[["DATE", "jbout", "jfighter", "p_vegas_f1", "dec_odds_f1", "dec_odds_f2"]],
            on=["DATE", "jbout", "jfighter"], how="left",
        )
        print(f"  Vegas-matched test fights: "
              f"{test_with_vegas['p_vegas_f1'].notna().sum()} / {len(test_with_vegas)}")

        # ─── Strategy A — IN-SAMPLE TAIL ────────────────────────────────────
        print("\n" + "=" * 78)
        print(f"STRATEGY A — In-sample tail  ({INSAMPLE_TAIL_START.date()} "
              f"→ {TRAIN_END.date()})")
        print("=" * 78)
        tail = train[train["DATE"] >= INSAMPLE_TAIL_START].copy()
        print(f"  tail rows: {len(tail)}")
        p_tail = predict_base(lr_final, imp_final, sc_final, tail, usable)
        y_tail = tail["win"].astype(int).values

        # ─── Strategy B — 5-FOLD CV-OOF ─────────────────────────────────────
        print("\n" + "=" * 78)
        print(f"STRATEGY B — 5-fold CV-OOF  (KFold shuffle=False on sorted train)")
        print("=" * 78)
        kf = KFold(n_splits=N_CV_FOLDS, shuffle=False)
        p_oof = np.zeros(len(train), dtype=np.float64)
        y_oof = train["win"].astype(int).values
        for k, (idx_tr, idx_te) in enumerate(kf.split(train)):
            t0 = time.time()
            tr_k = train.iloc[idx_tr]
            te_k = train.iloc[idx_te]
            lr_k, imp_k, sc_k = fit_base_en(tr_k, usable, LAM)
            p_oof[idx_te] = predict_base(lr_k, imp_k, sc_k, te_k, usable)
            print(f"  fold {k+1}/{N_CV_FOLDS}: "
                  f"train n={len(tr_k):>4d}  oof n={len(te_k):>4d}  "
                  f"({time.time()-t0:.0f}s)")
        print(f"  OOF predictions ready for {len(p_oof)} train rows")

        # ── Fit + evaluate every calibrator under each strategy ─────────────
        all_results = {}
        per_test_pred = {"A": {}, "B": {}}
        strategies = [("A", p_tail, y_tail, "in-sample tail"),
                      ("B", p_oof,  y_oof, "5-fold CV-OOF")]

        for tag, p_fit, y_fit, label in strategies:
            print("\n" + "─" * 78)
            print(f"Calibrator results — Strategy {tag}  ({label}, n_fit={len(p_fit)})")
            print("─" * 78)
            header = f"{'method':<14s}  {'acc':>6s}  {'logloss':>7s}  {'brier':>6s}  {'ECE':>6s}  {'AUC':>6s}    {'ROI(all)':>9s}  {'ROI(+EV)':>9s}  {'ROI(5pp)':>9s}"
            print(header); print("-" * len(header))

            res_strat = {}
            for name, cal in make_calibrators():
                if cal is None:
                    p_test_cal = p_test_raw
                    cal_params = None
                else:
                    cal.fit(p_fit, y_fit)
                    p_test_cal = cal.predict(p_test_raw)
                    cal_params = {k: v for k, v in vars(cal).items()
                                  if isinstance(v, (int, float, str, bool))}
                m = metrics_block(p_test_cal, y_test)
                r = roi_block(test_with_vegas, p_test_cal)
                res_strat[name] = {"params": cal_params, "metrics": m, "roi": r}
                per_test_pred[tag][name] = p_test_cal

                roi_all = (f"{r['all_picks']['roi_pct']:>+7.2f}%"
                           if r['all_picks'] else "    -   ")
                roi_ev  = (f"{r['ev_positive']['roi_pct']:>+7.2f}%"
                           if r['ev_positive'] else "    -   ")
                roi_5   = (f"{r['edge_5pp']['roi_pct']:>+7.2f}%"
                           if r['edge_5pp']    else "    -   ")
                print(f"  {name:<12s}  {m['accuracy']*100:>5.2f}%  {m['log_loss']:>7.4f}  "
                      f"{m['brier']:>6.4f}  {m['ece']:>6.4f}  "
                      f"{(m['auc'] or 0):>6.4f}    {roi_all}  {roi_ev}  {roi_5}")
            all_results[tag] = res_strat

        # ── Side-by-side delta vs raw ───────────────────────────────────────
        print("\n" + "=" * 78)
        print("Δ vs uncalibrated  (negative = improvement for log_loss/brier/ece)")
        print("=" * 78)
        print(f"  {'method':<14s}  | {'A: Δll':>9s} {'A: Δbri':>9s} {'A: Δece':>9s}  "
              f"|  {'B: Δll':>9s} {'B: Δbri':>9s} {'B: Δece':>9s}")
        raw_A = all_results["A"]["uncalibrated"]["metrics"]
        raw_B = all_results["B"]["uncalibrated"]["metrics"]
        for name, _ in make_calibrators():
            mA = all_results["A"][name]["metrics"]; mB = all_results["B"][name]["metrics"]
            d = lambda m, r, k: m[k] - r[k]
            print(f"  {name:<12s}  | "
                  f"{d(mA, raw_A, 'log_loss'):>+9.4f} "
                  f"{d(mA, raw_A, 'brier'):>+9.4f} "
                  f"{d(mA, raw_A, 'ece'):>+9.4f}  | "
                  f"{d(mB, raw_B, 'log_loss'):>+9.4f} "
                  f"{d(mB, raw_B, 'brier'):>+9.4f} "
                  f"{d(mB, raw_B, 'ece'):>+9.4f}")

        # ── Reliability plots: 2 rows (A / B) × 7 columns ───────────────────
        fig, axes = plt.subplots(2, 7, figsize=(22, 7), sharex=True, sharey=True)
        bins = np.linspace(0.0, 1.0, 11)
        for row, tag in enumerate(["A", "B"]):
            for col, (name, _) in enumerate(make_calibrators()):
                ax = axes[row, col]
                p = per_test_pred[tag][name]
                ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
                idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
                mp, ob, ns = [], [], []
                for k in range(10):
                    sel = idx == k
                    if sel.sum() == 0: continue
                    mp.append(p[sel].mean()); ob.append(y_test[sel].mean())
                    ns.append(int(sel.sum()))
                ax.plot(mp, ob, "o-",
                        color=("#2563eb" if tag == "A" else "#16a34a"),
                        markersize=5)
                m = all_results[tag][name]["metrics"]
                ax.set_title(f"{tag}: {name}\nECE={m['ece']:.3f}", fontsize=9)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
                if col == 0: ax.set_ylabel(f"Strategy {tag}\nObs rate")
                if row == 1: ax.set_xlabel("Predicted")
        fig.suptitle("Calibration comparison v2 — full 2016→2024-10 train, "
                     "test 2024-10→2026-04\n"
                     "Strategy A = in-sample tail (2024-04→2024-10)  |  "
                     "Strategy B = 5-fold CV-OOF on full train",
                     fontsize=12, y=1.02)
        plt.tight_layout()
        out_png = Path("results/calib_compare_v2.png")
        plt.savefig(out_png, dpi=140, bbox_inches="tight")
        print(f"\n✓ Saved {out_png}")

        # ── Save predictions parquet + JSON ─────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        out_pred = test[["DATE", "jevent", "jbout", "jfighter", "opp_jfighter", "win"]].copy()
        for tag in ("A", "B"):
            for name in per_test_pred[tag]:
                out_pred[f"p_{tag}_{name}"] = per_test_pred[tag][name]
        out_pred.to_parquet("results/train_calib_compare_v2_predictions.parquet", index=False)

        out = {
            "config": {
                "train_start": str(TRAIN_START.date()),
                "train_end":   str(TRAIN_END.date()),
                "test_end":    str(TEST_END.date()),
                "insample_tail_start": str(INSAMPLE_TAIL_START.date()),
                "n_cv_folds":  N_CV_FOLDS,
                "threshold": THRESHOLD, "recency_lambda": LAM,
                "model": "ElasticNet", "C": EN_C, "l1_ratio": EN_L1,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "n_features": int(len(usable)), "n_active": int(n_active),
            },
            "results": all_results,
            "audit":   "docs/audits/train_calib_compare_v2.md",
            "total_runtime_min": round((time.time() - overall_t0)/60, 1),
        }
        Path("results/calib_compare_v2.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"✓ Saved results/calib_compare_v2.json")
        print(f"\nTotal runtime: {(time.time() - overall_t0)/60:.1f} minutes")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
