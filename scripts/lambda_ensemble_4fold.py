"""Test λ=0.13 + λ=1.20 ensemble (simple average of calibrated probs).

Rationale: λ=1.20 maxes fold_4 performance (recent drift), λ=0.13 handles
older matchups. Average might hedge the era bet. Per-fold evaluation on
walk-forward + Vegas comparison.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from scipy.optimize import minimize_scalar

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, attach_vegas
from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features, flip_row_dataframe
from walk_forward_4fold import FOLDS, select_features, leakage_assertions
from build_walkforward_vegas_multi_threshold import attach_vegas_rich, fold_metrics

LAMBDA_A = 0.13
LAMBDA_B = 1.20


def temp_cal(p_train, y_train):
    p_train = np.clip(p_train, 1e-6, 1 - 1e-6)
    logit = np.log(p_train / (1 - p_train))
    def nll(T):
        if T <= 0: return 1e9
        pc = 1 / (1 + np.exp(-logit / T))
        pc = np.clip(pc, 1e-6, 1 - 1e-6)
        return -(y_train * np.log(pc) + (1 - y_train) * np.log(1 - pc)).mean()
    T = float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)
    return T


def apply_temp(p, T):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-logit / T))


def fit_and_predict(train, test, feats, lam):
    """Train symmetric LR at given λ, return calibrated test probs + T."""
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

    # Calibrate on undoubled train
    X_tr_orig = sc.transform(imp.transform(train[usable]))
    p_tr = lr.predict_proba(X_tr_orig)[:, 1]
    T = temp_cal(p_tr, train["win"].astype(int).values)

    Xte = sc.transform(imp.transform(test[usable]))
    p_raw = lr.predict_proba(Xte)[:, 1]
    p_cal = apply_temp(p_raw, T)
    return p_cal, T


def run_fold(df, fold, feats):
    train_start = pd.Timestamp(fold["train_start"])
    train_end   = pd.Timestamp(fold["train_end"])
    test_start  = pd.Timestamp(fold["test_start"])
    test_end    = pd.Timestamp(fold["test_end"])
    train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
    test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
    leakage_assertions(train, test, fold)

    p_a, T_a = fit_and_predict(train, test, feats, LAMBDA_A)
    p_b, T_b = fit_and_predict(train, test, feats, LAMBDA_B)
    p_ens = (p_a + p_b) / 2

    y = test["win"].astype(int).values
    def M(p, tag):
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        return {
            "tag": tag, "n": int(len(y)),
            "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "log_loss": float(log_loss(y, pc)),
            "brier":    float(brier_score_loss(y, pc)),
            "auc":      float(roc_auc_score(y, p)),
        }

    # Also compute +EV ROI on Vegas-matched subset for each
    test_ = test.copy()
    tv = attach_vegas_rich(test_[["DATE", "jbout", "jfighter"]].drop_duplicates())
    merged = test_.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                               "dec_odds_f1", "dec_odds_f2"]],
                          on=["DATE", "jbout", "jfighter"], how="left")
    merged = merged.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    def metrics_with_vegas(p):
        m = merged.copy(); m["p_model"] = p
        matched = m[m["p_vegas_f1"].notna()].copy()
        return fold_metrics(matched)

    return {
        "fold": fold["name"], "T_a": T_a, "T_b": T_b,
        "lambda_013":    M(p_a, "λ=0.13"),
        "lambda_120":    M(p_b, "λ=1.20"),
        "ensemble":      M(p_ens, "ensemble"),
        "vegas_013":     metrics_with_vegas(p_a),
        "vegas_120":     metrics_with_vegas(p_b),
        "vegas_ensemble": metrics_with_vegas(p_ens),
    }


def main():
    print("=" * 72)
    print(f"λ-ensemble test: {LAMBDA_A} + {LAMBDA_B} averaged (both calibrated)")
    print("=" * 72)
    base = load_base_both_elos()
    df = apply_threshold(base, 3)
    df = add_wc_features(df, load_wc_history_from_db())
    feats = select_features(df)

    results = []
    t0 = time.time()
    for fold in FOLDS:
        print(f"\n── {fold['name']}")
        r = run_fold(df, fold, feats)
        results.append(r)
        print(f"  λ=0.13       acc={r['lambda_013']['accuracy']*100:.2f}%  ll={r['lambda_013']['log_loss']:.4f}")
        print(f"  λ=1.20       acc={r['lambda_120']['accuracy']*100:.2f}%  ll={r['lambda_120']['log_loss']:.4f}")
        print(f"  ENSEMBLE     acc={r['ensemble']['accuracy']*100:.2f}%  ll={r['ensemble']['log_loss']:.4f}")
        print(f"  vs Vegas +EV: λ=0.13 {r['vegas_013'].get('roi_pos_ev')}%  "
              f"λ=1.20 {r['vegas_120'].get('roi_pos_ev')}%  "
              f"ensemble {r['vegas_ensemble'].get('roi_pos_ev')}%")

    # Aggregate
    print("\n" + "=" * 72)
    print("AGGREGATE across 4 folds")
    print("=" * 72)
    for key in ("lambda_013", "lambda_120", "ensemble"):
        accs = [r[key]["accuracy"] for r in results]
        lls  = [r[key]["log_loss"] for r in results]
        brs  = [r[key]["brier"] for r in results]
        print(f"  {key:<12s}  acc_mean={np.mean(accs)*100:.2f}%  "
              f"std={np.std(accs)*100:.2f}pp  ll={np.mean(lls):.4f}  "
              f"brier={np.mean(brs):.4f}")

    # Pooled Vegas metrics
    print()
    for key in ("vegas_013", "vegas_120", "vegas_ensemble"):
        n_pos_ev = sum(r[key].get("n_pos_ev") or 0 for r in results)
        # Weighted ROI by bet count
        total_bets = n_pos_ev
        total_pnl = sum((r[key].get("roi_pos_ev") or 0) / 100 * (r[key].get("n_pos_ev") or 0)
                         for r in results)
        roi = total_pnl / total_bets * 100 if total_bets else 0
        print(f"  {key:<16s}  pooled +EV ROI={roi:+.2f}% on {total_bets} bets")

    out = Path("results/lambda_ensemble_4fold.json")
    out.write_text(json.dumps({"results": results, "lambdas": [LAMBDA_A, LAMBDA_B],
                                "elapsed_s": time.time() - t0}, indent=2, default=str))
    print(f"\n✓ Saved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
