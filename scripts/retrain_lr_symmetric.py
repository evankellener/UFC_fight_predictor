"""Retrain LR on symmetrized training data.

Doubles the training set: each fight → original row + flipped row with
(jfighter ↔ opp_jfighter, win → 1-win, _diff features sign-flipped).
Special cases handled for ratio-of-two-fighters features.

ALSO adds 5 weight-class history features (see docs/feature_wc_history.md)
to the training set before fitting.

Saves new lr.pkl, lr_scaler.pkl, lr_imputer.pkl (overwrites blend_v2).
"""
import sys, json, pickle, sqlite3, warnings, math
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from run_threshold_sweep_both_elos import (
    load_base_both_elos, apply_threshold, TEST_FIRST, TEST_LAST, LAM,
)
FILTER_THRESHOLD = 3  # matches build_predictor_v2_artifacts.py:53
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

OUT = Path("app/models/blend_v2")

# Feature columns that are ratios of (f1/f2) rather than diffs of per-fighter ratios.
# Under swap, r = (f1/f2)-1 → r_new = (f2/f1)-1 = -r/(r+1).
# These are enumerated from mma_ai_pipeline.py: age_ratio_diff and reach_ratio_diff
# are the only two constructed as (f1/f2)-1.
RATIO_OF_FIGHTERS = {"age_ratio_diff", "reach_ratio_diff"}

# Weight-class history feature constants (match predictor_v2.py)
WC_HIST_DIFF_COLS = ["wc_native_winrate_diff", "wc_native_fights_diff",
                     "wc_native_ko_rate_diff", "days_since_this_wc_diff"]
WC_PAIR_COLS = ["cross_division_flag"]
WC_SHRINK_ALPHA = 3.0
LAM_WC = 0.13
DAYS_SINCE_NEVER = 9999


def load_wc_history_from_db():
    """Same query as scripts/build_wc_history_cache.py. Returns:
       {jfighter: [(np.datetime64, weightindex, win, ko), ...] sorted}"""
    conn = sqlite3.connect("data/sqlite_db/sqlite_scrapper.db")
    hist = pd.read_sql("""
        SELECT w.jfighter, e.DATE, w.jbout,
               fr.weightindex, w.win, w.ko
        FROM ufc_winlossko w
        JOIN ufc_event_details e ON e.jevent = w.jevent
        LEFT JOIN ufc_fight_results fr
          ON fr.jevent = w.jevent AND fr.jbout = w.jbout
    """, conn)
    conn.close()
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    hist = hist.dropna(subset=["weightindex", "DATE"])
    hist["weightindex"] = hist["weightindex"].astype(int)
    hist = hist.sort_values(["jfighter", "DATE"])
    out = {}
    for jf, grp in hist.groupby("jfighter"):
        out[jf] = [(np.datetime64(d), int(wc), int(w), int(k))
                   for d, wc, w, k in zip(grp["DATE"], grp["weightindex"],
                                            grp["win"], grp["ko"])]
    return out


def wc_state(jf, evt_ts, current_wc, wc_hist):
    """Compute wc-history state dict. Matches PredictorV2._wc_history_state."""
    from collections import Counter
    h = wc_hist.get(jf, [])
    evt64 = np.datetime64(evt_ts)
    prior_all = [(d, wc, w, k) for d, wc, w, k in h if d < evt64]
    if not prior_all:
        return dict(winrate=0.5, fights=0, ko_rate=0.0,
                    days_since=DAYS_SINCE_NEVER, modal_wc=0)
    # Modal = most-frequent career division
    modal_wc = int(Counter(wc for _, wc, _, _ in prior_all).most_common(1)[0][0])
    prior_wc = [(d, w, k) for d, wc, w, k in prior_all if wc == current_wc]
    if not prior_wc:
        return dict(winrate=0.5, fights=0, ko_rate=0.0,
                    days_since=DAYS_SINCE_NEVER, modal_wc=modal_wc)
    evt_pd = pd.Timestamp(evt_ts)
    weights = np.array([
        math.exp(-LAM_WC * (evt_pd - pd.Timestamp(d)).days / 365.25)
        for d, _, _ in prior_wc
    ])
    wins = np.array([w for _, w, _ in prior_wc], dtype=float)
    kos = np.array([k for _, _, k in prior_wc], dtype=float)
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return dict(winrate=0.5, fights=len(prior_wc), ko_rate=0.0,
                    days_since=DAYS_SINCE_NEVER, modal_wc=modal_wc)
    winrate = (float((weights * wins).sum()) + 0.5 * WC_SHRINK_ALPHA) / (w_sum + WC_SHRINK_ALPHA)
    ko_rate = float((weights * kos).sum()) / (w_sum + WC_SHRINK_ALPHA)
    days_since = float((evt_pd - pd.Timestamp(prior_wc[-1][0])).days)
    return dict(winrate=winrate, fights=len(prior_wc),
                ko_rate=ko_rate, days_since=days_since, modal_wc=modal_wc)


def add_wc_features(df, wc_hist):
    """Add 5 wc-history columns to df, keyed on (DATE, jfighter, opp_jfighter, weightindex).
    Uses each row's own weightindex as current_wc (training convention)."""
    out_rows = {c: [] for c in WC_HIST_DIFF_COLS + WC_PAIR_COLS}
    for _, r in df.iterrows():
        jf1, jf2 = r["jfighter"], r["opp_jfighter"]
        evt = r["DATE"]
        raw_wc = r.get("weightindex", 0)
        if pd.isna(raw_wc) or raw_wc in (None, ""):
            cur_wc = 0
        else:
            cur_wc = int(raw_wc)
        s1 = wc_state(jf1, evt, cur_wc, wc_hist)
        s2 = wc_state(jf2, evt, cur_wc, wc_hist)
        out_rows["wc_native_winrate_diff"].append(s1["winrate"] - s2["winrate"])
        out_rows["wc_native_fights_diff"].append(s1["fights"] - s2["fights"])
        out_rows["wc_native_ko_rate_diff"].append(s1["ko_rate"] - s2["ko_rate"])
        out_rows["days_since_this_wc_diff"].append(s1["days_since"] - s2["days_since"])
        crossed = int((s1["modal_wc"] not in (0, cur_wc))
                      or (s2["modal_wc"] not in (0, cur_wc)))
        out_rows["cross_division_flag"].append(float(crossed))
    for c, vals in out_rows.items():
        df[c] = vals
    return df


def flip_row_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a flipped copy of df where each row represents the same fight
    from the opposite corner's perspective.
    """
    out = df.copy()
    # Identity swaps
    if "jfighter" in out.columns and "opp_jfighter" in out.columns:
        out["jfighter"], out["opp_jfighter"] = df["opp_jfighter"].values, df["jfighter"].values
    if "win" in out.columns:
        out["win"] = 1 - df["win"].astype(int).values

    # Prior counts also swap
    if "f1_priors" in out.columns and "f2_priors" in out.columns:
        out["f1_priors"], out["f2_priors"] = df["f2_priors"].values, df["f1_priors"].values

    # Fighter-1-absolute: days_since_last_fight_f1 becomes f2's days.
    # f2 = f1 - diff  → new_f1 = old_f1 - old_diff (old_diff is f1 - f2)
    if "days_since_last_fight_f1" in out.columns and "days_since_last_fight_diff" in out.columns:
        out["days_since_last_fight_f1"] = (df["days_since_last_fight_f1"].values
                                            - df["days_since_last_fight_diff"].values)

    # Sign-flip all diff-like features EXCEPT ratio-of-fighters.
    # Three flavors of anti-symmetric columns:
    #   (a) `_diff`: sign-flip
    #   (b) `_ufc` / `_exp` (Elo features): sign-flip, EXCEPT elo_win_prob → 1-p
    #   (c) ratio-of-fighters (age_ratio_diff, reach_ratio_diff): handled separately below
    for c in df.columns:
        if c in RATIO_OF_FIGHTERS:
            continue  # handled below
        is_diff = c.endswith("_diff")
        is_elo_suffix = c.endswith("_ufc") or c.endswith("_exp")
        if not (is_diff or is_elo_suffix):
            continue
        if "elo_win_prob" in c:
            out[c] = 1.0 - df[c].values
        else:
            out[c] = -df[c].values

    # Ratio-of-fighters transform: r → -r / (r+1), guarding against r = -1
    for c in RATIO_OF_FIGHTERS:
        if c in out.columns:
            r = df[c].values.astype(float)
            denom = r + 1.0
            # Replace denom near zero with small epsilon (shouldn't happen in practice
            # since ages/reaches are > 0)
            denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
            out[c] = -r / denom

    return out


def verify_flip(df: pd.DataFrame, flipped: pd.DataFrame, n_samples: int = 3):
    """Sanity-check: for a few rows, confirm win and diffs inverted properly."""
    print("=== Flip verification (3 random rows) ===")
    idx = df.sample(n_samples, random_state=0).index
    for i in idx:
        orig = df.loc[i]; flip = flipped.loc[i]
        print(f"\n  Orig:  jf={orig['jfighter']:<18s} opp={orig['opp_jfighter']:<18s} win={orig['win']}")
        print(f"  Flip:  jf={flip['jfighter']:<18s} opp={flip['opp_jfighter']:<18s} win={flip['win']}")
        # Check a diff feature (should sum to 0 — sign-flipped correctly)
        for c in ["precomp_elo_diff_ufc", "precomp_elo_diff_exp", "age_diff",
                  "days_since_last_fight_diff", "striking_elo_diff",
                  "elo_win_prob_ufc"]:
            if c in df.columns:
                print(f"    {c}: orig={orig[c]:+.3f}  flip={flip[c]:+.3f}  sum={orig[c]+flip[c]:+.4f}")
        # Check ratio-of-fighters transform
        for c in RATIO_OF_FIGHTERS:
            if c in df.columns:
                o = orig[c]; f = flip[c]
                # -o/(o+1) should equal f
                expected = -o / (o + 1) if abs(o + 1) > 1e-9 else np.nan
                print(f"    {c}: orig={o:+.4f}  flip={f:+.4f}  expected(-o/(o+1))={expected:+.4f}")


def main():
    print("="*72)
    print("Retraining LR on symmetrized training data")
    print("="*72)

    # Load base features (single orientation)
    base = load_base_both_elos()
    df = apply_threshold(base, FILTER_THRESHOLD)

    # Add weight-class history features BEFORE splitting — each row's features
    # use strictly-prior bouts (filtered by d < row.DATE), no leakage.
    print("\nLoading wc_history from DB + computing features for all rows...")
    wc_hist = load_wc_history_from_db()
    print(f"  wc_history: {len(wc_hist):,} fighters")
    df = add_wc_features(df, wc_hist)
    # Report coverage of new features
    n_zero_fights = int((df["wc_native_fights_diff"] == 0).sum())
    n_cross = int(df["cross_division_flag"].sum())
    print(f"  wc_native_fights_diff == 0 (both fighters have same count at wc): {n_zero_fights:,}/{len(df):,}")
    print(f"  cross_division_flag == 1: {n_cross:,}/{len(df):,}  ({n_cross/len(df)*100:.1f}%)")

    train = df[df["DATE"] < TEST_FIRST].copy()
    test  = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)].copy()
    print(f"\nTrain fights (single-orient): {len(train):,}")
    print(f"Test fights (single-orient):  {len(test):,}")

    # Double the training set
    train_flipped = flip_row_dataframe(train)
    train_doubled = pd.concat([train, train_flipped], ignore_index=True)
    print(f"Train fights (doubled):       {len(train_doubled):,}")

    # Sanity-check the flip
    verify_flip(train.reset_index(drop=True), train_flipped.reset_index(drop=True), n_samples=2)

    # Also double the test set for symmetric evaluation
    test_flipped = flip_row_dataframe(test)
    test_doubled = pd.concat([test, test_flipped], ignore_index=True)

    # Feature selection — match build_predictor_v2_artifacts PLUS new wc features
    feats = [c for c in df.columns if (c.endswith("_diff") or c.endswith("_ufc")
             or c.endswith("_exp") or c in ("weightclass_encoded", "scheduled_rounds",
                                             "days_since_last_fight_f1",
                                             "cross_division_flag"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    print(f"\nFeatures: {len(usable)}")
    print(f"  (includes new wc features: "
          f"{[c for c in WC_HIST_DIFF_COLS + WC_PAIR_COLS if c in usable]})")

    # Fit on DOUBLED training set
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(train_doubled[usable])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    y = train_doubled["win"].astype(int).values
    # Recency weight relative to TEST_FIRST
    w = np.exp(-LAM * (TEST_FIRST - train_doubled["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs, y, sample_weight=w)

    # ── Evaluation ──────────────────────────────────────────────────────
    def eval_set(X_df, y_arr, tag):
        X_ = imp.transform(X_df[usable])
        X_ = sc.transform(X_)
        p = lr.predict_proba(X_)[:, 1]
        pc = np.clip(p, 0.02, 0.98)
        acc = accuracy_score(y_arr, (p >= 0.5).astype(int))
        ll = log_loss(y_arr, pc)
        auc = roc_auc_score(y_arr, p)
        brier = brier_score_loss(y_arr, pc)
        print(f"  {tag:<32s}  n={len(y_arr):>4d}  acc={acc:.4f}  ll={ll:.4f}  auc={auc:.4f}  brier={brier:.4f}")

    print("\n=== Evaluation (symmetric LR) ===")
    eval_set(test, test["win"].astype(int).values, "test (single-orient)")
    eval_set(test_doubled, test_doubled["win"].astype(int).values, "test (doubled)")
    # Also: split the doubled test into original vs flipped halves and
    # confirm acc is ~equal on both halves (symmetry check)
    mid = len(test)
    eval_set(test_doubled.iloc[:mid], test_doubled.iloc[:mid]["win"].astype(int).values,
             "test (doubled, orig half)")
    eval_set(test_doubled.iloc[mid:], test_doubled.iloc[mid:]["win"].astype(int).values,
             "test (doubled, flip half)")

    # Reference: deployed (asymmetric) LR on same test set, for comparison
    print("\n=== Reference: current deployed LR (asymmetric) ===")
    lr_old = pickle.load(open(OUT / "lr.pkl", "rb"))
    sc_old = pickle.load(open(OUT / "lr_scaler.pkl", "rb"))
    imp_old = pickle.load(open(OUT / "lr_imputer.pkl", "rb"))
    feats_old = json.loads((OUT / "feat_cols.json").read_text())
    X_o = imp_old.transform(test[feats_old])
    X_o = sc_old.transform(X_o)
    p_o = lr_old.predict_proba(X_o)[:, 1]
    y_o = test["win"].astype(int).values
    print(f"  deployed on test (single-orient)  n={len(y_o)}  "
          f"acc={accuracy_score(y_o, (p_o>=0.5).astype(int)):.4f}  "
          f"ll={log_loss(y_o, np.clip(p_o,0.02,0.98)):.4f}  "
          f"auc={roc_auc_score(y_o, p_o):.4f}")

    # Save new artifacts
    pickle.dump(lr,  open(OUT / "lr.pkl",         "wb"))
    pickle.dump(sc,  open(OUT / "lr_scaler.pkl",  "wb"))
    pickle.dump(imp, open(OUT / "lr_imputer.pkl", "wb"))
    (OUT / "feat_cols.json").write_text(json.dumps(usable, indent=2))
    print(f"\nSaved symmetric LR artifacts to {OUT}/ (lr.pkl, lr_scaler.pkl, lr_imputer.pkl)")


if __name__ == "__main__":
    main()
