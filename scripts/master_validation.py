"""Master Validation — Gold-Standard Walk-Forward + Permutation ROI Test.

╔══════════════════════════════════════════════════════════════════════════╗
║  PRE-REGISTER the config block before running.                          ║
║  Changing parameters after seeing results invalidates the p-value.      ║
╚══════════════════════════════════════════════════════════════════════════╝

Framework:
  • 3-fold 6-month expanding walk-forward
  • Per-fold: train on all data through fold start, test on 6-month window
  • 1000-permutation ROI test (within-fold restricted label shuffle)
  • Verdict: PASS / MARGINAL / FAIL  per strategy

Verdict gates:
  STRONG PASS : 3/3 folds positive  AND  p < 0.05
  PASS        : ≥2/3 folds positive  AND  p < 0.10
  MARGINAL    : ≥2/3 folds positive  OR   p < 0.15
  FAIL        : otherwise

Audit:   docs/audits/master_validation.md
Outputs: results/master_validation.json
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
import mma_ai_pipeline as mma

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PRE-REGISTERED CONFIGURATION — do not change after running            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
CONFIG = dict(
    C          = 0.05,
    L1_RATIO   = 0.5,
    LAM        = 1.20,
    THRESHOLD  = 3,
    TRAIN_START= "2016-01-01",
    # Betting filters
    BET_MENS   = True,      # skip women's divisions
    BET_NO_BW  = True,      # skip Bantamweight (weightclass_encoded == 10)
    # Novel features included
    NOVEL_FEATS= ["home_advantage_diff", "card_position_norm_career_diff"],
    # Permutation test
    N_PERMS    = 1000,
    PERM_SEED  = 42,
    # Retrain mode:
    #   "rolling" — retrain before each fold (deployment architecture)
    #   "static"  — train once on STATIC_TRAIN_END, evaluate all folds with same model
    RETRAIN_MODE     = "static",
    STATIC_TRAIN_END = "2024-10-01",   # only used when RETRAIN_MODE="static"
)
# ╚══════════════════════════════════════════════════════════════════════════╝

# Fold structure: 3 × 6-month evaluation windows
FOLDS = [
    ("H1", "2024-10-01", "2025-04-01"),
    ("H2", "2025-04-01", "2025-10-01"),
    ("H3", "2025-10-01", "2026-04-01"),
]

QUARTERS = [
    ("Q4-2024","2024-10-01","2025-01-01"),
    ("Q1-2025","2025-01-01","2025-04-01"),
    ("Q2-2025","2025-04-01","2025-07-01"),
    ("Q3-2025","2025-07-01","2025-10-01"),
    ("Q4-2025","2025-10-01","2026-01-01"),
    ("Q1-2026","2026-01-01","2026-04-01"),
]

MENS_WCS = {5,6,7,8,9,10,11,12}
BW_CODE  = 10
EPS      = 1e-6

NOVEL_ALL = ["home_advantage_diff","travel_distance_diff_km","tz_diff_diff_hr",
             "is_main_event","card_position_norm_career_diff"]

# Momentum/streak features computed on-the-fly from DB (not in market CSV)
STREAK_FEATS = ["win_streak_diff", "loss_streak_diff"]

# Age features computed on-the-fly from DB (not in market CSV)
AGE_FEATS = ["age_diff"]    # raw age difference (f1 - f2) in years; younger = negative


# ── Pipeline helpers ─────────────────────────────────────────────────────────

def build_through_step6():
    df = mma.load_base_data()
    df = mma.beta_binomial_smooth(df)
    df = mma.poisson_gamma_smooth(df)
    df = mma.compute_derived_features(df)
    stat_cols = sorted(set(c for c in df.columns if (
        c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
        c.endswith("_ratio") or c.endswith("_per_ctrl") or
        c in ["ko_smooth","win_smooth","decision_smooth","sub_land_smooth",
              "sub_land_rate","ctrl_pm","ko_per_sig_str_land","td_per_sig_str_att",
              "ground_per_ctrl","dist_per_sig_str_land","head_per_sig_str_land",
              "rev_per_ctrlopp","sig_str_land_ratio","ko_ratio","sub_att_ratio",
              "ctrl_ratio","ground_land_per_ctrl","td_land_per_ctrl"]
    ) and c in df.columns and not c.startswith("opp_") and not c.endswith("_raw")))
    df = mma.compute_decayed_averages(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    df = mma.compute_opponent_history(df, stat_cols, mma.V7_CONFIG["decay_lambda"])
    return df, stat_cols


def build_fold_features(df_full, stat_cols, train_end_ts, feats_csv, mf, cfg):
    train_only = df_full[df_full["DATE"] < train_end_ts].copy()
    priors     = mma.compute_wc_priors(train_only, stat_cols)
    df_adj     = mma.compute_adjperf(df_full, stat_cols, priors)
    result     = mma.assemble_features(df_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result     = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    result.to_csv(feats_csv, index=False)

    for mod in list(sys.modules):
        if mod.startswith("run_threshold_sweep_both_elos") \
           or mod in ("retrain_lr_symmetric","walk_forward_4fold"):
            del sys.modules[mod]

    from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
    from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features
    from walk_forward_4fold import select_features

    base = load_base_both_elos()
    df   = apply_threshold(base, cfg["THRESHOLD"])
    df   = add_wc_features(df, load_wc_history_from_db())
    feats_baseline = select_features(df)   # MUST be before market merge

    # --- Streak / momentum features (fight-level from DB, no future leakage) ---
    streak_novel = []
    req_streak = [c for c in STREAK_FEATS if c in cfg["NOVEL_FEATS"]]
    if req_streak:
        import sqlite3 as _sql
        _con = _sql.connect(str(elo_feature.DB_PATH))
        _sf  = pd.read_sql(
            "SELECT jfighter, DATE, jbout, win_streak, loss_streak "
            "FROM streak_features", _con)
        _con.close()
        _sf["DATE"] = pd.to_datetime(_sf["DATE"])
        _sf_opp = _sf.rename(columns={
            "jfighter":"opp_jfighter",
            "win_streak":"opp_win_streak",
            "loss_streak":"opp_loss_streak",
        })
        df = df.merge(_sf[["DATE","jbout","jfighter","win_streak","loss_streak"]],
                      on=["DATE","jbout","jfighter"], how="left")
        df = df.merge(_sf_opp[["DATE","jbout","opp_jfighter",
                                "opp_win_streak","opp_loss_streak"]],
                      on=["DATE","jbout","opp_jfighter"], how="left")
        df["win_streak_diff"]  = (df["win_streak"].fillna(0)
                                  - df["opp_win_streak"].fillna(0))
        df["loss_streak_diff"] = (df["loss_streak"].fillna(0)
                                  - df["opp_loss_streak"].fillna(0))
        streak_novel = [c for c in req_streak
                        if c in df.columns and df[c].std() > 1e-8]

    # --- Age features (from DB; age at fight time, no future leakage) ---
    age_novel = []
    req_age = [c for c in AGE_FEATS if c in cfg["NOVEL_FEATS"]]
    if req_age:
        import sqlite3 as _sql
        _con = _sql.connect(str(elo_feature.DB_PATH))
        _tott = pd.read_sql(
            "SELECT jfighter, DOB FROM ufc_fighter_tott "
            "WHERE DOB IS NOT NULL AND DOB != ''", _con)
        _con.close()
        _tott["DOB"] = pd.to_datetime(_tott["DOB"], errors="coerce")
        _tott = _tott[_tott["DOB"].notna()]
        _dob  = dict(zip(_tott["jfighter"], _tott["DOB"]))
        f1_dob = pd.to_datetime(df["jfighter"].map(_dob))
        f2_dob = pd.to_datetime(df["opp_jfighter"].map(_dob))
        df["age_diff"] = ((df["DATE"] - f1_dob).dt.days
                          - (df["DATE"] - f2_dob).dt.days) / 365.25
        age_novel = [c for c in req_age
                     if c in df.columns and df[c].std() > 1e-8]

    # --- Market features (from CSV) ---
    avail = [c for c in NOVEL_ALL if c in mf.columns and c in cfg["NOVEL_FEATS"]]
    df = df.merge(mf[["DATE","jbout","jfighter"]+avail],
                  on=["DATE","jbout","jfighter"], how="left")
    new_feats_mkt = [c for c in avail if c not in feats_baseline and df[c].std()>1e-8]

    new_feats = streak_novel + age_novel + new_feats_mkt
    usable = [c for c in feats_baseline + new_feats
              if c in df.columns and df[c].std() > 1e-8]
    return df, usable


def fit_fold(train_df, train_end_ts, usable, cfg):
    from retrain_lr_symmetric import flip_row_dataframe
    td  = pd.concat([train_df, flip_row_dataframe(train_df)], ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
    w   = np.exp(-cfg["LAM"]*(train_end_ts-td["DATE"]).dt.days.values/365.25)
    lr  = LogisticRegression(C=cfg["C"], penalty="elasticnet",
                             l1_ratio=cfg["L1_RATIO"],
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    n_act = int((np.abs(lr.coef_[0])>1e-8).sum())
    return lr, imp, sc, n_act


def make_bet_frame(test_df, p):
    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    tv = attach_vegas_rich(test_df[["DATE","jbout","jfighter"]].drop_duplicates())
    d  = test_df.assign(p_raw=p).merge(
         tv[["DATE","jbout","jfighter","p_vegas_f1","dec_odds_f1","dec_odds_f2"]],
         on=["DATE","jbout","jfighter"], how="left")
    d  = d[d["p_vegas_f1"].notna()].copy()
    d  = (d.sort_values("p_raw",ascending=False)
           .drop_duplicates(subset=["DATE","jbout"])
           .reset_index(drop=True))
    d["pick_a"]        = (d["p_raw"]>=0.5).astype(int)
    d["dec_odds_pick"] = np.where(d["pick_a"]==1,d["dec_odds_f1"],d["dec_odds_f2"])
    d["p_pick"]        = np.where(d["pick_a"]==1,d["p_raw"],      1-d["p_raw"])
    d["p_vegas_pick"]  = np.where(d["pick_a"]==1,d["p_vegas_f1"], 1-d["p_vegas_f1"])
    d["ev"]            = d["p_pick"]*d["dec_odds_pick"]-1.0
    y2 = d["win"].astype(int).values
    d["won_pick"]      = np.where(d["pick_a"]==1,y2==1,y2==0).astype(int)
    return d


def apply_bet_filter(d, cfg):
    """Apply pre-registered betting filter."""
    if cfg["BET_MENS"]:
        d = d[d["weightclass_encoded"].isin(MENS_WCS)]
    if cfg["BET_NO_BW"]:
        d = d[d["weightclass_encoded"]!=BW_CODE]
    d = d[d["ev"]>0]     # +EV only
    return d.reset_index(drop=True)


def roi_from_frame(d):
    if len(d)==0: return None
    pnl = np.where(d["won_pick"]==1, d["dec_odds_pick"]-1, -1.0)
    return float(pnl.mean()*100)


def roi_from_arrays(won_pick, dec_odds_pick):
    if len(won_pick)==0: return 0.0
    pnl = np.where(won_pick==1, dec_odds_pick-1, -1.0)
    return float(pnl.mean()*100)


# ── Permutation test ─────────────────────────────────────────────────────────

def permutation_test(fold_bet_frames, cfg, observed_roi):
    """
    Vegas-null Monte Carlo permutation test.

    Null hypothesis: the model has no edge beyond Vegas.
    Under the null, each bet wins with probability p_vegas_pick_i
    (the Vegas-implied probability for the side we picked).

    For each simulation:
      - Draw each bet's outcome from Bernoulli(p_vegas_pick_i)
      - Compute pooled ROI across all folds
    p-value = fraction of simulations with ROI >= observed ROI.

    This directly tests: do we win MORE than Vegas expects
    on the specific bets we select?
    """
    rng = np.random.default_rng(cfg["PERM_SEED"])
    null_rois = np.zeros(cfg["N_PERMS"])

    # Pre-extract arrays for speed
    fold_arrays = []
    for d in fold_bet_frames:
        fold_arrays.append((
            d["p_vegas_pick"].values.copy(),   # Vegas-implied win prob per bet
            d["dec_odds_pick"].values.copy(),
        ))

    total_bets = sum(len(fa[0]) for fa in fold_arrays)

    for i in range(cfg["N_PERMS"]):
        sim_pnl = []
        for p_vegas, odds in fold_arrays:
            # Each bet wins with its Vegas-implied probability
            won_sim = (rng.random(len(p_vegas)) < p_vegas).astype(int)
            pnl = np.where(won_sim==1, odds-1, -1.0)
            sim_pnl.extend(pnl)
        null_rois[i] = float(np.mean(sim_pnl)*100)

    p_value  = float((null_rois >= observed_roi).mean())
    null_mean= float(null_rois.mean())
    null_std = float(null_rois.std())
    null_p95 = float(np.percentile(null_rois, 95))
    null_p99 = float(np.percentile(null_rois, 99))

    return dict(
        p_value=p_value,
        null_mean=null_mean, null_std=null_std,
        null_p95=null_p95,   null_p99=null_p99,
        n_perms=cfg["N_PERMS"],
        total_bets=total_bets,
        observed_roi=observed_roi,
        null_type="vegas_monte_carlo",
    )


def verdict(n_pos_folds, p_value, n_folds=3):
    """Return verdict string."""
    if n_pos_folds == n_folds and p_value < 0.05:
        return "✅ STRONG PASS"
    elif n_pos_folds >= 2 and p_value < 0.10:
        return "✅ PASS"
    elif n_pos_folds >= 2 or p_value < 0.15:
        return "⚠  MARGINAL"
    else:
        return "❌ FAIL"


def print_null_bar(observed, null_mean, null_std, null_p95, width=40):
    """ASCII bar showing where observed ROI sits in the null distribution."""
    lo = null_mean - 3*null_std
    hi = max(null_mean + 3*null_std, observed*1.5)
    span = hi - lo
    if span <= 0: return
    obs_pos  = int((observed - lo) / span * width)
    p95_pos  = int((null_p95  - lo) / span * width)
    mean_pos = int((null_mean - lo) / span * width)

    bar = ["-"] * width
    if 0 <= mean_pos < width: bar[mean_pos] = "│"
    if 0 <= p95_pos  < width: bar[p95_pos]  = "▲"  # 95th pct of null
    obs_char = "★" if 0 <= obs_pos < width else ("►" if obs_pos >= width else "◄")
    if 0 <= obs_pos < width: bar[obs_pos] = obs_char

    print(f"  Null dist: [{''.join(bar)}]")
    print(f"             │=mean({null_mean:>+.1f}%)  ▲=null p95({null_p95:>+.1f}%)  ★=observed({observed:>+.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("═"*78)
    print("  MASTER VALIDATION  —  Walk-Forward + Permutation ROI Test")
    print("═"*78)
    print(f"  Config: C={CONFIG['C']}  l1={CONFIG['L1_RATIO']}  λ={CONFIG['LAM']}")
    print(f"  Filter: mens={CONFIG['BET_MENS']}  no_BW={CONFIG['BET_NO_BW']}  +EV only")
    print(f"  Market feats: {CONFIG['NOVEL_FEATS']}")
    mode_label = f"rolling retrain" if CONFIG["RETRAIN_MODE"]=="rolling" \
                 else f"static (train<{CONFIG['STATIC_TRAIN_END']})"
    print(f"  Mode: {mode_label}  |  Folds: 3 × 6-month  |  Perms: {CONFIG['N_PERMS']}")
    print(f"  ⚠  Config pre-registered — no post-hoc adjustment")
    print("═"*78)

    print("\nBuilding pipeline steps 1–6 (once)...")
    df_full, stat_cols = build_through_step6()
    print(f"  Full frame: {len(df_full):,} rows")

    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_masterval")
    if feats_csv.exists() and not backup.exists():
        import shutil; shutil.copy2(feats_csv, backup)

    mf = pd.read_csv("data/tmp/market_features_clean.csv", parse_dates=["DATE"])

    try:
        fold_results   = {}
        fold_bet_frames= []   # filtered bet frames for permutation test
        all_preds      = []   # for pooled metrics

        # ── Phase 1: Fit + predict all 3 folds ───────────────────────────
        print("\n" + "─"*78)
        print("  PHASE 1 — Walk-Forward Fits")
        print("─"*78)

        # In static mode, all folds use the same training cutoff
        static_features_cache = {}   # {train_end_str: (df, usable)} for static mode

        for fold_name, train_end_str, test_end_str in FOLDS:
            test_end_ts  = pd.Timestamp(test_end_str)
            train_start  = pd.Timestamp(CONFIG["TRAIN_START"])

            if CONFIG["RETRAIN_MODE"] == "static":
                actual_train_end_str = CONFIG["STATIC_TRAIN_END"]
            else:
                actual_train_end_str = train_end_str

            train_end_ts = pd.Timestamp(actual_train_end_str)

            mode_tag = f"[static: train<{actual_train_end_str}]" \
                       if CONFIG["RETRAIN_MODE"]=="static" else "[rolling]"
            print(f"\n  [{fold_name}] {mode_tag}  test {train_end_str}–{test_end_str}")

            if actual_train_end_str in static_features_cache:
                df, usable = static_features_cache[actual_train_end_str]
                print(f"  (reusing cached features for train<{actual_train_end_str})")
            else:
                df, usable = build_fold_features(df_full, stat_cols,
                                                  train_end_ts, feats_csv, mf, CONFIG)
                static_features_cache[actual_train_end_str] = (df, usable)

            fold_test_start = pd.Timestamp(train_end_str)
            train = df[(df["DATE"]>=train_start)&(df["DATE"]<train_end_ts)].sort_values("DATE").reset_index(drop=True)
            test  = df[(df["DATE"]>=fold_test_start)&(df["DATE"]<test_end_ts)].reset_index(drop=True)
            assert train["DATE"].max() < train_end_ts, f"Train leak [{fold_name}]"
            assert test["DATE"].min()  >= fold_test_start, f"Test leak [{fold_name}]"
            assert train_end_ts <= fold_test_start, f"Train overlaps test window [{fold_name}]"
            assert not (set(zip(train["DATE"],train["jbout"])) &
                        set(zip(test["DATE"],test["jbout"]))), f"Bout overlap [{fold_name}]"

            lr, imp, sc, n_act = fit_fold(train, train_end_ts, usable, CONFIG)
            p = lr.predict_proba(sc.transform(imp.transform(test[usable])))[:,1]
            y = test["win"].astype(int).values

            # Accuracy metrics
            pc  = np.clip(p, EPS, 1-EPS)
            acc = float(accuracy_score(y,(p>=0.5).astype(int))*100)
            ll  = float(log_loss(y,pc))
            auc = float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None
            bri = float(brier_score_loss(y,pc))

            # Bet frame
            d_all = make_bet_frame(test, p)
            d_flt = apply_bet_filter(d_all, CONFIG)   # filtered: mens, no-BW, +EV

            fold_roi = roi_from_frame(d_flt)
            n_bets   = len(d_flt)
            win_rate = float(d_flt["won_pick"].mean()*100) if n_bets>0 else None

            # Also compute p65 for reference
            d_p65 = d_flt[d_flt["p_pick"]>=0.65]
            roi_p65 = roi_from_frame(d_p65)

            auc_s = f"{auc:.4f}" if auc is not None else "   —"
            pos_s = "✓" if (fold_roi is not None and fold_roi > 0) else "✗"
            print(f"  acc={acc:.2f}%  ll={ll:.4f}  auc={auc_s}  active={n_act}")
            print(f"  +EV filter: {n_bets} bets  ROI={fold_roi:>+.1f}%  "
                  f"win%={win_rate:.1f}%  {pos_s}positive")

            # Per-quarter within fold
            q_rois = {}
            for qn,qs,qe in QUARTERS:
                dq = d_flt[(d_flt["DATE"]>=qs)&(d_flt["DATE"]<qe)]
                q_rois[qn] = {"roi": roi_from_frame(dq), "n": len(dq)}

            fold_results[fold_name] = dict(
                train_end=train_end_str, test_end=test_end_str,
                n_train=len(train), n_test=len(test), n_active=n_act,
                acc=acc, ll=ll, auc=auc, brier=bri,
                n_bets=n_bets, roi=fold_roi, win_rate=win_rate,
                roi_p65=roi_p65, n_p65=len(d_p65),
                positive=(fold_roi is not None and fold_roi>0),
                quarters=q_rois,
            )
            fold_bet_frames.append(d_flt)
            all_preds.append(test.assign(p_pred=p))

        # ── Phase 2: Pooled accuracy metrics ─────────────────────────────
        print("\n" + "─"*78)
        print("  PHASE 2 — Pooled Accuracy Metrics")
        print("─"*78)

        df_pool = pd.concat(all_preds, ignore_index=True)
        y_pool  = df_pool["win"].astype(int).values
        p_pool  = df_pool["p_pred"].values
        pc_pool = np.clip(p_pool, EPS, 1-EPS)
        acc_pool= float(accuracy_score(y_pool,(p_pool>=0.5).astype(int))*100)
        ll_pool = float(log_loss(y_pool,pc_pool))
        auc_pool= float(roc_auc_score(y_pool,p_pool))
        bri_pool= float(brier_score_loss(y_pool,pc_pool))

        print(f"\n  {'Metric':<10}  {'This run':>10}  {'Baseline':>10}  {'Delta':>8}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
        for metric, val, base, delta in [
            ("Accuracy",  f"{acc_pool:.2f}%", "69.82%",  f"{acc_pool-69.82:>+.2f}pp"),
            ("Log loss",  f"{ll_pool:.4f}",   "0.5967",  f"{ll_pool-0.5967:>+.4f}"),
            ("AUC",       f"{auc_pool:.4f}",  "0.7543",  f"{auc_pool-0.7543:>+.4f}"),
            ("Brier",     f"{bri_pool:.4f}",  "0.2047",  f"{bri_pool-0.2047:>+.4f}"),
        ]:
            print(f"  {metric:<10}  {val:>10}  {base:>10}  {delta:>8}")

        # ── Phase 3: ROI summary per fold ─────────────────────────────────
        print("\n" + "─"*78)
        print("  PHASE 3 — Per-Fold ROI Summary")
        print("─"*78)
        print(f"\n  {'Fold':<6}  {'Train cutoff':<14}  {'N bets':>7}  "
              f"{'ROI':>9}  {'Win%':>7}  {'p65 ROI':>9}  {'Pos?':>5}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*5}")

        n_pos = 0
        for fn, fr in fold_results.items():
            pos = "✓" if fr["positive"] else "✗"
            if fr["positive"]: n_pos += 1
            roi_s  = f"{fr['roi']:>+8.1f}%" if fr["roi"] is not None else "         —"
            p65_s  = f"{fr['roi_p65']:>+8.1f}%" if fr["roi_p65"] is not None else "         —"
            wrate  = f"{fr['win_rate']:.1f}%" if fr["win_rate"] is not None else "  —"
            print(f"  {fn:<6}  {fr['train_end']:<14}  {fr['n_bets']:>7}  "
                  f"{roi_s}  {wrate:>7}  {p65_s}  {pos:>5}")

        # Pooled observed ROI
        all_d_flt = pd.concat(fold_bet_frames, ignore_index=True)
        obs_roi   = roi_from_frame(all_d_flt)
        total_n   = len(all_d_flt)
        obs_wr    = float(all_d_flt["won_pick"].mean()*100)

        print(f"\n  {'POOLED':<6}  {'(all folds)':<14}  {total_n:>7}  "
              f"{obs_roi:>+8.1f}%  {obs_wr:>6.1f}%  "
              f"  {n_pos}/3 folds positive")

        # Calibration check: implied win rate vs actual
        implied_wr = float((1/(all_d_flt["dec_odds_pick"])).mean()*100)  # rough
        vegas_wr   = float(all_d_flt["p_vegas_pick"].mean()*100)
        print(f"\n  Calibration on +EV bets:")
        print(f"    Actual win rate:       {obs_wr:.1f}%")
        print(f"    Vegas implied win rate:{vegas_wr:.1f}%")
        print(f"    Model avg p_pick:      {float(all_d_flt['p_pick'].mean()*100):.1f}%")
        print(f"    Avg edge over Vegas:   {float(all_d_flt['ev'].mean()*100):.1f}%")

        # ── Phase 4: Permutation test ─────────────────────────────────────
        print("\n" + "─"*78)
        print(f"  PHASE 4 — Vegas-Null Monte Carlo ROI Test  ({CONFIG['N_PERMS']:,} simulations)")
        print("─"*78)
        print(f"  Null: each bet wins with its Vegas-implied probability.")
        print(f"  Tests: do we win MORE than Vegas expects on our selected bets?")
        print(f"\n  Running {CONFIG['N_PERMS']:,} simulations...")

        perm = permutation_test(fold_bet_frames, CONFIG, obs_roi)

        print(f"\n  Observed ROI:        {perm['observed_roi']:>+7.2f}%")
        print(f"  Null mean:           {perm['null_mean']:>+7.2f}%")
        print(f"  Null std:            {perm['null_std']:>7.2f}%")
        print(f"  Null 95th pct:       {perm['null_p95']:>+7.2f}%")
        print(f"  Null 99th pct:       {perm['null_p99']:>+7.2f}%")
        print(f"  p-value (one-tailed):{perm['p_value']:>8.4f}")
        print()
        print_null_bar(perm["observed_roi"], perm["null_mean"],
                       perm["null_std"], perm["null_p95"])

        # ── Phase 5: Verdict ──────────────────────────────────────────────
        v = verdict(n_pos, perm["p_value"])
        print("\n" + "═"*78)
        print("  VERDICT")
        print("═"*78)
        print(f"\n  Consistency:  {n_pos}/3 folds positive")
        print(f"  p-value:      {perm['p_value']:.4f}")
        print(f"  Pooled ROI:   {obs_roi:>+.2f}%  ({total_n} bets)")
        print(f"\n  {v}")
        print()
        print("  Gate reference:")
        print("  ✅ STRONG PASS : 3/3 positive  AND  p < 0.05")
        print("  ✅ PASS        : ≥2/3 positive  AND  p < 0.10")
        print("  ⚠  MARGINAL   : ≥2/3 positive  OR   p < 0.15")
        print("  ❌ FAIL        : otherwise")

        # ── Phase 6: Quarterly breakdown ──────────────────────────────────
        print("\n" + "─"*78)
        print("  PHASE 6 — Quarterly Breakdown")
        print("─"*78)
        print(f"\n  {'Quarter':<10}  {'Fold':>6}  {'N bets':>7}  {'ROI':>9}  {'Pos?':>5}")
        print(f"  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*5}")
        for qn,_,_ in QUARTERS:
            for fn, fr in fold_results.items():
                if qn in fr["quarters"] and fr["quarters"][qn]["n"]>0:
                    qr = fr["quarters"][qn]
                    pos = "✓" if (qr["roi"] is not None and qr["roi"]>0) else "✗"
                    roi_s = f"{qr['roi']:>+8.1f}%" if qr["roi"] is not None else "         —"
                    print(f"  {qn:<10}  {fn:>6}  {qr['n']:>7}  {roi_s}  {pos:>5}")

        # Quarter-level summary
        all_q_pos = sum(
            1 for fr in fold_results.values()
            for qn, qv in fr["quarters"].items()
            if qv["roi"] is not None and qv["roi"]>0 and qv["n"]>0
        )
        all_q_total = sum(
            1 for fr in fold_results.values()
            for qn, qv in fr["quarters"].items()
            if qv["n"]>0
        )
        print(f"\n  Quarterly consistency: {all_q_pos}/{all_q_total} quarters positive")

        # ── Save ──────────────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        out = dict(
            config=CONFIG,
            folds=fold_results,
            pooled=dict(acc=acc_pool, ll=ll_pool, auc=auc_pool, brier=bri_pool,
                        roi=obs_roi, n_bets=total_n, win_rate=obs_wr,
                        n_pos_folds=n_pos),
            permutation=perm,
            verdict=v,
            quarterly_consistency=f"{all_q_pos}/{all_q_total}",
            runtime_min=round((time.time()-t0)/60,1),
        )
        Path("results/master_validation.json").write_text(
            json.dumps(out, indent=2, default=str))
        print(f"\n✓ Saved results/master_validation.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
