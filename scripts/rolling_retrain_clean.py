"""Quarterly expanding-window rolling retrain on the clean pipeline.

Each quarter gets its own model trained on all data through the quarter start:
  Q4-2024: train < 2024-10-01, test 2024-10-01 → 2025-01-01
  Q1-2025: train < 2025-01-01, test 2025-01-01 → 2025-04-01
  Q2-2025: train < 2025-04-01, test 2025-04-01 → 2025-07-01
  Q3-2025: train < 2025-07-01, test 2025-07-01 → 2025-10-01
  Q4-2025: train < 2025-10-01, test 2025-10-01 → 2026-01-01
  Q1-2026: train < 2026-01-01, test 2026-01-01 → 2026-04-01

Compare per-quarter ROI to static model (single train through 2024-10).

Audit:   docs/audits/rolling_retrain_clean.md
Outputs: results/rolling_retrain_clean.json
"""
import sys, json, time, warnings, copy
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

C          = 0.05
L1_RATIO   = 0.5
LAM        = 1.20
TRAIN_START= pd.Timestamp("2016-01-01")
TEST_END   = pd.Timestamp("2026-04-01")
THRESHOLD  = 3
EPS        = 1e-6

NOVEL = ["home_advantage_diff","travel_distance_diff_km","tz_diff_diff_hr",
         "is_main_event","card_position_norm_career_diff"]

MENS_WCS  = {5,6,7,8,9,10,11,12}

# Quarterly folds — (name, train_end, test_end)
FOLDS = [
    ("Q4-2024", "2024-10-01", "2025-01-01"),
    ("Q1-2025", "2025-01-01", "2025-04-01"),
    ("Q2-2025", "2025-04-01", "2025-07-01"),
    ("Q3-2025", "2025-07-01", "2025-10-01"),
    ("Q4-2025", "2025-10-01", "2026-01-01"),
    ("Q1-2026", "2026-01-01", "2026-04-01"),
]


def build_through_step6():
    """Steps 1–6: run once. Returns full df + stat_cols."""
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


def build_fold_features(df_full, stat_cols, train_end_ts, feats_csv, mf):
    """
    Steps 7–9 + downstream: run per fold with train_end_ts as the WC prior cutoff.
    Returns assembled feature df (all dates).
    """
    train_only = df_full[df_full["DATE"] < train_end_ts].copy()
    priors     = mma.compute_wc_priors(train_only, stat_cols)
    df_adj     = mma.compute_adjperf(df_full, stat_cols, priors)
    result     = mma.assemble_features(df_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result     = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
    result.to_csv(feats_csv, index=False)

    # Reload modules so they pick up fresh mmaai_features.csv
    for mod in list(sys.modules):
        if mod.startswith("run_threshold_sweep_both_elos") \
           or mod in ("retrain_lr_symmetric","walk_forward_4fold"):
            del sys.modules[mod]

    from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
    from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features
    from walk_forward_4fold import select_features

    base = load_base_both_elos()
    df   = apply_threshold(base, THRESHOLD)
    df   = add_wc_features(df, load_wc_history_from_db())
    feats_baseline = select_features(df)          # BEFORE market merge

    avail = [c for c in NOVEL if c in mf.columns]
    df = df.merge(mf[["DATE","jbout","jfighter"]+avail],
                  on=["DATE","jbout","jfighter"], how="left")
    new_feats = [c for c in avail if c not in feats_baseline and df[c].std()>1e-8]
    usable = [c for c in feats_baseline+new_feats
              if c in df.columns and df[c].std()>1e-8]

    return df, usable


def fit_fold(train_df, train_end_ts, usable):
    from retrain_lr_symmetric import flip_row_dataframe
    td  = pd.concat([train_df, flip_row_dataframe(train_df)], ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
    # Recency weights anchored at fold train_end
    w   = np.exp(-LAM*(train_end_ts - td["DATE"]).dt.days.values/365.25)
    lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1_RATIO,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    n_act = int((np.abs(lr.coef_[0])>1e-8).sum())
    return lr, imp, sc, n_act


def predict_fold(lr, imp, sc, test_df, usable):
    return lr.predict_proba(sc.transform(imp.transform(test_df[usable])))[:,1]


def roi_stats(sub):
    if len(sub)==0: return {"roi":None,"n":0,"win_rate":None}
    pnl = np.where(sub["won_pick"]==1, sub["dec_odds_pick"]-1, -1.0)
    return {"roi":float(pnl.mean()*100), "n":int(len(sub)),
            "win_rate":float(sub["won_pick"].mean()*100)}


def build_bet_frame(test_df, p):
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


def main():
    t0 = time.time()
    print("="*78)
    print("QUARTERLY ROLLING RETRAIN  (clean pipeline)")
    print(f"  C={C}, l1={L1_RATIO}, λ={LAM}  |  6 quarterly folds")
    print("="*78)

    print("\nBuilding steps 1–6 (once)...")
    df_full, stat_cols = build_through_step6()
    print(f"  Full frame: {len(df_full):,} rows")

    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_rolling")
    if feats_csv.exists() and not backup.exists():
        import shutil; shutil.copy2(feats_csv, backup)

    mf = pd.read_csv("data/tmp/market_features_clean.csv", parse_dates=["DATE"])

    try:
        # ── Run each fold ─────────────────────────────────────────────────
        fold_results = {}
        all_preds    = []   # collect for pooled stats

        for fold_name, train_end_str, test_end_str in FOLDS:
            train_end_ts = pd.Timestamp(train_end_str)
            test_end_ts  = pd.Timestamp(test_end_str)

            print(f"\n{'─'*60}")
            print(f"  Fold {fold_name}: train < {train_end_str}  |  test {train_end_str} → {test_end_str}")

            df, usable = build_fold_features(df_full, stat_cols, train_end_ts,
                                              feats_csv, mf)

            train = df[(df["DATE"]>=TRAIN_START)&(df["DATE"]<train_end_ts)].sort_values("DATE").reset_index(drop=True)
            test  = df[(df["DATE"]>=train_end_ts)&(df["DATE"]<test_end_ts)].reset_index(drop=True)

            assert train["DATE"].max() < train_end_ts, f"Train leak in {fold_name}"
            assert test["DATE"].min()  >= train_end_ts, f"Test leak in {fold_name}"
            assert not (set(zip(train["DATE"],train["jbout"])) &
                        set(zip(test["DATE"],test["jbout"]))), f"Bout overlap in {fold_name}"

            print(f"  Train: {len(train):,} rows  Test: {len(test):,} rows  Features: {len(usable)}")

            if len(test)==0:
                print(f"  ⚠ No test rows — skipping")
                continue

            lr, imp, sc, n_act = fit_fold(train, train_end_ts, usable)
            p = predict_fold(lr, imp, sc, test, usable)
            y = test["win"].astype(int).values

            # Metrics
            pc  = np.clip(p, EPS, 1-EPS)
            acc = float(accuracy_score(y,(p>=0.5).astype(int))*100)
            ll  = float(log_loss(y,pc))
            auc = float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None
            bri = float(brier_score_loss(y,pc))

            # Bet frame (all, men's, men's no-BW)
            d_all  = build_bet_frame(test, p)
            d_mens = d_all[d_all["weightclass_encoded"].isin(MENS_WCS)]
            d_nobw = d_mens[d_mens["weightclass_encoded"]!=10]   # drop BW

            r_all  = roi_stats(d_all)
            r_ev   = roi_stats(d_all[d_all["ev"]>0])
            r_m_ev = roi_stats(d_mens[d_mens["ev"]>0])
            r_n_ev = roi_stats(d_nobw[d_nobw["ev"]>0])

            auc_s = f"{auc:.4f}" if auc is not None else "   —"
            print(f"  acc={acc:.2f}%  ll={ll:.4f}  auc={auc_s}  brier={bri:.4f}  active={n_act}")
            def fmt(s): return f"{s['roi']:>+7.1f}%({s['n']:>3})" if s["roi"] is not None else "          —"
            print(f"  ROI → all={fmt(r_all)}  +EV(all)={fmt(r_ev)}  +EV(mens)={fmt(r_m_ev)}  +EV(no-BW)={fmt(r_n_ev)}")

            fold_results[fold_name] = dict(
                train_end=train_end_str, test_end=test_end_str,
                n_train=len(train), n_test=len(test), n_active=n_act,
                acc=acc, ll=ll, auc=auc, brier=bri,
                roi_all=r_all, roi_ev=r_ev,
                roi_mens_ev=r_m_ev, roi_nobw_ev=r_n_ev,
            )
            all_preds.append(test.assign(p_pred=p))

        # ── Pooled stats across all folds ─────────────────────────────────
        print("\n" + "="*78)
        print("POOLED RESULTS  (all 6 folds concatenated)")
        print("="*78)

        df_all_preds = pd.concat(all_preds, ignore_index=True)
        y_pool = df_all_preds["win"].astype(int).values
        p_pool = df_all_preds["p_pred"].values

        pc   = np.clip(p_pool, EPS, 1-EPS)
        acc  = float(accuracy_score(y_pool,(p_pool>=0.5).astype(int))*100)
        ll   = float(log_loss(y_pool,pc))
        auc  = float(roc_auc_score(y_pool,p_pool))
        bri  = float(brier_score_loss(y_pool,pc))
        print(f"  Rolling retrain:  acc={acc:.2f}%  ll={ll:.4f}  auc={auc:.4f}  brier={bri:.4f}")
        print(f"  Static baseline:  acc=69.82%  ll=0.5967  auc=0.7543  brier=0.2047")
        print(f"  Delta:            acc={acc-69.82:>+.2f}pp  ll={ll-0.5967:>+.4f}  "
              f"auc={auc-0.7543:>+.4f}  brier={bri-0.2047:>+.4f}")

        # Pooled ROI
        d_pool     = build_bet_frame(df_all_preds.rename(columns={"p_pred":"p_raw_override"}),
                                      p_pool)
        # rebuild properly
        d_pool2    = build_bet_frame(df_all_preds, p_pool)
        d_mens2    = d_pool2[d_pool2["weightclass_encoded"].isin(MENS_WCS)]
        d_nobw2    = d_mens2[d_mens2["weightclass_encoded"]!=10]

        r_all2  = roi_stats(d_pool2)
        r_ev2   = roi_stats(d_pool2[d_pool2["ev"]>0])
        r_mev2  = roi_stats(d_mens2[d_mens2["ev"]>0])
        r_nev2  = roi_stats(d_nobw2[d_nobw2["ev"]>0])

        def fmt(s): return f"{s['roi']:>+7.1f}%({s['n']:>3})" if s["roi"] is not None else "          —"
        print(f"\n  ROI rolling retrain:")
        print(f"    all bets:       {fmt(r_all2)}")
        print(f"    +EV (all WCs):  {fmt(r_ev2)}")
        print(f"    +EV (mens):     {fmt(r_mev2)}")
        print(f"    +EV (no BW):    {fmt(r_nev2)}")
        print(f"\n  ROI static baseline (from mens_only_model.py):")
        print(f"    all bets:          +4.7%(386)")
        print(f"    +EV (all WCs):     +5.9%(146)")
        print(f"    +EV (mens):       +10.7%(123)")
        print(f"    +EV (no BW):      +19.0%( 97)")

        # ── Quarter-by-quarter comparison ─────────────────────────────────
        print("\n" + "="*78)
        print("QUARTER-BY-QUARTER COMPARISON  (rolling vs static)")
        print("="*78)
        print(f"  {'Quarter':<10}  {'Static +EV ROI':>15}  {'Rolling +EV ROI':>16}  "
              f"{'Delta':>8}  {'n bets':>7}")
        # Static per-quarter +EV ROI from previous run
        static_ev = {
            "Q4-2024": (+9.8, 12),   # from mens_only_model quarterly (men's +EV)
            "Q1-2025": (+6.9, 11),
            "Q2-2025": (+7.0, 19),
            "Q3-2025": (+10.4, 17),
            "Q4-2025": (+7.2, 17),
            "Q1-2026": (-5.3, 11),
        }
        # actually use the per-quarter +EV from per_wc filter (all men's)
        # from mens_only_model: men's +EV qtrs: +9.8%, +6.9%, +7.0%, +10.4%, +7.2%, -5.3%
        for fold_name, train_end_str, test_end_str in FOLDS:
            if fold_name not in fold_results:
                continue
            fr  = fold_results[fold_name]
            rev = fr["roi_mens_ev"]
            static_r, static_n = static_ev.get(fold_name, (None, 0))
            delta = (rev["roi"] - static_r) if (rev["roi"] is not None and static_r is not None) else None
            def f(v,n): return f"{v:>+7.1f}%({n:>3})" if v is not None else "            —"
            d_str = f"{delta:>+7.1f}pp" if delta is not None else "        —"
            print(f"  {fold_name:<10}  {f(static_r,static_n)}       {f(rev['roi'],rev['n'])}  {d_str}")

        # ── Save ──────────────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        Path("results/rolling_retrain_clean.json").write_text(json.dumps({
            "config": dict(C=C, l1_ratio=L1_RATIO, lam=LAM),
            "folds": fold_results,
            "pooled": dict(acc=acc, ll=ll, auc=auc, brier=bri,
                           roi_all=r_all2, roi_ev=r_ev2,
                           roi_mens_ev=r_mev2, roi_nobw_ev=r_nev2),
            "runtime_min": round((time.time()-t0)/60,1),
        }, indent=2, default=str))
        print(f"\n✓ Saved results/rolling_retrain_clean.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
