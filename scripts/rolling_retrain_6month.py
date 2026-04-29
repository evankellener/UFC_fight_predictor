"""6-month expanding-window rolling retrain on the clean pipeline.

3 folds of 6 months each:
  Fold 1: train < 2024-10-01  test 2024-10-01 → 2025-04-01
  Fold 2: train < 2025-04-01  test 2025-04-01 → 2025-10-01
  Fold 3: train < 2025-10-01  test 2025-10-01 → 2026-04-01
           └─ Q1-2026 model only 3 months stale (vs 15 months static)

Per-quarter breakdown within each fold for apples-to-apples vs static/quarterly.

Audit:   docs/audits/rolling_retrain_6month.md
Outputs: results/rolling_retrain_6month.json
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

C          = 0.05
L1_RATIO   = 0.5
LAM        = 1.20
TRAIN_START= pd.Timestamp("2016-01-01")
EPS        = 1e-6
THRESHOLD  = 3

NOVEL = ["home_advantage_diff","travel_distance_diff_km","tz_diff_diff_hr",
         "is_main_event","card_position_norm_career_diff"]

MENS_WCS = {5,6,7,8,9,10,11,12}

FOLDS = [
    ("H1",  "2024-10-01", "2025-04-01"),
    ("H2",  "2025-04-01", "2025-10-01"),
    ("H3",  "2025-10-01", "2026-04-01"),
]

QUARTERS = [
    ("Q4-2024","2024-10-01","2025-01-01"),
    ("Q1-2025","2025-01-01","2025-04-01"),
    ("Q2-2025","2025-04-01","2025-07-01"),
    ("Q3-2025","2025-07-01","2025-10-01"),
    ("Q4-2025","2025-10-01","2026-01-01"),
    ("Q1-2026","2026-01-01","2026-04-01"),
]

# Static model per-quarter +EV ROI on men's (from mens_only_model.py)
STATIC_Q = {
    "Q4-2024": (+9.8,  12),
    "Q1-2025": (+6.9,  11),
    "Q2-2025": (+7.0,  19),
    "Q3-2025": (+10.4, 17),
    "Q4-2025": (+7.2,  17),
    "Q1-2026": (-5.3,  11),
}


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


def build_fold_features(df_full, stat_cols, train_end_ts, feats_csv, mf):
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
    df   = apply_threshold(base, THRESHOLD)
    df   = add_wc_features(df, load_wc_history_from_db())
    feats_baseline = select_features(df)   # BEFORE market merge

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
    w   = np.exp(-LAM*(train_end_ts-td["DATE"]).dt.days.values/365.25)
    lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1_RATIO,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    n_act = int((np.abs(lr.coef_[0])>1e-8).sum())
    Xte   = sc.transform(imp.transform(test_df_placeholder := td[usable][:0]))  # placeholder
    return lr, imp, sc, n_act


def predict(lr, imp, sc, df, usable):
    return lr.predict_proba(sc.transform(imp.transform(df[usable])))[:,1]


def roi_stats(sub):
    if len(sub)==0: return {"roi":None,"n":0}
    pnl = np.where(sub["won_pick"]==1, sub["dec_odds_pick"]-1, -1.0)
    return {"roi":float(pnl.mean()*100), "n":int(len(sub))}


def bet_frame(test_df, p):
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
    print("6-MONTH ROLLING RETRAIN  (3 folds)")
    print(f"  C={C}  l1={L1_RATIO}  λ={LAM}")
    print("="*78)

    print("\nBuilding steps 1–6 (once)...")
    df_full, stat_cols = build_through_step6()
    print(f"  Full frame: {len(df_full):,} rows")

    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_6m")
    if feats_csv.exists() and not backup.exists():
        import shutil; shutil.copy2(feats_csv, backup)

    mf = pd.read_csv("data/tmp/market_features_clean.csv", parse_dates=["DATE"])

    try:
        fold_results = {}
        all_preds    = []

        for fold_name, train_end_str, test_end_str in FOLDS:
            train_end_ts = pd.Timestamp(train_end_str)
            test_end_ts  = pd.Timestamp(test_end_str)

            print(f"\n{'─'*60}")
            print(f"  Fold {fold_name}: train < {train_end_str}  →  test {train_end_str}–{test_end_str}")

            df, usable = build_fold_features(df_full, stat_cols,
                                              train_end_ts, feats_csv, mf)

            train = df[(df["DATE"]>=TRAIN_START)&(df["DATE"]<train_end_ts)].sort_values("DATE").reset_index(drop=True)
            test  = df[(df["DATE"]>=train_end_ts)&(df["DATE"]<test_end_ts)].reset_index(drop=True)

            assert train["DATE"].max() < train_end_ts
            assert test["DATE"].min()  >= train_end_ts
            assert not (set(zip(train["DATE"],train["jbout"])) &
                        set(zip(test["DATE"],test["jbout"])))

            print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(usable)}")

            # Fit
            from retrain_lr_symmetric import flip_row_dataframe
            td  = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
            imp = SimpleImputer(strategy="median")
            sc  = StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
            w   = np.exp(-LAM*(train_end_ts-td["DATE"]).dt.days.values/365.25)
            lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1_RATIO,
                                     solver="saga", max_iter=8000, random_state=42)
            lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
            n_act = int((np.abs(lr.coef_[0])>1e-8).sum())

            p = lr.predict_proba(sc.transform(imp.transform(test[usable])))[:,1]
            y = test["win"].astype(int).values

            pc  = np.clip(p, EPS, 1-EPS)
            acc = float(accuracy_score(y,(p>=0.5).astype(int))*100)
            ll  = float(log_loss(y,pc))
            auc = float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None
            bri = float(brier_score_loss(y,pc))
            auc_s = f"{auc:.4f}" if auc is not None else "—"
            print(f"  acc={acc:.2f}%  ll={ll:.4f}  auc={auc_s}  active={n_act}")

            d     = bet_frame(test, p)
            d_m   = d[d["weightclass_encoded"].isin(MENS_WCS)]
            d_nobw= d_m[d_m["weightclass_encoded"]!=10]

            # Per-quarter breakdown within fold
            q_details = {}
            for qn,qs,qe in QUARTERS:
                dq    = d[(d["DATE"]>=qs)&(d["DATE"]<qe)]
                dq_m  = dq[dq["weightclass_encoded"].isin(MENS_WCS)]
                dq_nb = dq_m[dq_m["weightclass_encoded"]!=10]
                if len(dq)==0: continue
                q_details[qn] = dict(
                    all   = roi_stats(dq),
                    mens_ev  = roi_stats(dq_m[dq_m["ev"]>0]),
                    nobw_ev  = roi_stats(dq_nb[dq_nb["ev"]>0]),
                )

            fold_results[fold_name] = dict(
                train_end=train_end_str, test_end=test_end_str,
                n_train=len(train), n_test=len(test), n_active=n_act,
                acc=acc, ll=ll, auc=auc, brier=bri,
                roi_all  = roi_stats(d),
                roi_ev   = roi_stats(d[d["ev"]>0]),
                roi_m_ev = roi_stats(d_m[d_m["ev"]>0]),
                roi_nb_ev= roi_stats(d_nobw[d_nobw["ev"]>0]),
                quarters = q_details,
            )
            all_preds.append(test.assign(p_pred=p))

        # ── Pooled ────────────────────────────────────────────────────────
        df_pool = pd.concat(all_preds, ignore_index=True)
        y_p = df_pool["win"].astype(int).values
        p_p = df_pool["p_pred"].values
        pc  = np.clip(p_p, EPS, 1-EPS)
        acc_p = float(accuracy_score(y_p,(p_p>=0.5).astype(int))*100)
        ll_p  = float(log_loss(y_p,pc))
        auc_p = float(roc_auc_score(y_p,p_p))
        bri_p = float(brier_score_loss(y_p,pc))

        d_p    = bet_frame(df_pool, p_p)
        d_pm   = d_p[d_p["weightclass_encoded"].isin(MENS_WCS)]
        d_pnb  = d_pm[d_pm["weightclass_encoded"]!=10]

        print("\n" + "="*78)
        print("POOLED  (all 3 folds)")
        print("="*78)
        print(f"  {'Model':<28}  {'Acc':>7}  {'LL':>7}  {'AUC':>7}  {'Brier':>7}")
        print(f"  {'6-month rolling':<28}  {acc_p:>6.2f}%  {ll_p:>7.4f}  {auc_p:>7.4f}  {bri_p:>7.4f}")
        print(f"  {'Static baseline':<28}  {'69.82':>6}%  {'0.5967':>7}  {'0.7543':>7}  {'0.2047':>7}")
        print(f"  {'Quarterly rolling':<28}  {'67.77':>6}%  {'0.5955':>7}  {'0.7530':>7}  {'0.2045':>7}")

        def fmt(s): return f"{s['roi']:>+7.1f}%({s['n']:>3})" if s["roi"] is not None else "           —"
        print(f"\n  {'Strategy':<28}  {'6-month':>16}  {'Static':>16}  {'Quarterly':>16}")
        print(f"  {'-'*28}  {'-'*16}  {'-'*16}  {'-'*16}")

        r_rows = [
            ("all bets",    roi_stats(d_p),                      "+4.7%(386)",  "+0.2%(386)"),
            ("+EV all WCs", roi_stats(d_p[d_p["ev"]>0]),         "+5.9%(146)",  "-1.9%(158)"),
            ("+EV mens",    roi_stats(d_pm[d_pm["ev"]>0]),        "+10.7%(123)", "-0.0%(130)"),
            ("+EV no-BW",   roi_stats(d_pnb[d_pnb["ev"]>0]),     "+19.0%( 97)", "+2.6%(105)"),
        ]
        for label, r, static_s, qtr_s in r_rows:
            print(f"  {label:<28}  {fmt(r):>16}  {static_s:>16}  {qtr_s:>16}")

        # ── Quarter-by-quarter comparison ─────────────────────────────────
        print("\n" + "="*78)
        print("QUARTER-BY-QUARTER  (+EV mens bets)")
        print("="*78)
        print(f"  {'Quarter':<10}  {'Static':>12}  {'Quarterly':>12}  {'6-month':>12}  {'Delta vs static':>16}")

        # Pull quarterly rolling from fold_results
        qtr_rolling = {
            "Q4-2024": (-5.3,  11),   # from quarterly results (mens +EV)  Actually: +10.7(17) in Q4 fold
            "Q1-2025": (+9.4,  29),
            "Q2-2025": (-28.3, 27),
            "Q3-2025": (+19.0, 20),
            "Q4-2025": (-4.0,  23),
            "Q1-2026": (+1.2,  14),
        }
        # Override with actual quarterly rolling results
        qtr_rolling = {
            "Q4-2024": (+10.7, 17),
            "Q1-2025": (+9.4,  29),
            "Q2-2025": (-28.3, 27),
            "Q3-2025": (+19.0, 20),
            "Q4-2025": (-4.0,  23),
            "Q1-2026": (+1.2,  14),
        }

        for qn,qs,qe in QUARTERS:
            # Find 6-month result for this quarter
            r6 = None
            for fn, fr in fold_results.items():
                if qn in fr["quarters"]:
                    r6 = fr["quarters"][qn]["mens_ev"]
                    break
            stat_r, stat_n = STATIC_Q.get(qn, (None,0))
            qtr_r,  qtr_n  = qtr_rolling.get(qn, (None,0))
            r6_r = r6["roi"] if r6 else None
            r6_n = r6["n"]   if r6 else 0
            delta = (r6_r - stat_r) if (r6_r is not None and stat_r is not None) else None

            def f(v,n): return f"{v:>+6.1f}%({n:>2})" if v is not None else "          —"
            d_str = f"{delta:>+6.1f}pp" if delta is not None else "          —"
            print(f"  {qn:<10}  {f(stat_r,stat_n)}  {f(qtr_r,qtr_n)}  {f(r6_r,r6_n)}  {d_str}")

        # ── Focus: Q1-2026 deep dive ──────────────────────────────────────
        print("\n" + "="*78)
        print("Q1-2026 DEEP DIVE  (Jan–Apr 2026)")
        print("="*78)
        q1_26 = None
        for fn, fr in fold_results.items():
            if "Q1-2026" in fr["quarters"]:
                q1_26 = fr["quarters"]["Q1-2026"]
                print(f"  From fold {fn} (model trained through {fr['train_end']})")
                break
        if q1_26:
            for label, r in [
                ("All bets", q1_26["all"]),
                ("+EV mens", q1_26["mens_ev"]),
                ("+EV no-BW", q1_26["nobw_ev"]),
            ]:
                def f(s): return f"{s['roi']:>+7.1f}%  ({s['n']} bets)" if s["roi"] is not None else "—"
                print(f"  {label:<14}: {f(r)}")
            print(f"\n  Static:    -5.3%  (11 bets)")
            print(f"  Quarterly: +1.2%  (14 bets)")
            m6_r = q1_26["mens_ev"]["roi"]
            if m6_r is not None:
                print(f"  6-month:  {m6_r:>+5.1f}%  ({q1_26['mens_ev']['n']} bets)")

        # ── Save ──────────────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        Path("results/rolling_retrain_6month.json").write_text(json.dumps({
            "config": dict(C=C, l1_ratio=L1_RATIO, lam=LAM),
            "folds": fold_results,
            "pooled": dict(acc=acc_p, ll=ll_p, auc=auc_p, brier=bri_p,
                           roi_all=roi_stats(d_p),
                           roi_ev=roi_stats(d_p[d_p["ev"]>0]),
                           roi_mens_ev=roi_stats(d_pm[d_pm["ev"]>0]),
                           roi_nobw_ev=roi_stats(d_pnb[d_pnb["ev"]>0])),
            "runtime_min": round((time.time()-t0)/60,1),
        }, indent=2, default=str))
        print(f"\n✓ Saved results/rolling_retrain_6month.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
