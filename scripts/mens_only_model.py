"""Men's-only EN model — ROI hypothesis test.

Trains a men's-only EN (weightclass_encoded ∈ {5..12}) and compares:
  1. Men's-only EN  vs  Mixed EN  on men's test fights
  2. Mixed EN ROI on men's-only  vs  mixed EN ROI on all fights

Tests hypothesis: women's fights drag down ROI; removing them improves betting.

Audit:   docs/audits/mens_only_model.md
Outputs: results/mens_only_model.json
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
TRAIN_END  = pd.Timestamp("2024-10-01")
TEST_END   = pd.Timestamp("2026-04-01")
THRESHOLD  = 3
EPS        = 1e-6

NOVEL = ["home_advantage_diff","travel_distance_diff_km","tz_diff_diff_hr",
         "is_main_event","card_position_norm_career_diff"]

MENS_WCS   = {5,6,7,8,9,10,11,12}
WOMENS_WCS = {1,2,3,4}
WC_NAMES   = {5:"LHW",6:"MW",7:"WW",8:"LW",9:"FW",10:"BW",11:"FlyW",12:"StraW"}

QUARTERS = [
    ("Q4-2024","2024-10-01","2025-01-01"),
    ("Q1-2025","2025-01-01","2025-04-01"),
    ("Q2-2025","2025-04-01","2025-07-01"),
    ("Q3-2025","2025-07-01","2025-10-01"),
    ("Q4-2025","2025-10-01","2026-01-01"),
    ("Q1-2026","2026-01-01","2026-04-01"),
]


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


def load_full_dataset():
    df_full, stat_cols = build_through_step6()
    train_only = df_full[df_full["DATE"] < TRAIN_END].copy()
    priors     = mma.compute_wc_priors(train_only, stat_cols)
    df_adj     = mma.compute_adjperf(df_full, stat_cols, priors)
    result     = mma.assemble_features(df_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result     = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])

    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_mens")
    if feats_csv.exists() and not backup.exists():
        import shutil; shutil.copy2(feats_csv, backup)
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

    feats_baseline = select_features(df)  # MUST be before merge

    mf = pd.read_csv("data/tmp/market_features_clean.csv", parse_dates=["DATE"])
    avail = [c for c in NOVEL if c in mf.columns]
    df = df.merge(mf[["DATE","jbout","jfighter"]+avail],
                  on=["DATE","jbout","jfighter"], how="left")
    new_feats = [c for c in avail if c not in feats_baseline and df[c].std()>1e-8]
    usable = [c for c in feats_baseline+new_feats
              if c in df.columns and df[c].std()>1e-8]

    if any(c in feats_baseline for c in new_feats):
        raise RuntimeError("ORDERING BUG: market cols already in baseline")

    return df, usable, backup, feats_csv


def fit_en(train_df, usable):
    from retrain_lr_symmetric import flip_row_dataframe
    td  = pd.concat([train_df, flip_row_dataframe(train_df)], ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
    w   = np.exp(-LAM*(TRAIN_END-td["DATE"]).dt.days.values/365.25)
    lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1_RATIO,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    return lr, imp, sc


def predict(lr, imp, sc, test_df, usable):
    return lr.predict_proba(sc.transform(imp.transform(test_df[usable])))[:,1]


def metrics(y, p):
    pc = np.clip(p, EPS, 1-EPS)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y))>1 else None
    return dict(acc=float(accuracy_score(y,(p>=0.5).astype(int))*100),
                ll=float(log_loss(y,pc)), auc=auc,
                brier=float(brier_score_loss(y,pc)), n=int(len(y)))


def roi_table(test_df, p):
    """Full ROI breakdown: all, +EV, p65, per quarter."""
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
    d["edge"]          = d["p_pick"]-d["p_vegas_pick"]
    d["ev"]            = d["p_pick"]*d["dec_odds_pick"]-1.0
    y2 = d["win"].astype(int).values
    d["won_pick"]      = np.where(d["pick_a"]==1,y2==1,y2==0).astype(int)

    def roi(sub):
        if len(sub)==0: return None
        return float(np.where(sub["won_pick"]==1,
                               sub["dec_odds_pick"]-1,-1.0).mean()*100)

    def n_bets(sub): return int(len(sub))

    r_all  = roi(d);      n_all  = n_bets(d)
    r_ev   = roi(d[d["ev"]>0]);       n_ev   = n_bets(d[d["ev"]>0])
    r_p65  = roi(d[(d["ev"]>0)&(d["p_pick"]>=0.65)])
    n_p65  = n_bets(d[(d["ev"]>0)&(d["p_pick"]>=0.65)])

    q_rois = {}
    for qn,qs,qe in QUARTERS:
        dq = d[(d["DATE"]>=qs)&(d["DATE"]<qe)]
        q_rois[qn] = {"roi": roi(dq), "n": n_bets(dq)}

    pos = sum(1 for v in q_rois.values() if v["roi"] is not None and v["roi"]>0)

    return dict(n_vegas=len(d),
                roi_all=r_all, n_all=n_all,
                roi_ev=r_ev,   n_ev=n_ev,
                roi_p65=r_p65, n_p65=n_p65,
                quarters=q_rois, pos_quarters=pos)


def print_roi_row(label, r):
    def fmt(v,n): return f"{v:>+7.1f}% ({n:>3})" if v is not None else "         —"
    q_str = "  ".join(
        f"{v['roi']:>+6.1f}%" if v["roi"] is not None and v["n"]>0 else "      —"
        for v in r["quarters"].values()
    )
    print(f"  {label:<28}  all={fmt(r['roi_all'],r['n_all'])}  "
          f"+EV={fmt(r['roi_ev'],r['n_ev'])}  "
          f"p65={fmt(r['roi_p65'],r['n_p65'])}  "
          f"pos={r['pos_quarters']}/6")
    print(f"  {'':28}  qtrs: {q_str}")


def main():
    t0 = time.time()
    print("="*78)
    print("MEN'S-ONLY EN MODEL — ROI HYPOTHESIS TEST")
    print(f"  Hypothesis: women's fights drag down ROI")
    print(f"  C={C}, l1_ratio={L1_RATIO}, λ={LAM}")
    print("="*78)

    print("\nBuilding pipeline...")
    df_all, usable, backup, feats_csv = load_full_dataset()
    wc_col = "weightclass_encoded"
    print(f"  Full frame: {len(df_all):,} rows, {len(usable)} features")

    try:
        # ── Subsets ────────────────────────────────────────────────────────
        df_m = df_all[df_all[wc_col].isin(MENS_WCS)].copy()
        df_w = df_all[df_all[wc_col].isin(WOMENS_WCS)].copy()

        train_all = df_all[(df_all["DATE"]>=TRAIN_START)&(df_all["DATE"]<TRAIN_END)].sort_values("DATE").reset_index(drop=True)
        test_all  = df_all[(df_all["DATE"]>=TRAIN_END) &(df_all["DATE"]<TEST_END)].reset_index(drop=True)

        train_m = df_m[(df_m["DATE"]>=TRAIN_START)&(df_m["DATE"]<TRAIN_END)].sort_values("DATE").reset_index(drop=True)
        test_m  = df_m[(df_m["DATE"]>=TRAIN_END) &(df_m["DATE"]<TEST_END)].reset_index(drop=True)

        test_w  = df_w[(df_w["DATE"]>=TRAIN_END) &(df_w["DATE"]<TEST_END)].reset_index(drop=True)

        assert train_m["DATE"].max() < TRAIN_END
        assert test_m["DATE"].min()  >= TRAIN_END
        assert not (set(zip(train_m["DATE"],train_m["jbout"])) &
                    set(zip(test_m["DATE"], test_m["jbout"])))

        print(f"\n  Men's   train: {len(train_m):,} rows  ({train_m['jbout'].nunique()} bouts)")
        print(f"  Men's   test:  {len(test_m):,} rows   ({test_m['jbout'].nunique()} bouts)")
        print(f"  Women's test:  {len(test_w):,} rows   ({test_w['jbout'].nunique()} bouts)")
        print(f"  Mixed   test:  {len(test_all):,} rows   ({test_all['jbout'].nunique()} bouts)")

        # ── Fit models ────────────────────────────────────────────────────
        print("\nFitting men's-only EN...")
        lr_m, imp_m, sc_m = fit_en(train_m, usable)
        n_act_m = int((np.abs(lr_m.coef_[0])>1e-8).sum())
        print(f"  Active: {n_act_m}/{len(usable)}")

        print("Fitting mixed EN...")
        lr_all, imp_all, sc_all = fit_en(train_all, usable)
        n_act_all = int((np.abs(lr_all.coef_[0])>1e-8).sum())
        print(f"  Active: {n_act_all}/{len(usable)}")

        # ── Predictions ───────────────────────────────────────────────────
        p_mens_on_mens  = predict(lr_m,   imp_m,   sc_m,   test_m,   usable)
        p_mixed_on_mens = predict(lr_all, imp_all, sc_all, test_m,   usable)
        p_mixed_on_all  = predict(lr_all, imp_all, sc_all, test_all, usable)

        # Also get mixed on women's for the full picture
        p_mixed_on_womens = predict(lr_all, imp_all, sc_all, test_w, usable)

        # ── Accuracy metrics ──────────────────────────────────────────────
        y_m   = test_m["win"].astype(int).values
        y_all = test_all["win"].astype(int).values
        y_w   = test_w["win"].astype(int).values

        m_mens_on_mens  = metrics(y_m,   p_mens_on_mens)
        m_mixed_on_mens = metrics(y_m,   p_mixed_on_mens)
        m_mixed_on_all  = metrics(y_all, p_mixed_on_all)
        m_mixed_on_wom  = metrics(y_w,   p_mixed_on_womens)

        print("\n" + "="*78)
        print("ACCURACY METRICS")
        print("="*78)
        print(f"  {'Model / Eval subset':<34}  {'N':>5}  {'Acc':>7}  "
              f"{'LL':>7}  {'AUC':>7}  {'Brier':>7}")
        print(f"  {'-'*34}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for name, m in [
            ("Men's EN  → men's test",   m_mens_on_mens),
            ("Mixed EN  → men's test",   m_mixed_on_mens),
            ("Mixed EN  → all test",     m_mixed_on_all),
            ("Mixed EN  → women's test", m_mixed_on_wom),
        ]:
            auc_str = f"{m['auc']:>7.4f}" if m['auc'] is not None else "      —"
            print(f"  {name:<34}  {m['n']:>5}  {m['acc']:>6.2f}%  "
                  f"{m['ll']:>7.4f}  {auc_str}  {m['brier']:>7.4f}")

        delta_acc = m_mens_on_mens["acc"] - m_mixed_on_mens["acc"]
        delta_ll  = m_mens_on_mens["ll"]  - m_mixed_on_mens["ll"]
        print(f"\n  Men's EN − Mixed EN (on men's): "
              f"acc {delta_acc:>+.2f}pp  ll {delta_ll:>+.4f}")

        # ── ROI ───────────────────────────────────────────────────────────
        print("\n" + "="*78)
        print("ROI ANALYSIS")
        print("="*78)

        r_mens_on_mens  = roi_table(test_m,   p_mens_on_mens)
        r_mixed_on_mens = roi_table(test_m,   p_mixed_on_mens)
        r_mixed_on_all  = roi_table(test_all, p_mixed_on_all)
        r_mixed_on_wom  = roi_table(test_w,   p_mixed_on_womens)

        for label, r in [
            ("Men's EN  → men's bets",   r_mens_on_mens),
            ("Mixed EN  → men's bets",   r_mixed_on_mens),
            ("Mixed EN  → all bets",     r_mixed_on_all),
            ("Mixed EN  → women's bets", r_mixed_on_wom),
        ]:
            print_roi_row(label, r)
            print()

        # ── Per quarter comparison: mixed-all vs mixed-mens ───────────────
        print("="*78)
        print("QUARTERLY ROI  —  Mixed EN:  all bets vs men's-only bets")
        print("="*78)
        print(f"  {'Quarter':<10}  {'All n':>6}  {'All ROI':>8}  "
              f"{'Men n':>6}  {'Men ROI':>8}  {'Delta':>8}")
        for qn in [q[0] for q in QUARTERS]:
            ra = r_mixed_on_all["quarters"].get(qn,  {"roi":None,"n":0})
            rm = r_mixed_on_mens["quarters"].get(qn, {"roi":None,"n":0})
            def f(v): return f"{v:>+7.1f}%" if v is not None else "       —"
            delta_q = (rm["roi"] - ra["roi"]) if (rm["roi"] is not None and ra["roi"] is not None) else None
            print(f"  {qn:<10}  {ra['n']:>6}  {f(ra['roi'])}  "
                  f"{rm['n']:>6}  {f(rm['roi'])}  "
                  f"{f(delta_q)}")

        # ── Save ──────────────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        Path("results/mens_only_model.json").write_text(json.dumps({
            "config": dict(C=C, l1_ratio=L1_RATIO, lam=LAM),
            "n_train": {"mens": len(train_m), "mixed": len(train_all)},
            "n_test":  {"mens": len(test_m),  "mixed": len(test_all), "womens": len(test_w)},
            "n_active": {"mens_en": n_act_m, "mixed_en": n_act_all},
            "metrics": {
                "mens_en_on_mens":   m_mens_on_mens,
                "mixed_en_on_mens":  m_mixed_on_mens,
                "mixed_en_on_all":   m_mixed_on_all,
                "mixed_en_on_womens":m_mixed_on_wom,
            },
            "roi": {
                "mens_en_on_mens":   r_mens_on_mens,
                "mixed_en_on_mens":  r_mixed_on_mens,
                "mixed_en_on_all":   r_mixed_on_all,
                "mixed_en_on_womens":r_mixed_on_wom,
            },
            "runtime_min": round((time.time()-t0)/60,1),
        }, indent=2, default=str))
        print(f"\n✓ Saved results/mens_only_model.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
