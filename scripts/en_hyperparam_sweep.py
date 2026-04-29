"""Sweep EN hyperparameters C × l1_ratio on the clean market-features pipeline.

C        ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}
l1_ratio ∈ {0.3, 0.5, 0.7}
λ = 1.20 (fixed — isolated from lambda sweep)

⚠ Exploratory — C/l1_ratio swept on held-out test set.
  Treat best cell as a hypothesis to pre-register, not a validated result.

Audit:   docs/audits/en_hyperparam_sweep.md
Outputs: results/en_hyperparam_sweep.json
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

C_VALUES   = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
L1_VALUES  = [0.3, 0.5, 0.7]
LAM        = 1.20
TRAIN_START= pd.Timestamp("2016-01-01")
TRAIN_END  = pd.Timestamp("2024-10-01")
TEST_END   = pd.Timestamp("2026-04-01")
THRESHOLD  = 3
EPS        = 1e-6

NOVEL = ["home_advantage_diff","travel_distance_diff_km","tz_diff_diff_hr",
         "is_main_event","card_position_norm_career_diff"]

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


def load_dataset():
    df_full, stat_cols = build_through_step6()
    train_only = df_full[df_full["DATE"] < TRAIN_END].copy()
    priors     = mma.compute_wc_priors(train_only, stat_cols)
    df_adj     = mma.compute_adjperf(df_full, stat_cols, priors)
    result     = mma.assemble_features(df_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result     = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])

    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_ensweep")
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
    feats_baseline = select_features(df)

    mf = pd.read_csv("data/tmp/market_features_clean.csv", parse_dates=["DATE"])
    avail = [c for c in NOVEL if c in mf.columns]
    df = df.merge(mf[["DATE","jbout","jfighter"]+avail],
                  on=["DATE","jbout","jfighter"], how="left")
    new_feats = [c for c in avail if c not in feats_baseline and df[c].std()>1e-8]
    usable = [c for c in feats_baseline+new_feats
              if c in df.columns and df[c].std()>1e-8]

    train = df[(df["DATE"]>=TRAIN_START)&(df["DATE"]<TRAIN_END)].copy()
    test  = df[(df["DATE"]>=TRAIN_END)  &(df["DATE"]<TEST_END)].copy()
    train = train.sort_values("DATE").reset_index(drop=True)

    assert train["DATE"].max() < TRAIN_END
    assert test["DATE"].min()  >= TRAIN_END
    assert not (set(zip(train["DATE"],train["jbout"])) &
                set(zip(test["DATE"], test["jbout"])))

    return train, test, usable, backup, feats_csv


def fit_predict(train, test, usable, C, l1_ratio):
    from retrain_lr_symmetric import flip_row_dataframe
    td = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    imp=SimpleImputer(strategy="median"); sc=StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
    w   = np.exp(-LAM*(TRAIN_END-td["DATE"]).dt.days.values/365.25)
    lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=l1_ratio,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    n_active = int((np.abs(lr.coef_[0])>1e-8).sum())
    Xte = sc.transform(imp.transform(test[usable]))
    return lr.predict_proba(Xte)[:,1], n_active, lr.coef_[0], usable


def main():
    t0 = time.time()
    print("="*78)
    print("EN HYPERPARAMETER SWEEP  C × l1_ratio  (λ=1.2 fixed)")
    print(f"  C values:       {C_VALUES}")
    print(f"  l1_ratio values:{L1_VALUES}")
    print("  ⚠  exploratory — swept on test set")
    print("  Audit: docs/audits/en_hyperparam_sweep.md")
    print("="*78)

    print("\nBuilding pipeline (once)...")
    train, test, usable, backup, feats_csv = load_dataset()
    y_full = test["win"].astype(int).values
    print(f"  Train:{len(train):,}  Test:{len(test):,}  Candidates:{len(usable)}")

    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    tv = attach_vegas_rich(test[["DATE","jbout","jfighter"]].drop_duplicates())
    test_v = test.merge(
        tv[["DATE","jbout","jfighter","p_vegas_f1","dec_odds_f1","dec_odds_f2"]],
        on=["DATE","jbout","jfighter"], how="left"
    )
    test_v = test_v[test_v["p_vegas_f1"].notna()].copy()

    try:
        all_results = {}

        # ── Header ───────────────────────────────────────────────────────
        print(f"\n{'C':>5}  {'l1':>4}  {'act':>4}  "
              f"{'acc':>6}  {'ll':>6}  {'auc':>6}  {'brier':>6}  "
              f"{'ROI_all':>8}  {'ROI_+EV':>8}  {'ROI_p65':>8}  "
              + "  ".join(f"{q[0]:>8}" for q in QUARTERS)
              + "  pos")
        print("-"*(5+4+4+6+6+6+6+8+8+8+5 + 10*6 + 10))

        for C in C_VALUES:
            for l1 in L1_VALUES:
                p, n_act, coefs, feats = fit_predict(train, test, usable, C, l1)

                # Pooled accuracy metrics
                pc   = np.clip(p, EPS, 1-EPS)
                acc  = float(accuracy_score(y_full,(p>=0.5).astype(int))*100)
                ll   = float(log_loss(y_full, pc))
                auc  = float(roc_auc_score(y_full, p))
                brier= float(brier_score_loss(y_full, pc))

                # Build deduped betting frame
                p_map = (test.assign(p_raw=p)
                             .drop_duplicates(subset=["DATE","jbout"])
                             [["DATE","jbout","p_raw"]])
                d = test_v.merge(p_map, on=["DATE","jbout"], how="left")
                d = d.dropna(subset=["p_raw"])
                d = d.sort_values("p_raw",ascending=False)\
                     .drop_duplicates(subset=["DATE","jbout"]).reset_index(drop=True)

                d["pick_a"]        = (d["p_raw"]>=0.5).astype(int)
                d["dec_odds_pick"] = np.where(d["pick_a"]==1,d["dec_odds_f1"],d["dec_odds_f2"])
                d["p_pick"]        = np.where(d["pick_a"]==1,d["p_raw"],      1-d["p_raw"])
                d["p_vegas_pick"]  = np.where(d["pick_a"]==1,d["p_vegas_f1"], 1-d["p_vegas_f1"])
                d["edge"]          = d["p_pick"]-d["p_vegas_pick"]
                d["ev"]            = d["p_pick"]*d["dec_odds_pick"]-1.0
                y2 = d["win"].astype(int).values
                d["won_pick"] = np.where(d["pick_a"]==1,y2==1,y2==0).astype(int)

                def roi(sub):
                    if len(sub)==0: return None
                    return float(np.where(sub["won_pick"]==1,
                                         sub["dec_odds_pick"]-1,-1.0).mean()*100)

                r_all = roi(d)
                r_ev  = roi(d[d["ev"]>0])
                r_p65 = roi(d[(d["ev"]>0)&(d["p_pick"]>=0.65)])

                q_rois=[]
                for _,qs,qe in QUARTERS:
                    dq = d[(d["DATE"]>=qs)&(d["DATE"]<qe)]
                    q_rois.append(roi(dq))
                pos = sum(1 for r in q_rois if r is not None and r>0)

                key = f"C={C}_l1={l1}"
                all_results[key] = dict(
                    C=C, l1_ratio=l1, n_active=n_act,
                    acc=acc, ll=ll, auc=auc, brier=brier,
                    roi_all=r_all, roi_ev=r_ev, roi_p65=r_p65,
                    quarters={q[0]:r for q,r in zip(QUARTERS,q_rois)},
                    pos_quarters=pos,
                )

                q_str = "  ".join(
                    f"{r:>+7.1f}%" if r is not None else f"{'—':>8}" for r in q_rois
                )
                cur = " ←" if C==0.05 and l1==0.5 else ""
                def fmt(v): return f"{v:>+7.1f}%" if v is not None else f"{'—':>8}"
                print(f"{C:>5g}  {l1:>4.1f}  {n_act:>4d}  "
                      f"{acc:>5.2f}%  {ll:>6.4f}  {auc:>6.4f}  {brier:>6.4f}  "
                      f"{fmt(r_all)}  {fmt(r_ev)}  {fmt(r_p65)}  "
                      f"{q_str}  {pos}/6{cur}")

        # ── Summary tables ────────────────────────────────────────────────
        print()
        print("="*78)
        print("BEST CELL PER METRIC  (exploratory)")
        print("="*78)
        for metric, key, better in [
            ("Accuracy  ↑","acc",max), ("Log loss  ↓","ll",min),
            ("AUC       ↑","auc",max), ("Brier     ↓","brier",min),
            ("ROI all   ↑","roi_all",max), ("ROI +EV   ↑","roi_ev",max),
            ("ROI p≥0.65↑","roi_p65",max),
            ("Pos qtrs  ↑","pos_quarters",max),
        ]:
            vals = {k:v[key] for k,v in all_results.items() if v[key] is not None}
            best_k = better(vals, key=lambda k: vals[k])
            bv = all_results[best_k]
            print(f"  {metric:<14}  C={bv['C']:<5g}  l1={bv['l1_ratio']:.1f}  "
                  f"active={bv['n_active']:>3d}  value={vals[best_k]:.4f}")

        # ── Active feature count vs C ─────────────────────────────────────
        print()
        print("Active features by C (l1_ratio=0.5):")
        print(f"  {'C':>6}  {'active':>7}  {'zeroed':>7}  {'acc':>7}  {'ll':>7}  {'auc':>7}")
        for C in C_VALUES:
            r = all_results[f"C={C}_l1=0.5"]
            zeroed = len(usable) - r["n_active"]
            print(f"  {C:>6g}  {r['n_active']:>7d}  {zeroed:>7d}  "
                  f"{r['acc']:>6.2f}%  {r['ll']:>7.4f}  {r['auc']:>7.4f}")

        Path("results").mkdir(exist_ok=True)
        Path("results/en_hyperparam_sweep.json").write_text(
            json.dumps({"C_values":C_VALUES,"l1_values":L1_VALUES,
                        "lambda":LAM,"n_candidates":len(usable),
                        "warning":"exploratory — swept on test set",
                        "results":all_results,
                        "audit":"docs/audits/en_hyperparam_sweep.md",
                        "runtime_min":round((time.time()-t0)/60,1)},
                       indent=2, default=str))
        print(f"\n✓ Saved results/en_hyperparam_sweep.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__=="__main__":
    main()
