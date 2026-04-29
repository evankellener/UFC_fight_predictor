"""Hierarchical gender+WC prior for women's divisions.

Men's WC priors (weightindex 5–12): unchanged.
Women's WC priors (weightindex 1–4):
    hybrid_mean = α * per_wc_mean + (1-α) * pooled_women_mean
    hybrid_mad  = α * per_wc_mad  + (1-α) * pooled_women_mad
    α = n_wc / (n_wc + κ)   — more WC data → trust WC prior more

κ = 50 (pre-specified; ~25 bouts worth of shrinkage pressure).
  W.Feather   n≈9   → α≈0.15  (nearly all pooled-women prior)
  W.Bantam    n≈67  → α≈0.57  (blend)
  W.Fly       n≈107 → α≈0.68  (mostly WC)
  W.Straw     n≈113 → α≈0.69  (mostly WC)

Audit:   docs/audits/womens_hybrid_prior_model.md
Outputs: results/womens_hybrid_prior_model.json
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
KAPPA      = 50        # shrinkage constant
TRAIN_START= pd.Timestamp("2016-01-01")
TRAIN_END  = pd.Timestamp("2024-10-01")
TEST_END   = pd.Timestamp("2026-04-01")
THRESHOLD  = 3
EPS        = 1e-6

NOVEL = ["home_advantage_diff","travel_distance_diff_km","tz_diff_diff_hr",
         "is_main_event","card_position_norm_career_diff"]

WOMENS_WCS = {1, 2, 3, 4}
WC_NAMES   = {1:"W.Straw",2:"W.Fly",3:"W.Bantam",4:"W.Feather",
              5:"LHW",6:"MW",7:"WW",8:"LW",9:"FW",10:"BW",11:"FlyW",12:"StraW"}

QUARTERS = [
    ("Q4-2024","2024-10-01","2025-01-01"),
    ("Q1-2025","2025-01-01","2025-04-01"),
    ("Q2-2025","2025-04-01","2025-07-01"),
    ("Q3-2025","2025-07-01","2025-10-01"),
    ("Q4-2025","2025-10-01","2026-01-01"),
    ("Q1-2026","2026-01-01","2026-04-01"),
]


# ── Hybrid prior ────────────────────────────────────────────────────────────

def compute_hybrid_priors(train_df: pd.DataFrame, stat_cols: list,
                           kappa: float = KAPPA) -> dict:
    """
    Compute per-WC priors then shrink women's WC priors toward a pooled
    all-women prior.  Men's WC priors are untouched.
    """
    # Step 1: standard per-WC priors (same as mma_ai_pipeline.compute_wc_priors)
    priors = mma.compute_wc_priors(train_df, stat_cols)

    # Step 2: pooled "all-women" prior (train rows with weightindex ∈ {1,2,3,4})
    women_df = train_df[train_df["weightindex"].isin(WOMENS_WCS)]
    if len(women_df) == 0:
        print("  ⚠ No women's training rows — hybrid prior skipped")
        return priors

    # Count per-WC training rows for alpha computation
    wc_counts = women_df["weightindex"].value_counts().to_dict()

    # Compute pooled means/MADs for each stat (same formula as compute_wc_priors)
    pooled = {}
    for col in stat_cols:
        vals = women_df[col].dropna().values
        if len(vals) < 3:
            continue
        med = np.median(vals)
        pooled[col] = {
            "mean": float(np.mean(vals)),
            "mad":  float(max(np.median(np.abs(vals - med)), 1e-6)),
            "n":    int(len(vals)),
        }

    # Step 3: blend women's per-WC priors with pooled prior
    print(f"\n  Hybrid prior blending (κ={kappa}):")
    print(f"  {'WC':<12}  {'n_rows':>7}  {'alpha':>7}  (higher alpha = more WC-specific)")
    for wc in sorted(WOMENS_WCS):
        n_wc = wc_counts.get(wc, 0)
        alpha = n_wc / (n_wc + kappa)
        name  = WC_NAMES.get(wc, f"WC{wc}")
        print(f"  {name:<12}  {n_wc:>7}  {alpha:>7.3f}")

        for col in stat_cols:
            if col not in pooled:
                continue
            wc_prior    = priors[col].get(wc)
            pool_prior  = pooled[col]
            if wc_prior is None:
                # WC had < 3 rows — use pooled entirely
                priors[col][wc] = {"mean": pool_prior["mean"],
                                   "mad":  pool_prior["mad"]}
            else:
                priors[col][wc] = {
                    "mean": alpha * wc_prior["mean"] + (1-alpha) * pool_prior["mean"],
                    "mad":  alpha * wc_prior["mad"]  + (1-alpha) * pool_prior["mad"],
                }

    return priors


# ── Pipeline ─────────────────────────────────────────────────────────────────

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


def load_datasets():
    """
    Returns two full assembled feature frames:
      baseline_df  — standard per-WC priors (original pipeline)
      hybrid_df    — hybrid gender+WC priors for women's WCs
    Plus usable feature list, backup paths.
    """
    df_full, stat_cols = build_through_step6()
    train_only = df_full[df_full["DATE"] < TRAIN_END].copy()

    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_hybrid")

    results = {}
    for label, prior_fn in [
        ("baseline", lambda df, sc: mma.compute_wc_priors(df, sc)),
        ("hybrid",   lambda df, sc: compute_hybrid_priors(df, sc)),
    ]:
        print(f"\n=== Building pipeline: {label} priors ===")
        if feats_csv.exists() and not backup.exists():
            import shutil; shutil.copy2(feats_csv, backup)

        priors  = prior_fn(train_only, stat_cols)
        df_adj  = mma.compute_adjperf(df_full, stat_cols, priors)
        result  = mma.assemble_features(df_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
        result  = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
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

        results[label] = (df, usable)

    # Both runs should produce same usable list
    assert results["baseline"][1] == results["hybrid"][1], "Feature list mismatch!"
    usable = results["baseline"][1]
    return results["baseline"][0], results["hybrid"][0], usable, backup, feats_csv


def fit_predict(train_df, test_df, usable):
    from retrain_lr_symmetric import flip_row_dataframe
    td  = pd.concat([train_df, flip_row_dataframe(train_df)], ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
    w   = np.exp(-LAM*(TRAIN_END-td["DATE"]).dt.days.values/365.25)
    lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1_RATIO,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    n_act = int((np.abs(lr.coef_[0])>1e-8).sum())
    Xte   = sc.transform(imp.transform(test_df[usable]))
    return lr.predict_proba(Xte)[:,1], n_act


def compute_metrics(y, p):
    pc  = np.clip(p, EPS, 1-EPS)
    auc = float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None
    return dict(acc=float(accuracy_score(y,(p>=0.5).astype(int))*100),
                ll=float(log_loss(y,pc)), auc=auc,
                brier=float(brier_score_loss(y,pc)), n=int(len(y)))


def roi_full(test_df, p):
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

    def roi(sub):
        if len(sub)==0: return None
        return float(np.where(sub["won_pick"]==1,
                               sub["dec_odds_pick"]-1,-1.0).mean()*100)

    q_rois = {}
    for qn,qs,qe in QUARTERS:
        dq = d[(d["DATE"]>=qs)&(d["DATE"]<qe)]
        q_rois[qn] = {"roi": roi(dq), "n": int(len(dq))}

    pos = sum(1 for v in q_rois.values() if v["roi"] is not None and v["roi"]>0)
    return dict(n=len(d),
                roi_all=roi(d),           n_all=len(d),
                roi_ev=roi(d[d["ev"]>0]), n_ev=int((d["ev"]>0).sum()),
                roi_p65=roi(d[(d["ev"]>0)&(d["p_pick"]>=0.65)]),
                n_p65=int(((d["ev"]>0)&(d["p_pick"]>=0.65)).sum()),
                quarters=q_rois, pos_quarters=pos)


def main():
    t0 = time.time()
    print("="*78)
    print("WOMEN'S HYBRID PRIOR MODEL (gender + WC hierarchical shrinkage)")
    print(f"  κ={KAPPA}  C={C}  l1={L1_RATIO}  λ={LAM}")
    print("="*78)

    df_base, df_hyb, usable, backup, feats_csv = load_datasets()

    try:
        wc_col    = "weightclass_encoded"
        mens_mask = df_base[wc_col].isin({5,6,7,8,9,10,11,12})

        def split(df):
            tr = df[(df["DATE"]>=TRAIN_START)&(df["DATE"]<TRAIN_END)].sort_values("DATE").reset_index(drop=True)
            te = df[(df["DATE"]>=TRAIN_END)  &(df["DATE"]<TEST_END)].reset_index(drop=True)
            assert tr["DATE"].max() < TRAIN_END
            assert te["DATE"].min() >= TRAIN_END
            return tr, te

        train_base, test_base = split(df_base)
        train_hyb,  test_hyb  = split(df_hyb)

        print(f"\n  Train rows: baseline={len(train_base):,}  hybrid={len(train_hyb):,}")
        print(f"  Test  rows: baseline={len(test_base):,}  hybrid={len(test_hyb):,}")

        # Sanity: same row count (only adjperf z-scores differ, not row count)
        assert len(train_base)==len(train_hyb) and len(test_base)==len(test_hyb), \
            "Row count mismatch between baseline and hybrid frames!"

        # ── Fit both models ───────────────────────────────────────────────
        print("\nFitting baseline EN...")
        p_base, n_act_base = fit_predict(train_base, test_base, usable)
        print(f"  Active: {n_act_base}")

        print("Fitting hybrid EN...")
        p_hyb, n_act_hyb = fit_predict(train_hyb, test_hyb, usable)
        print(f"  Active: {n_act_hyb}")

        y_all  = test_base["win"].astype(int).values
        y_mens = y_all[test_base[wc_col].isin({5,6,7,8,9,10,11,12}).values]
        y_wom  = y_all[test_base[wc_col].isin(WOMENS_WCS).values]
        p_base_mens = p_base[test_base[wc_col].isin({5,6,7,8,9,10,11,12}).values]
        p_hyb_mens  = p_hyb[test_base[wc_col].isin({5,6,7,8,9,10,11,12}).values]
        p_base_wom  = p_base[test_base[wc_col].isin(WOMENS_WCS).values]
        p_hyb_wom   = p_hyb[test_base[wc_col].isin(WOMENS_WCS).values]

        # ── Accuracy metrics ──────────────────────────────────────────────
        print("\n" + "="*78)
        print("ACCURACY METRICS")
        print("="*78)
        print(f"  {'Model':<28}  {'Subset':<8}  {'N':>5}  {'Acc':>7}  "
              f"{'LL':>7}  {'AUC':>7}  {'Brier':>7}")
        print(f"  {'-'*28}  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for (name, subset, y, p) in [
            ("Baseline", "All",     y_all,  p_base),
            ("Hybrid",   "All",     y_all,  p_hyb),
            ("Baseline", "Men's",   y_mens, p_base_mens),
            ("Hybrid",   "Men's",   y_mens, p_hyb_mens),
            ("Baseline", "Women's", y_wom,  p_base_wom),
            ("Hybrid",   "Women's", y_wom,  p_hyb_wom),
        ]:
            m = compute_metrics(y, p)
            auc_s = f"{m['auc']:>7.4f}" if m['auc'] is not None else "      —"
            print(f"  {name:<28}  {subset:<8}  {m['n']:>5}  {m['acc']:>6.2f}%  "
                  f"{m['ll']:>7.4f}  {auc_s}  {m['brier']:>7.4f}")

        # Delta
        m_base_all = compute_metrics(y_all,  p_base)
        m_hyb_all  = compute_metrics(y_all,  p_hyb)
        m_base_wom = compute_metrics(y_wom,  p_base_wom)
        m_hyb_wom  = compute_metrics(y_wom,  p_hyb_wom)
        print(f"\n  Delta hybrid−baseline (all):     "
              f"acc {m_hyb_all['acc']-m_base_all['acc']:>+.2f}pp  "
              f"ll {m_hyb_all['ll']-m_base_all['ll']:>+.4f}  "
              f"auc {(m_hyb_all['auc'] or 0)-(m_base_all['auc'] or 0):>+.4f}")
        print(f"  Delta hybrid−baseline (women's): "
              f"acc {m_hyb_wom['acc']-m_base_wom['acc']:>+.2f}pp  "
              f"ll {m_hyb_wom['ll']-m_base_wom['ll']:>+.4f}  "
              f"auc {(m_hyb_wom['auc'] or 0)-(m_base_wom['auc'] or 0):>+.4f}")

        # Per-WC accuracy
        print("\nPer WC accuracy delta (hybrid − baseline):")
        print(f"  {'WC':<12}  {'N':>5}  {'Baseline':>10}  {'Hybrid':>10}  {'Delta':>8}")
        for wc in sorted(range(1,13)):
            mask = (test_base[wc_col]==wc).values
            if mask.sum()==0: continue
            yw = y_all[mask]; pb = p_base[mask]; ph = p_hyb[mask]
            ab = float(accuracy_score(yw,(pb>=0.5).astype(int))*100)
            ah = float(accuracy_score(yw,(ph>=0.5).astype(int))*100)
            marker = " ◀ women's" if wc in WOMENS_WCS else ""
            print(f"  {WC_NAMES.get(wc,f'WC{wc}'):<12}  {mask.sum():>5}  "
                  f"{ab:>9.2f}%  {ah:>9.2f}%  {ah-ab:>+7.2f}pp{marker}")

        # ── ROI ───────────────────────────────────────────────────────────
        print("\n" + "="*78)
        print("ROI ANALYSIS")
        print("="*78)

        r_base = roi_full(test_base, p_base)
        r_hyb  = roi_full(test_hyb,  p_hyb)

        def fmt(v,n): return f"{v:>+7.1f}% ({n:>3})" if v is not None else "         —"
        for label, r in [("Baseline", r_base), ("Hybrid  ", r_hyb)]:
            print(f"  {label}:  all={fmt(r['roi_all'],r['n_all'])}  "
                  f"+EV={fmt(r['roi_ev'],r['n_ev'])}  "
                  f"p65={fmt(r['roi_p65'],r['n_p65'])}  "
                  f"pos={r['pos_quarters']}/6")
            q_str = "  ".join(
                f"{v['roi']:>+6.1f}%" if v["roi"] is not None else "      —"
                for v in r["quarters"].values()
            )
            print(f"  {'':8}  qtrs: {q_str}")
            print()

        print("Quarterly detail:")
        print(f"  {'Quarter':<10}  {'Base n':>6}  {'Base ROI':>9}  "
              f"{'Hyb n':>6}  {'Hyb ROI':>9}  {'Delta':>8}")
        for qn in [q[0] for q in QUARTERS]:
            rb = r_base["quarters"].get(qn, {"roi":None,"n":0})
            rh = r_hyb["quarters"].get(qn,  {"roi":None,"n":0})
            d  = (rh["roi"]-rb["roi"]) if (rh["roi"] is not None and rb["roi"] is not None) else None
            def f(v): return f"{v:>+8.1f}%" if v is not None else "         —"
            print(f"  {qn:<10}  {rb['n']:>6}  {f(rb['roi'])}  "
                  f"{rh['n']:>6}  {f(rh['roi'])}  {f(d)}")

        # ── Save ──────────────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        Path("results/womens_hybrid_prior_model.json").write_text(json.dumps({
            "config": dict(C=C, l1_ratio=L1_RATIO, lam=LAM, kappa=KAPPA),
            "n_active": {"baseline": n_act_base, "hybrid": n_act_hyb},
            "metrics": {
                "baseline_all":    compute_metrics(y_all,  p_base),
                "hybrid_all":      compute_metrics(y_all,  p_hyb),
                "baseline_womens": compute_metrics(y_wom,  p_base_wom),
                "hybrid_womens":   compute_metrics(y_wom,  p_hyb_wom),
                "baseline_mens":   compute_metrics(y_mens, p_base_mens),
                "hybrid_mens":     compute_metrics(y_mens, p_hyb_mens),
            },
            "roi": {"baseline": r_base, "hybrid": r_hyb},
            "runtime_min": round((time.time()-t0)/60,1),
        }, indent=2, default=str))
        print(f"\n✓ Saved results/womens_hybrid_prior_model.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
