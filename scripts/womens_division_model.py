"""Women's division separate EN model.

Filters train/test to women's weight classes (weightclass_encoded ∈ {1,2,3,4}):
  1 = W.Strawweight   2 = W.Flyweight
  3 = W.Bantamweight  4 = W.Featherweight

Trains a separate EN (same C=0.05, l1=0.5, λ=1.2 as mixed model) on
women's-only fights and compares its test metrics to the MIXED model's
performance on those same women's test fights.

Audit:   docs/audits/womens_division_model.md
Outputs: results/womens_division_model.json
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

# ── Constants (same as walkforward_market_features / en_hyperparam_sweep) ──
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

# Women's WC codes (1-indexed, matching DB weightindex encoding)
WOMENS_WCS = {1, 2, 3, 4}
WC_NAMES   = {1:"W.Straw", 2:"W.Fly", 3:"W.Bantam", 4:"W.Feather"}

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
    """Load the full (all-division) feature frame — same as walkforward_market_features."""
    df_full, stat_cols = build_through_step6()
    train_only = df_full[df_full["DATE"] < TRAIN_END].copy()
    priors     = mma.compute_wc_priors(train_only, stat_cols)
    df_adj     = mma.compute_adjperf(df_full, stat_cols, priors)
    result     = mma.assemble_features(df_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
    result     = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])

    feats_csv = Path("data/tmp/mmaai_features.csv")
    backup    = Path("data/tmp/mmaai_features.csv.before_womens")
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

    # CRITICAL: call select_features BEFORE merging novel market cols
    feats_baseline = select_features(df)

    mf = pd.read_csv("data/tmp/market_features_clean.csv", parse_dates=["DATE"])
    avail = [c for c in NOVEL if c in mf.columns]
    df = df.merge(mf[["DATE","jbout","jfighter"]+avail],
                  on=["DATE","jbout","jfighter"], how="left")
    new_feats = [c for c in avail if c not in feats_baseline and df[c].std()>1e-8]
    usable = [c for c in feats_baseline+new_feats
              if c in df.columns and df[c].std()>1e-8]

    # Safety check: market cols must NOT be in baseline
    market_in_baseline = [c for c in new_feats if c in feats_baseline]
    if market_in_baseline:
        raise RuntimeError(f"ORDERING BUG: {market_in_baseline} already in baseline")

    return df, usable, backup, feats_csv


def fit_predict_model(train_df, test_df, usable, label=""):
    """Fit EN and return (p_test, n_active, coef_series)."""
    from retrain_lr_symmetric import flip_row_dataframe
    td = pd.concat([train_df, flip_row_dataframe(train_df)], ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
    w   = np.exp(-LAM*(TRAIN_END-td["DATE"]).dt.days.values/365.25)
    lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1_RATIO,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    n_active = int((np.abs(lr.coef_[0])>1e-8).sum())
    Xte = sc.transform(imp.transform(test_df[usable]))
    p   = lr.predict_proba(Xte)[:,1]
    return p, n_active, pd.Series(lr.coef_[0], index=usable)


def compute_metrics(y, p):
    pc = np.clip(p, EPS, 1-EPS)
    return dict(
        acc  = float(accuracy_score(y,(p>=0.5).astype(int))*100),
        ll   = float(log_loss(y, pc)),
        auc  = float(roc_auc_score(y, p)) if len(np.unique(y))>1 else None,
        brier= float(brier_score_loss(y, pc)),
        n    = int(len(y)),
    )


def roi_analysis(test_df, p, label):
    """ROI vs Vegas for a prediction set."""
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
    d["p_pick"]        = np.where(d["pick_a"]==1,d["p_raw"],1-d["p_raw"])
    d["p_vegas_pick"]  = np.where(d["pick_a"]==1,d["p_vegas_f1"],1-d["p_vegas_f1"])
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
    q_rois = {}
    for qn,qs,qe in QUARTERS:
        dq = d[(d["DATE"]>=qs)&(d["DATE"]<qe)]
        q_rois[qn] = roi(dq)

    return dict(n_vegas=len(d), roi_all=r_all, roi_ev=r_ev, roi_p65=r_p65,
                quarters=q_rois)


def main():
    t0 = time.time()
    print("="*78)
    print("WOMEN'S DIVISION SEPARATE EN MODEL")
    print(f"  WCs: W.Straw(1) W.Fly(2) W.Bantam(3) W.Feather(4)")
    print(f"  C={C}, l1_ratio={L1_RATIO}, λ={LAM}")
    print(f"  Train: {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  Test:  {TRAIN_END.date()} → {TEST_END.date()}")
    print("="*78)

    print("\nBuilding pipeline (once)...")
    df_all, usable, backup, feats_csv = load_full_dataset()
    print(f"  All-division frame: {len(df_all):,} rows, {len(usable)} features")

    # ── Check weightclass column ───────────────────────────────────────────
    if "weightclass_encoded" not in df_all.columns:
        # Fallback: try wc_encoded or weight_class
        wc_col = next((c for c in df_all.columns if "wc" in c.lower()
                       or "weight" in c.lower()), None)
        print(f"  ⚠ weightclass_encoded not found; trying: {wc_col}")
        print(f"  Available cols containing 'weight': "
              f"{[c for c in df_all.columns if 'weight' in c.lower() or 'wc' in c.lower()]}")
    else:
        wc_col = "weightclass_encoded"
    print(f"  Using WC column: {wc_col}")
    if wc_col:
        wc_dist = df_all[wc_col].value_counts().sort_index()
        print(f"  WC distribution (all rows):")
        for wc, n in wc_dist.items():
            tag = WC_NAMES.get(int(wc), f"WC{int(wc)}")
            print(f"    {wc:>4.0f} ({tag:<12}) : {n:>5,}")

    try:
        # ── Split: women's only ────────────────────────────────────────────
        if wc_col and wc_col in df_all.columns:
            df_w = df_all[df_all[wc_col].isin(WOMENS_WCS)].copy()
        else:
            raise ValueError(f"Cannot find women's WC column. Available: {list(df_all.columns)[:20]}")

        train_w = df_w[(df_w["DATE"]>=TRAIN_START)&(df_w["DATE"]<TRAIN_END)].copy()
        test_w  = df_w[(df_w["DATE"]>=TRAIN_END)  &(df_w["DATE"]<TEST_END)].copy()
        train_w = train_w.sort_values("DATE").reset_index(drop=True)

        assert train_w["DATE"].max() < TRAIN_END, "Train/test leak!"
        assert test_w["DATE"].min() >= TRAIN_END, "Train/test leak!"
        assert not (set(zip(train_w["DATE"],train_w["jbout"])) &
                    set(zip(test_w["DATE"], test_w["jbout"]))), "Bout overlap!"

        print(f"\n  Women's train: {len(train_w):,} rows  "
              f"({train_w['jbout'].nunique()} bouts)")
        print(f"  Women's test:  {len(test_w):,} rows  "
              f"({test_w['jbout'].nunique()} bouts)")

        y_test_w = test_w["win"].astype(int).values

        # Per-WC breakdown in test
        print("\n  Test set per WC:")
        for wc in sorted(WOMENS_WCS):
            tw = test_w[test_w[wc_col]==wc]
            if len(tw)>0:
                print(f"    {WC_NAMES[wc]:<12}: {len(tw):>4} rows  "
                      f"({tw['jbout'].nunique()} bouts)")

        # ── Also keep the full mixed model split for comparison ────────────
        train_all = df_all[(df_all["DATE"]>=TRAIN_START)&(df_all["DATE"]<TRAIN_END)].copy()
        test_all  = df_all[(df_all["DATE"]>=TRAIN_END)  &(df_all["DATE"]<TEST_END)].copy()
        train_all = train_all.sort_values("DATE").reset_index(drop=True)

        # ── Fit women's model ──────────────────────────────────────────────
        print("\nFitting women's-only EN...")
        p_womens, n_act_w, coef_w = fit_predict_model(train_w, test_w, usable, "women")
        print(f"  Active features: {n_act_w}/{len(usable)}")

        # ── Fit mixed model (same hyperparams) ────────────────────────────
        print("Fitting mixed (all-division) EN for comparison...")
        p_mixed_all, n_act_m, coef_m = fit_predict_model(train_all, test_all, usable, "mixed")
        # Subset mixed predictions to women's test rows
        # (test_w rows are a subset of test_all rows — match by index or key)
        test_all_reset = test_all.reset_index(drop=True)
        test_all_reset["p_mixed"] = p_mixed_all
        test_all_reset["row_key"] = list(zip(test_all_reset["DATE"],
                                             test_all_reset["jbout"],
                                             test_all_reset["jfighter"]))
        test_w_reset = test_w.reset_index(drop=True)
        test_w_reset["row_key"] = list(zip(test_w_reset["DATE"],
                                           test_w_reset["jbout"],
                                           test_w_reset["jfighter"]))
        key_to_pmixed = dict(zip(test_all_reset["row_key"],
                                  test_all_reset["p_mixed"]))
        p_mixed_w = np.array([key_to_pmixed[k] for k in test_w_reset["row_key"]])

        # ── Metrics ───────────────────────────────────────────────────────
        m_womens = compute_metrics(y_test_w, p_womens)
        m_mixed  = compute_metrics(y_test_w, p_mixed_w)

        print("\n" + "="*78)
        print("RESULTS  (women's test fights only)")
        print("="*78)
        print(f"  {'Model':<22}  {'N':>5}  {'Acc':>7}  {'LL':>7}  "
              f"{'AUC':>7}  {'Brier':>7}")
        print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for name, m in [("Women's-only EN", m_womens), ("Mixed EN (women's rows)", m_mixed)]:
            auc_str = f"{m['auc']:>7.4f}" if m['auc'] is not None else "     —"
            print(f"  {name:<22}  {m['n']:>5}  {m['acc']:>6.2f}%  "
                  f"{m['ll']:>7.4f}  {auc_str}  {m['brier']:>7.4f}")

        delta = dict(
            acc   = m_womens["acc"]   - m_mixed["acc"],
            ll    = m_womens["ll"]    - m_mixed["ll"],
            auc   = (m_womens["auc"] or 0) - (m_mixed["auc"] or 0),
            brier = m_womens["brier"] - m_mixed["brier"],
        )
        print(f"\n  Delta (women's − mixed):  "
              f"acc {delta['acc']:>+.2f}pp  "
              f"ll {delta['ll']:>+.4f}  "
              f"auc {delta['auc']:>+.4f}  "
              f"brier {delta['brier']:>+.4f}")
        print(f"  (negative ll/brier = women's model BETTER)")

        # ── Per WC breakdown ───────────────────────────────────────────────
        print("\nPer weight-class accuracy:")
        print(f"  {'WC':<12}  {'N':>5}  {'Womens-EN':>10}  {'Mixed-EN':>10}  {'Delta':>8}")
        per_wc = {}
        for wc in sorted(WOMENS_WCS):
            mask = (test_w_reset[wc_col]==wc).values
            if mask.sum() == 0:
                continue
            yw  = y_test_w[mask]
            pw  = p_womens[mask]
            pm  = p_mixed_w[mask]
            aw  = float(accuracy_score(yw,(pw>=0.5).astype(int))*100)
            am  = float(accuracy_score(yw,(pm>=0.5).astype(int))*100)
            per_wc[WC_NAMES[wc]] = dict(n=int(mask.sum()),
                                         acc_womens=aw, acc_mixed=am)
            print(f"  {WC_NAMES[wc]:<12}  {mask.sum():>5}  "
                  f"{aw:>9.2f}%  {am:>9.2f}%  {aw-am:>+7.2f}pp")

        # ── Active features comparison ─────────────────────────────────────
        print(f"\nActive features:")
        print(f"  Women's EN:  {n_act_w} active / {len(usable)} total")
        print(f"  Mixed EN:    {n_act_m} active / {len(usable)} total")

        active_w = set(usable[i] for i,c in enumerate(coef_w.values) if abs(c)>1e-8)
        active_m = set(usable[i] for i,c in enumerate(coef_m.values) if abs(c)>1e-8)
        only_in_w = active_w - active_m
        only_in_m = active_m - active_w
        in_both   = active_w & active_m

        print(f"  In both:           {len(in_both)}")
        print(f"  Women's-only active: {len(only_in_w)}")
        if only_in_w:
            for f in sorted(only_in_w)[:10]:
                print(f"    {f:<40}  w={coef_w[f]:>+.4f}")
        print(f"  Mixed-only active: {len(only_in_m)}")

        # ── Top 10 coefficients (women's model) ───────────────────────────
        print("\nTop 10 features by |coef| (women's EN):")
        top10 = coef_w.abs().nlargest(10).index
        for i,f in enumerate(top10,1):
            wc = coef_w[f]; wm = coef_m.get(f,0.0)
            print(f"  {i:>2}. {f:<40}  w={wc:>+.4f}  mixed={wm:>+.4f}")

        # ── Quarter breakdown (women's) ────────────────────────────────────
        print("\nQuarterly accuracy (women's test rows):")
        print(f"  {'Quarter':<10}  {'N':>5}  {'Womens-EN':>10}  {'Mixed-EN':>10}")
        q_details = {}
        for qn,qs,qe in QUARTERS:
            mask = ((test_w_reset["DATE"]>=qs)&(test_w_reset["DATE"]<qe)).values
            if mask.sum()==0:
                print(f"  {qn:<10}  {'—':>5}")
                q_details[qn] = None
                continue
            yw = y_test_w[mask]; pw = p_womens[mask]; pm = p_mixed_w[mask]
            aw = float(accuracy_score(yw,(pw>=0.5).astype(int))*100)
            am = float(accuracy_score(yw,(pm>=0.5).astype(int))*100)
            q_details[qn] = dict(n=int(mask.sum()), acc_womens=aw, acc_mixed=am)
            print(f"  {qn:<10}  {mask.sum():>5}  {aw:>9.2f}%  {am:>9.2f}%  {aw-am:>+7.2f}pp")

        # ── ROI (women's model vs mixed on women's fights) ────────────────
        print("\nROI analysis (women's test, Vegas-matched bouts):")
        roi_w = roi_analysis(test_w_reset, p_womens, "womens")
        roi_m = roi_analysis(test_w_reset, p_mixed_w, "mixed_on_womens")
        for name, r in [("Women's EN", roi_w), ("Mixed EN", roi_m)]:
            def fmt(v): return f"{v:>+7.1f}%" if v is not None else "      —"
            print(f"  {name:<22}: all={fmt(r['roi_all'])}  "
                  f"+EV={fmt(r['roi_ev'])}  p65={fmt(r['roi_p65'])}  "
                  f"n={r['n_vegas']}")

        # ── Save results ───────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        output = {
            "config": dict(C=C, l1_ratio=L1_RATIO, lam=LAM,
                           threshold=THRESHOLD, womens_wcs=list(WOMENS_WCS)),
            "train_n_womens": len(train_w),
            "test_n_womens":  len(test_w),
            "metrics": {
                "womens_model": m_womens,
                "mixed_model_on_womens": m_mixed,
                "delta": delta,
            },
            "per_wc_accuracy": per_wc,
            "quarterly": q_details,
            "n_active": {"womens": n_act_w, "mixed": n_act_m},
            "features_only_womens_active": sorted(only_in_w),
            "roi": {
                "womens_model": roi_w,
                "mixed_model_on_womens": roi_m,
            },
            "runtime_min": round((time.time()-t0)/60,1),
        }
        Path("results/womens_division_model.json").write_text(
            json.dumps(output, indent=2, default=str))
        print(f"\n✓ Saved results/womens_division_model.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
