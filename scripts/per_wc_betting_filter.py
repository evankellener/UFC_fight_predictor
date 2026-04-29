"""Per weight-class ROI breakdown and betting filter analysis.

For each men's WC (5–12), computes:
  - Accuracy, LL, AUC
  - ROI: all picks, +EV, p≥0.65
  - Quarterly ROI

Then tests composite filter strategies:
  - Men's all (baseline)
  - Men's top-N WCs by ROI
  - Men's positive-ROI WCs only
  - Specific WC combos

⚠ Exploratory — WC filter selected from test-set ROI.
  Audit: docs/audits/per_wc_betting_filter.md
Outputs: results/per_wc_betting_filter.json
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

MENS_WCS  = [5,6,7,8,9,10,11,12]
WC_NAMES  = {5:"LHW",6:"MW",7:"WW",8:"LW",9:"FW",10:"BW",11:"FlyW",12:"StraW"}

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
    backup    = Path("data/tmp/mmaai_features.csv.before_wcfilter")
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

    train = df[(df["DATE"]>=TRAIN_START)&(df["DATE"]<TRAIN_END)].sort_values("DATE").reset_index(drop=True)
    test  = df[(df["DATE"]>=TRAIN_END)  &(df["DATE"]<TEST_END)].reset_index(drop=True)

    assert train["DATE"].max() < TRAIN_END
    assert test["DATE"].min()  >= TRAIN_END
    assert not (set(zip(train["DATE"],train["jbout"])) &
                set(zip(test["DATE"], test["jbout"])))

    return train, test, usable, backup, feats_csv


def fit_predict(train, test, usable):
    from retrain_lr_symmetric import flip_row_dataframe
    td  = pd.concat([train, flip_row_dataframe(train)], ignore_index=True)
    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
    w   = np.exp(-LAM*(TRAIN_END-td["DATE"]).dt.days.values/365.25)
    lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1_RATIO,
                             solver="saga", max_iter=8000, random_state=42)
    lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
    Xte = sc.transform(imp.transform(test[usable]))
    return lr.predict_proba(Xte)[:,1]


def build_betting_frame(test, p):
    """Attach Vegas, compute edge/EV, deduplicate to one row per bout."""
    from build_walkforward_vegas_multi_threshold import attach_vegas_rich
    tv = attach_vegas_rich(test[["DATE","jbout","jfighter"]].drop_duplicates())
    d  = test.assign(p_raw=p).merge(
         tv[["DATE","jbout","jfighter","p_vegas_f1","dec_odds_f1","dec_odds_f2"]],
         on=["DATE","jbout","jfighter"], how="left")
    d  = d[d["p_vegas_f1"].notna()].copy()
    d  = (d.sort_values("p_raw", ascending=False)
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
    return d


def roi_stats(sub):
    if len(sub)==0: return {"roi":None, "n":0, "win_rate":None}
    pnl = np.where(sub["won_pick"]==1, sub["dec_odds_pick"]-1, -1.0)
    return {"roi": float(pnl.mean()*100),
            "n":   int(len(sub)),
            "win_rate": float(sub["won_pick"].mean()*100)}


def strategy_roi(d):
    """all, +EV, p65, per quarter."""
    r = {}
    r["all"]  = roi_stats(d)
    r["ev"]   = roi_stats(d[d["ev"]>0])
    r["p65"]  = roi_stats(d[(d["ev"]>0)&(d["p_pick"]>=0.65)])
    r["quarters"] = {}
    for qn,qs,qe in QUARTERS:
        dq = d[(d["DATE"]>=qs)&(d["DATE"]<qe)]
        r["quarters"][qn] = roi_stats(dq)
    r["pos_quarters"] = sum(1 for v in r["quarters"].values()
                            if v["roi"] is not None and v["roi"]>0)
    return r


def print_strategy(label, r, indent="  "):
    def fmt(s): return f"{s['roi']:>+7.1f}%({s['n']:>3})" if s["roi"] is not None else "         —"
    q_str = "  ".join(
        f"{v['roi']:>+6.1f}%" if v["roi"] is not None and v["n"]>0 else "      —"
        for v in r["quarters"].values()
    )
    print(f"{indent}{label:<32}  all={fmt(r['all'])}  "
          f"+EV={fmt(r['ev'])}  p65={fmt(r['p65'])}  "
          f"pos={r['pos_quarters']}/6")
    print(f"{indent}{'':32}  qtrs: {q_str}")


def main():
    t0 = time.time()
    print("="*78)
    print("PER WEIGHT-CLASS BETTING FILTER")
    print("  ⚠ exploratory — WC filter selected from test-set ROI")
    print("="*78)

    print("\nBuilding pipeline...")
    train, test, usable, backup, feats_csv = load_dataset()
    print(f"  Train:{len(train):,}  Test:{len(test):,}  Features:{len(usable)}")

    try:
        p = fit_predict(train, test, usable)
        d = build_betting_frame(test, p)
        d_mens = d[d["weightclass_encoded"].isin(MENS_WCS)].copy()

        # ── Per-WC breakdown ───────────────────────────────────────────────
        print("\n" + "="*78)
        print("PER WEIGHT-CLASS ROI  (men's, +EV bets)")
        print("="*78)
        print(f"  {'WC':<8}  {'N_test':>6}  {'N_vegas':>7}  {'Acc':>7}  "
              f"{'all ROI':>9}  {'n':>4}  {'+EV ROI':>9}  {'n':>4}  "
              f"{'p65 ROI':>9}  {'n':>4}  {'pos/6':>6}")
        print(f"  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  "
              f"{'-'*9}  {'-'*4}  {'-'*9}  {'-'*4}  {'-'*9}  {'-'*4}  {'-'*6}")

        wc_results = {}
        for wc in MENS_WCS:
            name    = WC_NAMES[wc]
            d_wc    = d_mens[d_mens["weightclass_encoded"]==wc]
            test_wc = test[test["weightclass_encoded"]==wc]
            n_test  = test_wc["jbout"].nunique()
            n_vegas = len(d_wc)

            if n_vegas == 0:
                print(f"  {name:<8}  {n_test:>6}  {'—':>7}")
                wc_results[name] = None
                continue

            y_wc = test_wc["win"].astype(int).values
            p_wc = p[test["weightclass_encoded"]==wc]
            acc  = float(accuracy_score(y_wc,(p_wc>=0.5).astype(int))*100)

            r_all = roi_stats(d_wc)
            r_ev  = roi_stats(d_wc[d_wc["ev"]>0])
            r_p65 = roi_stats(d_wc[(d_wc["ev"]>0)&(d_wc["p_pick"]>=0.65)])

            # quarterly positives
            pos = sum(1 for _,qs,qe in QUARTERS
                      if roi_stats(d_wc[(d_wc["DATE"]>=qs)&(d_wc["DATE"]<qe)])["roi"] is not None
                      and roi_stats(d_wc[(d_wc["DATE"]>=qs)&(d_wc["DATE"]<qe)])["roi"] > 0)

            def f(s):
                if s["roi"] is None: return f"{'—':>9}  {'—':>4}"
                return f"{s['roi']:>+8.1f}%  {s['n']:>4}"

            print(f"  {name:<8}  {n_test:>6}  {n_vegas:>7}  {acc:>6.2f}%  "
                  f"{f(r_all)}  {f(r_ev)}  {f(r_p65)}  {pos:>5}/6")

            wc_results[name] = dict(
                wc=wc, n_test=n_test, n_vegas=n_vegas, acc=acc,
                roi_all=r_all, roi_ev=r_ev, roi_p65=r_p65, pos_quarters=pos
            )

        # ── Quarterly breakdown per WC ─────────────────────────────────────
        print("\n" + "="*78)
        print("QUARTERLY ROI PER WC  (+EV bets)")
        print("="*78)
        q_names = [q[0] for q in QUARTERS]
        print(f"  {'WC':<8}  " + "  ".join(f"{q:>10}" for q in q_names))
        print(f"  {'-'*8}  " + "  ".join(f"{'-'*10}" for _ in q_names))
        for wc in MENS_WCS:
            name = WC_NAMES[wc]
            d_wc = d_mens[d_mens["weightclass_encoded"]==wc]
            if len(d_wc)==0:
                continue
            q_vals = []
            for _,qs,qe in QUARTERS:
                dq = d_wc[(d_wc["DATE"]>=qs)&(d_wc["DATE"]<qe)&(d_wc["ev"]>0)]
                rs = roi_stats(dq)
                if rs["roi"] is None or rs["n"]==0:
                    q_vals.append(f"{'—':>10}")
                else:
                    q_vals.append(f"{rs['roi']:>+8.1f}%({rs['n']:1})")
            print(f"  {name:<8}  " + "  ".join(q_vals))

        # ── Filter strategies ─────────────────────────────────────────────
        print("\n" + "="*78)
        print("FILTER STRATEGIES  (⚠ all selected from test set — exploratory)")
        print("="*78)

        # Rank WCs by +EV ROI
        ev_rois = {wc: wc_results[WC_NAMES[wc]]["roi_ev"]["roi"]
                   for wc in MENS_WCS
                   if wc_results.get(WC_NAMES[wc]) and
                      wc_results[WC_NAMES[wc]]["roi_ev"]["roi"] is not None}
        ranked = sorted(ev_rois.keys(), key=lambda w: ev_rois[w], reverse=True)

        strategies = {}

        # Baseline: all bets
        r_all_bets = strategy_roi(d)
        print_strategy("All bets (incl. women's)", r_all_bets)
        strategies["all_bets"] = r_all_bets

        # Men's all
        r_mens = strategy_roi(d_mens)
        print_strategy("Men's all WCs", r_mens)
        strategies["mens_all"] = r_mens

        print()

        # Positive +EV ROI WCs
        pos_ev_wcs = [wc for wc in MENS_WCS
                      if wc_results.get(WC_NAMES[wc]) and
                         wc_results[WC_NAMES[wc]]["roi_ev"]["roi"] is not None and
                         wc_results[WC_NAMES[wc]]["roi_ev"]["roi"] > 0]
        neg_ev_wcs = [wc for wc in MENS_WCS if wc not in pos_ev_wcs]
        print(f"  +EV positive WCs: {[WC_NAMES[w] for w in pos_ev_wcs]}")
        print(f"  +EV negative WCs: {[WC_NAMES[w] for w in neg_ev_wcs]}")
        print()

        d_pos_ev = d_mens[d_mens["weightclass_encoded"].isin(pos_ev_wcs)]
        r_pos_ev = strategy_roi(d_pos_ev)
        print_strategy(f"Men's +EV-positive WCs ({len(pos_ev_wcs)} WCs)", r_pos_ev)
        strategies[f"mens_pos_ev_wcs"] = {"wcs": [WC_NAMES[w] for w in pos_ev_wcs], **r_pos_ev}

        # Positive all-picks ROI WCs
        pos_all_wcs = [wc for wc in MENS_WCS
                       if wc_results.get(WC_NAMES[wc]) and
                          wc_results[WC_NAMES[wc]]["roi_all"]["roi"] is not None and
                          wc_results[WC_NAMES[wc]]["roi_all"]["roi"] > 0]
        d_pos_all = d_mens[d_mens["weightclass_encoded"].isin(pos_all_wcs)]
        r_pos_all = strategy_roi(d_pos_all)
        print_strategy(f"Men's all-positive WCs ({len(pos_all_wcs)} WCs)", r_pos_all)
        strategies["mens_pos_all_wcs"] = {"wcs": [WC_NAMES[w] for w in pos_all_wcs], **r_pos_all}

        print()

        # Top N WCs by +EV ROI
        for n_top in [3, 4, 5, 6]:
            top_wcs = ranked[:n_top]
            d_top   = d_mens[d_mens["weightclass_encoded"].isin(top_wcs)]
            r_top   = strategy_roi(d_top)
            label   = f"Top-{n_top} WCs ({', '.join(WC_NAMES[w] for w in top_wcs)})"
            print_strategy(label, r_top)
            strategies[f"top_{n_top}_wcs"] = {"wcs": [WC_NAMES[w] for w in top_wcs], **r_top}

        print()

        # Best and worst single WC
        best_wc = ranked[0]
        worst_wc = ranked[-1]
        print_strategy(f"Best single WC: {WC_NAMES[best_wc]}",
                       strategy_roi(d_mens[d_mens["weightclass_encoded"]==best_wc]))
        print_strategy(f"Worst single WC: {WC_NAMES[worst_wc]}",
                       strategy_roi(d_mens[d_mens["weightclass_encoded"]==worst_wc]))

        print()

        # Remove worst WC only
        d_drop_worst = d_mens[d_mens["weightclass_encoded"]!=worst_wc]
        r_drop_worst = strategy_roi(d_drop_worst)
        print_strategy(f"Men's drop {WC_NAMES[worst_wc]} only (7 WCs)", r_drop_worst)
        strategies["mens_drop_worst"] = {"dropped": WC_NAMES[worst_wc], **r_drop_worst}

        # ── Summary ranked by +EV ROI ──────────────────────────────────────
        print("\n" + "="*78)
        print("WC RANKING BY +EV ROI")
        print("="*78)
        print(f"  {'Rank':<5}  {'WC':<8}  {'+EV ROI':>9}  {'n':>4}  {'acc':>7}  {'pos/6':>6}")
        for i, wc in enumerate(ranked, 1):
            rs = wc_results[WC_NAMES[wc]]
            print(f"  {i:<5}  {WC_NAMES[wc]:<8}  "
                  f"{rs['roi_ev']['roi']:>+8.1f}%  {rs['roi_ev']['n']:>4}  "
                  f"{rs['acc']:>6.2f}%  {rs['pos_quarters']:>5}/6")

        # ── Save ──────────────────────────────────────────────────────────
        Path("results").mkdir(exist_ok=True)
        Path("results/per_wc_betting_filter.json").write_text(json.dumps({
            "config": dict(C=C, l1_ratio=L1_RATIO, lam=LAM),
            "wc_breakdown": wc_results,
            "wc_ranking_by_ev_roi": [WC_NAMES[w] for w in ranked],
            "strategies": strategies,
            "warning": "exploratory — WC filter selected from test-set ROI",
            "runtime_min": round((time.time()-t0)/60,1),
        }, indent=2, default=str))
        print(f"\n✓ Saved results/per_wc_betting_filter.json")
        print(f"Total runtime: {(time.time()-t0)/60:.1f} min")

    finally:
        if backup.exists():
            import shutil; shutil.copy2(backup, feats_csv); backup.unlink()


if __name__ == "__main__":
    main()
