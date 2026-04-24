"""Multi-threshold walk-forward model-vs-Vegas comparison.

Runs the full 4-fold walk-forward + Vegas comparison at THREE prior-fight
thresholds: t=1, t=2, t=3 (current production). Saves a unified JSON
with all three so the UI can toggle between them.

Per finding_threshold_matters.md, lowering the threshold brings in more
rookie fighters (noisier predictions) but grows the bettable fight set.

Saves: results/walkforward_vegas_multi_threshold.json
"""
import sys, json, warnings, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
warnings.filterwarnings("ignore")

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, american_to_decimal
from retrain_lr_symmetric import (
    load_wc_history_from_db, add_wc_features,
    load_recent_form_from_db, add_recent_form_features,
)
from walk_forward_4fold import FOLDS, select_features
from build_walkforward_vegas_comparison import run_fold_with_vegas, fold_metrics


# ── Richer attach_vegas using both data/tmp/odds_table.csv + DB ─────────
def _norm_name(s):
    """Convert 'Adam Fugitt' → 'AdamFugitt' (jfighter format)."""
    if not isinstance(s, str): return ""
    return "".join(s.split())


def _norm_bout(s):
    """Convert 'Adam Fugitt vs Ty Miller' → 'AdamFugittvs.TyMiller'."""
    if not isinstance(s, str): return ""
    parts = s.split(" vs ")
    if len(parts) != 2: return _norm_name(s)
    return f"{_norm_name(parts[0])}vs.{_norm_name(parts[1])}"


def load_odds_csv():
    """Load odds_table.csv and return in the shape attach_vegas expects.

    Input: DATE, BOUT (human), FIGHTER (human), prob_norm (devigged), odds (American)
    Output: one row per (DATE, jbout, jfighter) with devigged-probability columns.
    """
    csv_path = Path("data/tmp/odds_table.csv")
    if not csv_path.exists():
        return None
    raw = pd.read_csv(csv_path, parse_dates=["DATE"])
    raw = raw.dropna(subset=["BOUT", "FIGHTER", "prob_norm", "odds"])
    # Normalize names to jfighter/jbout format
    raw["jbout"] = raw["BOUT"].apply(_norm_bout)
    raw["jfighter"] = raw["FIGHTER"].apply(_norm_name)
    # Decimal odds from American
    raw["dec_odds"] = raw["odds"].apply(american_to_decimal)
    raw = raw.dropna(subset=["dec_odds"])
    # Each fight has 2 rows (one per fighter). Pivot to wide on jbout:
    # for each (DATE, jbout), we want p_f1/p_f2/dec_f1/dec_f2 keyed by
    # which fighter.
    return raw[["DATE", "jbout", "jfighter", "prob_norm", "dec_odds"]].copy()


def _canonical_fight_key(f1, f2):
    """Order-independent fight key: tuple of sorted fighter names."""
    a, b = sorted([str(f1), str(f2)])
    return f"{a}||{b}"


def attach_vegas_rich(test):
    """Attach Vegas odds from odds_table.csv (fresh through 2026-04) using
    an order-independent fighter-pair key. The CSV's BOUT string and
    winlossko's jbout sometimes order the two fighters differently
    (e.g. CSV: 'DerrickLewisvs.WaldoCortesAcosta' vs winlossko:
    'WaldoCortesAcostavs.DerrickLewis'). Joining on raw jbout misses
    these — we join on (DATE, sorted-fighter-pair) instead.
    """
    csv_odds = load_odds_csv()
    if csv_odds is None or len(csv_odds) == 0:
        from run_threshold_sweep_both_elos import attach_vegas as _old
        return _old(test)

    # For CSV: for each (DATE, jbout), get both fighters → wide format
    # with one row per fight keyed on sorted pair.
    csv_wide = csv_odds.groupby(["DATE", "jbout"]).agg(list).reset_index()
    csv_wide = csv_wide[csv_wide["jfighter"].apply(len) == 2].copy()
    # Each row now has 2 fighters. Build canonical pair key + store both
    # fighters' probs/odds keyed by fighter name.
    records = []
    for _, r in csv_wide.iterrows():
        jfs = r["jfighter"]
        probs = r["prob_norm"]
        decs = r["dec_odds"]
        key = _canonical_fight_key(jfs[0], jfs[1])
        records.append({
            "DATE":         r["DATE"],
            "fight_key":    key,
            "csv_jf_a":     jfs[0], "csv_jf_b": jfs[1],
            "csv_p_a":      probs[0], "csv_p_b": probs[1],
            "csv_dec_a":    decs[0], "csv_dec_b": decs[1],
        })
    csv_by_key = pd.DataFrame(records).drop_duplicates(subset=["DATE","fight_key"])

    # For test: each row has jfighter + opp_jfighter → canonical key.
    t = test.copy()
    # If we only have jfighter (no opp), need to infer from jbout. Parse jbout
    # which is "FighterAvs.FighterB".
    if "opp_jfighter" not in t.columns and "jbout" in t.columns:
        split = t["jbout"].str.split("vs.", n=1, expand=True)
        if split.shape[1] == 2:
            # Opp is whichever name in the jbout isn't jfighter
            def _opp(row):
                a, b = row["jbout"].split("vs.", 1)
                return b if row["jfighter"] == a else a
            t["opp_jfighter"] = t.apply(_opp, axis=1)

    t["fight_key"] = t.apply(lambda r: _canonical_fight_key(r["jfighter"],
                                                              r.get("opp_jfighter","")), axis=1)
    # Join
    m = t.merge(csv_by_key, on=["DATE", "fight_key"], how="left")

    # Align — p_vegas_f1 = CSV prob for THIS row's jfighter;
    # dec_odds_f1 = CSV dec odds for THIS row's jfighter; opposite for _f2.
    def _align(row):
        jf = row["jfighter"]
        if pd.isna(row.get("csv_jf_a")):
            return pd.Series({"p_vegas_f1": np.nan, "dec_odds_f1": np.nan,
                               "dec_odds_f2": np.nan})
        if jf == row["csv_jf_a"]:
            return pd.Series({"p_vegas_f1": row["csv_p_a"],
                               "dec_odds_f1": row["csv_dec_a"],
                               "dec_odds_f2": row["csv_dec_b"]})
        elif jf == row["csv_jf_b"]:
            return pd.Series({"p_vegas_f1": row["csv_p_b"],
                               "dec_odds_f1": row["csv_dec_b"],
                               "dec_odds_f2": row["csv_dec_a"]})
        return pd.Series({"p_vegas_f1": np.nan, "dec_odds_f1": np.nan,
                           "dec_odds_f2": np.nan})
    aligned = m.apply(_align, axis=1)
    m[["p_vegas_f1","dec_odds_f1","dec_odds_f2"]] = aligned
    m = m.drop(columns=["csv_jf_a","csv_jf_b","csv_p_a","csv_p_b",
                         "csv_dec_a","csv_dec_b","fight_key"], errors="ignore")
    m = m.drop_duplicates(subset=["DATE","jbout","jfighter"]).reset_index(drop=True)
    return m

THRESHOLDS = [1, 2, 3]


def _run_fold_rich(df, fold, feats):
    """Drop-in for run_fold_with_vegas that uses attach_vegas_rich instead
    of the DB-only attach_vegas."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from retrain_lr_symmetric import flip_row_dataframe
    from run_threshold_sweep_both_elos import LAM
    from walk_forward_4fold import leakage_assertions

    train_start = pd.Timestamp(fold["train_start"])
    train_end   = pd.Timestamp(fold["train_end"])
    test_start  = pd.Timestamp(fold["test_start"])
    test_end    = pd.Timestamp(fold["test_end"])
    train = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].copy()
    test  = df[(df["DATE"] >= test_start) & (df["DATE"] < test_end)].copy()
    leakage_assertions(train, test, fold)

    train_flipped = flip_row_dataframe(train)
    train_doubled = pd.concat([train, train_flipped], ignore_index=True)
    usable = [c for c in feats if c in train_doubled.columns and train_doubled[c].std() > 1e-8]

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(train_doubled[usable]))
    ytr = train_doubled["win"].astype(int).values
    w = np.exp(-LAM * (train_end - train_doubled["DATE"]).dt.days.values / 365.25)
    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xtr, ytr, sample_weight=w)

    Xte = sc.transform(imp.transform(test[usable]))
    p_model = lr.predict_proba(Xte)[:, 1]
    test = test.copy()
    test["p_model"] = p_model

    # ── Rich Vegas attach (CSV preferred, DB fallback) ──
    tv = attach_vegas_rich(test[["DATE", "jbout", "jfighter"]].drop_duplicates())
    merged = test.merge(tv[["DATE", "jbout", "jfighter", "p_vegas_f1",
                              "dec_odds_f1", "dec_odds_f2"]],
                         on=["DATE", "jbout", "jfighter"], how="left")
    merged = merged.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)
    return merged


def run_for_threshold(threshold, base, wc_hist, rf_hist=None):
    print(f"\n{'='*72}")
    print(f"Threshold = {threshold} prior UFC fights on both sides")
    print(f"{'='*72}")

    df = apply_threshold(base, threshold)
    df = add_wc_features(df, wc_hist)
    # recent-form features tried & rejected — see retrain_lr_symmetric.py
    feats = select_features(df)

    per_fold_raw = {}
    all_matched = []

    for fold in FOLDS:
        merged = _run_fold_rich(df, fold, feats)
        matched = merged[merged["p_vegas_f1"].notna()].copy()
        n_total = len(merged)
        n_matched = len(matched)
        pct = 100 * n_matched / n_total if n_total > 0 else 0

        m = fold_metrics(matched)
        m["n_total"] = n_total
        m["n_matched"] = n_matched
        m["pct_matched"] = round(pct, 1)
        m["test_start"] = fold["test_start"]
        m["test_end"] = fold["test_end"]
        per_fold_raw[fold["name"]] = m

        matched = matched.copy()
        matched["fold"] = fold["name"]
        all_matched.append(matched)

        print(f"  {fold['name']}  n_test={n_total:>4d}  matched={n_matched:>3d} "
              f"({pct:>4.1f}%)  acc_model={m.get('acc_model', 0)*100:>5.2f}%  "
              f"acc_vegas={m.get('acc_vegas', 0)*100:>5.2f}%  "
              f"roi_flat={m.get('roi_flat_model', 0):>+6.2f}%  "
              f"+EV={m.get('roi_pos_ev')}")

    pooled_df = pd.concat(all_matched, ignore_index=True) if all_matched else pd.DataFrame()
    pooled = fold_metrics(pooled_df)
    pooled["n_total"] = sum(per_fold_raw[f]["n_total"] for f in per_fold_raw)
    pooled["n_matched"] = len(pooled_df)
    pooled["pct_matched"] = round(100 * pooled["n_matched"] / max(pooled["n_total"], 1), 1)

    print(f"\n  POOLED: n_total={pooled['n_total']}  matched={pooled['n_matched']}  "
          f"acc_model={pooled['acc_model']*100:.2f}%  acc_vegas={pooled['acc_vegas']*100:.2f}%")
    print(f"          flat-model ROI={pooled['roi_flat_model']:+.2f}%  "
          f"+EV ROI={pooled['roi_pos_ev']}% on {pooled['n_pos_ev']} bets")
    return {"folds": [{"name": f["name"], **per_fold_raw[f["name"]]} for f in FOLDS],
            "pooled": pooled}


def main():
    print("Loading base features + wc_history...")
    base = load_base_both_elos()
    wc_hist = load_wc_history_from_db()

    by_threshold = {}
    for t in THRESHOLDS:
        by_threshold[str(t)] = run_for_threshold(t, base, wc_hist)

    # Save
    out = Path("results/walkforward_vegas_multi_threshold.json")
    out.write_text(json.dumps({
        "methodology": "4-fold walk-forward (7yr train / 6mo test, symmetric LR refit per fold), "
                       "evaluated at three minimum prior-fights thresholds: 1, 2, 3. "
                       "Lower threshold = more rookies included, noisier per-fight predictions, "
                       "larger test set. Higher threshold = cleaner signal on veterans, fewer fights. "
                       "Per finding_threshold_matters.md: ROI is monotonic in threshold.",
        "thresholds": THRESHOLDS,
        "by_threshold": by_threshold,
        "test_window": f"{FOLDS[0]['test_start']} → {FOLDS[-1]['test_end']}",
    }, indent=2, default=str))
    print(f"\n✓ Saved {out}")

    # Quick side-by-side summary at the end
    print(f"\n{'='*90}")
    print("SIDE-BY-SIDE POOLED METRICS")
    print(f"{'='*90}")
    print(f"{'threshold':<12s} {'n_matched':>10s} {'acc_model':>10s} {'acc_vegas':>10s} "
          f"{'ll_model':>9s} {'ll_vegas':>9s} {'roi_flat':>9s} {'+EV_ROI':>9s} {'+EV_n':>6s}")
    print("-" * 90)
    for t in THRESHOLDS:
        p = by_threshold[str(t)]["pooled"]
        print(f"t={t:<10d} {p['n_matched']:>10d} {p['acc_model']*100:>9.2f}% "
              f"{p['acc_vegas']*100:>9.2f}%  {p['ll_model']:>8.4f} {p['ll_vegas']:>8.4f} "
              f"{p['roi_flat_model']:>+8.2f}% {p['roi_pos_ev'] if p['roi_pos_ev'] is not None else 0:>+8.2f}% "
              f"{p['n_pos_ev']:>6d}")


if __name__ == "__main__":
    main()
