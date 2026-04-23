"""Build all artifacts for predictor_v2 — the clean inference path.

Outputs (under app/models/blend_v2/):
  - fighter_mma_history.parquet  : per-fighter per-fight ABSOLUTE feature rows
                                    (pre-diff). Used at inference to find a
                                    fighter's state as of any `event_date`.
  - fighter_elo_ufc.json         : UFC-only Elo trajectory per fighter
  - fighter_elo_exp.json         : Expanded Elo trajectory per fighter (UFC + pre-UFC)
  - fighter_style_elo.json       : Striking + grappling Elo trajectory per fighter
  - fighter_bios.json            : DOB, stance, weightindex per fighter
  - lr.pkl + lr_scaler.pkl       : trained model on 202-feature stack
  - feat_cols.json               : ordered feature list (matches scaler.transform input)

Training set: threshold=3, 2016-01-01 → 2024-05-04 (pre-test-window).
  Matches `scripts/run_threshold_sweep_both_elos.py` single-shot LR.

Leakage guardrails (LEAKAGE_REFERENCE.md §1-§11):
  §1 Train strictly < 2024-05-04.
  §2/§5 All pipeline steps internal to compute_elo / compute_decayed_averages
        / compute_opponent_history use strict `d < fight_date`. Verified.
  §4 Scaler fit on train-only.
  §6 LR hyperparams frozen (C=0.05, l1=0.5, saga).
"""
import json, pickle, sqlite3, sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
from elo_feature import compute_elo
from mma_ai_pipeline import (load_base_data, beta_binomial_smooth,
                              poisson_gamma_smooth, compute_derived_features,
                              compute_decayed_averages, compute_opponent_history,
                              compute_wc_priors, compute_adjperf)
from mma_ai_config import V7_CONFIG

DT = Path("data/tmp")
OUT = Path("app/models/blend_v2")
OUT.mkdir(parents=True, exist_ok=True)
SCRAPER_DB = "data/sqlite_db/sqlite_scrapper.db"

TEST_FIRST = pd.Timestamp("2024-05-04")   # nothing post this enters training
TRAIN_START = pd.Timestamp("2016-01-01")
FILTER_THRESHOLD = 3
LAM = 0.13

ELO_COLS_BASE = ["precomp_elo_diff", "elo_win_prob", "elo_momentum_diff",
                 "peak_elo_diff", "avg_opp_elo_diff", "elo_consist_diff"]
ELO_PARAMS = dict(K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
                  decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
                  logistic_scale=449.205,
                  opp_quality_k=True, sliding_k=True, upset_momentum=True,
                  champ_mult=1.2)
LOGISTIC_SCALE = 449.205


# ─────────────────────────────────────────────────────────────────────────
# Part 1: build per-fighter per-fight absolute-stat history (for MMA-AI diff)
# ─────────────────────────────────────────────────────────────────────────
def build_mma_history():
    """Run the v7 pipeline to the point right before `assemble_features` does
    the diff, and emit a per-fighter per-fight DataFrame with absolute values.
    Downstream inference will lookup + diff these directly.
    """
    print("="*70)
    print("PART 1: Per-fighter MMA-AI feature history (pre-diff)")
    print("="*70)

    config = V7_CONFIG
    decay_lambda = config["decay_lambda"]

    print("\nStep 1-4: base data, BB, PG, derived...")
    df = load_base_data()
    df = beta_binomial_smooth(df)
    df = poisson_gamma_smooth(df)
    df = compute_derived_features(df)

    # Same stat_cols discovery logic as build_features
    stat_cols = [c for c in df.columns if
                 c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
                 c.endswith("_ratio") or c.endswith("_per_ctrl") or
                 c in ["ko_smooth", "win_smooth", "decision_smooth",
                       "sub_land_smooth", "sub_land_rate", "ctrl_pm",
                       "ko_per_sig_str_land", "td_per_sig_str_att",
                       "ground_per_ctrl", "dist_per_sig_str_land",
                       "head_per_sig_str_land", "rev_per_ctrlopp",
                       "sig_str_land_ratio", "ko_ratio", "sub_att_ratio",
                       "ctrl_ratio", "ground_land_per_ctrl", "td_land_per_ctrl"]]
    stat_cols = sorted(set(c for c in stat_cols if c in df.columns and
                           not c.startswith("opp_") and not c.endswith("_raw")))

    print(f"\nStep 5: decayed averages ({len(stat_cols)} stats)")
    df = compute_decayed_averages(df, stat_cols, decay_lambda)

    print("\nStep 6: opponent history + WC priors")
    df = compute_opponent_history(df, stat_cols, decay_lambda)
    priors = compute_wc_priors(df, stat_cols)

    print("\nStep 7: AdjPerf")
    df = compute_adjperf(df, stat_cols, priors)

    # Static features (replicating assemble_features 720-738)
    print("\nStep 8: static + dec_avg of static (age, ufc_age, days_since_last_fight)")
    debut_dates = df.groupby("jfighter")["DATE"].min().to_dict()
    df["ufc_age"] = df.apply(
        lambda r: (r["DATE"] - debut_dates.get(r["jfighter"], r["DATE"])).days / 365.25,
        axis=1
    )
    df["reach_ratio"] = df["REACH"] / df["REACH"].median()
    df["days_since_last_fight"] = df.groupby("jfighter")["DATE"].diff().dt.days.fillna(365)
    static_da_cols = ["age", "ufc_age", "days_since_last_fight"]
    df = compute_decayed_averages(df, static_da_cols, decay_lambda)

    # Layer 1: dec_avg of adjperf scores
    adjperf_cols = [f"{col}_adjperf" for col in stat_cols if f"{col}_adjperf" in df.columns]
    print(f"\nStep 9: dec_avg of {len(adjperf_cols)} adjperf columns")
    df = compute_decayed_averages(df, adjperf_cols, decay_lambda)

    # Layer 2: opp dec_avg
    opp_cols = [f"{col}_opp_mean" for col in stat_cols if f"{col}_opp_mean" in df.columns]
    print(f"\nStep 10: dec_avg of {len(opp_cols)} opp_mean columns")
    df = compute_decayed_averages(df, opp_cols, decay_lambda)

    # At this point df has per-fighter per-fight rows with all absolute
    # feature values computed. `is_fighter1` distinguishes f1 (red corner)
    # and f2 rows for the same bout — both have the SAME feature values
    # from that fighter's perspective (not diffed).

    # Save ALL columns of interest per fighter per fight
    keep_cols = ["DATE", "jevent", "jbout", "jfighter", "opp_jfighter",
                 "win", "weightindex", "REACH", "WEIGHT", "STANCE", "DOB",
                 "age", "ufc_age", "reach_ratio", "days_since_last_fight",
                 "is_fighter1"]
    # + all the stat_cols, adjperf_dec_avg, opp_dec_avg, own dec_avg
    keep_cols += [c for c in df.columns if
                   c.endswith("_dec_avg") or
                   c.endswith("_adjperf_dec_avg") or
                   c.endswith("_opp_mean_dec_avg")]
    keep_cols = [c for c in keep_cols if c in df.columns]
    keep_cols = list(dict.fromkeys(keep_cols))  # de-dup preserving order
    history = df[keep_cols].copy()
    history["DATE"] = pd.to_datetime(history["DATE"])

    # CRITICAL: dedup on (jevent, jbout, jfighter) — a few fights trigger
    # cartesian blowups in upstream merges (~30 fights × 4096 rows each).
    # `assemble_features` handles this with drop_duplicates before diffing;
    # we have to do it explicitly here since we stop before assemble_features.
    before = len(history)
    history = history.drop_duplicates(subset=["jevent", "jbout", "jfighter"],
                                        keep="first").reset_index(drop=True)
    print(f"\n  Deduplicated: {before:,} → {len(history):,} rows "
          f"(dropped {before - len(history):,})")

    # Save to parquet
    path = OUT / "fighter_mma_history.parquet"
    history.to_parquet(path, index=False)
    print(f"Saved {len(history):,} rows × {history.shape[1]} cols to {path}")
    print(f"  Unique fighters: {history['jfighter'].nunique()}")
    print(f"  Date range: {history['DATE'].min().date()} → {history['DATE'].max().date()}")
    return history


# ─────────────────────────────────────────────────────────────────────────
# Part 2: per-fighter Elo trajectories (UFC-only, expanded, striking, grappling)
# ─────────────────────────────────────────────────────────────────────────
def build_elo_cache(bout_file_name, label):
    """Run compute_elo on a bout CSV, extract per-fighter per-fight rating
    trajectory (date → pre-fight rating). Save as JSON.
    Each bout contributes TWO rows (one per fighter)."""
    print(f"\n── {label} Elo trajectory ──")
    bouts = pd.read_csv(DT / bout_file_name, parse_dates=["DATE"])
    elo, ratings, last_date, extra = compute_elo(bouts, **ELO_PARAMS)
    if "source" in elo.columns:
        elo = elo[elo["source"] == "ufc"].copy()
    # Per-fighter trajectory: list of (date_iso, precomp_elo, momentum, peak, avg_opp, consist)
    traj = defaultdict(list)
    for r in elo.itertuples():
        traj[r.f1].append(dict(
            date=str(pd.Timestamp(r.DATE).date()),
            precomp_elo=float(r.precomp_elo_f1),
            elo_momentum=float(r.elo_momentum_f1),
            peak_elo=float(r.peak_elo_f1),
            avg_opp_elo=float(r.avg_opp_elo_f1),
            elo_consist=float(r.elo_consist_f1),
        ))
        traj[r.f2].append(dict(
            date=str(pd.Timestamp(r.DATE).date()),
            precomp_elo=float(r.precomp_elo_f2),
            elo_momentum=float(r.elo_momentum_f2),
            peak_elo=float(r.peak_elo_f2),
            avg_opp_elo=float(r.avg_opp_elo_f2),
            elo_consist=float(r.elo_consist_f2),
        ))
    # Also save POST-final-fight ratings (current state after last fight) and last_date
    # These come from `ratings` and `last_date` dicts returned by compute_elo.
    final_state = {}
    for jf in traj.keys():
        traj[jf] = sorted(traj[jf], key=lambda r: r["date"])
        final_state[jf] = dict(
            current_elo=float(ratings.get(jf, 1500.0)),
            last_date=str(pd.Timestamp(last_date.get(jf, pd.Timestamp("1900-01-01"))).date())
                if jf in last_date else None,
            # Peak may not be in extra for every fighter; lookup if present
            peak_elo=float(extra.get("peak_elo", {}).get(jf, ratings.get(jf, 1500.0))),
        )

    out = dict(trajectory=dict(traj), final_state=final_state)
    return out


def build_style_elo_cache():
    """Rebuild striking + grappling Elos (UFC-only, as in build_style_elos.py)."""
    print("\n── Striking + Grappling Elo trajectory ──")
    conn = sqlite3.connect(SCRAPER_DB)
    stats = pd.read_sql("""
        SELECT jevent, jbout, jfighter, sigstracc, tdacc, ctrl, subatt
        FROM ufc_fighter_match_stats_smooth
    """, conn)
    conn.close()
    ufc = pd.read_csv(DT / "elo_bouts.csv", parse_dates=["DATE"])
    b = ufc.merge(stats, on=["jevent", "jbout"], how="inner")
    f1 = b[b["jfighter"] == b["f1"]][["jevent", "jbout", "DATE", "f1", "f2",
                                        "sigstracc", "tdacc", "ctrl", "subatt"]].rename(
        columns={"sigstracc": "f1_sigstr", "tdacc": "f1_td",
                 "ctrl": "f1_ctrl", "subatt": "f1_sub"})
    f2 = b[b["jfighter"] == b["f2"]][["jevent", "jbout",
                                        "sigstracc", "tdacc", "ctrl", "subatt"]].rename(
        columns={"sigstracc": "f2_sigstr", "tdacc": "f2_td",
                 "ctrl": "f2_ctrl", "subatt": "f2_sub"})
    m = f1.merge(f2, on=["jevent", "jbout"], how="inner").sort_values(
        ["DATE", "jevent", "jbout"]).reset_index(drop=True)

    def sig(x, s=5.0): return 1.0 / (1.0 + np.exp(-x / s))
    m["strk_actual_f1"] = sig(m["f1_sigstr"] - m["f2_sigstr"], 5.0)
    m["grp_f1"] = m["f1_td"] + m["f1_ctrl"] / 60 + 0.3 * m["f1_sub"]
    m["grp_f2"] = m["f2_td"] + m["f2_ctrl"] / 60 + 0.3 * m["f2_sub"]
    m["grp_actual_f1"] = sig(m["grp_f1"] - m["grp_f2"], 1.5)

    strike = defaultdict(lambda: 1500.0)
    grapple = defaultdict(lambda: 1500.0)
    traj = defaultdict(list)
    K = 20; SCALE = 400.0

    def exp_sc(a, b): return 1.0 / (1.0 + 10 ** ((b - a) / SCALE))

    for r in m.itertuples():
        sf1, sf2 = strike[r.f1], strike[r.f2]
        gf1, gf2 = grapple[r.f1], grapple[r.f2]
        d = str(pd.Timestamp(r.DATE).date())
        # Record pre-fight ratings
        traj[r.f1].append(dict(date=d, striking=sf1, grappling=gf1))
        traj[r.f2].append(dict(date=d, striking=sf2, grappling=gf2))
        # Update
        if not (np.isnan(r.strk_actual_f1) or np.isnan(r.grp_actual_f1)):
            e_s = exp_sc(sf1, sf2)
            strike[r.f1] = sf1 + K*(r.strk_actual_f1 - e_s)
            strike[r.f2] = sf2 + K*((1-r.strk_actual_f1) - (1-e_s))
            e_g = exp_sc(gf1, gf2)
            grapple[r.f1] = gf1 + K*(r.grp_actual_f1 - e_g)
            grapple[r.f2] = gf2 + K*((1-r.grp_actual_f1) - (1-e_g))

    final_state = {jf: dict(striking=float(strike[jf]), grappling=float(grapple[jf]))
                   for jf in traj.keys()}
    for jf in traj.keys():
        traj[jf] = sorted(traj[jf], key=lambda r: r["date"])
    return dict(trajectory=dict(traj), final_state=final_state)


# ─────────────────────────────────────────────────────────────────────────
# Part 3: Fighter bios (DOB, stance, weightindex)
# ─────────────────────────────────────────────────────────────────────────
def build_bios():
    print("\n── Fighter bios ──")
    conn = sqlite3.connect(SCRAPER_DB)
    tott = pd.read_sql(
        "SELECT jfighter, DOB, STANCE, weightindex, HEIGHT, WEIGHT, REACH "
        "FROM ufc_fighter_tott", conn)
    conn.close()
    # Robustly parse DOB — some rows have "--" or other junk
    tott["DOB_parsed"] = pd.to_datetime(tott["DOB"], errors="coerce")
    for col in ("HEIGHT", "WEIGHT", "REACH", "weightindex"):
        tott[col] = pd.to_numeric(tott[col], errors="coerce")
    bios = {}
    for r in tott.itertuples():
        bios[r.jfighter] = dict(
            dob=str(r.DOB_parsed.date()) if pd.notna(r.DOB_parsed) else None,
            stance=(r.STANCE or "").strip().lower() if isinstance(r.STANCE, str) else "",
            weightindex=int(r.weightindex) if pd.notna(r.weightindex) else 0,
            height=float(r.HEIGHT) if pd.notna(r.HEIGHT) else None,
            weight=float(r.WEIGHT) if pd.notna(r.WEIGHT) else None,
            reach=float(r.REACH) if pd.notna(r.REACH) else None,
        )
    print(f"  {len(bios)} fighter bios  "
          f"(DOB parsed: {tott['DOB_parsed'].notna().sum()})")
    return bios


# ─────────────────────────────────────────────────────────────────────────
# Part 4: Train final LR on 202-feature stack
# ─────────────────────────────────────────────────────────────────────────
def train_final_lr():
    """Replicates run_threshold_sweep_both_elos.py single-shot training.
    Saves lr.pkl, lr_scaler.pkl, feat_cols.json.
    """
    print("\n" + "="*70)
    print("PART 4: Train final LR on 202-feature stack")
    print("="*70)

    # Use the already-built mmaai_features.csv + feature layers
    df = pd.read_csv(DT / "mmaai_features.csv", parse_dates=["DATE"])
    # Apply filter and all feature layers EXACTLY as the final production script does.
    from run_threshold_sweep_both_elos import (
        compute_elo_suffixed as _elo_suf,
        build_style_elos as _style_elos,
        apply_threshold, TIER_1C, STYLE_COLS, ELO_COLS_BASE as _ecb,
    )

    # UFC prior counts
    conn = sqlite3.connect(SCRAPER_DB)
    hist = pd.read_sql("SELECT w.jfighter, e.DATE FROM ufc_winlossko w "
                       "JOIN ufc_event_details e ON e.jevent=w.jevent", conn)
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fd = {f: grp["DATE"].values for f, grp in
          hist.sort_values(["jfighter", "DATE"]).groupby("jfighter")}

    def prior(j, d):
        dates = fd.get(j, np.array([], dtype="datetime64[ns]"))
        return int((dates < np.datetime64(d)).sum()) if len(dates) else 0

    df["f1_priors"] = df.apply(lambda r: prior(r["jfighter"], r["DATE"]), axis=1)
    df["f2_priors"] = df.apply(lambda r: prior(r["opp_jfighter"], r["DATE"]), axis=1)
    res = pd.read_sql("SELECT jevent, jbout, METHOD FROM ufc_fight_results", conn)
    res["METHOD_norm"] = res["METHOD"].str.lower().fillna("")
    conn.close()
    df = df.merge(res[["jevent", "jbout", "METHOD_norm"]], on=["jevent", "jbout"], how="left")
    unwanted = ["dq", "other", "overturned", "decision - split", "decision - majority"]
    m = df["METHOD_norm"].apply(lambda x: any(u in str(x) for u in unwanted)
                                 if pd.notna(x) else False)
    df = df[~m].drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    # Merge both Elo sources
    print("  Computing both Elos...")
    elo_ufc = _elo_suf("elo_bouts.csv", "ufc")
    elo_exp = _elo_suf("elo_bouts_expanded.csv", "exp")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.merge(elo_ufc, on=["jbout", "DATE"], how="left")
    df = df.rename(columns={"f1": "f1_tmp", "f2": "f2_tmp"})
    df = df.merge(elo_exp[["jbout", "DATE"] + [c for c in elo_exp.columns
                                                 if c.endswith("_exp")]],
                  on=["jbout", "DATE"], how="left")
    flip = df["jfighter"] != df["f1_tmp"]
    for suffix in ("ufc", "exp"):
        for c in _ecb:
            col = f"{c}_{suffix}"
            if c == "elo_win_prob":
                df.loc[flip, col] = 1 - df.loc[flip, col]
                df[col] = df[col].fillna(0.5)
            else:
                df.loc[flip, col] = -df.loc[flip, col]
                df[col] = df[col].fillna(0.0)
    df.drop(columns=["f1_tmp", "f2_tmp"], inplace=True, errors="ignore")

    # Tier 1c + style
    rs = pd.read_csv(DT / "recency_stance_features.csv", parse_dates=["DATE"])
    df = df.merge(rs[["DATE", "jbout", "jfighter"] + TIER_1C],
                  on=["DATE", "jbout", "jfighter"], how="left")
    for c in TIER_1C: df[c] = df[c].fillna(0)
    se = _style_elos()
    se["DATE"] = pd.to_datetime(se["DATE"])
    df = df.merge(se, on=["DATE", "jbout", "jfighter"], how="left")
    for c in STYLE_COLS: df[c] = df[c].fillna(0.0)
    df = df.drop_duplicates(subset=["DATE", "jbout", "jfighter"]).reset_index(drop=True)

    # Filter threshold=3 + train slice
    df = apply_threshold(df, FILTER_THRESHOLD)
    train = df[df["DATE"] < TEST_FIRST].copy()
    print(f"  Train rows: {len(train):,}")

    # Feature selection (matches run_threshold_sweep_both_elos)
    feats = [c for c in df.columns if (c.endswith("_diff") or c.endswith("_ufc")
             or c.endswith("_exp") or c in ("weightclass_encoded", "scheduled_rounds",
                                             "days_since_last_fight_f1"))
             and c not in ("f1_priors", "f2_priors")
             and not c.startswith("ix_")
             and c not in ("sos_last3_diff", "sos_last5_diff", "sos_trajectory_diff",
                           "form_winrate3_diff", "form_winrate5_diff",
                           "elo_trajectory_diff", "career_fights_diff",
                           "stance_mismatch", "southpaw_advantage_diff")]
    usable = [c for c in feats if c in train.columns and train[c].std() > 1e-8]
    print(f"  Features: {len(usable)}")

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(train[usable])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    y = train["win"].astype(int).values
    # Recency weight: relative to TEST_FIRST (the "cutover" date)
    w = np.exp(-LAM * (TEST_FIRST - train["DATE"]).dt.days.values / 365.25)

    lr = LogisticRegression(C=0.05, penalty="elasticnet", l1_ratio=0.5,
                            solver="saga", max_iter=6000, random_state=42)
    lr.fit(Xs, y, sample_weight=w)

    # Save
    with open(OUT / "lr.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(OUT / "lr_scaler.pkl", "wb") as f:
        pickle.dump(sc, f)
    with open(OUT / "lr_imputer.pkl", "wb") as f:
        pickle.dump(imp, f)
    (OUT / "feat_cols.json").write_text(json.dumps(usable, indent=2))
    print(f"  Saved LR + scaler + imputer + feat_cols ({len(usable)} features)")


def main():
    # Part 1: MMA-AI per-fighter history
    _ = build_mma_history()

    # Part 2: Elo caches
    print("\n" + "="*70)
    print("PART 2: Elo trajectories (UFC-only, Expanded, Style)")
    print("="*70)
    elo_ufc = build_elo_cache("elo_bouts.csv", "UFC-only")
    with open(OUT / "fighter_elo_ufc.json", "w") as f:
        json.dump(elo_ufc, f, default=str)
    print(f"  UFC-only: {len(elo_ufc['trajectory'])} fighters")

    elo_exp = build_elo_cache("elo_bouts_expanded.csv", "Expanded")
    with open(OUT / "fighter_elo_exp.json", "w") as f:
        json.dump(elo_exp, f, default=str)
    print(f"  Expanded: {len(elo_exp['trajectory'])} fighters")

    style = build_style_elo_cache()
    with open(OUT / "fighter_style_elo.json", "w") as f:
        json.dump(style, f, default=str)
    print(f"  Style: {len(style['trajectory'])} fighters")

    # Part 3: Bios
    print("\n" + "="*70)
    print("PART 3: Fighter bios")
    print("="*70)
    bios = build_bios()
    with open(OUT / "fighter_bios.json", "w") as f:
        json.dump(bios, f, default=str)

    # Part 4: Train final LR
    train_final_lr()

    print("\n" + "="*70)
    print(f"DONE. All artifacts in {OUT}")
    print("="*70)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.name:<40s}  {p.stat().st_size / 1024:>6.1f} KB")


if __name__ == "__main__":
    main()
