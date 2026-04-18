"""Per-fold comparison: blended model (LR+XGB) vs Vegas devig'd market.

Joins app/models/blend/backtest_predictions.json (the exact predictions
powering the Model Insights page) with ufc_fight_odds from slim_scrapper.db.

Metrics per fold (on matched rows only):
  - n_matched, coverage
  - accuracy, log loss, brier  (model and vegas)
  - model ROI flat-betting its pick at market odds
  - model ROI with edge filter (bet only when model_prob - vegas_prob > 0.05)
"""
import json, sqlite3
import numpy as np, pandas as pd
from pathlib import Path

PRED_PATH = Path("app/models/blend/backtest_predictions.json")
DB = "data/sqlite_db/slim_scrapper.db"
BLEND_W = 0.5         # xgb weight used by prod
EDGE_THRESH = 0.05    # edge filter for the "smart" ROI column

# ── Load predictions ────────────────────────────────────────────────────
d = json.load(open(PRED_PATH))
folds = pd.DataFrame(d["folds"])
pred = pd.DataFrame(d["predictions"])
pred["bout_date"] = pd.to_datetime(pred["bout_date"])
pred["p_model"]   = (1-BLEND_W)*pred["p_lr"] + BLEND_W*pred["p_xgb"]
pred["y_true"]    = (pred["actual_winner"] == pred["fighter_a"]).astype(int)

# ── Load odds ──────────────────────────────────────────────────────────
con = sqlite3.connect(DB)
odds = pd.read_sql(
    "SELECT DATE as bout_date, jfighter, opp_jfighter, avg_odds_f1, avg_odds_f2 "
    "FROM ufc_fight_odds", con, parse_dates=["bout_date"])
con.close()

# f1 = jfighter, f2 = opp_jfighter (verified from sample inspection)
def am_to_implied(a):
    return np.where(a < 0, -a/(-a+100), 100/(a+100))

odds["p_vegas_raw_jf"]  = am_to_implied(odds["avg_odds_f1"])
odds["p_vegas_raw_opp"] = am_to_implied(odds["avg_odds_f2"])
# devig
s = odds["p_vegas_raw_jf"] + odds["p_vegas_raw_opp"]
odds["p_vegas_jf"]  = odds["p_vegas_raw_jf"]  / s
odds["p_vegas_opp"] = odds["p_vegas_raw_opp"] / s
odds["vig"]          = s - 1
# decimal odds (for ROI)
def am_to_decimal(a):
    return np.where(a < 0, 1 + 100/(-a), 1 + a/100)
odds["dec_odds_jf"]  = am_to_decimal(odds["avg_odds_f1"])
odds["dec_odds_opp"] = am_to_decimal(odds["avg_odds_f2"])

# ── Join on (date, unordered pair) ─────────────────────────────────────
# Strategy: make a canonical key (sorted pair) on both sides.
def pair_key(a, b):
    return [tuple(sorted([x, y])) for x, y in zip(a, b)]

pred["pair"] = pair_key(pred["fighter_a"], pred["fighter_b"])
odds["pair"] = pair_key(odds["jfighter"],  odds["opp_jfighter"])

m = pred.merge(
    odds[["bout_date","pair","jfighter","opp_jfighter",
          "p_vegas_jf","p_vegas_opp","dec_odds_jf","dec_odds_opp","vig",
          "avg_odds_f1","avg_odds_f2"]],
    on=["bout_date","pair"], how="left")

# Orient vegas prob/odds to fighter_a
same = m["fighter_a"] == m["jfighter"]
m["p_vegas_a"]   = np.where(same, m["p_vegas_jf"],   m["p_vegas_opp"])
m["p_vegas_b"]   = np.where(same, m["p_vegas_opp"],  m["p_vegas_jf"])
m["dec_odds_a"]  = np.where(same, m["dec_odds_jf"],  m["dec_odds_opp"])
m["dec_odds_b"]  = np.where(same, m["dec_odds_opp"], m["dec_odds_jf"])

have_odds = m["p_vegas_a"].notna()
# Filter corrupt odds: both sides same-sign negative, or raw vig > 10% (impossible book)
both_neg = (m["avg_odds_f1"] < 0) & (m["avg_odds_f2"] < 0)
bad_vig  = m["vig"] > 0.10
bad_odds = have_odds & (both_neg | bad_vig)
print(f"Matched rows with odds: {have_odds.sum()}/{len(m)} "
      f"({have_odds.mean():.1%})")
print(f"Corrupt odds dropped:   {bad_odds.sum()}  "
      f"(both-neg={both_neg.sum()}, vig>10%={bad_vig.sum()})")
have_odds = have_odds & ~bad_odds
print(f"Odds date range:         {odds['bout_date'].min().date()} → "
      f"{odds['bout_date'].max().date()}")

# ── Metrics ───────────────────────────────────────────────────────────
def metrics_row(sub):
    n = len(sub)
    if n == 0: return dict(n=0)
    y   = sub["y_true"].values
    pm  = sub["p_model"].clip(1e-6, 1-1e-6).values
    pv  = sub["p_vegas_a"].clip(1e-6, 1-1e-6).values
    acc_m = ((pm >= 0.5).astype(int) == y).mean()
    acc_v = ((pv >= 0.5).astype(int) == y).mean()
    ll_m  = -np.mean(y*np.log(pm) + (1-y)*np.log(1-pm))
    ll_v  = -np.mean(y*np.log(pv) + (1-y)*np.log(1-pv))
    br_m  = np.mean((pm-y)**2)
    br_v  = np.mean((pv-y)**2)

    # ROI: bet 1u on model's pick at market odds, for every bout
    pick_a = pm >= 0.5
    odds_pick = np.where(pick_a, sub["dec_odds_a"], sub["dec_odds_b"])
    won      = np.where(pick_a, y==1, y==0)
    pnl_flat = np.where(won, odds_pick - 1.0, -1.0)
    roi_flat = pnl_flat.mean() * 100

    # ROI with edge filter: only bet when model vs vegas edge > EDGE_THRESH
    vegas_pick_prob = np.where(pick_a, pv, 1-pv)
    model_pick_prob = np.where(pick_a, pm, 1-pm)
    edge = model_pick_prob - vegas_pick_prob
    mask = edge > EDGE_THRESH
    if mask.sum() > 0:
        roi_edge = pnl_flat[mask].mean() * 100
        n_edge = int(mask.sum())
    else:
        roi_edge = np.nan; n_edge = 0

    return dict(n=n,
                acc_model=acc_m, acc_vegas=acc_v,
                ll_model=ll_m,  ll_vegas=ll_v,
                br_model=br_m,  br_vegas=br_v,
                roi_flat_pct=roi_flat,
                n_edge_bets=n_edge, roi_edge_pct=roi_edge,
                vig_mean=sub["vig"].mean())

matched = m[have_odds].copy()

# Per fold
rows = []
for fn in sorted(matched["fold_num"].unique()):
    sub = matched[matched["fold_num"] == fn]
    info = folds[folds["fold_num"] == fn].iloc[0]
    r = metrics_row(sub)
    r = dict(fold=fn,
             test_window=f"{info.test_start[5:]}→{info.test_end[5:]}",
             n_total=info.n_bouts, **r)
    rows.append(r)

# Pooled
r = metrics_row(matched); r = dict(fold="ALL",
         test_window=f"{folds.iloc[0].test_start[5:]}→{folds.iloc[-1].test_end[5:]}",
         n_total=len(pred), **r); rows.append(r)

df = pd.DataFrame(rows)

# ── Pretty print ──────────────────────────────────────────────────────
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

print("\n" + "="*110)
print("PER-FOLD: MODEL vs VEGAS  (matched rows only; folds 7-8 drop — odds coverage ends 2025-12-13)")
print("="*110)
disp = df.copy()
for c in ["acc_model","acc_vegas"]:
    disp[c] = disp[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
for c in ["ll_model","ll_vegas","br_model","br_vegas"]:
    disp[c] = disp[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
for c in ["roi_flat_pct","roi_edge_pct"]:
    disp[c] = disp[c].apply(lambda v: f"{v:+6.2f}%" if pd.notna(v) else "—")
disp["vig_mean"] = disp["vig_mean"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
disp["coverage"] = disp.apply(lambda r: f"{r['n']}/{r['n_total']}" if r['n_total'] else "—", axis=1)
cols = ["fold","test_window","coverage",
        "acc_model","acc_vegas",
        "ll_model","ll_vegas",
        "br_model","br_vegas",
        "roi_flat_pct","n_edge_bets","roi_edge_pct","vig_mean"]
print(disp[cols].to_string(index=False))

df.to_csv("data/tmp/model_vs_vegas_per_fold.csv", index=False)
print(f"\nSaved: data/tmp/model_vs_vegas_per_fold.csv")
