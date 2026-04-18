"""Generate app/models/blend/model_vs_vegas.json for the Model Insights page.

Reads:
  app/models/blend/backtest_predictions.json (predictions)
  data/tmp/odds_table.csv                    (devig'd market odds, clamped ±100)

Writes a compact JSON summary: per-fold metrics, pooled totals, decomposition
(agree/disagree, favorite/underdog), and generation metadata.
"""
import json, re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/evankellener/Desktop/UFC_fight_predictor")
PRED_PATH = REPO / "app/models/blend/backtest_predictions.json"
ODDS_CSV  = REPO / "data/tmp/odds_table.csv"
OUT_PATH  = REPO / "app/models/blend/model_vs_vegas.json"

BLEND_W     = 0.5
EDGE_THRESH = 0.05

NAME_ALIASES = {"patriciofreire": "patriciopitbull"}
def norm_jf(name):
    n = re.sub(r"\W+", "", str(name)).lower()
    return NAME_ALIASES.get(n, n)

def am_to_implied(a):
    return np.where(a < 0, -a/(-a+100), 100/(a+100))
def am_to_decimal(a):
    return np.where(a < 0, 1 + 100/(-a), 1 + a/100)
def clamp(a):
    if a > 0 and a <= 110:  return 100.0
    if a < 0 and a >= -110: return -100.0
    return a

# ── Load predictions ───────────────────────────────────────────────────────
d = json.load(open(PRED_PATH))
folds = pd.DataFrame(d["folds"])
pred  = pd.DataFrame(d["predictions"])
pred["bout_date"] = pd.to_datetime(pred["bout_date"])
pred["p_model"]   = (1-BLEND_W)*pred["p_lr"] + BLEND_W*pred["p_xgb"]
pred["y_true"]    = (pred["actual_winner"] == pred["fighter_a"]).astype(int)
pred["jf_a_n"]    = pred["fighter_a"].apply(norm_jf)
pred["jf_b_n"]    = pred["fighter_b"].apply(norm_jf)
pred["pair"]      = [tuple(sorted([a,b])) for a,b in zip(pred["jf_a_n"], pred["jf_b_n"])]

# ── Load odds ──────────────────────────────────────────────────────────────
raw = pd.read_csv(ODDS_CSV, parse_dates=["DATE"])
raw["jfighter"]     = raw["FIGHTER"].apply(norm_jf)
raw["odds_clamped"] = raw["odds"].apply(clamp)

wide = raw.pivot_table(
    index=["DATE","BOUT"],
    columns=raw.groupby(["DATE","BOUT"]).cumcount(),
    values=["jfighter","odds_clamped"],
    aggfunc="first",
).reset_index()
wide.columns = ["DATE","BOUT","jf_a","jf_b","odds_a","odds_b"]
pa, pb = am_to_implied(wide["odds_a"]), am_to_implied(wide["odds_b"])
s = pa + pb
wide["prob_a_clamped"] = pa/s
wide["prob_b_clamped"] = pb/s
wide["vig"]            = s - 1
wide["dec_a"]          = am_to_decimal(wide["odds_a"])
wide["dec_b"]          = am_to_decimal(wide["odds_b"])
wide["am_a"]           = wide["odds_a"]
wide["am_b"]           = wide["odds_b"]
wide["pair"]           = [tuple(sorted([a,b])) for a,b in zip(wide["jf_a"], wide["jf_b"])]
wide = wide.rename(columns={"DATE":"bout_date"}).drop_duplicates(
    subset=["bout_date","pair"], keep="first")

m = pred.merge(
    wide[["bout_date","pair","jf_a","jf_b",
          "prob_a_clamped","prob_b_clamped","dec_a","dec_b","am_a","am_b","vig"]],
    on=["bout_date","pair"], how="left")
same = m["jf_a_n"] == m["jf_a"]
m["p_vegas_a"]  = np.where(same, m["prob_a_clamped"], m["prob_b_clamped"])
m["dec_odds_a"] = np.where(same, m["dec_a"], m["dec_b"])
m["dec_odds_b"] = np.where(same, m["dec_b"], m["dec_a"])
m["am_odds_a"]  = np.where(same, m["am_a"], m["am_b"])
m["am_odds_b"]  = np.where(same, m["am_b"], m["am_a"])

M = m[m["p_vegas_a"].notna()].copy()

# ── Metrics ────────────────────────────────────────────────────────────────
def metrics(sub):
    if len(sub) == 0:
        return dict(n=0)
    y  = sub["y_true"].values
    pm = sub["p_model"].clip(1e-6, 1-1e-6).values
    pv = sub["p_vegas_a"].clip(1e-6, 1-1e-6).values
    acc_m = float(((pm >= .5) == y).mean())
    acc_v = float(((pv >= .5) == y).mean())
    ll_m  = float(-np.mean(y*np.log(pm) + (1-y)*np.log(1-pm)))
    ll_v  = float(-np.mean(y*np.log(pv) + (1-y)*np.log(1-pv)))
    br_m  = float(np.mean((pm-y)**2))
    br_v  = float(np.mean((pv-y)**2))

    pick_a = pm >= 0.5
    odds_p = np.where(pick_a, sub["dec_odds_a"], sub["dec_odds_b"])
    won    = np.where(pick_a, y == 1, y == 0)
    pnl    = np.where(won, odds_p - 1.0, -1.0)
    roi    = float(pnl.mean() * 100)
    vpick  = np.where(pick_a, pv, 1-pv)
    mpick  = np.where(pick_a, pm, 1-pm)
    edge   = mpick - vpick
    mask   = edge > EDGE_THRESH
    if mask.sum() > 0:
        roi_e = float(pnl[mask].mean() * 100)
        n_e   = int(mask.sum())
    else:
        roi_e = None
        n_e   = 0
    return dict(
        n=int(len(sub)),
        acc_model=acc_m, acc_vegas=acc_v,
        ll_model=ll_m,   ll_vegas=ll_v,
        br_model=br_m,   br_vegas=br_v,
        roi_flat=roi,
        n_edge=n_e, roi_edge=roi_e,
        vig_mean=float(sub["vig"].mean()),
    )

per_fold = []
for fn in sorted(M["fold_num"].unique()):
    sub  = M[M["fold_num"] == fn]
    info = folds[folds["fold_num"] == fn].iloc[0]
    per_fold.append(dict(
        fold=int(fn),
        test_start=info["test_start"],
        test_end=info["test_end"],
        n_total=int(info["n_bouts"]),
        **metrics(sub),
    ))

pooled = metrics(M)
pooled["test_start"] = folds.iloc[0]["test_start"]
pooled["test_end"]   = folds.iloc[-1]["test_end"]
pooled["n_total"]    = int(len(pred))

# ── Decomposition: agree / disagree and fav / dog ──────────────────────────
M = M.copy()
M["model_pick_a"] = M["p_model"]   >= 0.5
M["vegas_pick_a"] = M["p_vegas_a"] >= 0.5
M["agree"]        = M["model_pick_a"] == M["vegas_pick_a"]
M["odds_pick"]    = np.where(M["model_pick_a"], M["dec_odds_a"], M["dec_odds_b"])
M["am_pick"]      = np.where(M["model_pick_a"], M["am_odds_a"],  M["am_odds_b"])
M["won"]          = np.where(M["model_pick_a"], M["y_true"] == 1, M["y_true"] == 0)
M["pnl"]          = np.where(M["won"], M["odds_pick"] - 1.0, -1.0)
M["v_odds_pick"]  = np.where(M["vegas_pick_a"], M["dec_odds_a"], M["dec_odds_b"])
M["v_won"]        = np.where(M["vegas_pick_a"], M["y_true"] == 1, M["y_true"] == 0)
M["v_pnl"]        = np.where(M["v_won"], M["v_odds_pick"] - 1.0, -1.0)

def bucket(sub):
    if len(sub) == 0:
        return dict(n=0)
    return dict(
        n=int(len(sub)),
        win_rate=float(sub["won"].mean()),
        avg_american_odds=float(sub["am_pick"].mean()),
        avg_decimal=float(sub["odds_pick"].mean()),
        roi=float(sub["pnl"].mean() * 100),
        total_pnl=float(sub["pnl"].sum()),
    )

decomposition = dict(
    agree=bucket(M[M["agree"]]),
    disagree=bucket(M[~M["agree"]]),
    picks_favorite=bucket(M[M["am_pick"] < 0]),
    picks_underdog=bucket(M[M["am_pick"] >= 0]),
    vegas_pick_roi=float(M["v_pnl"].mean() * 100),
    vegas_pick_win_rate=float(M["v_won"].mean()),
)

# ── Final payload ─────────────────────────────────────────────────────────
payload = dict(
    generated_at=datetime.now(timezone.utc).isoformat(),
    config=dict(
        blend_weight_xgb=BLEND_W,
        edge_threshold=EDGE_THRESH,
        odds_clamp="±100",
        odds_source="data/tmp/odds_table.csv (multi-book average, devig'd)",
    ),
    coverage=dict(
        matched=int(len(M)),
        total=int(len(pred)),
        pct=float(len(M) / len(pred)),
        odds_start=str(raw["DATE"].min().date()),
        odds_end=str(raw["DATE"].max().date()),
    ),
    pooled=pooled,
    per_fold=per_fold,
    decomposition=decomposition,
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(payload, f, indent=2)

print(f"Wrote {OUT_PATH}")
print(f"  coverage:   {payload['coverage']['matched']}/{payload['coverage']['total']} "
      f"({payload['coverage']['pct']:.1%})")
print(f"  pooled ROI: {pooled['roi_flat']:+.2f}%  "
      f"(edge: {pooled['roi_edge']:+.2f}% on {pooled['n_edge']})")
