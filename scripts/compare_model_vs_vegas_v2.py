"""Per-fold comparison: blended model vs Vegas.

Sources:
  predictions: app/models/blend/backtest_predictions.json (what the app shows)
  odds:        ../../data/tmp/odds_table.csv (long format, devig'd already)

Applies the user's odds-clamping convention: american odds in [-100, +100]
are clamped to ±100 (to floor the devig).
"""
import json, sqlite3, re
import numpy as np, pandas as pd
from pathlib import Path

REPO_MAIN = Path("/Users/evankellener/Desktop/UFC_fight_predictor")
PRED_PATH = Path("app/models/blend/backtest_predictions.json")
ODDS_CSV  = REPO_MAIN / "data/tmp/odds_table.csv"
SLIM_DB   = "data/sqlite_db/slim_scrapper.db"
BLEND_W = 0.5
EDGE_THRESH = 0.05

# Nickname/alias map: how bouts appear in odds feed vs predictions jfighter key
NAME_ALIASES = {
    "patriciofreire": "patriciopitbull",
}

def norm_jf(name):
    n = re.sub(r"\W+", "", str(name)).lower()
    return NAME_ALIASES.get(n, n)

# ── Predictions ────────────────────────────────────────────────────────
d = json.load(open(PRED_PATH))
folds = pd.DataFrame(d["folds"])
pred = pd.DataFrame(d["predictions"])
pred["bout_date"] = pd.to_datetime(pred["bout_date"])
pred["p_model"]   = (1-BLEND_W)*pred["p_lr"] + BLEND_W*pred["p_xgb"]
pred["y_true"]    = (pred["actual_winner"] == pred["fighter_a"]).astype(int)

# ── Odds CSV → wide per-bout ──────────────────────────────────────────
raw = pd.read_csv(ODDS_CSV, parse_dates=["DATE"])
raw["jfighter"] = raw["FIGHTER"].apply(norm_jf)

# Apply clamping: odds in [-100, +100] → ±100
def clamp(a):
    if a > 0 and a <= 110:  return 100.0
    if a < 0 and a >= -110: return -100.0
    return a
raw["odds_clamped"] = raw["odds"].apply(clamp)

# Pivot: one row per (DATE, BOUT) with both sides
wide = raw.pivot_table(index=["DATE", "BOUT"], columns=raw.groupby(["DATE","BOUT"]).cumcount(),
                       values=["jfighter", "odds_clamped", "prob_norm"],
                       aggfunc="first").reset_index()
wide.columns = ["DATE","BOUT","jf_a","jf_b","odds_a","odds_b","prob_a","prob_b"]

# Recompute devig from clamped american odds (because clamping changes implied probs)
def am_to_implied(a):
    return np.where(a<0, -a/(-a+100), 100/(a+100))
pa = am_to_implied(wide["odds_a"])
pb = am_to_implied(wide["odds_b"])
s = pa + pb
wide["prob_a_clamped"] = pa/s
wide["prob_b_clamped"] = pb/s
wide["vig"] = s - 1
# decimal odds on clamped american
def am_to_decimal(a):
    return np.where(a<0, 1+100/(-a), 1+a/100)
wide["dec_a"] = am_to_decimal(wide["odds_a"])
wide["dec_b"] = am_to_decimal(wide["odds_b"])

# Canonical unordered-pair key
wide["pair"] = [tuple(sorted([a,b])) for a,b in zip(wide["jf_a"], wide["jf_b"])]
wide = wide.rename(columns={"DATE":"bout_date"})

# Apply SAME normalization to both sides (strip non-word, lowercase, alias-map)
pred["jf_a_n"] = pred["fighter_a"].apply(norm_jf)
pred["jf_b_n"] = pred["fighter_b"].apply(norm_jf)
pred["pair"]   = [tuple(sorted([a,b])) for a,b in zip(pred["jf_a_n"], pred["jf_b_n"])]

# Drop duplicates on pair (seen when a bout appears twice)
wide = wide.drop_duplicates(subset=["bout_date","pair"], keep="first")

m = pred.merge(
    wide[["bout_date","pair","jf_a","jf_b",
          "prob_a_clamped","prob_b_clamped","dec_a","dec_b","vig"]],
    on=["bout_date","pair"], how="left")

# Orient to fighter_a (compare via normalized form)
same = m["jf_a_n"] == m["jf_a"]
m["p_vegas_a"]  = np.where(same, m["prob_a_clamped"], m["prob_b_clamped"])
m["dec_odds_a"] = np.where(same, m["dec_a"], m["dec_b"])
m["dec_odds_b"] = np.where(same, m["dec_b"], m["dec_a"])

have = m["p_vegas_a"].notna()
print(f"Predictions: {len(m)}")
print(f"Matched with odds: {have.sum()} ({have.mean():.1%})")
print(f"Odds coverage: {raw['DATE'].min().date()} → {raw['DATE'].max().date()}")

# Show unmatched per fold
unmatched = m[~have]
print(f"\nUnmatched per fold:")
for fn, sub in unmatched.groupby("fold_num"):
    print(f"  fold {fn}: {len(sub)}")

# ── Metrics ───────────────────────────────────────────────────────────
def metrics_row(sub):
    if len(sub) == 0: return dict(n=0)
    y   = sub["y_true"].values
    pm  = sub["p_model"].clip(1e-6,1-1e-6).values
    pv  = sub["p_vegas_a"].clip(1e-6,1-1e-6).values
    acc_m = ((pm>=.5)==y).mean()
    acc_v = ((pv>=.5)==y).mean()
    ll_m  = -np.mean(y*np.log(pm)+(1-y)*np.log(1-pm))
    ll_v  = -np.mean(y*np.log(pv)+(1-y)*np.log(1-pv))
    br_m  = np.mean((pm-y)**2)
    br_v  = np.mean((pv-y)**2)
    pick_a = pm >= 0.5
    odds_p = np.where(pick_a, sub["dec_odds_a"], sub["dec_odds_b"])
    won    = np.where(pick_a, y==1, y==0)
    pnl    = np.where(won, odds_p - 1.0, -1.0)
    roi    = pnl.mean()*100
    vpick  = np.where(pick_a, pv, 1-pv)
    mpick  = np.where(pick_a, pm, 1-pm)
    edge   = mpick - vpick
    mask   = edge > EDGE_THRESH
    if mask.sum():
        roi_e = pnl[mask].mean()*100; n_e = int(mask.sum())
    else: roi_e = np.nan; n_e = 0
    return dict(n=len(sub), acc_m=acc_m, acc_v=acc_v,
                ll_m=ll_m, ll_v=ll_v, br_m=br_m, br_v=br_v,
                roi=roi, n_edge=n_e, roi_edge=roi_e,
                vig=sub["vig"].mean())

M = m[have].copy()
rows=[]
for fn in sorted(M["fold_num"].unique()):
    sub = M[M["fold_num"]==fn]
    info = folds[folds["fold_num"]==fn].iloc[0]
    rows.append(dict(fold=int(fn),
                     test_window=f"{info.test_start[5:]}→{info.test_end[5:]}",
                     n_total=int(info.n_bouts),
                     **metrics_row(sub)))

rows.append(dict(fold="ALL",
                 test_window=f"{folds.iloc[0].test_start[5:]}→{folds.iloc[-1].test_end[5:]}",
                 n_total=len(pred), **metrics_row(M)))
df = pd.DataFrame(rows)

pd.set_option("display.width",240); pd.set_option("display.max_columns",30)
print("\n" + "="*120)
print("PER-FOLD: MODEL vs VEGAS  (source: odds_table.csv, clamped ±100)")
print("="*120)
disp = df.copy()
for c in ["acc_m","acc_v"]:
    disp[c] = disp[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
for c in ["ll_m","ll_v","br_m","br_v"]:
    disp[c] = disp[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
for c in ["roi","roi_edge"]:
    disp[c] = disp[c].apply(lambda v: f"{v:+7.2f}%" if pd.notna(v) else "—")
disp["vig"] = disp["vig"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
disp["coverage"] = disp.apply(lambda r: f"{r['n']}/{r['n_total']}", axis=1)
cols = ["fold","test_window","coverage","acc_m","acc_v","ll_m","ll_v",
        "br_m","br_v","roi","n_edge","roi_edge","vig"]
print(disp[cols].to_string(index=False))

df.to_csv("data/tmp/model_vs_vegas_per_fold_v2.csv", index=False)
print(f"\nSaved: data/tmp/model_vs_vegas_per_fold_v2.csv")
