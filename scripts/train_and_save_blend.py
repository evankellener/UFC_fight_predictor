"""Train the final LR + XGB blend on ALL available data and serialize artifacts
to app/models/blend/ for the Flask app to load at startup.

Artifacts saved:
  app/models/blend/lr.pkl              — fitted LogisticRegression
  app/models/blend/lr_scaler.pkl       — fitted StandardScaler (for LR features)
  app/models/blend/xgb.json            — fitted XGBClassifier (native format)
  app/models/blend/feat_lists.json     — { "lr_cols":[...], "xgb_cols":[...] }
  app/models/blend/training_df.csv     — full DataFrame used for training (for refit/diagnostics)
  app/models/blend/fighter_lookup.json — per-fighter latest aggregated diff features

This mirrors the notebook pipeline (cells 18+23) but trains on ALL data, not just
a walk-forward fold.
"""
import json, os, pickle, sqlite3, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
from elo_feature import compute_elo
from new_market_features import COUNTRY_GEO, EVENT_CITY_GEO, haversine, get_event_geo

DATA_TMP = Path("data/tmp")
APP_MODELS = Path("app/models/blend")
APP_MODELS.mkdir(parents=True, exist_ok=True)
SCRAPER_DB = "data/sqlite_db/sqlite_scrapper.db"
APP_DB     = "data/sqlite_db/app.db"

# ──────── Build features (same logic as notebook cell 18, condensed) ────────
df = pd.read_csv(DATA_TMP/"mmaai_features.csv", parse_dates=["DATE"])
bouts = pd.read_csv(DATA_TMP/"elo_bouts.csv", parse_dates=["DATE"])
elo_df, *_ = compute_elo(bouts, K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
    decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0, logistic_scale=449.205,
    opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2)
ELO = ["precomp_elo_diff","elo_win_prob","elo_momentum_diff","peak_elo_diff",
       "avg_opp_elo_diff","elo_consist_diff"]
em = elo_df[["jbout","DATE","f1","f2"]+ELO].copy(); em["DATE"] = pd.to_datetime(em["DATE"])
df = df.merge(em, on=["jbout","DATE"], how="left")
flip = df["jfighter"] == df["f2"]
for c in ELO:
    df.loc[flip,c] = (1-df.loc[flip,c]) if c=="elo_win_prob" else -df.loc[flip,c]
df.drop(columns=["f1","f2"], inplace=True, errors="ignore")
for c in ELO: df[c] = df[c].fillna(0.5 if c=="elo_win_prob" else 0.0)

# Market features
sc_db = sqlite3.connect(SCRAPER_DB); ac_db = sqlite3.connect(APP_DB)
tott = pd.read_sql("SELECT jfighter,STANCE FROM ufc_fighter_tott", sc_db)
stance_map = {r.jfighter:(r.STANCE or "").strip().lower() for r in tott.itertuples()}
events = pd.read_sql("SELECT jevent,DATE,LOCATION FROM ufc_event_details", sc_db)
events["DATE"] = pd.to_datetime(events["DATE"])
loc_map = dict(zip(events["jevent"], events["LOCATION"]))
nat = pd.read_sql("SELECT jfighter,country FROM ufc_fighter_nationality", ac_db)
nat_map = dict(zip(nat["jfighter"], nat["country"]))
cp = pd.read_sql("SELECT jevent,jbout,position,total_fights FROM ufc_card_position", ac_db)
cp_map = {(r.jevent, r.jbout):(((r.position-1)/max(r.total_fights-1,1)), int(r.position==1))
          for r in cp.itertuples()}
wlk = pd.read_sql("SELECT jfighter,jevent,jbout FROM ufc_winlossko", sc_db)
fp = wlk.merge(cp[["jevent","jbout","position","total_fights"]], on=["jevent","jbout"], how="inner")
fp = fp.merge(events[["jevent","DATE"]], on="jevent", how="left").dropna(subset=["DATE"])
fp["DATE"] = pd.to_datetime(fp["DATE"]); fp["pos_norm"] = (fp["position"]-1)/(fp["total_fights"]-1).clip(lower=1)
fp = fp.sort_values(["jfighter","DATE"])
career_cp = {j:list(zip(g["DATE"].values, g["pos_norm"].values)) for j,g in fp.groupby("jfighter")}
def career_pos_before(jf, dt):
    h = career_cp.get(jf, [])
    prior = [p for d,p in h if d < dt]
    return float(np.mean(prior)) if prior else 0.5

hist = []
for r in df[["DATE","jfighter","win"]].itertuples():
    hist.append((r.DATE, r.jfighter, int(r.win)))
hist_df = pd.DataFrame(hist, columns=["DATE","jfighter","won"]).sort_values(["jfighter","DATE"])
prior_loss, prior_streak, prior_12m = {}, {}, {}
for jf, g in hist_df.groupby("jfighter"):
    dates = g["DATE"].values; wins = g["won"].values
    for i, dt in enumerate(dates):
        key = (pd.Timestamp(dt), jf)
        if i == 0:
            prior_loss[key]=0; prior_streak[key]=0; prior_12m[key]=0
        else:
            prior_loss[key] = 0 if wins[i-1] else 1
            s=0
            for j in range(i-1,-1,-1):
                if wins[j]: s+=1
                else: break
            prior_streak[key] = s
            cutoff = dates[i] - np.timedelta64(365,"D")
            prior_12m[key] = int(sum(1 for d in dates[:i] if d >= cutoff))

def stance_code(s):
    if not isinstance(s,str): return 0
    s=s.lower()
    return {"orthodox":1,"southpaw":2,"switch":3}.get(s,0)

mrows=[]
for r in df[["DATE","jevent","jbout","jfighter","opp_jfighter"]].itertuples():
    jf, opp, dt, jevent, jbout = r.jfighter, r.opp_jfighter, r.DATE, r.jevent, r.jbout
    loc = loc_map.get(jevent, "")
    evt_geo = get_event_geo(loc)
    f1c, f2c = nat_map.get(jf), nat_map.get(opp)
    f1g = COUNTRY_GEO.get(f1c) if f1c else None
    f2g = COUNTRY_GEO.get(f2c) if f2c else None
    evt_country = (loc.split(",")[-1].strip() if isinstance(loc,str) and "," in loc else "") or ""
    evt_country = {"USA":"United States","UK":"United Kingdom","England":"United Kingdom",
                   "Scotland":"United Kingdom","Wales":"United Kingdom"}.get(evt_country, evt_country)
    f1_home = int(f1c == evt_country) if f1c else 0
    f2_home = int(f2c == evt_country) if f2c else 0
    td1, tz1, td2, tz2 = np.nan, np.nan, np.nan, np.nan
    if evt_geo and f1g: td1, tz1 = haversine(f1g[0],f1g[1],evt_geo[0],evt_geo[1]), abs(f1g[2]-evt_geo[2])
    if evt_geo and f2g: td2, tz2 = haversine(f2g[0],f2g[1],evt_geo[0],evt_geo[1]), abs(f2g[2]-evt_geo[2])
    travel_diff = (td1-td2) if not (np.isnan(td1) or np.isnan(td2)) else 0.0
    tz_diff     = (tz1-tz2) if not (np.isnan(tz1) or np.isnan(tz2)) else 0.0
    pn, im = cp_map.get((jevent, jbout), (0.5, 0))
    cp1 = career_pos_before(jf, dt); cp2 = career_pos_before(opp, dt)
    s1 = stance_code(stance_map.get(jf,"")); s2 = stance_code(stance_map.get(opp,""))
    mrows.append({"DATE":dt,"jbout":jbout,"jfighter":jf,
        "home_advantage_diff":f1_home-f2_home,"travel_distance_diff_km":travel_diff,"tz_diff_diff_hr":tz_diff,
        "is_main_event":int(im),"card_position_norm_career_diff":cp1-cp2,
        "coming_off_loss_diff":prior_loss.get((dt,jf),0)-prior_loss.get((dt,opp),0),
        "win_streak_entering_diff":prior_streak.get((dt,jf),0)-prior_streak.get((dt,opp),0),
        "fights_last_12m_diff":prior_12m.get((dt,jf),0)-prior_12m.get((dt,opp),0),
        "stance_mismatch":int(s1!=s2 and s1>0 and s2>0),
        "southpaw_advantage_diff":int(s1==2)-int(s2==2)})
df = df.merge(pd.DataFrame(mrows), on=["DATE","jbout","jfighter"], how="left")

# SoS + form
per_fight=[]
for r in elo_df.itertuples():
    per_fight.append((r.DATE, r.f1, r.precomp_elo_f2, int(r.f1_win==1), r.precomp_elo_f1))
    per_fight.append((r.DATE, r.f2, r.precomp_elo_f1, int(r.f1_win!=1), r.precomp_elo_f2))
ph = pd.DataFrame(per_fight, columns=["DATE","jfighter","opp_elo","won","own_elo"]).sort_values(["jfighter","DATE"])
fhist = {j:list(zip(g["DATE"].values, g["opp_elo"].values, g["won"].values, g["own_elo"].values))
         for j,g in ph.groupby("jfighter")}
def sos_before(jf, dt):
    h = fhist.get(jf, [])
    prior = [(d,oe,w,me) for d,oe,w,me in h if d < dt]
    n = len(prior)
    if n == 0: return (np.nan,)*6 + (0,)
    last3, last5 = prior[-3:], prior[-5:]
    sos3 = float(np.mean([oe for _,oe,_,_ in last3]))
    sos5 = float(np.mean([oe for _,oe,_,_ in last5]))
    if n >= 6:
        prior3 = prior[-6:-3]
        sos_traj = sos3 - float(np.mean([oe for _,oe,_,_ in prior3]))
        elo_traj = prior[-1][3] - prior[-6][3]
    else:
        sos_traj, elo_traj = 0.0, 0.0
    wr3 = float(np.mean([w for _,_,w,_ in last3]))
    wr5 = float(np.mean([w for _,_,w,_ in last5]))
    return sos3, sos5, sos_traj, wr3, wr5, elo_traj, n

srows=[]
for r in df[["DATE","jbout","jfighter","opp_jfighter"]].itertuples():
    sj = sos_before(r.jfighter, r.DATE); so = sos_before(r.opp_jfighter, r.DATE)
    srows.append({"DATE":r.DATE,"jbout":r.jbout,"jfighter":r.jfighter,
        "sos_last3_diff":(sj[0]-so[0]) if not (np.isnan(sj[0]) or np.isnan(so[0])) else 0.0,
        "sos_last5_diff":(sj[1]-so[1]) if not (np.isnan(sj[1]) or np.isnan(so[1])) else 0.0,
        "sos_trajectory_diff":sj[2]-so[2],
        "form_winrate3_diff":(sj[3]-so[3]) if not (np.isnan(sj[3]) or np.isnan(so[3])) else 0.0,
        "form_winrate5_diff":(sj[4]-so[4]) if not (np.isnan(sj[4]) or np.isnan(so[4])) else 0.0,
        "elo_trajectory_diff":sj[5]-so[5],
        "career_fights_diff":sj[6]-so[6]})
df = df.merge(pd.DataFrame(srows), on=["DATE","jbout","jfighter"], how="left")

# Column lists
with open(DATA_TMP/"model_feat_cols.json") as f: feat_cols = json.load(f)
feat_cols = [c for c in feat_cols if c in df.columns]
market_cols = ["home_advantage_diff","travel_distance_diff_km","tz_diff_diff_hr","is_main_event",
               "card_position_norm_career_diff","coming_off_loss_diff","win_streak_entering_diff",
               "fights_last_12m_diff","stance_mismatch","southpaw_advantage_diff"]
sos_cols = ["sos_last3_diff","sos_last5_diff","sos_trajectory_diff","form_winrate3_diff",
            "form_winrate5_diff","elo_trajectory_diff","career_fights_diff"]

def S(col): return df[col] if col in df.columns else pd.Series(0.0, index=df.index)
df["ix_age_x_elo"]=S("age_diff")*S("elo_win_prob")
df["ix_age_x_streak"]=S("age_diff")*S("win_streak_entering_diff")
df["ix_elo_x_streak"]=S("precomp_elo_diff")*S("win_streak_entering_diff")
df["ix_age_x_fights12m"]=S("age_diff")*S("fights_last_12m_diff")
df["ix_reach_x_stance"]=S("reach_ratio_diff")*S("stance_mismatch")
df["ix_elo_x_layoff"]=S("precomp_elo_diff")*S("days_since_last_fight_diff")
df["ix_age_x_layoff"]=S("age_diff")*S("days_since_last_fight_diff")
df["ix_kd_x_ko_smooth"]=S("kd_pm_dec_avg_diff")*S("ko_smooth_dec_avg_diff")
df["ix_td_x_ground_acc"]=S("td_land_pm_dec_avg_diff")*S("ground_acc_dec_avg_diff")
df["ix_sig_x_dist_acc"]=S("sig_str_land_pm_dec_avg_diff")*S("dist_acc_dec_avg_diff")
df["ix_home_x_main"]=S("home_advantage_diff")*S("is_main_event")
df["ix_age_x_main"]=S("age_diff")*S("is_main_event")
df["ix_elo_x_age_ratio"]=S("elo_win_prob")*S("age_ratio_diff")
df["ix_elo_x_card"]=S("elo_win_prob")*S("card_position_norm_career_diff")
df["ix_sos_x_age"]=S("sos_last5_diff")*S("age_diff")
df["ix_sos_x_elo"]=S("sos_last5_diff")*S("elo_win_prob")
df["ix_form_x_layoff"]=S("form_winrate5_diff")*S("days_since_last_fight_diff")
df["ix_traj_x_age"]=S("elo_trajectory_diff")*S("age_diff")
ix_cols = [c for c in df.columns if c.startswith("ix_")]

xgb_cols = feat_cols + market_cols + sos_cols + ix_cols
all_cols = list(set(feat_cols + market_cols + sos_cols + ix_cols))
df[all_cols] = df[all_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
df = df.dropna(subset=["win"]).copy()
df["win"] = df["win"].astype(int)
print(f"Training rows: {len(df)}")

# ──────── Train on ALL data ────────
with open(DATA_TMP/"tau_optimized.json") as f: tau = json.load(f)
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]

TRAIN_START = pd.Timestamp("2018-01-01")
tr = df[df.DATE >= TRAIN_START].copy()
end_date = tr["DATE"].max() + pd.Timedelta(days=1)
w_tr = np.exp(-LAM*(end_date - tr["DATE"]).dt.days.values/365.0)
ytr = tr["win"].values

scaler = StandardScaler(); Xtr = scaler.fit_transform(tr[feat_cols])
lr = LogisticRegression(C=LR_C, penalty="elasticnet", l1_ratio=LR_L1, solver="saga", max_iter=4000)
lr.fit(Xtr, ytr, sample_weight=w_tr)
xb = XGBClassifier(n_estimators=1200, max_depth=4, learning_rate=0.015, subsample=0.7,
                   colsample_bytree=0.6, reg_lambda=4.0, min_child_weight=20,
                   eval_metric="logloss", tree_method="hist", random_state=42)
xb.fit(tr[xgb_cols], ytr, sample_weight=w_tr)
print(f"Fit LR (elastic net, C={LR_C}, l1={LR_L1}) + XGB (depth=4, n=1200)")

# ──────── Save artifacts ────────
with open(APP_MODELS/"lr.pkl","wb") as f: pickle.dump(lr, f)
with open(APP_MODELS/"lr_scaler.pkl","wb") as f: pickle.dump(scaler, f)
xb.save_model(str(APP_MODELS/"xgb.json"))
with open(APP_MODELS/"feat_lists.json","w") as f:
    json.dump({"lr_cols":feat_cols, "xgb_cols":xgb_cols,
               "market_cols":market_cols, "sos_cols":sos_cols, "ix_cols":ix_cols,
               "blend_weight_xgb":0.5, "trained_on_rows":int(len(tr)),
               "train_start":str(TRAIN_START.date()),
               "train_end":str(tr["DATE"].max().date())}, f, indent=2)

# Training dataframe (so the app can rebuild or inspect)
df.to_csv(APP_MODELS/"training_df.csv", index=False)

# Per-fighter lookup: for each fighter, find their MOST RECENT row (as jfighter perspective)
# and store their feature vector. At inference we'll diff jf-vs-opp by looking up both.
# NOTE: since the pipeline stores DIFF features (jf - opp), we cannot separate jf-only from opp-only easily.
# Instead, persist the full training rows keyed by (DATE, jfighter); for inference, find the LATEST
# available row the fighter appeared in and use it as a proxy.
latest_by_fighter = df.sort_values("DATE").groupby("jfighter").tail(1).set_index("jfighter")
latest_by_fighter.to_csv(APP_MODELS/"fighter_latest_row.csv")
print(f"Saved {len(latest_by_fighter)} per-fighter latest rows")

print(f"\nArtifacts written to {APP_MODELS}")
for p in sorted(APP_MODELS.iterdir()):
    print(f"  {p.name:<28} ({p.stat().st_size:>10,} bytes)")
