"""Build leakage-free strength-of-schedule + recent-form features.

Outputs data/tmp/sos_form_features.csv keyed by (DATE, jbout, jfighter).

Features:
  sos_last3_diff         : avg pre-fight Elo of last 3 opponents (jf - opp)
  sos_last5_diff         : avg pre-fight Elo of last 5 opponents (jf - opp)
  sos_trajectory_diff    : (last3 SoS - prior3 SoS); rising=positive
  form_winrate3_diff     : last-3 win rate (jf - opp)
  form_winrate5_diff     : last-5 win rate
  elo_trajectory_diff    : current pre-elo - elo 5 fights ago (jf - opp)
  career_fights_diff     : total prior fight count (jf - opp)
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo

DT="data/tmp"
base = pd.read_csv(f"{DT}/mmaai_features.csv",
                   usecols=["DATE","jevent","jbout","jfighter","opp_jfighter","win"],
                   parse_dates=["DATE"])
base['win']=base['win'].astype(int)
print(f"Base rows: {len(base)}")

# Compute Elo to get per-bout pre-fight Elos for both sides
b=pd.read_csv(f"{DT}/elo_bouts.csv", parse_dates=["DATE"])
ed,*_=compute_elo(b,K=48.0,ko_mult=1.80,sub_mult=1.20,decay_lambda=0.923,decay_max=0.25,decay_midpoint=730.0,decay_steepness=80.0,logistic_scale=449.205,opp_quality_k=True,sliding_k=True,upset_momentum=True,champ_mult=1.2)
ed['DATE']=pd.to_datetime(ed['DATE'])

# Build per-fighter timeline: (DATE, opp_pre_elo, won, own_pre_elo)
# Each bout produces 2 fighter-rows.
rows=[]
for r in ed.itertuples():
    rows.append((r.DATE, r.f1, r.precomp_elo_f2, int(r.f1_win==1), r.precomp_elo_f1))
    rows.append((r.DATE, r.f2, r.precomp_elo_f1, int(r.f1_win!=1), r.precomp_elo_f2))
hist = pd.DataFrame(rows, columns=['DATE','jfighter','opp_pre_elo','won','own_pre_elo'])
hist = hist.sort_values(['jfighter','DATE'])
print(f"Per-fighter fight rows: {len(hist)}")

# For each fighter, build chronological lists
fhist = {}
for jf, g in hist.groupby('jfighter'):
    fhist[jf] = list(zip(g['DATE'].values, g['opp_pre_elo'].values, g['won'].values, g['own_pre_elo'].values))

def stats_before(jf, dt):
    """Return (sos3, sos5, sos_traj, wr3, wr5, elo_traj, n_fights) using only fights with d < dt."""
    h = fhist.get(jf, [])
    prior = [(d,oe,w,me) for d,oe,w,me in h if d < dt]
    n = len(prior)
    if n == 0:
        return (np.nan,)*6 + (0,)
    last3 = prior[-3:]; last5 = prior[-5:]
    sos3 = float(np.mean([oe for _,oe,_,_ in last3]))
    sos5 = float(np.mean([oe for _,oe,_,_ in last5]))
    if n >= 6:
        prior3 = prior[-6:-3]
        sos_traj = sos3 - float(np.mean([oe for _,oe,_,_ in prior3]))
    else:
        sos_traj = 0.0
    wr3 = float(np.mean([w for _,_,w,_ in last3]))
    wr5 = float(np.mean([w for _,_,w,_ in last5]))
    if n >= 6:
        # current pre-elo (most recent fight's own_pre_elo, but that's the elo BEFORE the prior fight)
        # we want elo trajectory: elo_now (pre this fight) ~= last fight's post-elo, approximated via own_pre_elo at i=-1
        # Simpler: own_pre_elo at -1 minus own_pre_elo at -6
        elo_traj = prior[-1][3] - prior[-6][3]
    else:
        elo_traj = 0.0
    return sos3, sos5, sos_traj, wr3, wr5, elo_traj, n

out=[]
for r in base.itertuples():
    s_jf  = stats_before(r.jfighter,    r.DATE)
    s_opp = stats_before(r.opp_jfighter, r.DATE)
    out.append({
        'DATE':r.DATE,'jbout':r.jbout,'jfighter':r.jfighter,
        'sos_last3_diff':       (s_jf[0] - s_opp[0]) if not (np.isnan(s_jf[0]) or np.isnan(s_opp[0])) else 0.0,
        'sos_last5_diff':       (s_jf[1] - s_opp[1]) if not (np.isnan(s_jf[1]) or np.isnan(s_opp[1])) else 0.0,
        'sos_trajectory_diff':  s_jf[2] - s_opp[2],
        'form_winrate3_diff':   (s_jf[3] - s_opp[3]) if not (np.isnan(s_jf[3]) or np.isnan(s_opp[3])) else 0.0,
        'form_winrate5_diff':   (s_jf[4] - s_opp[4]) if not (np.isnan(s_jf[4]) or np.isnan(s_opp[4])) else 0.0,
        'elo_trajectory_diff':  s_jf[5] - s_opp[5],
        'career_fights_diff':   s_jf[6] - s_opp[6],
    })
df=pd.DataFrame(out)
df.to_csv(f"{DT}/sos_form_features.csv", index=False)
print(f"\nWrote {len(df)} rows × {df.shape[1]} cols -> {DT}/sos_form_features.csv")

print("\nSanity:")
for c in df.columns:
    if c in ('DATE','jbout','jfighter'): continue
    nz=(df[c]!=0).sum()
    print(f"  {c:<28} nonzero={nz}/{len(df)}  mean={df[c].mean():+.4f}  std={df[c].std():.4f}")
