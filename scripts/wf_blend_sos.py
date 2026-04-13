"""WF blend with new SoS + form features added.

Tests whether adding the 7 SoS/form features to LR and XGB improves WF metrics
vs the prior best blend. Also re-runs the past-1-year ROI test.
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

DT="data/tmp"
TEST_FIRST=pd.Timestamp("2022-05-01"); TEST_LAST=pd.Timestamp("2026-04-05")
FOLD_MONTHS=4; TRAIN_YEARS=8; DATA_START=pd.Timestamp("2016-01-01")
ROI_START=pd.Timestamp("2025-04-05")
tau=json.load(open(f"{DT}/tau_optimized.json"))
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
fc=json.load(open(f"{DT}/model_feat_cols.json"))

print("Loading + Elo…")
df=pd.read_csv(f"{DT}/mmaai_features.csv",parse_dates=["DATE"])
b=pd.read_csv(f"{DT}/elo_bouts.csv",parse_dates=["DATE"])
ed,*_=compute_elo(b,K=48.0,ko_mult=1.80,sub_mult=1.20,decay_lambda=0.923,decay_max=0.25,decay_midpoint=730.0,decay_steepness=80.0,logistic_scale=449.205,opp_quality_k=True,sliding_k=True,upset_momentum=True,champ_mult=1.2)
EL=['precomp_elo_diff','elo_win_prob','elo_momentum_diff','peak_elo_diff','avg_opp_elo_diff','elo_consist_diff']
em=ed[['jbout','DATE','f1','f2']+EL].copy(); em['DATE']=pd.to_datetime(em['DATE'])
df=df.merge(em,on=['jbout','DATE'],how='left')
fl=df['jfighter']==df['f2']
for c in EL: df.loc[fl,c]=(1-df.loc[fl,c]) if c=='elo_win_prob' else -df.loc[fl,c]
df.drop(columns=['f1','f2'],inplace=True,errors='ignore')
for c in EL: df[c]=df[c].fillna(0.5 if c=='elo_win_prob' else 0.0)

mf=pd.read_csv(f"{DT}/market_features_clean.csv",parse_dates=["DATE"])
df=df.merge(mf,on=['DATE','jbout','jfighter'],how='left')
sf=pd.read_csv(f"{DT}/sos_form_features.csv",parse_dates=["DATE"])
df=df.merge(sf,on=['DATE','jbout','jfighter'],how='left')

mc=['home_advantage_diff','travel_distance_diff_km','tz_diff_diff_hr','is_main_event','card_position_norm_career_diff','coming_off_loss_diff','win_streak_entering_diff','fights_last_12m_diff','stance_mismatch','southpaw_advantage_diff']
sos_cols=['sos_last3_diff','sos_last5_diff','sos_trajectory_diff','form_winrate3_diff','form_winrate5_diff','elo_trajectory_diff','career_fights_diff']
fc=[c for c in fc if c in df.columns]
def s(col): return df[col] if col in df.columns else pd.Series(0.0, index=df.index)
df['ix_age_x_elo']=s('age_diff')*s('elo_win_prob')
df['ix_age_x_streak']=s('age_diff')*s('win_streak_entering_diff')
df['ix_elo_x_streak']=s('precomp_elo_diff')*s('win_streak_entering_diff')
df['ix_age_x_fights12m']=s('age_diff')*s('fights_last_12m_diff')
df['ix_reach_x_stance']=s('reach_ratio_diff')*s('stance_mismatch')
df['ix_elo_x_layoff']=s('precomp_elo_diff')*s('days_since_last_fight_diff')
df['ix_age_x_layoff']=s('age_diff')*s('days_since_last_fight_diff')
df['ix_kd_x_ko_smooth']=s('kd_pm_dec_avg_diff')*s('ko_smooth_dec_avg_diff')
df['ix_td_x_ground_acc']=s('td_land_pm_dec_avg_diff')*s('ground_acc_dec_avg_diff')
df['ix_sig_x_dist_acc']=s('sig_str_land_pm_dec_avg_diff')*s('dist_acc_dec_avg_diff')
df['ix_home_x_main']=s('home_advantage_diff')*s('is_main_event')
df['ix_age_x_main']=s('age_diff')*s('is_main_event')
df['ix_elo_x_age_ratio']=s('elo_win_prob')*s('age_ratio_diff')
df['ix_elo_x_card']=s('elo_win_prob')*s('card_position_norm_career_diff')
# New: SoS-driven interactions for XGB
df['ix_sos_x_age']        = s('sos_last5_diff') * s('age_diff')
df['ix_sos_x_elo']        = s('sos_last5_diff') * s('elo_win_prob')
df['ix_form_x_layoff']    = s('form_winrate5_diff') * s('days_since_last_fight_diff')
df['ix_traj_x_age']       = s('elo_trajectory_diff') * s('age_diff')
ix_cols=[c for c in df.columns if c.startswith('ix_')]

xgb_cols_old = fc + mc + [c for c in ix_cols if not any(t in c for t in ['sos','form','traj'])]
xgb_cols_new = fc + mc + sos_cols + ix_cols
lr_cols_old  = fc
lr_cols_new  = fc + sos_cols
all_needed = list(set(fc + mc + sos_cols + ix_cols))
df[all_needed]=df[all_needed].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)

def metrics(y,p):
    pred=(p>=0.5).astype(int)
    return accuracy_score(y,pred), log_loss(y,p), brier_score_loss(y,p), roc_auc_score(y,p)

folds=[]; cur=TEST_FIRST
while cur<TEST_LAST:
    nxt=cur+pd.DateOffset(months=FOLD_MONTHS)
    folds.append((cur, min(nxt, TEST_LAST))); cur=nxt
print(f"\n{len(folds)} folds × {FOLD_MONTHS} months")

# 4 systems: LR_old / LR_new / Blend_old / Blend_new
y_all={'lr_old':[], 'lr_new':[], 'blend_old':[], 'blend_new':[]}
p_all={k:[] for k in y_all}
preds_per_row=[]  # for ROI: keep DATE/jbout/jfighter/p_blend_new/win

for fs, fe in folds:
    train_start=max(DATA_START, fs - pd.DateOffset(years=TRAIN_YEARS))
    tr=df[(df.DATE>=train_start)&(df.DATE<fs)].copy()
    te=df[(df.DATE>=fs)&(df.DATE<fe)].copy()
    if len(te)==0: continue
    ytr=tr['win'].values; yte=te['win'].values
    w_tr=np.exp(-LAM*(fs-tr['DATE']).dt.days.values/365.0)

    # LR old (199)
    sc=StandardScaler(); X1=sc.fit_transform(tr[lr_cols_old]); X1t=sc.transform(te[lr_cols_old])
    m1=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
    m1.fit(X1, ytr, sample_weight=w_tr); p_lr_old=m1.predict_proba(X1t)[:,1]
    # LR new (199 + 7)
    sc=StandardScaler(); X2=sc.fit_transform(tr[lr_cols_new]); X2t=sc.transform(te[lr_cols_new])
    m2=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
    m2.fit(X2, ytr, sample_weight=w_tr); p_lr_new=m2.predict_proba(X2t)[:,1]

    # XGB old (baseline + market + 14 ix)
    xb=XGBClassifier(n_estimators=1200,max_depth=4,learning_rate=0.015,subsample=0.7,colsample_bytree=0.6,reg_lambda=4.0,min_child_weight=20,eval_metric='logloss',tree_method='hist',random_state=42)
    xb.fit(tr[xgb_cols_old], ytr, sample_weight=w_tr)
    p_xb_old=xb.predict_proba(te[xgb_cols_old])[:,1]
    # XGB new (+ sos + sos-interactions)
    xb2=XGBClassifier(n_estimators=1200,max_depth=4,learning_rate=0.015,subsample=0.7,colsample_bytree=0.6,reg_lambda=4.0,min_child_weight=20,eval_metric='logloss',tree_method='hist',random_state=42)
    xb2.fit(tr[xgb_cols_new], ytr, sample_weight=w_tr)
    p_xb_new=xb2.predict_proba(te[xgb_cols_new])[:,1]

    p_bl_old = 0.5*p_lr_old + 0.5*p_xb_old
    p_bl_new = 0.5*p_lr_new + 0.5*p_xb_new

    for name, p in [('lr_old',p_lr_old),('lr_new',p_lr_new),('blend_old',p_bl_old),('blend_new',p_bl_new)]:
        y_all[name].append(yte); p_all[name].append(p)

    fold_df=te[['DATE','jbout','jfighter','opp_jfighter','win']].copy()
    fold_df['p_blend_new']=p_bl_new; fold_df['p_blend_old']=p_bl_old
    fold_df['p_lr_new']=p_lr_new
    preds_per_row.append(fold_df)

print("\n=== Pooled metrics across all 12 folds (n=2011) ===")
for name in ['lr_old','lr_new','blend_old','blend_new']:
    y=np.concatenate(y_all[name]); p=np.concatenate(p_all[name])
    a,l,br,au=metrics(y,p)
    print(f"  {name:<11} acc={a:.4f}  ll={l:.4f}  brier={br:.4f}  auc={au:.4f}")

print("\nDeltas vs blend_old:")
y=np.concatenate(y_all['blend_old']); a0,l0,br0,au0=metrics(y, np.concatenate(p_all['blend_old']))
for name in ['lr_new','blend_new']:
    p=np.concatenate(p_all[name]); a,l,br,au=metrics(y,p)
    print(f"  {name:<11} Δacc={a-a0:+.4f}  Δll={l-l0:+.5f}  Δbrier={br-br0:+.5f}  Δauc={au-au0:+.4f}")

# === Past-1-year ROI ===
allp=pd.concat(preds_per_row, ignore_index=True)
yr=allp[allp['DATE']>=ROI_START].copy()
print(f"\n=== Past-1-year ROI (since {ROI_START.date()}, n={len(yr)}) ===")

def n(x): return ''.join(str(x).lower().split())
o=pd.read_csv(f"{DT}/odds_table.csv",parse_dates=["DATE"]); o['fkey']=o['FIGHTER'].map(n)
yr['fkey']=yr['jfighter'].map(n); yr['okey']=yr['opp_jfighter'].map(n)
of=o[['DATE','fkey','prob_norm','odds']].rename(columns={'prob_norm':'vp_f','odds':'odds_f'})
oo=o[['DATE','fkey','prob_norm','odds']].rename(columns={'fkey':'okey','prob_norm':'vp_o','odds':'odds_o'})
md=yr.merge(of,on=['DATE','fkey'],how='left').merge(oo,on=['DATE','okey'],how='left').dropna(subset=['vp_f','vp_o']).copy()
def pay(x): x=float(x); return x/100.0 if x>0 else 100.0/abs(x)
md['payout_f']=md['odds_f'].map(pay); md['payout_o']=md['odds_o'].map(pay); md['won']=md['win'].astype(int)

def roi(label, prob_col):
    md['edge_f']=md[prob_col]-md['vp_f']
    md['edge_o']=(1-md[prob_col])-md['vp_o']
    recs=[]
    for _,r in md.iterrows():
        if r['edge_f']>r['edge_o']:
            e,p,w_,od=r['edge_f'],r['payout_f'],r['won']==1,r['odds_f']
        else:
            e,p,w_,od=r['edge_o'],r['payout_o'],r['won']==0,r['odds_o']
        recs.append({'edge':e,'payout':p,'won':w_,'profit':p if w_ else -1.0,'fav':od<0})
    bb=pd.DataFrame(recs)
    print(f"\n[{label}]")
    for thr in [0,0.03,0.05,0.08,0.10]:
        s=bb[bb['fav'] & (bb['edge']>thr)]
        if len(s)>0:
            print(f"  fav + edge>{thr*100:>4.1f}%: n={len(s):>4}  wr={s['won'].mean()*100:>5.1f}%  ROI={s['profit'].sum()/len(s)*100:>+6.2f}%  $={s['profit'].sum():+.2f}")
roi("Blend OLD (no SoS)", "p_blend_old")
roi("Blend NEW (with SoS)", "p_blend_new")
