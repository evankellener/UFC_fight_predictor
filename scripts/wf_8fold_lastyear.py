"""8-fold WF on past 1 year only (2025-04-05 → 2026-04-05).
Each fold ~1.5 months. Sliding 8-yr training window.
Compares Blend OLD (no SoS) vs Blend NEW (with SoS).
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

DT="data/tmp"
TEST_FIRST=pd.Timestamp("2025-04-05"); TEST_LAST=pd.Timestamp("2026-04-05")
N_FOLDS=8; TRAIN_YEARS=8; DATA_START=pd.Timestamp("2016-01-01")
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
df['ix_sos_x_age']=s('sos_last5_diff')*s('age_diff')
df['ix_sos_x_elo']=s('sos_last5_diff')*s('elo_win_prob')
df['ix_form_x_layoff']=s('form_winrate5_diff')*s('days_since_last_fight_diff')
df['ix_traj_x_age']=s('elo_trajectory_diff')*s('age_diff')
ix_old=[c for c in df.columns if c.startswith('ix_') and not any(t in c for t in ['sos','form','traj'])]
ix_new=[c for c in df.columns if c.startswith('ix_') and any(t in c for t in ['sos','form','traj'])]
xgb_old   = fc + mc + ix_old
xgb_full  = fc + mc + sos_cols + ix_old + ix_new

all_needed=list(set(fc+mc+sos_cols+ix_old+ix_new))
df[all_needed]=df[all_needed].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)

def metrics(y,p):
    pred=(p>=0.5).astype(int)
    return accuracy_score(y,pred), log_loss(y,p), brier_score_loss(y,p), roc_auc_score(y,p)

# 8 equal folds spanning the year
span = (TEST_LAST - TEST_FIRST).days
folds=[]
for i in range(N_FOLDS):
    fs = TEST_FIRST + pd.Timedelta(days=int(round(i*span/N_FOLDS)))
    fe = TEST_FIRST + pd.Timedelta(days=int(round((i+1)*span/N_FOLDS))) if i<N_FOLDS-1 else TEST_LAST
    folds.append((fs, fe))
print(f"\n{N_FOLDS} folds across {TEST_FIRST.date()}..{TEST_LAST.date()}\n")

def run(label, xgb_cols):
    y_pool=[]; p_lr=[]; p_xb=[]; preds=[]
    print(f"\n{'fold':<26}{'n_te':>5}  {'lr_ll':>7} {'xgb_ll':>7} {'bl_ll':>7} {'bl_acc':>7}")
    for fs, fe in folds:
        train_start=max(DATA_START, fs - pd.DateOffset(years=TRAIN_YEARS))
        tr=df[(df.DATE>=train_start)&(df.DATE<fs)].copy()
        te=df[(df.DATE>=fs)&(df.DATE<fe)].copy()
        if len(te)==0: continue
        ytr=tr['win'].values; yte=te['win'].values
        w_tr=np.exp(-LAM*(fs-tr['DATE']).dt.days.values/365.0)
        sc=StandardScaler(); X1=sc.fit_transform(tr[fc]); X1t=sc.transform(te[fc])
        m=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
        m.fit(X1, ytr, sample_weight=w_tr); plr=m.predict_proba(X1t)[:,1]
        xb=XGBClassifier(n_estimators=1200,max_depth=4,learning_rate=0.015,subsample=0.7,colsample_bytree=0.6,reg_lambda=4.0,min_child_weight=20,eval_metric='logloss',tree_method='hist',random_state=42)
        xb.fit(tr[xgb_cols], ytr, sample_weight=w_tr)
        pxb=xb.predict_proba(te[xgb_cols])[:,1]
        pbl=0.5*plr+0.5*pxb
        y_pool.append(yte); p_lr.append(plr); p_xb.append(pxb)
        fdf=te[['DATE','jbout','jfighter','opp_jfighter','win']].copy(); fdf['p_blend']=pbl
        preds.append(fdf)
        a_lr=log_loss(yte,plr); a_xb=log_loss(yte,pxb); a_bl=log_loss(yte,pbl); ac_bl=accuracy_score(yte,(pbl>=0.5).astype(int))
        print(f"{fs.date()}..{fe.date()}    {len(te):>5}  {a_lr:>7.4f} {a_xb:>7.4f} {a_bl:>7.4f} {ac_bl:>7.4f}")
    y=np.concatenate(y_pool); plr=np.concatenate(p_lr); pxb=np.concatenate(p_xb); pbl=0.5*plr+0.5*pxb
    print(f"\n  POOLED {label}:")
    for n,p in [('lr',plr),('xgb',pxb),('blend',pbl)]:
        a,l,br,au=metrics(y,p)
        print(f"    {n:<6} acc={a:.4f}  ll={l:.4f}  brier={br:.4f}  auc={au:.4f}")
    return pd.concat(preds, ignore_index=True), pbl, y

print("="*78)
print("Variant 1: Blend OLD (no SoS in XGB)")
preds_old, p_old, y_old = run("OLD", xgb_old)

print("\n"+"="*78)
print("Variant 2: Blend NEW (XGB with SoS+ix)")
preds_new, p_new, y_new = run("NEW", xgb_full)

# ROI
def roi_for(allp, label):
    def n(x): return ''.join(str(x).lower().split())
    o=pd.read_csv(f"{DT}/odds_table.csv",parse_dates=["DATE"]); o['fkey']=o['FIGHTER'].map(n)
    yr=allp.copy()
    yr['fkey']=yr['jfighter'].map(n); yr['okey']=yr['opp_jfighter'].map(n)
    of=o[['DATE','fkey','prob_norm','odds']].rename(columns={'prob_norm':'vp_f','odds':'odds_f'})
    oo=o[['DATE','fkey','prob_norm','odds']].rename(columns={'fkey':'okey','prob_norm':'vp_o','odds':'odds_o'})
    md=yr.merge(of,on=['DATE','fkey'],how='left').merge(oo,on=['DATE','okey'],how='left').dropna(subset=['vp_f','vp_o']).copy()
    def pay(x): x=float(x); return x/100.0 if x>0 else 100.0/abs(x)
    md['payout_f']=md['odds_f'].map(pay); md['payout_o']=md['odds_o'].map(pay); md['won']=md['win'].astype(int)
    md['edge_f']=md['p_blend']-md['vp_f']; md['edge_o']=(1-md['p_blend'])-md['vp_o']
    recs=[]
    for _,r in md.iterrows():
        if r['edge_f']>r['edge_o']:
            e,p,w_,od=r['edge_f'],r['payout_f'],r['won']==1,r['odds_f']
        else:
            e,p,w_,od=r['edge_o'],r['payout_o'],r['won']==0,r['odds_o']
        recs.append({'edge':e,'payout':p,'won':w_,'profit':p if w_ else -1.0,'fav':od<0})
    bb=pd.DataFrame(recs)
    print(f"\n  [{label}]")
    for thr in [0,0.03,0.05,0.08,0.10]:
        s_=bb[bb['fav']&(bb['edge']>thr)]
        if len(s_)>0:
            print(f"    fav+edge>{thr*100:>4.1f}%: n={len(s_):>4}  wr={s_['won'].mean()*100:>5.1f}%  ROI={s_['profit'].sum()/len(s_)*100:>+6.2f}%  $={s_['profit'].sum():+.2f}")

print("\n"+"="*78)
print("ROI on past 1 year:")
roi_for(preds_old, "Blend OLD")
roi_for(preds_new, "Blend NEW")
