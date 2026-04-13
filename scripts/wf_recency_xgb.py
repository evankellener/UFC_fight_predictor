"""Recency-weighted XGB experiment: shorter training window for XGB only.

LR keeps 8-yr window (it likes data; elastic net regularizes noise).
XGB: try 4-yr and 6-yr sliding windows. Same SoS+market+ix features.

Compares blend metrics + past-1-year ROI across XGB window lengths.
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
FOLD_MONTHS=4; LR_TRAIN_YEARS=8; DATA_START=pd.Timestamp("2016-01-01")
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
df['ix_sos_x_age']=s('sos_last5_diff')*s('age_diff')
df['ix_sos_x_elo']=s('sos_last5_diff')*s('elo_win_prob')
df['ix_form_x_layoff']=s('form_winrate5_diff')*s('days_since_last_fight_diff')
df['ix_traj_x_age']=s('elo_trajectory_diff')*s('age_diff')
ix_cols=[c for c in df.columns if c.startswith('ix_')]
xgb_cols = fc + mc + sos_cols + ix_cols
all_needed=list(set(fc+mc+sos_cols+ix_cols))
df[all_needed]=df[all_needed].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)

def metrics(y,p):
    pred=(p>=0.5).astype(int)
    return accuracy_score(y,pred), log_loss(y,p), brier_score_loss(y,p), roc_auc_score(y,p)

folds=[]; cur=TEST_FIRST
while cur<TEST_LAST:
    nxt=cur+pd.DateOffset(months=FOLD_MONTHS)
    folds.append((cur, min(nxt, TEST_LAST))); cur=nxt
print(f"\n{len(folds)} folds × {FOLD_MONTHS}mo, LR train={LR_TRAIN_YEARS}yr\n")

def run(xgb_train_years):
    y_pool=[]; p_lr_pool=[]; p_xb_pool=[]; preds=[]
    for fs, fe in folds:
        lr_start  = max(DATA_START, fs - pd.DateOffset(years=LR_TRAIN_YEARS))
        xgb_start = max(DATA_START, fs - pd.DateOffset(years=xgb_train_years))
        lr_tr  = df[(df.DATE>=lr_start)&(df.DATE<fs)].copy()
        xgb_tr = df[(df.DATE>=xgb_start)&(df.DATE<fs)].copy()
        te     = df[(df.DATE>=fs)&(df.DATE<fe)].copy()
        if len(te)==0: continue
        ytr_lr=lr_tr['win'].values; ytr_xb=xgb_tr['win'].values; yte=te['win'].values
        w_lr=np.exp(-LAM*(fs-lr_tr['DATE']).dt.days.values/365.0)
        w_xb=np.exp(-LAM*(fs-xgb_tr['DATE']).dt.days.values/365.0)
        sc=StandardScaler(); X1=sc.fit_transform(lr_tr[fc]); X1t=sc.transform(te[fc])
        m=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
        m.fit(X1, ytr_lr, sample_weight=w_lr); plr=m.predict_proba(X1t)[:,1]
        xb=XGBClassifier(n_estimators=1200,max_depth=4,learning_rate=0.015,subsample=0.7,colsample_bytree=0.6,reg_lambda=4.0,min_child_weight=20,eval_metric='logloss',tree_method='hist',random_state=42)
        xb.fit(xgb_tr[xgb_cols], ytr_xb, sample_weight=w_xb)
        pxb=xb.predict_proba(te[xgb_cols])[:,1]
        pbl=0.5*plr+0.5*pxb
        y_pool.append(yte); p_lr_pool.append(plr); p_xb_pool.append(pxb)
        fdf=te[['DATE','jbout','jfighter','opp_jfighter','win']].copy()
        fdf['p_blend']=pbl
        preds.append(fdf)
    y=np.concatenate(y_pool); plr=np.concatenate(p_lr_pool); pxb=np.concatenate(p_xb_pool); pbl=0.5*plr+0.5*pxb
    return pd.concat(preds, ignore_index=True), metrics(y, pbl), metrics(y, pxb), metrics(y, plr)

def show(label, mb, mx, ml):
    print(f"  {label:<22} blend acc={mb[0]:.4f} ll={mb[1]:.4f} brier={mb[2]:.4f} auc={mb[3]:.4f}  | xgb_alone ll={mx[1]:.4f} | lr_alone ll={ml[1]:.4f}")

print("=== Pooled metrics (n=2011, 12 folds) ===")
results = {}
for yr in [3, 4, 6, 8]:
    preds, mb, mx, ml = run(yr)
    results[yr] = (preds, mb, mx, ml)
    show(f"XGB train={yr}yr", mb, mx, ml)

# Past-1-year ROI for each
def roi_for(allp, label):
    yr_df = allp[allp['DATE']>=ROI_START].copy()
    def n(x): return ''.join(str(x).lower().split())
    o=pd.read_csv(f"{DT}/odds_table.csv",parse_dates=["DATE"]); o['fkey']=o['FIGHTER'].map(n)
    yr_df['fkey']=yr_df['jfighter'].map(n); yr_df['okey']=yr_df['opp_jfighter'].map(n)
    of=o[['DATE','fkey','prob_norm','odds']].rename(columns={'prob_norm':'vp_f','odds':'odds_f'})
    oo=o[['DATE','fkey','prob_norm','odds']].rename(columns={'fkey':'okey','prob_norm':'vp_o','odds':'odds_o'})
    md=yr_df.merge(of,on=['DATE','fkey'],how='left').merge(oo,on=['DATE','okey'],how='left').dropna(subset=['vp_f','vp_o']).copy()
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
    line=f"  [{label:<18}] "
    for thr in [0,0.03,0.05]:
        s_=bb[bb['fav']&(bb['edge']>thr)]
        if len(s_)>0:
            line += f"e>{thr*100:.0f}%:n={len(s_)}/wr={s_['won'].mean()*100:.1f}%/ROI={s_['profit'].sum()/len(s_)*100:+.2f}%  "
    print(line)

print("\n=== Past-1-year ROI (favorite-only, multiple thresholds) ===")
for yr, (preds, *_ ) in results.items():
    roi_for(preds, f"XGB={yr}yr")
