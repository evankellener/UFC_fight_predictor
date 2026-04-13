"""ROI test for the past 1 year (2025-04-05 → 2026-04-05) using the blend model.

3 walk-forward folds × 4 months. Each fold:
  - Train LR on baseline 199 + XGB-deep on baseline + market + interactions
  - Predict via 50/50 blend
  - Match predictions to odds, apply favorite+edge>3% strategy
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
FOLD_MONTHS=4; TRAIN_YEARS=8; DATA_START=pd.Timestamp("2016-01-01")
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
mc=['home_advantage_diff','travel_distance_diff_km','tz_diff_diff_hr','is_main_event','card_position_norm_career_diff','coming_off_loss_diff','win_streak_entering_diff','fights_last_12m_diff','stance_mismatch','southpaw_advantage_diff']
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
ix_cols=[c for c in df.columns if c.startswith('ix_')]
xgb_cols = fc + mc + ix_cols
all_needed = list(set(fc + mc + ix_cols))
df[all_needed]=df[all_needed].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)

# Folds
folds=[]; cur=TEST_FIRST
while cur<TEST_LAST:
    nxt=cur+pd.DateOffset(months=FOLD_MONTHS)
    folds.append((cur, min(nxt, TEST_LAST))); cur=nxt
print(f"\n{len(folds)} folds × {FOLD_MONTHS} months, sliding {TRAIN_YEARS}-yr train")

# Generate blended preds for all test bouts
preds=[]
for fs, fe in folds:
    train_start=max(DATA_START, fs - pd.DateOffset(years=TRAIN_YEARS))
    tr=df[(df.DATE>=train_start)&(df.DATE<fs)].copy()
    te=df[(df.DATE>=fs)&(df.DATE<fe)].copy()
    if len(te)==0: continue
    ytr=tr['win'].values
    w_tr=np.exp(-LAM*(fs-tr['DATE']).dt.days.values/365.0)
    sc=StandardScaler(); Xtr=sc.fit_transform(tr[fc]); Xte=sc.transform(te[fc])
    m=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
    m.fit(Xtr, ytr, sample_weight=w_tr)
    p_lr=m.predict_proba(Xte)[:,1]
    xb=XGBClassifier(n_estimators=1200,max_depth=4,learning_rate=0.015,subsample=0.7,colsample_bytree=0.6,reg_lambda=4.0,min_child_weight=20,eval_metric='logloss',tree_method='hist',random_state=42)
    xb.fit(tr[xgb_cols], ytr, sample_weight=w_tr)
    p_xb=xb.predict_proba(te[xgb_cols])[:,1]
    p_bl=0.5*p_lr + 0.5*p_xb
    fold_df=te[['DATE','jbout','jfighter','opp_jfighter','win']].copy()
    fold_df['p_lr']=p_lr; fold_df['p_xgb']=p_xb; fold_df['p_blend']=p_bl
    fold_df['fold']=f"{fs.date()}..{fe.date()}"
    preds.append(fold_df)
    print(f"  fold {fs.date()}..{fe.date()}: trained on {len(tr)}, predicted {len(te)}")

allp=pd.concat(preds, ignore_index=True)
print(f"\nTotal predictions: {len(allp)}  bouts unique: {allp[['DATE','jbout']].drop_duplicates().shape[0]}")

# Match odds
def n(x): return ''.join(str(x).lower().split())
o=pd.read_csv(f"{DT}/odds_table.csv",parse_dates=["DATE"])
o['fkey']=o['FIGHTER'].map(n)
allp['fkey']=allp['jfighter'].map(n); allp['okey']=allp['opp_jfighter'].map(n)
of=o[['DATE','fkey','prob_norm','odds']].rename(columns={'prob_norm':'vp_f','odds':'odds_f'})
oo=o[['DATE','fkey','prob_norm','odds']].rename(columns={'fkey':'okey','prob_norm':'vp_o','odds':'odds_o'})
md=allp.merge(of,on=['DATE','fkey'],how='left').merge(oo,on=['DATE','okey'],how='left')
md=md.dropna(subset=['vp_f','vp_o']).copy()
print(f"Odds-matched: {len(md)}")

def pay(x): x=float(x); return x/100.0 if x>0 else 100.0/abs(x)
md['payout_f']=md['odds_f'].map(pay); md['payout_o']=md['odds_o'].map(pay)
md['won']=md['win'].astype(int)

def roi_table(label, prob_col):
    md['edge_f']=md[prob_col]-md['vp_f']
    md['edge_o']=(1-md[prob_col])-md['vp_o']
    recs=[]
    for _,r in md.iterrows():
        if r['edge_f']>r['edge_o']:
            side,e,p,w_,od='f',r['edge_f'],r['payout_f'],r['won']==1,r['odds_f']
        else:
            side,e,p,w_,od='o',r['edge_o'],r['payout_o'],r['won']==0,r['odds_o']
        recs.append({'edge':e,'payout':p,'won':w_,'profit':p if w_ else -1.0,'underdog':od>0,'odds':od})
    bb=pd.DataFrame(recs); bb['fav']=~bb['underdog']
    print(f"\n[{label}]  {prob_col}")
    print(f"  bet-all:                    n={len(bb):>4}  wr={bb['won'].mean()*100:>5.1f}%  ROI={bb['profit'].sum()/len(bb)*100:>+6.2f}%")
    for thr in [0,0.03,0.05,0.08,0.10]:
        s=bb[bb['fav'] & (bb['edge']>thr)]
        if len(s)>0:
            print(f"  fav + edge>{thr*100:>4.1f}%:        n={len(s):>4}  wr={s['won'].mean()*100:>5.1f}%  ROI={s['profit'].sum()/len(s)*100:>+6.2f}%  $profit={s['profit'].sum():+.2f}")

# Compare LR-only vs Blend
roi_table("LR only", "p_lr")
roi_table("Blend",   "p_blend")

# Metric summary on past year
print("\n=== Model metrics on past-year bouts ===")
for col in ['p_lr','p_xgb','p_blend']:
    p=allp[col].values; y=allp['win'].values
    pred=(p>=0.5).astype(int)
    print(f"  {col:<10} acc={accuracy_score(y,pred):.4f}  ll={log_loss(y,p):.4f}  brier={brier_score_loss(y,p):.4f}  auc={roc_auc_score(y,p):.4f}")
