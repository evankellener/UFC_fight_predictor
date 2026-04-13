"""Diagnose: which features does XGB rank highly but LR shrinks to 0?
Those are the non-linear/interaction features. Conversely, features where
LR has a big coef but XGB rates low are 'pure linear'.
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DT="data/tmp"; TS=pd.Timestamp("2018-01-01"); TE=pd.Timestamp("2024-05-01")
tau=json.load(open(f"{DT}/tau_optimized.json"))
fc=json.load(open(f"{DT}/model_feat_cols.json"))
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
fc=[c for c in fc if c in df.columns]
df[fc]=df[fc].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)
tr=df[(df.DATE>=TS)&(df.DATE<TE)].copy(); ytr=tr['win'].values
w=np.exp(-tau['recency_lambda']*(TE-tr['DATE']).dt.days.values/365.0)

# LR coefs (standardized)
sc=StandardScaler(); Xtr=sc.fit_transform(tr[fc])
m=LogisticRegression(C=tau['lr_C'],penalty='elasticnet',l1_ratio=tau['lr_l1'],solver='saga',max_iter=4000)
m.fit(Xtr, ytr, sample_weight=w)
lr_abs={f:abs(c) for f,c in zip(fc, m.coef_[0])}
# Normalize to rank pct
lr_ranks={f:(sum(1 for v in lr_abs.values() if v>lr_abs[f])+1) for f in fc}

# XGB importance (gain)
x=XGBClassifier(n_estimators=800,max_depth=3,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,reg_lambda=5.0,min_child_weight=30,eval_metric='logloss',tree_method='hist',random_state=42)
x.fit(tr[fc], ytr, sample_weight=w)
xgb_imp = dict(zip(fc, x.feature_importances_))
xgb_ranks={f:(sum(1 for v in xgb_imp.values() if v>xgb_imp[f])+1) for f in fc}

# Compare
out=[]
for f in fc:
    out.append({'feat':f,'lr_abs':lr_abs[f],'lr_rank':lr_ranks[f],
                'xgb_imp':xgb_imp[f],'xgb_rank':xgb_ranks[f],
                'rank_gap': lr_ranks[f] - xgb_ranks[f]})  # +ve = XGB likes more
o=pd.DataFrame(out)
N=len(fc)

print(f"Features XGB ranks much higher than LR (likely non-linear / interaction-heavy):")
print("-"*90)
top = o.nlargest(20,'rank_gap')
print(top[['feat','lr_abs','lr_rank','xgb_imp','xgb_rank','rank_gap']].to_string(index=False))

print(f"\nFeatures LR ranks much higher than XGB (likely 'pure linear'):")
print("-"*90)
top = o.nsmallest(20,'rank_gap')
print(top[['feat','lr_abs','lr_rank','xgb_imp','xgb_rank','rank_gap']].to_string(index=False))

print(f"\nFeatures LR zeroed AND XGB heavily uses (XGB-only candidates):")
print("-"*90)
zeros = o[o['lr_abs']<1e-6].nlargest(15,'xgb_imp')
print(zeros[['feat','xgb_imp','xgb_rank']].to_string(index=False))
print(f"\nLR zeroed total: {(o['lr_abs']<1e-6).sum()}/{N}")
