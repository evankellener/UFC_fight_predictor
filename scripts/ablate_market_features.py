"""Per-feature ablation: add ONE market feature at a time to the baseline,
measure ΔLL, ΔAcc, ΔBrier, ΔAUC. Keep features that improve LL on the test set.

Final pass: train with the kept subset and report metrics + walk-forward.
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

DT="data/tmp"; TS=pd.Timestamp("2018-01-01"); TE=pd.Timestamp("2024-05-01")
tau=json.load(open(f"{DT}/tau_optimized.json"))
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
fc_full=json.load(open(f"{DT}/model_feat_cols.json"))

df=pd.read_csv(f"{DT}/mmaai_features.csv", parse_dates=["DATE"])
b=pd.read_csv(f"{DT}/elo_bouts.csv", parse_dates=["DATE"])
ed,*_=compute_elo(b,K=48.0,ko_mult=1.80,sub_mult=1.20,decay_lambda=0.923,decay_max=0.25,decay_midpoint=730.0,decay_steepness=80.0,logistic_scale=449.205,opp_quality_k=True,sliding_k=True,upset_momentum=True,champ_mult=1.2)
EL=['precomp_elo_diff','elo_win_prob','elo_momentum_diff','peak_elo_diff','avg_opp_elo_diff','elo_consist_diff']
em=ed[['jbout','DATE','f1','f2']+EL].copy(); em['DATE']=pd.to_datetime(em['DATE'])
df=df.merge(em,on=['jbout','DATE'],how='left')
fl=df['jfighter']==df['f2']
for c in EL: df.loc[fl,c]=(1-df.loc[fl,c]) if c=='elo_win_prob' else -df.loc[fl,c]
df.drop(columns=['f1','f2'],inplace=True,errors='ignore')
for c in EL: df[c]=df[c].fillna(0.5 if c=='elo_win_prob' else 0.0)

# Merge market features
mf=pd.read_csv(f"{DT}/market_features_clean.csv", parse_dates=["DATE"])
df=df.merge(mf, on=['DATE','jbout','jfighter'], how='left')

market_cols=['home_advantage_diff','travel_distance_diff_km','tz_diff_diff_hr',
             'is_main_event','card_position_norm_career_diff',
             'coming_off_loss_diff','win_streak_entering_diff','fights_last_12m_diff',
             'stance_mismatch','southpaw_advantage_diff']

fc_full=[c for c in fc_full if c in df.columns]
all_needed=fc_full+market_cols
df[all_needed]=df[all_needed].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy()
df['win']=df['win'].astype(int)

train=df[(df.DATE>=TS)&(df.DATE<TE)].copy()
test =df[df.DATE>=TE].copy()
w=np.exp(-LAM*(TE-train['DATE']).dt.days.values/365.0)

def fit_eval(cols):
    sc=StandardScaler()
    Xtr=sc.fit_transform(train[cols].values); Xte=sc.transform(test[cols].values)
    m=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
    m.fit(Xtr, train['win'].values, sample_weight=w)
    p=m.predict_proba(Xte)[:,1]; pred=(p>=0.5).astype(int); y=test['win'].values
    return dict(acc=accuracy_score(y,pred), ll=log_loss(y,p), brier=brier_score_loss(y,p), auc=roc_auc_score(y,p))

base=fit_eval(fc_full)
print(f"\nBaseline (199 cols): acc={base['acc']:.4f}  ll={base['ll']:.4f}  brier={base['brier']:.4f}  auc={base['auc']:.4f}")
print(f"\n{'feature':<35}{'Δacc':>8}{'Δll':>9}{'Δbrier':>10}{'Δauc':>8}  keep?")
keepers=[]
for f in market_cols:
    r=fit_eval(fc_full+[f])
    dacc=r['acc']-base['acc']; dll=r['ll']-base['ll']; dbr=r['brier']-base['brier']; dau=r['auc']-base['auc']
    keep = dll < -0.0001
    if keep: keepers.append(f)
    flag = "YES" if keep else ""
    print(f"{f:<35}{dacc:>+7.4f}{dll:>+9.5f}{dbr:>+10.5f}{dau:>+8.4f}  {flag}")

print(f"\nKeepers ({len(keepers)}): {keepers}")
final=fit_eval(fc_full+keepers)
print(f"\nFinal (baseline + keepers): acc={final['acc']:.4f}  ll={final['ll']:.4f}  brier={final['brier']:.4f}  auc={final['auc']:.4f}")
print(f"  delta vs baseline: acc{final['acc']-base['acc']:+.4f}  ll{final['ll']-base['ll']:+.5f}  brier{final['brier']-base['brier']:+.5f}  auc{final['auc']-base['auc']:+.4f}")

# Also try ALL market cols dumped in (for comparison)
all_in=fit_eval(fc_full+market_cols)
print(f"\nAll 10 market cols dumped in: acc={all_in['acc']:.4f}  ll={all_in['ll']:.4f}  brier={all_in['brier']:.4f}  auc={all_in['auc']:.4f}")

with open(f"{DT}/market_keepers.json","w") as f:
    json.dump(keepers, f, indent=2)
print(f"\nSaved keepers list to {DT}/market_keepers.json")
