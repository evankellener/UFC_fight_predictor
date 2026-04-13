"""XGBoost on baseline features ± market features. Compares to LR baseline."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

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

mf=pd.read_csv(f"{DT}/market_features_clean.csv", parse_dates=["DATE"])
df=df.merge(mf, on=['DATE','jbout','jfighter'], how='left')

market_cols=['home_advantage_diff','travel_distance_diff_km','tz_diff_diff_hr',
             'is_main_event','card_position_norm_career_diff',
             'coming_off_loss_diff','win_streak_entering_diff','fights_last_12m_diff',
             'stance_mismatch','southpaw_advantage_diff']

fc_full=[c for c in fc_full if c in df.columns]
allcols=fc_full+market_cols
df[allcols]=df[allcols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)

train=df[(df.DATE>=TS)&(df.DATE<TE)].copy()
test =df[df.DATE>=TE].copy()
ytr=train['win'].values; yte=test['win'].values
w=np.exp(-LAM*(TE-train['DATE']).dt.days.values/365.0)

def report(label, p):
    pred=(p>=0.5).astype(int)
    print(f"  {label:<35} acc={accuracy_score(yte,pred):.4f}  ll={log_loss(yte,p):.4f}  brier={brier_score_loss(yte,p):.4f}  auc={roc_auc_score(yte,p):.4f}")

print("BASELINE LR (199 cols):")
sc=StandardScaler(); Xtr=sc.fit_transform(train[fc_full]); Xte=sc.transform(test[fc_full])
m=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
m.fit(Xtr, ytr, sample_weight=w)
p_lr=m.predict_proba(Xte)[:,1]
report("LR baseline", p_lr)

# XGB grid: a few configs
configs = [
    dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=10),
    dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=20),
    dict(n_estimators=800, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, min_child_weight=30),
]

print("\nXGB on baseline 199 cols:")
best_xgb_p = None; best_xgb_ll = 9e9; best_cfg = None
for cfg in configs:
    x=XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=42, **cfg)
    x.fit(train[fc_full], ytr, sample_weight=w)
    p=x.predict_proba(test[fc_full])[:,1]
    ll=log_loss(yte,p)
    report(f"XGB {cfg['n_estimators']}/{cfg['max_depth']}/{cfg['learning_rate']}", p)
    if ll < best_xgb_ll: best_xgb_ll=ll; best_xgb_p=p; best_cfg=cfg

print("\nXGB on baseline + market (10 extra cols):")
best_xgbm_p = None; best_xgbm_ll = 9e9
for cfg in configs:
    x=XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=42, **cfg)
    x.fit(train[allcols], ytr, sample_weight=w)
    p=x.predict_proba(test[allcols])[:,1]
    ll=log_loss(yte,p)
    report(f"XGB+market {cfg['n_estimators']}/{cfg['max_depth']}/{cfg['learning_rate']}", p)
    if ll < best_xgbm_ll: best_xgbm_ll=ll; best_xgbm_p=p

# Blend best XGB with LR
print("\nBlends (best XGB + LR):")
for w_xgb in [0.3, 0.5, 0.7]:
    p_blend = w_xgb*best_xgb_p + (1-w_xgb)*p_lr
    report(f"blend xgb={w_xgb:.1f} lr={1-w_xgb:.1f}", p_blend)

# Blend XGB+market with LR
print("\nBlends (best XGB+market + LR):")
for w_xgb in [0.3, 0.5, 0.7]:
    p_blend = w_xgb*best_xgbm_p + (1-w_xgb)*p_lr
    report(f"blend xgbm={w_xgb:.1f} lr={1-w_xgb:.1f}", p_blend)
