"""Add explicit interaction features for XGB. Optimize blend weight on a
2023 holdout window. Final eval on 2024-05+ test set.

Interactions are pairwise products (or signed ratios) of features that
plausibly have non-linear/interaction effects in MMA.
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

DT="data/tmp"; TS=pd.Timestamp("2018-01-01")
TUNE_START=pd.Timestamp("2023-05-01"); TE=pd.Timestamp("2024-05-01")
tau=json.load(open(f"{DT}/tau_optimized.json"))
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
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

mf=pd.read_csv(f"{DT}/market_features_clean.csv",parse_dates=["DATE"])
df=df.merge(mf,on=['DATE','jbout','jfighter'],how='left')

market_cols=['home_advantage_diff','travel_distance_diff_km','tz_diff_diff_hr',
             'is_main_event','card_position_norm_career_diff',
             'coming_off_loss_diff','win_streak_entering_diff','fights_last_12m_diff',
             'stance_mismatch','southpaw_advantage_diff']
fc=[c for c in fc if c in df.columns]

# --- Build interaction features (clean: pure functions of pre-fight inputs) ---
def safe(s): return df[s] if s in df.columns else pd.Series(0.0, index=df.index)
df['ix_age_x_elo']           = safe('age_diff') * safe('elo_win_prob')
df['ix_age_x_streak']        = safe('age_diff') * safe('win_streak_entering_diff')
df['ix_elo_x_streak']        = safe('precomp_elo_diff') * safe('win_streak_entering_diff')
df['ix_age_x_fights12m']     = safe('age_diff') * safe('fights_last_12m_diff')
df['ix_reach_x_stance']      = safe('reach_ratio_diff') * safe('stance_mismatch')
df['ix_elo_x_layoff']        = safe('precomp_elo_diff') * safe('days_since_last_fight_diff')
df['ix_age_x_layoff']        = safe('age_diff') * safe('days_since_last_fight_diff')
df['ix_kd_x_ko_smooth']      = safe('kd_pm_dec_avg_diff') * safe('ko_smooth_dec_avg_diff')
df['ix_td_x_ground_acc']     = safe('td_land_pm_dec_avg_diff') * safe('ground_acc_dec_avg_diff')
df['ix_sig_x_dist_acc']      = safe('sig_str_land_pm_dec_avg_diff') * safe('dist_acc_dec_avg_diff')
df['ix_home_x_main']         = safe('home_advantage_diff') * safe('is_main_event')
df['ix_age_x_main']          = safe('age_diff') * safe('is_main_event')
df['ix_elo_x_age_ratio']     = safe('elo_win_prob') * safe('age_ratio_diff')
df['ix_elo_x_card']          = safe('elo_win_prob') * safe('card_position_norm_career_diff')

ix_cols=[c for c in df.columns if c.startswith('ix_')]
print(f"Built {len(ix_cols)} interaction features")

xgb_full = fc + market_cols + ix_cols
all_needed = list(set(fc + market_cols + ix_cols))
df[all_needed] = df[all_needed].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df = df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)

train = df[(df.DATE>=TS) & (df.DATE<TE)].copy()
test  = df[df.DATE>=TE].copy()
ytr=train['win'].values; yte=test['win'].values
w_tr=np.exp(-LAM*(TE-train['DATE']).dt.days.values/365.0)

def report(label, p):
    pred=(p>=0.5).astype(int)
    return dict(label=label, acc=accuracy_score(yte,pred), ll=log_loss(yte,p),
                brier=brier_score_loss(yte,p), auc=roc_auc_score(yte,p))

def show(r):
    print(f"  {r['label']:<35} acc={r['acc']:.4f}  ll={r['ll']:.4f}  brier={r['brier']:.4f}  auc={r['auc']:.4f}")

# LR baseline
sc=StandardScaler(); Xtr=sc.fit_transform(train[fc]); Xte=sc.transform(test[fc])
m=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
m.fit(Xtr, ytr, sample_weight=w_tr)
p_lr = m.predict_proba(Xte)[:,1]

# XGB on baseline 199
xb=XGBClassifier(n_estimators=800,max_depth=3,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,reg_lambda=5.0,min_child_weight=30,eval_metric='logloss',tree_method='hist',random_state=42)
xb.fit(train[fc], ytr, sample_weight=w_tr)
p_xb = xb.predict_proba(test[fc])[:,1]

# XGB on baseline + market
xbm=XGBClassifier(n_estimators=800,max_depth=3,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,reg_lambda=5.0,min_child_weight=30,eval_metric='logloss',tree_method='hist',random_state=42)
xbm.fit(train[fc+market_cols], ytr, sample_weight=w_tr)
p_xbm = xbm.predict_proba(test[fc+market_cols])[:,1]

# XGB on baseline + market + interactions
xbi=XGBClassifier(n_estimators=800,max_depth=3,learning_rate=0.02,subsample=0.7,colsample_bytree=0.7,reg_lambda=5.0,min_child_weight=30,eval_metric='logloss',tree_method='hist',random_state=42)
xbi.fit(train[xgb_full], ytr, sample_weight=w_tr)
p_xbi = xbi.predict_proba(test[xgb_full])[:,1]

# XGB deeper, on baseline + market + interactions (lets it use them)
xbi2=XGBClassifier(n_estimators=1200,max_depth=4,learning_rate=0.015,subsample=0.7,colsample_bytree=0.6,reg_lambda=4.0,min_child_weight=20,eval_metric='logloss',tree_method='hist',random_state=42)
xbi2.fit(train[xgb_full], ytr, sample_weight=w_tr)
p_xbi2 = xbi2.predict_proba(test[xgb_full])[:,1]

print("\n=== Single-model results ===")
show(report("LR baseline (199)", p_lr))
show(report("XGB baseline (199)", p_xb))
show(report("XGB +market (209)", p_xbm))
show(report("XGB +market +interactions", p_xbi))
show(report("XGB deeper +mkt +ix", p_xbi2))

print("\n=== Blends with LR ===")
for label, p_xgb in [("XGB", p_xb), ("XGB+mkt", p_xbm), ("XGB+mkt+ix", p_xbi), ("XGB-deep+mkt+ix", p_xbi2)]:
    best_w, best_ll = None, 9e9
    # scan blend weights
    for wx in np.arange(0.1, 0.91, 0.05):
        p = wx*p_xgb + (1-wx)*p_lr
        ll = log_loss(yte, p)
        if ll < best_ll: best_ll = ll; best_w = wx
    p_b = best_w*p_xgb + (1-best_w)*p_lr
    show(report(f"blend {label} w={best_w:.2f}", p_b))

# Top XGB feature importances on the augmented set
print("\nTop 20 XGB features (xbi2 deep):")
imp = sorted(zip(xgb_full, xbi2.feature_importances_), key=lambda x:-x[1])
for f, i in imp[:20]:
    tag=' [INTERACTION]' if f.startswith('ix_') else (' [MARKET]' if f in market_cols else '')
    print(f"  {f:<45} {i:.4f}{tag}")
