"""8-fold walk-forward, each fold trained on the most recent 8 years of data.

Test span: 2024-05-01 → 2026-04-05 (24 months) split into 8 × 3-month folds.
Training window: sliding 8 years ending at fold start.
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

DT="data/tmp"
TEST_FIRST=pd.Timestamp("2024-05-01"); TEST_LAST=pd.Timestamp("2026-04-05")
N_FOLDS=8
TRAIN_YEARS=8
tau=json.load(open(f"{DT}/tau_optimized.json"))
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
fc=json.load(open(f"{DT}/model_feat_cols.json"))

print("Loading + Elo (once)…")
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

def s(col): return df[col] if col in df.columns else pd.Series(0.0, index=df.index)
df['ix_age_x_elo']           = s('age_diff')*s('elo_win_prob')
df['ix_age_x_streak']        = s('age_diff')*s('win_streak_entering_diff')
df['ix_elo_x_streak']        = s('precomp_elo_diff')*s('win_streak_entering_diff')
df['ix_age_x_fights12m']     = s('age_diff')*s('fights_last_12m_diff')
df['ix_reach_x_stance']      = s('reach_ratio_diff')*s('stance_mismatch')
df['ix_elo_x_layoff']        = s('precomp_elo_diff')*s('days_since_last_fight_diff')
df['ix_age_x_layoff']        = s('age_diff')*s('days_since_last_fight_diff')
df['ix_kd_x_ko_smooth']      = s('kd_pm_dec_avg_diff')*s('ko_smooth_dec_avg_diff')
df['ix_td_x_ground_acc']     = s('td_land_pm_dec_avg_diff')*s('ground_acc_dec_avg_diff')
df['ix_sig_x_dist_acc']      = s('sig_str_land_pm_dec_avg_diff')*s('dist_acc_dec_avg_diff')
df['ix_home_x_main']         = s('home_advantage_diff')*s('is_main_event')
df['ix_age_x_main']          = s('age_diff')*s('is_main_event')
df['ix_elo_x_age_ratio']     = s('elo_win_prob')*s('age_ratio_diff')
df['ix_elo_x_card']          = s('elo_win_prob')*s('card_position_norm_career_diff')
ix_cols=[c for c in df.columns if c.startswith('ix_')]
xgb_cols = fc + market_cols + ix_cols
all_needed = list(set(fc + market_cols + ix_cols))
df[all_needed]=df[all_needed].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)

def metrics(y,p):
    pred=(p>=0.5).astype(int)
    return accuracy_score(y,pred), log_loss(y,p), brier_score_loss(y,p), roc_auc_score(y,p)

# 8 folds of equal duration spanning [TEST_FIRST, TEST_LAST]
test_span_days = (TEST_LAST - TEST_FIRST).days
fold_days = test_span_days / N_FOLDS
folds = []
for i in range(N_FOLDS):
    fs = TEST_FIRST + pd.Timedelta(days=int(round(i*fold_days)))
    fe = TEST_FIRST + pd.Timedelta(days=int(round((i+1)*fold_days))) if i<N_FOLDS-1 else TEST_LAST
    folds.append((fs, fe))

print(f"\n{N_FOLDS}-fold walk-forward, sliding {TRAIN_YEARS}-year training window")
print(f"Test span: {TEST_FIRST.date()} → {TEST_LAST.date()} ({test_span_days} days)")
print(f"\n{'fold':<26}{'train_start':<14}{'n_tr':>6}{'n_te':>6}  {'model':<7}{'acc':>8}{'ll':>9}{'brier':>9}{'auc':>9}")
y_all={'lr':[], 'xgb':[], 'blend':[]}; p_all={'lr':[], 'xgb':[], 'blend':[]}
for fs, fe in folds:
    train_start = fs - pd.DateOffset(years=TRAIN_YEARS)
    tr=df[(df.DATE>=train_start)&(df.DATE<fs)].copy()
    te=df[(df.DATE>=fs)&(df.DATE<fe)].copy()
    if len(te)==0: continue
    ytr=tr['win'].values; yte=te['win'].values
    w_tr=np.exp(-LAM*(fs-tr['DATE']).dt.days.values/365.0)

    sc=StandardScaler(); Xtr=sc.fit_transform(tr[fc]); Xte=sc.transform(te[fc])
    m=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
    m.fit(Xtr, ytr, sample_weight=w_tr)
    p_lr=m.predict_proba(Xte)[:,1]

    xb=XGBClassifier(n_estimators=1200,max_depth=4,learning_rate=0.015,subsample=0.7,colsample_bytree=0.6,reg_lambda=4.0,min_child_weight=20,eval_metric='logloss',tree_method='hist',random_state=42)
    xb.fit(tr[xgb_cols], ytr, sample_weight=w_tr)
    p_xb=xb.predict_proba(te[xgb_cols])[:,1]

    p_bl = 0.5*p_lr + 0.5*p_xb
    label=f"{fs.date()}..{fe.date()}"
    for name, p in [('lr',p_lr),('xgb',p_xb),('blend',p_bl)]:
        a,l,br,au = metrics(yte, p)
        y_all[name].append(yte); p_all[name].append(p)
        print(f"{label:<26}{str(train_start.date()):<14}{len(tr):>6}{len(te):>6}  {name:<7}{a:>7.4f}{l:>9.4f}{br:>9.4f}{au:>9.4f}")
    print()

print("="*78)
print("Pooled (all 8 folds concatenated):")
for name in ('lr','xgb','blend'):
    y=np.concatenate(y_all[name]); p=np.concatenate(p_all[name])
    a,l,br,au=metrics(y,p)
    print(f"  {name:<7} n={len(y):>4}  acc={a:.4f}  ll={l:.4f}  brier={br:.4f}  auc={au:.4f}")

print("\nDeltas vs LR pooled:")
y=np.concatenate(y_all['lr']); a0,l0,br0,au0=metrics(y, np.concatenate(p_all['lr']))
for name in ('xgb','blend'):
    p=np.concatenate(p_all[name]); a,l,br,au=metrics(y,p)
    print(f"  {name:<7} Δacc={a-a0:+.4f}  Δll={l-l0:+.5f}  Δbrier={br-br0:+.5f}  Δauc={au-au0:+.4f}")
