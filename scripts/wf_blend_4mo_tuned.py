"""4-month WF with (1) adaptive blend weight per fold via inner validation,
and (2) globally-tuned XGB hyperparams.

Step A: Tune XGB hyperparams once via grid search on a fixed validation
        period (2022-01-01 to 2022-05-01), train ≤ 2022-01-01.
Step B: For each test fold, hold out last 4 months of training as inner
        val set; fit LR + XGB on rest; choose blend weight w that minimizes
        inner-val LL; refit on FULL pre-fold data with chosen w; predict test.
"""
import json, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

DT="data/tmp"
TEST_FIRST=pd.Timestamp("2022-05-01"); TEST_LAST=pd.Timestamp("2026-04-05")
FOLD_MONTHS=4; TRAIN_YEARS=8; DATA_START=pd.Timestamp("2016-01-01")
INNER_VAL_MONTHS=4
TUNE_VAL_START=pd.Timestamp("2022-01-01"); TUNE_VAL_END=TEST_FIRST
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
xgb_cols = fc + market_cols + ix_cols
all_needed = list(set(fc + market_cols + ix_cols))
df[all_needed]=df[all_needed].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df=df.dropna(subset=['win']).copy(); df['win']=df['win'].astype(int)

def metrics(y,p):
    pred=(p>=0.5).astype(int)
    return accuracy_score(y,pred), log_loss(y,p), brier_score_loss(y,p), roc_auc_score(y,p)

def fit_lr(tr):
    sc=StandardScaler(); Xtr=sc.fit_transform(tr[fc])
    w=np.exp(-LAM*(tr['DATE'].max()+pd.Timedelta(days=1)-tr['DATE']).dt.days.values/365.0)
    m=LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
    m.fit(Xtr, tr['win'].values, sample_weight=w)
    return sc, m

def fit_xgb(tr, params):
    w=np.exp(-LAM*(tr['DATE'].max()+pd.Timedelta(days=1)-tr['DATE']).dt.days.values/365.0)
    xb=XGBClassifier(eval_metric='logloss',tree_method='hist',random_state=42,**params)
    xb.fit(tr[xgb_cols], tr['win'].values, sample_weight=w)
    return xb

# === Step A: Tune XGB hyperparams globally on 2022-01..2022-05 validation ===
print(f"\n[Step A] Tuning XGB hyperparams on {TUNE_VAL_START.date()}..{TUNE_VAL_END.date()}")
tune_train = df[(df.DATE>=DATA_START)&(df.DATE<TUNE_VAL_START)].copy()
tune_val   = df[(df.DATE>=TUNE_VAL_START)&(df.DATE<TUNE_VAL_END)].copy()
print(f"  Tune train: {len(tune_train)}  val: {len(tune_val)}")

grid = list(itertools.product(
    [800, 1500],                # n_estimators
    [3, 4, 5],                  # max_depth
    [0.015, 0.03],              # learning_rate
    [0.7],                      # subsample
    [0.6, 0.8],                 # colsample_bytree
    [2.0, 5.0, 10.0],           # reg_lambda
    [15, 30],                   # min_child_weight
))
print(f"  Grid size: {len(grid)}")

best=(9e9, None)
for i,(ne,md,lr_,ss,cs,rl,mc) in enumerate(grid):
    p=dict(n_estimators=ne,max_depth=md,learning_rate=lr_,subsample=ss,colsample_bytree=cs,reg_lambda=rl,min_child_weight=mc)
    xb=fit_xgb(tune_train, p)
    p_val=xb.predict_proba(tune_val[xgb_cols])[:,1]
    ll=log_loss(tune_val['win'].values, p_val)
    if ll < best[0]: best=(ll, p)
    if (i+1) % 50 == 0: print(f"    [{i+1}/{len(grid)}] best so far LL={best[0]:.5f}")
print(f"\n  Best XGB params: {best[1]}")
print(f"  Best val LL: {best[0]:.5f}")
XGB_PARAMS = best[1]

# === Step B: Walk-forward with adaptive blend weight per fold ===
folds=[]; cur=TEST_FIRST
while cur<TEST_LAST:
    nxt=cur+pd.DateOffset(months=FOLD_MONTHS)
    folds.append((cur, min(nxt, TEST_LAST))); cur=nxt

print(f"\n[Step B] {len(folds)}-fold WF with adaptive blend weight")
print(f"\n{'fold':<26}{'n_te':>6}{'w*':>6}  {'model':<7}{'acc':>8}{'ll':>9}{'brier':>9}{'auc':>9}")
y_all={'lr':[], 'xgb':[], 'blend_fixed':[], 'blend_adapt':[]}
p_all={'lr':[], 'xgb':[], 'blend_fixed':[], 'blend_adapt':[]}
fold_results=[]; chosen_ws=[]

for fs, fe in folds:
    train_start = max(DATA_START, fs - pd.DateOffset(years=TRAIN_YEARS))
    inner_split = fs - pd.DateOffset(months=INNER_VAL_MONTHS)
    inner_tr = df[(df.DATE>=train_start)&(df.DATE<inner_split)].copy()
    inner_val= df[(df.DATE>=inner_split)&(df.DATE<fs)].copy()
    full_tr  = df[(df.DATE>=train_start)&(df.DATE<fs)].copy()
    te=df[(df.DATE>=fs)&(df.DATE<fe)].copy()
    if len(te)==0 or len(inner_val)==0: continue

    # Choose blend weight on inner val
    sc_i, lr_i = fit_lr(inner_tr)
    xb_i = fit_xgb(inner_tr, XGB_PARAMS)
    p_lr_v = lr_i.predict_proba(sc_i.transform(inner_val[fc]))[:,1]
    p_xb_v = xb_i.predict_proba(inner_val[xgb_cols])[:,1]
    yv = inner_val['win'].values
    best_w, best_ll=0.5, 9e9
    for w in np.arange(0.0, 1.001, 0.05):
        p = w*p_xb_v + (1-w)*p_lr_v
        ll=log_loss(yv, p)
        if ll < best_ll: best_ll=ll; best_w=w
    chosen_ws.append(best_w)

    # Retrain on full pre-fold
    sc_f, lr_f = fit_lr(full_tr)
    xb_f = fit_xgb(full_tr, XGB_PARAMS)
    p_lr=lr_f.predict_proba(sc_f.transform(te[fc]))[:,1]
    p_xb=xb_f.predict_proba(te[xgb_cols])[:,1]
    p_bl_fix = 0.5*p_lr + 0.5*p_xb
    p_bl_ad  = best_w*p_xb + (1-best_w)*p_lr

    yte=te['win'].values
    label=f"{fs.date()}..{fe.date()}"
    fr={}
    for name, p in [('lr',p_lr),('xgb',p_xb),('blend_fixed',p_bl_fix),('blend_adapt',p_bl_ad)]:
        a,l,br,au = metrics(yte, p)
        y_all[name].append(yte); p_all[name].append(p)
        fr[name]=(a,l,br,au)
    fold_results.append((label, fr))
    a,l,br,au = fr['blend_adapt']
    print(f"{label:<26}{len(te):>6}{best_w:>5.2f}  blend  {a:>7.4f}{l:>9.4f}{br:>9.4f}{au:>9.4f}")

print("="*78)
print("Pooled (all folds concatenated):")
for name in ('lr','xgb','blend_fixed','blend_adapt'):
    y=np.concatenate(y_all[name]); p=np.concatenate(p_all[name])
    a,l,br,au=metrics(y,p)
    print(f"  {name:<13} n={len(y):>4}  acc={a:.4f}  ll={l:.4f}  brier={br:.4f}  auc={au:.4f}")

y=np.concatenate(y_all['lr']); a0,l0,br0,au0=metrics(y, np.concatenate(p_all['lr']))
print("\nDeltas vs LR pooled:")
for name in ('xgb','blend_fixed','blend_adapt'):
    p=np.concatenate(p_all[name]); a,l,br,au=metrics(y,p)
    print(f"  {name:<13} Δacc={a-a0:+.4f}  Δll={l-l0:+.5f}  Δbrier={br-br0:+.5f}  Δauc={au-au0:+.4f}")

ll_wins_a=sum(1 for _,fr in fold_results if fr['blend_adapt'][1] < fr['lr'][1])
ll_wins_f=sum(1 for _,fr in fold_results if fr['blend_fixed'][1] < fr['lr'][1])
print(f"\nLL wins vs LR — adaptive: {ll_wins_a}/{len(fold_results)}  fixed: {ll_wins_f}/{len(fold_results)}")
print(f"Chosen blend weights per fold: {[f'{w:.2f}' for w in chosen_ws]}")
print(f"Mean adaptive weight on XGB: {np.mean(chosen_ws):.2f}")
