"""Stacking experiment: combine LR model prob + Vegas prob via 5-fold CV
on matched test bouts (2024-05+). Compares: model only, vegas only, stacked."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

DT = "data/tmp"
TRAIN_START = pd.Timestamp("2018-01-01"); TEST_START = pd.Timestamp("2024-05-01")

with open(f"{DT}/tau_optimized.json") as f: tau = json.load(f)
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
with open(f"{DT}/model_feat_cols.json") as f: feat_cols = json.load(f)

df = pd.read_csv(f"{DT}/mmaai_features.csv", parse_dates=["DATE"])
bouts = pd.read_csv(f"{DT}/elo_bouts.csv", parse_dates=["DATE"])
elo_df, *_ = compute_elo(bouts, K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
    decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0, logistic_scale=449.205,
    opp_quality_k=True, sliding_k=True, upset_momentum=True, champ_mult=1.2)
ELO=['precomp_elo_diff','elo_win_prob','elo_momentum_diff','peak_elo_diff','avg_opp_elo_diff','elo_consist_diff']
em = elo_df[['jbout','DATE','f1','f2']+ELO].copy(); em['DATE']=pd.to_datetime(em['DATE'])
df = df.merge(em, on=['jbout','DATE'], how='left')
flip = df['jfighter']==df['f2']
for c in ELO: df.loc[flip,c] = (1-df.loc[flip,c]) if c=='elo_win_prob' else -df.loc[flip,c]
df.drop(columns=['f1','f2'], inplace=True, errors='ignore')
for c in ELO: df[c] = df[c].fillna(0.5 if c=='elo_win_prob' else 0.0)

feat_cols = [c for c in feat_cols if c in df.columns]

train = df[(df.DATE>=TRAIN_START)&(df.DATE<TEST_START)].dropna(subset=['win']).copy()
test  = df[df.DATE>=TEST_START].dropna(subset=['win']).copy()
for d in (train, test):
    d[feat_cols] = d[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)

w = np.exp(-LAM*(TEST_START-train['DATE']).dt.days.values/365.0)
sc = StandardScaler(); Xtr=sc.fit_transform(train[feat_cols]); Xte=sc.transform(test[feat_cols])
m = LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
m.fit(Xtr, train['win'].astype(int).values, sample_weight=w)
test = test.assign(model_prob=m.predict_proba(Xte)[:,1])

# Match odds
def norm(s): return ''.join(str(s).lower().split())
odds = pd.read_csv(f"{DT}/odds_table.csv", parse_dates=["DATE"])
odds['fkey'] = odds['FIGHTER'].map(norm)
test['fkey']=test['jfighter'].map(norm); test['okey']=test['opp_jfighter'].map(norm)
of = odds[['DATE','fkey','prob_norm','odds']].rename(columns={'prob_norm':'vp_f','odds':'odds_f'})
oo = odds[['DATE','fkey','prob_norm','odds']].rename(columns={'fkey':'okey','prob_norm':'vp_o','odds':'odds_o'})
m_df = test.merge(of, on=['DATE','fkey'], how='left').merge(oo, on=['DATE','okey'], how='left')
m_df = m_df.dropna(subset=['vp_f','vp_o']).copy()
print(f"Matched bouts: {len(m_df)}")

y = m_df['win'].astype(int).values
mp = m_df['model_prob'].values
vp = m_df['vp_f'].values

# Also build stacking features: model_prob, vp, model_prob - vp, model_prob*vp
X_stack = np.column_stack([mp, vp, mp-vp, mp*vp])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
stacked_pred = np.zeros(len(y))
for tr_i, te_i in kf.split(X_stack):
    s = LogisticRegression(C=1.0, max_iter=2000)
    s.fit(X_stack[tr_i], y[tr_i])
    stacked_pred[te_i] = s.predict_proba(X_stack[te_i])[:,1]

def report(name, p):
    pred = (p>=0.5).astype(int)
    print(f"  {name:<14} acc={accuracy_score(y,pred):.4f}  ll={log_loss(y,p):.4f}  auc={roc_auc_score(y,p):.4f}")

print("\nOn the 962 matched bouts:")
report("model only", mp)
report("vegas only", vp)
report("stacked CV", stacked_pred)

def pay(o): o=float(o); return o/100.0 if o>0 else 100.0/abs(o)
m_df['payout_f']=m_df['odds_f'].map(pay); m_df['payout_o']=m_df['odds_o'].map(pay)
m_df['won']=y

def roi(label, mp_arr):
    print(f"\n[{label}]")
    print(f"{'thr':>6}{'bets':>6}{'wr':>8}{'roi':>8}")
    for thr in [0.0,0.03,0.05,0.08,0.10,0.15,0.20]:
        bets=wins=0; profit=0.0
        for i,(_,r) in enumerate(m_df.iterrows()):
            ef = mp_arr[i] - r['vp_f']; eo = (1-mp_arr[i]) - r['vp_o']
            if ef>thr:
                bets+=1
                if r['won']==1: wins+=1; profit+=r['payout_f']
                else: profit-=1
            elif eo>thr:
                bets+=1
                if r['won']==0: wins+=1; profit+=r['payout_o']
                else: profit-=1
        if bets:
            print(f"{thr:>6.2f}{bets:>6d}{wins/bets:>7.3f} {profit/bets*100:>7.2f}%")

roi("model only", mp)
roi("stacked",    stacked_pred)
