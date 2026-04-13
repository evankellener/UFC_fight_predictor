"""ROI test with augmented features."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

DATA_TMP = "data/tmp"
TRAIN_START = pd.Timestamp("2018-01-01")
TEST_START  = pd.Timestamp("2024-05-01")

with open(f"{DATA_TMP}/tau_optimized.json") as f: tau = json.load(f)
LR_C, LR_L1, LAM = tau.get("lr_C",0.05), tau.get("lr_l1",0.5), tau.get("recency_lambda",0.05)
with open(f"{DATA_TMP}/model_feat_cols.json") as f: feat_cols = json.load(f)
with open(f"{DATA_TMP}/aug_new_cols.json") as f: new_cols = json.load(f)

df = pd.read_csv(f"{DATA_TMP}/mmaai_features_aug.csv", parse_dates=["DATE"])

# Elo
bouts = pd.read_csv(f"{DATA_TMP}/elo_bouts.csv", parse_dates=["DATE"])
elo_df, ratings, *_ = compute_elo(
    bouts, K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
    decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
    logistic_scale=449.205, opp_quality_k=True, sliding_k=True,
    upset_momentum=True, champ_mult=1.2)
ELO_COLS = ['precomp_elo_diff','elo_win_prob','elo_momentum_diff','peak_elo_diff','avg_opp_elo_diff','elo_consist_diff']
em = elo_df[['jbout','DATE','f1','f2']+ELO_COLS].copy(); em['DATE']=pd.to_datetime(em['DATE'])
df = df.merge(em, on=['jbout','DATE'], how='left')
flip = df['jfighter']==df['f2']
for c in ELO_COLS:
    df.loc[flip,c] = (1.0 - df.loc[flip,c]) if c=='elo_win_prob' else (-df.loc[flip,c])
df.drop(columns=['f1','f2'], inplace=True, errors='ignore')
for c in ELO_COLS: df[c] = df[c].fillna(0.5 if c=='elo_win_prob' else 0.0)

feat_cols = [c for c in feat_cols if c in df.columns]

def evaluate(label, cols):
    train = df[(df.DATE>=TRAIN_START)&(df.DATE<TEST_START)].dropna(subset=['win']).copy()
    test  = df[df.DATE>=TEST_START].dropna(subset=['win']).copy()
    train[cols] = train[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    test[cols]  = test[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    days = (TEST_START - train['DATE']).dt.days.values
    w = np.exp(-LAM*days/365.0)
    sc = StandardScaler()
    Xtr = sc.fit_transform(train[cols].values); Xte = sc.transform(test[cols].values)
    ytr = train['win'].astype(int).values; yte = test['win'].astype(int).values
    m = LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
    m.fit(Xtr, ytr, sample_weight=w)
    p = m.predict_proba(Xte)[:,1]
    pred = (p>=0.5).astype(int)
    acc = accuracy_score(yte, pred); ll = log_loss(yte, p); auc = roc_auc_score(yte, p)
    print(f"\n[{label}] cols={len(cols)}  Acc={acc:.4f}  LL={ll:.4f}  AUC={auc:.4f}  n={len(yte)}")
    test = test.assign(model_prob=p)
    return test, m

def roi_table(test, label):
    def norm(s): return ''.join(str(s).lower().split())
    odds = pd.read_csv(f"{DATA_TMP}/odds_table.csv", parse_dates=["DATE"])
    odds['fkey'] = odds['FIGHTER'].map(norm); odds['DATE']=pd.to_datetime(odds['DATE'])
    test=test.copy()
    test['fkey']=test['jfighter'].map(norm); test['okey']=test['opp_jfighter'].map(norm)
    of = odds[['DATE','fkey','prob_norm','odds']].rename(columns={'prob_norm':'vp_f','odds':'odds_f'})
    oo = odds[['DATE','fkey','prob_norm','odds']].rename(columns={'fkey':'okey','prob_norm':'vp_o','odds':'odds_o'})
    t = test.merge(of, on=['DATE','fkey'], how='left').merge(oo, on=['DATE','okey'], how='left')
    m = t.dropna(subset=['vp_f','vp_o']).copy()
    def pay(o): o=float(o); return o/100.0 if o>0 else 100.0/abs(o)
    m['payout_f']=m['odds_f'].map(pay); m['payout_o']=m['odds_o'].map(pay)
    m['edge_f']=m['model_prob']-m['vp_f']; m['edge_o']=(1-m['model_prob'])-m['vp_o']
    m['won']=m['win'].astype(int)
    print(f"\nROI [{label}]  matched={len(m)}")
    print(f"{'thr':>6}{'bets':>6}{'wr':>8}{'roi':>8}")
    for thr in [0.0,0.03,0.05,0.08,0.10,0.15,0.20]:
        bets=wins=0; profit=0.0
        for _,r in m.iterrows():
            if r['edge_f']>thr:
                bets+=1
                if r['won']: wins+=1; profit+=r['payout_f']
                else: profit-=1
            elif r['edge_o']>thr:
                bets+=1
                if not r['won']: wins+=1; profit+=r['payout_o']
                else: profit-=1
        if bets:
            print(f"{thr:>6.2f}{bets:>6d}{wins/bets:>7.3f} {profit/bets*100:>7.2f}%")
        else:
            print(f"{thr:>6.2f}{bets:>6d}     -       -")

# Baseline
test_b, m_b = evaluate("baseline", feat_cols)
roi_table(test_b, "baseline")

# Augmented
aug_cols = feat_cols + new_cols
test_a, m_a = evaluate("augmented", aug_cols)
roi_table(test_a, "augmented")

# Coefficient inspection on new cols
print("\nNew-col coefficients (augmented model):")
sc = StandardScaler()
train = df[(df.DATE>=TRAIN_START)&(df.DATE<TEST_START)].dropna(subset=['win']).copy()
train[aug_cols] = train[aug_cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
sc.fit(train[aug_cols].values)
coefs = dict(zip(aug_cols, m_a.coef_[0]))
for c in new_cols:
    print(f"  {c:<26} {coefs.get(c,0):+.4f}")
