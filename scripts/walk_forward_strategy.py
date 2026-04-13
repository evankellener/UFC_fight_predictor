"""Walk-forward validation of the favorite+edge>thr strategy.

Refits LR every 6 months on all data up to fold start, predicts next 6 months,
applies favorite+edge filter, reports ROI per fold and overall.
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DT = "data/tmp"
TRAIN_START = pd.Timestamp("2018-01-01")
TEST_FIRST  = pd.Timestamp("2024-05-01")
TEST_LAST   = pd.Timestamp("2026-04-05")
FOLD_MONTHS = 6

with open(f"{DT}/tau_optimized.json") as f: tau = json.load(f)
LR_C, LR_L1, LAM = tau["lr_C"], tau["lr_l1"], tau["recency_lambda"]
with open(f"{DT}/model_feat_cols.json") as f: feat_cols = json.load(f)

print("Loading features + computing Elo (once)...")
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
df[feat_cols] = df[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
df = df.dropna(subset=['win']).copy()
df['win'] = df['win'].astype(int)

# Odds prep
def norm(s): return ''.join(str(s).lower().split())
odds = pd.read_csv(f"{DT}/odds_table.csv", parse_dates=["DATE"])
odds['fkey'] = odds['FIGHTER'].map(norm)
of = odds[['DATE','fkey','prob_norm','odds']].rename(columns={'prob_norm':'vp_f','odds':'odds_f'})
oo = odds[['DATE','fkey','prob_norm','odds']].rename(columns={'fkey':'okey','prob_norm':'vp_o','odds':'odds_o'})
def pay(o): o=float(o); return o/100.0 if o>0 else 100.0/abs(o)

# Build folds
folds = []
cur = TEST_FIRST
while cur < TEST_LAST:
    nxt = cur + pd.DateOffset(months=FOLD_MONTHS)
    folds.append((cur, min(nxt, TEST_LAST)))
    cur = nxt

print(f"\nFolds ({len(folds)} × {FOLD_MONTHS} months):")
all_bets = []

for f_start, f_end in folds:
    train = df[(df.DATE >= TRAIN_START) & (df.DATE < f_start)].copy()
    test  = df[(df.DATE >= f_start) & (df.DATE < f_end)].copy()
    if len(test) == 0: continue
    days = (f_start - train['DATE']).dt.days.values
    w = np.exp(-LAM * days / 365.0)
    sc = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols].values)
    Xte = sc.transform(test[feat_cols].values)
    m = LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1, solver='saga', max_iter=4000)
    m.fit(Xtr, train['win'].values, sample_weight=w)
    test = test.assign(model_prob=m.predict_proba(Xte)[:,1])
    test['fkey']=test['jfighter'].map(norm); test['okey']=test['opp_jfighter'].map(norm)
    md = test.merge(of, on=['DATE','fkey'], how='left').merge(oo, on=['DATE','okey'], how='left')
    md = md.dropna(subset=['vp_f','vp_o']).copy()
    md['payout_f']=md['odds_f'].map(pay); md['payout_o']=md['odds_o'].map(pay)
    md['edge_f']=md['model_prob']-md['vp_f']; md['edge_o']=(1-md['model_prob'])-md['vp_o']
    md['won']=md['win']
    for _, r in md.iterrows():
        if r['edge_f'] > r['edge_o']:
            side, edge, payout, won, ods = 'f', r['edge_f'], r['payout_f'], r['won']==1, r['odds_f']
        else:
            side, edge, payout, won, ods = 'o', r['edge_o'], r['payout_o'], r['won']==0, r['odds_o']
        all_bets.append({'fold_start':f_start,'fold_end':f_end,'date':r['DATE'],
                         'edge':edge,'payout':payout,'won':won,
                         'profit':payout if won else -1.0,
                         'odds':ods,'underdog':ods>0})

b = pd.DataFrame(all_bets)
b['fav'] = ~b['underdog']

print(f"\n{'fold':<24}{'all_bets':>10}{'fav_e>3%':>12}{'wr':>8}{'roi':>9}")
for f_start, f_end in folds:
    s = b[b['fold_start']==f_start]
    fs = s[s['fav'] & (s['edge']>0.03)]
    label = f"{f_start.date()}..{f_end.date()}"
    if len(fs)>0:
        print(f"{label:<24}{len(s):>10d}{len(fs):>12d}{fs['won'].mean()*100:>7.1f}%{fs['profit'].sum()/len(fs)*100:>+8.2f}%")
    else:
        print(f"{label:<24}{len(s):>10d}{0:>12d}     -        -")

print(f"\n=== Walk-forward strategy summary (favorite + edge > X) ===")
for thr in [0, 0.03, 0.05, 0.08, 0.10]:
    s = b[b['fav'] & (b['edge']>thr)]
    if len(s):
        print(f"  edge>{thr*100:>4.1f}%  n={len(s):>4d}  wr={s['won'].mean()*100:>5.1f}%  roi={s['profit'].sum()/len(s)*100:>+6.2f}%  total_profit=${s['profit'].sum():>+7.2f}")

print(f"\nFor reference — bet-everything baseline: n={len(b)}  roi={b['profit'].sum()/len(b)*100:+.2f}%")
print(f"All-favorites (no edge filter):       n={(~b['underdog']).sum()}  roi={b[b['fav']]['profit'].sum()/b['fav'].sum()*100:+.2f}%")

b.to_csv(f"{DT}/walk_forward_bets.csv", index=False)
print(f"\nSaved per-bet log: {DT}/walk_forward_bets.csv")
