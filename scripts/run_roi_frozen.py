"""ROI test on frozen-prior MMA-AI features.

Loads saved features, computes Elo, trains LR on 2018-01-01..2024-04-30,
evaluates on 2024-05-01+, matches predictions to odds, reports ROI by edge.
"""
import json, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from elo_feature import compute_elo

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

DATA_TMP = "data/tmp"
TRAIN_START = pd.Timestamp("2018-01-01")
TEST_START = pd.Timestamp("2024-05-01")

with open(f"{DATA_TMP}/tau_optimized.json") as f:
    tau = json.load(f)
LR_C, LR_L1, LAM = tau.get("lr_C", 0.05), tau.get("lr_l1", 0.5), tau.get("recency_lambda", 0.05)

with open(f"{DATA_TMP}/model_feat_cols.json") as f:
    feat_cols = json.load(f)

print(f"LR C={LR_C} l1={LR_L1} lam={LAM}  | feat_cols={len(feat_cols)}")

# --- Load features ---
df = pd.read_csv(f"{DATA_TMP}/mmaai_features.csv", parse_dates=["DATE"])
print(f"Features: {len(df)} rows  {df.DATE.min().date()} -> {df.DATE.max().date()}")

# --- Compute Elo and merge ---
bouts = pd.read_csv(f"{DATA_TMP}/elo_bouts.csv", parse_dates=["DATE"])
elo_df, ratings, last_date, extra = compute_elo(
    bouts, K=48.0, ko_mult=1.80, sub_mult=1.20, decay_lambda=0.923,
    decay_max=0.25, decay_midpoint=730.0, decay_steepness=80.0,
    logistic_scale=449.205, opp_quality_k=True, sliding_k=True,
    upset_momentum=True, champ_mult=1.2)

ELO_COLS = ['precomp_elo_diff','elo_win_prob','elo_momentum_diff',
            'peak_elo_diff','avg_opp_elo_diff','elo_consist_diff']
elo_merge = elo_df[['jbout','DATE','f1','f2'] + ELO_COLS].copy()
elo_merge['DATE'] = pd.to_datetime(elo_merge['DATE'])
df = df.merge(elo_merge, on=['jbout','DATE'], how='left')
flip = (df['jfighter'] == df['f2'])
for c in ELO_COLS:
    if c == 'elo_win_prob': df.loc[flip, c] = 1.0 - df.loc[flip, c]
    else:                   df.loc[flip, c] = -df.loc[flip, c]
df.drop(columns=['f1','f2'], inplace=True, errors='ignore')
for c in ELO_COLS: df[c] = df[c].fillna(0.5 if c == 'elo_win_prob' else 0.0)
print(f"Elo coverage: {(df['precomp_elo_diff']!=0).sum()}/{len(df)}")

# --- Build train/test ---
missing = [c for c in feat_cols if c not in df.columns]
if missing:
    print(f"WARN: {len(missing)} feat_cols missing from df. First few: {missing[:5]}")
    feat_cols = [c for c in feat_cols if c in df.columns]

train = df[(df.DATE >= TRAIN_START) & (df.DATE < TEST_START)].copy()
test  = df[df.DATE >= TEST_START].copy()
train = train.dropna(subset=['win'])
test  = test.dropna(subset=['win'])
train[feat_cols] = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
test[feat_cols]  = test[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
print(f"Train: {len(train)}  Test: {len(test)}")

# Recency weights
days = (TEST_START - train['DATE']).dt.days.values
w = np.exp(-LAM * days / 365.0)

scaler = StandardScaler()
Xtr = scaler.fit_transform(train[feat_cols].values)
Xte = scaler.transform(test[feat_cols].values)
ytr = train['win'].astype(int).values
yte = test['win'].astype(int).values

penalty = 'elasticnet' if 0 < LR_L1 < 1 else ('l1' if LR_L1 == 1 else 'l2')
solver = 'saga'
model = LogisticRegression(C=LR_C, penalty='elasticnet', l1_ratio=LR_L1,
                           solver='saga', max_iter=4000)
model.fit(Xtr, ytr, sample_weight=w)
prob = model.predict_proba(Xte)[:, 1]
pred = (prob >= 0.5).astype(int)

print(f"\nTest metrics:")
print(f"  Acc:     {accuracy_score(yte, pred):.4f}")
print(f"  LogLoss: {log_loss(yte, prob):.4f}")
print(f"  AUC:     {roc_auc_score(yte, prob):.4f}  (n={len(yte)})")

test = test.assign(model_prob=prob)

# --- Match to odds ---
def norm(s): return ''.join(str(s).lower().split())
odds = pd.read_csv(f"{DATA_TMP}/odds_table.csv", parse_dates=["DATE"])
odds['fkey'] = odds['FIGHTER'].map(norm)
odds['DATE'] = pd.to_datetime(odds['DATE'])

def american_to_payout(o):
    o = float(o)
    return o/100.0 if o > 0 else 100.0/abs(o)

# Need both fighter and opponent's vegas prob for paired bets.
# Strategy: for each test row (jfighter perspective), pull jfighter odds and opp odds.
test['fkey']  = test['jfighter'].map(norm)
test['okey']  = test['opp_jfighter'].map(norm)

odds_lite = odds[['DATE','fkey','prob_norm','odds']].rename(
    columns={'prob_norm':'vegas_prob_f','odds':'odds_f'})
test = test.merge(odds_lite, on=['DATE','fkey'], how='left')
odds_opp = odds[['DATE','fkey','prob_norm','odds']].rename(
    columns={'fkey':'okey','prob_norm':'vegas_prob_o','odds':'odds_o'})
test = test.merge(odds_opp, on=['DATE','okey'], how='left')

matched = test.dropna(subset=['vegas_prob_f','vegas_prob_o']).copy()
# Dedupe: each bout has two rows (one per fighter perspective). Keep alphabetically-first jfighter.
matched = matched.sort_values(['DATE','jbout','jfighter']).drop_duplicates(subset=['DATE','jbout'], keep='first')
print(f"\nOdds-matched test rows: {len(matched)}/{len(test)}")

matched['edge_f'] = matched['model_prob']     - matched['vegas_prob_f']
matched['edge_o'] = (1 - matched['model_prob']) - matched['vegas_prob_o']
matched['payout_f'] = matched['odds_f'].map(american_to_payout)
matched['payout_o'] = matched['odds_o'].map(american_to_payout)
matched['won'] = matched['win'].astype(int)

print("\nROI by edge threshold (paired bets, $1 stake each):")
print(f"{'thr':>6} {'bets':>6} {'wins':>6} {'wr':>7} {'profit':>9} {'roi':>8}")
for thr in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    bets, wins, profit = 0, 0, 0.0
    for _, r in matched.iterrows():
        if r['edge_f'] > thr:
            bets += 1
            if r['won'] == 1: wins += 1; profit += r['payout_f']
            else:             profit -= 1
        elif r['edge_o'] > thr:  # i.e. edge_f < -thr (model favors opponent)
            bets += 1
            if r['won'] == 0: wins += 1; profit += r['payout_o']
            else:             profit -= 1
    if bets:
        print(f"{thr:>6.2f} {bets:>6d} {wins:>6d} {wins/bets:>7.3f} {profit:>9.2f} {profit/bets*100:>7.2f}%")
    else:
        print(f"{thr:>6.2f} {bets:>6d}    -      -        -        -")
