"""Profile model_only ROI across many slices to find profitable subsets.
Slices examined: underdog-only, favorite-only, weight class, year, model-confidence bucket,
disagreement-with-vegas magnitude, both-fighter-elo level."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from elo_feature import compute_elo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

def norm(s): return ''.join(str(s).lower().split())
odds = pd.read_csv(f"{DT}/odds_table.csv", parse_dates=["DATE"])
odds['fkey'] = odds['FIGHTER'].map(norm)
test['fkey']=test['jfighter'].map(norm); test['okey']=test['opp_jfighter'].map(norm)
of = odds[['DATE','fkey','prob_norm','odds']].rename(columns={'prob_norm':'vp_f','odds':'odds_f'})
oo = odds[['DATE','fkey','prob_norm','odds']].rename(columns={'fkey':'okey','prob_norm':'vp_o','odds':'odds_o'})
m_df = test.merge(of, on=['DATE','fkey'], how='left').merge(oo, on=['DATE','okey'], how='left')
m_df = m_df.dropna(subset=['vp_f','vp_o']).copy()

def pay(o): o=float(o); return o/100.0 if o>0 else 100.0/abs(o)
m_df['payout_f']=m_df['odds_f'].map(pay); m_df['payout_o']=m_df['odds_o'].map(pay)
m_df['won']=m_df['win'].astype(int)
m_df['edge_f']=m_df['model_prob']-m_df['vp_f']; m_df['edge_o']=(1-m_df['model_prob'])-m_df['vp_o']
m_df['year']=m_df['DATE'].dt.year

# Per-bet records: choose side with max edge (single bet per fight)
records = []
for _, r in m_df.iterrows():
    if r['edge_f'] > r['edge_o']:
        side, edge, payout, won = 'f', r['edge_f'], r['payout_f'], r['won']==1
        odds_us = r['odds_f']
    else:
        side, edge, payout, won = 'o', r['edge_o'], r['payout_o'], r['won']==0
        odds_us = r['odds_o']
    records.append({
        'date': r['DATE'], 'year': r['year'], 'side': side,
        'edge': edge, 'payout': payout, 'won': won,
        'profit': payout if won else -1.0,
        'underdog': odds_us > 0,
        'odds': odds_us,
        'model_prob_picked': r['model_prob'] if side=='f' else (1-r['model_prob']),
        'vegas_prob_picked': r['vp_f'] if side=='f' else r['vp_o'],
        'weightindex': r.get('weightindex', np.nan),
        'main_event_proxy': 0,  # placeholder
    })
b = pd.DataFrame(records)
print(f"Total bouts: {len(b)}  baseline ROI (bet all): {b['profit'].sum()/len(b)*100:.2f}%")
print(f"  underdogs: {b['underdog'].sum()}  favorites: {(~b['underdog']).sum()}")

def slice_roi(name, mask):
    s = b[mask]
    if len(s)<10: return f"{name:<40} n={len(s)} (skip)"
    roi = s['profit'].sum()/len(s)*100
    wr = s['won'].mean()*100
    return f"{name:<40} n={len(s):>4d}  wr={wr:>5.1f}%  roi={roi:>+6.2f}%"

print("\n=== Underdog vs favorite ===")
print(slice_roi("all underdogs",        b['underdog']))
print(slice_roi("all favorites",        ~b['underdog']))
print(slice_roi("underdog edge>3%",     b['underdog'] & (b['edge']>0.03)))
print(slice_roi("underdog edge>5%",     b['underdog'] & (b['edge']>0.05)))
print(slice_roi("underdog edge>10%",    b['underdog'] & (b['edge']>0.10)))
print(slice_roi("underdog edge>15%",    b['underdog'] & (b['edge']>0.15)))
print(slice_roi("underdog edge>20%",    b['underdog'] & (b['edge']>0.20)))
print(slice_roi("favorite edge>3%",     ~b['underdog'] & (b['edge']>0.03)))
print(slice_roi("favorite edge>5%",     ~b['underdog'] & (b['edge']>0.05)))
print(slice_roi("favorite edge>10%",    ~b['underdog'] & (b['edge']>0.10)))

print("\n=== Underdog by odds bucket ===")
for lo, hi in [(100,150),(150,200),(200,300),(300,500),(500,9999)]:
    m = b['underdog'] & (b['odds'].between(lo, hi))
    print(slice_roi(f"+{lo}..+{hi}", m))

print("\n=== By year ===")
for y in sorted(b['year'].unique()):
    print(slice_roi(f"year={y}", b['year']==y))

print("\n=== By weight class ===")
for wi in sorted(b['weightindex'].dropna().unique()):
    print(slice_roi(f"weightindex={int(wi)}", b['weightindex']==wi))

print("\n=== Model-prob bucket (picked side) ===")
for lo,hi in [(0.30,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.70),(0.70,0.85)]:
    m = b['model_prob_picked'].between(lo, hi)
    print(slice_roi(f"model_prob {lo:.2f}-{hi:.2f}", m))

print("\n=== Model vs Vegas disagreement (picked side) ===")
b['disagree'] = b['model_prob_picked'] - b['vegas_prob_picked']
for lo,hi in [(-0.10,0),(0,0.05),(0.05,0.10),(0.10,0.20),(0.20,1.0)]:
    print(slice_roi(f"disagree {lo:.2f}..{hi:.2f}", b['disagree'].between(lo,hi)))

print("\n=== Combined: underdog AND disagree>0.05 ===")
print(slice_roi("underdog & disagree>0.05",  b['underdog'] & (b['disagree']>0.05)))
print(slice_roi("underdog & disagree>0.10",  b['underdog'] & (b['disagree']>0.10)))
print(slice_roi("underdog & disagree>0.15",  b['underdog'] & (b['disagree']>0.15)))
print(slice_roi("underdog & 100<odds<300 & disagree>0.05",
                b['underdog'] & b['odds'].between(100,300) & (b['disagree']>0.05)))
print(slice_roi("underdog & 100<odds<200 & disagree>0.05",
                b['underdog'] & b['odds'].between(100,200) & (b['disagree']>0.05)))
