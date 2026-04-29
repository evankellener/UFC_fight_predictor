"""Build deploy_v1 — master_validation pipeline trained on ALL data through April 2026.

Saves to app/models/deploy_v1/:
  lr.pkl, lr_scaler.pkl, lr_imputer.pkl, feat_cols.json

Feature set: master_validation select_features() (same as validated STRONG PASS config).
Training:    C=0.05, l1_ratio=0.5, λ=1.20, threshold=3, ALL data < 2026-05-01.
"""
import sys, warnings, json, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"app")
from pathlib import Path
import elo_feature, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")
import mma_ai_pipeline as mma

TRAIN_END = pd.Timestamp("2026-05-01")
LAM       = 1.20
C, L1     = 0.05, 0.5
THRESHOLD = 3
NOVEL     = ["home_advantage_diff", "card_position_norm_career_diff"]
OUT       = Path("app/models/deploy_v1")
OUT.mkdir(parents=True, exist_ok=True)

print("="*60)
print("BUILD DEPLOY MODEL — all data through April 2026")
print("="*60)

# ── Step 1: MMA-AI pipeline ──────────────────────────────────────
print("\n[1/4] MMA-AI pipeline steps 1-6...")
df_full = mma.load_base_data()
df_full = mma.beta_binomial_smooth(df_full)
df_full = mma.poisson_gamma_smooth(df_full)
df_full = mma.compute_derived_features(df_full)
stat_cols = sorted(set(c for c in df_full.columns if (
    c.endswith("_pm") or c.endswith("_acc") or c.endswith("_def") or
    c.endswith("_ratio") or c.endswith("_per_ctrl") or
    c in ["ko_smooth","win_smooth","decision_smooth","sub_land_smooth",
          "sub_land_rate","ctrl_pm","ko_per_sig_str_land","td_per_sig_str_att",
          "ground_per_ctrl","dist_per_sig_str_land","head_per_sig_str_land",
          "rev_per_ctrlopp","sig_str_land_ratio","ko_ratio","sub_att_ratio",
          "ctrl_ratio","ground_land_per_ctrl","td_land_per_ctrl"]
) and c in df_full.columns and not c.startswith("opp_") and not c.endswith("_raw")))
df_full = mma.compute_decayed_averages(df_full, stat_cols, mma.V7_CONFIG["decay_lambda"])
df_full = mma.compute_opponent_history(df_full, stat_cols, mma.V7_CONFIG["decay_lambda"])

print("\n[2/4] WC priors (all data) + adjperf + assemble...")
priors = mma.compute_wc_priors(df_full, stat_cols)
df_adj = mma.compute_adjperf(df_full, stat_cols, priors)
result = mma.assemble_features(df_adj, stat_cols, mma.V7_CONFIG["decay_lambda"])
result = mma.filter_to_era(result, mma.V7_CONFIG["start_date"])
result.to_csv("data/tmp/market_features_clean.csv", index=False)

# ── Step 2: Load full feature df ─────────────────────────────────
print("\n[3/4] Load Elo + WC features + select_features...")
for mod in list(sys.modules):
    if mod.startswith("run_threshold_sweep_both_elos") or \
       mod in ("retrain_lr_symmetric","walk_forward_4fold"):
        del sys.modules[mod]

from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold
from retrain_lr_symmetric import load_wc_history_from_db, add_wc_features, flip_row_dataframe
from walk_forward_4fold import select_features

base = load_base_both_elos()
df   = apply_threshold(base, THRESHOLD)
df   = add_wc_features(df, load_wc_history_from_db())
feats_baseline = select_features(df)

mf = pd.read_csv("data/tmp/market_features_clean.csv", parse_dates=["DATE"])
avail = [c for c in NOVEL if c in mf.columns]
df = df.merge(mf[["DATE","jbout","jfighter"]+avail],
              on=["DATE","jbout","jfighter"], how="left")
new_feats = [c for c in avail if c not in feats_baseline and df[c].std()>1e-8]
usable = [c for c in feats_baseline+new_feats if c in df.columns and df[c].std()>1e-8]
print(f"  Feature count: {len(usable)}")

# ── Step 3: Train ────────────────────────────────────────────────
print("\n[4/4] Train LR (C=0.05, l1=0.5, λ=1.20) on all data through Apr 2026...")
train_df = df[df["DATE"] < TRAIN_END].copy()
print(f"  Train rows: {len(train_df)} ({train_df['DATE'].min().date()} → {train_df['DATE'].max().date()})")

td  = pd.concat([train_df, flip_row_dataframe(train_df)], ignore_index=True)
imp = SimpleImputer(strategy="median")
sc  = StandardScaler()
Xtr = sc.fit_transform(imp.fit_transform(td[usable]))
w   = np.exp(-LAM * (TRAIN_END - td["DATE"]).dt.days.values / 365.25)

lr  = LogisticRegression(C=C, penalty="elasticnet", l1_ratio=L1,
                          solver="saga", max_iter=8000, random_state=42)
lr.fit(Xtr, td["win"].astype(int).values, sample_weight=w)
n_act = int((np.abs(lr.coef_[0])>1e-8).sum())
print(f"  Active features: {n_act} / {len(usable)}")

# Top active features
active_idx = sorted(range(len(usable)),
                    key=lambda i: abs(lr.coef_[0][i]), reverse=True)
print("  Top 10 active:")
for i in active_idx[:10]:
    if abs(lr.coef_[0][i]) > 1e-8:
        print(f"    {usable[i]:<45s} {lr.coef_[0][i]:+.4f}")

# ── Save ─────────────────────────────────────────────────────────
with open(OUT/"lr.pkl","wb") as f:      pickle.dump(lr, f)
with open(OUT/"lr_scaler.pkl","wb") as f: pickle.dump(sc, f)
with open(OUT/"lr_imputer.pkl","wb") as f: pickle.dump(imp, f)
(OUT/"feat_cols.json").write_text(json.dumps(usable, indent=2))
print(f"\n  ✓ Saved {OUT}/lr.pkl  ({len(usable)} feats, {n_act} active)")
print("  ✓ Deploy model ready.")
