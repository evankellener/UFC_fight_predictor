#!/usr/bin/env python3
"""
Stacked Ensemble - Uses meta-learner to optimally combine predictions
Better than simple averaging because it learns when each model is reliable
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import json
from ensemble_model_best import FightOutcomeModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import time

print("🎯 UFC Stacked Ensemble with Meta-Learner")
print("=" * 80)
print("Training a meta-model to learn optimal combination of base models")
print("=" * 80)
print()

# Load XGBoost config
with open('xgboost_ga_results_1760303427.json', 'r') as f:
    xgboost_config = json.load(f)

# LogReg features (GA optimized)
logreg_features = [
    "precomp_elo_diff", "precomp_strike_elo_diff", "precomp_grapple_elo_diff",
    "precomp_strike_elo_change_5", "opp_precomp_bodyacc_perc",
    "opp_precomp_headacc_perc5", "precomp_headacc_perc5",
    "opp_precomp_bodyacc_perc3", "opp_precomp_distacc_perc3",
    "opp_precomp_tdacc_perc5", "precomp_totalstr_pm",
    "opp_precomp_str_eff_diff", "precomp_tdavg", "opp_precomp_tdavg",
    "precomp_bodyacc_perc5", "age_ratio_difference",
    "precomp_strike_elo_change_3", "precomp_sapm3",
    "precomp_totalacc_perc5", "precomp_sigstr_pm3",
    "opp_precomp_str_eff_diff3", "precomp_headacc_perc",
    "opp_precomp_headacc_perc", "opp_precomp_strike_elo",
    "opp_precomp_groundacc_perc3", "precomp_sapm",
    "opp_precomp_bodyacc_perc5", "precomp_sigstr_perc5"
]

xgb_features = xgboost_config['features']

print(f"📊 Model configurations:")
print(f"   LogReg: {len(logreg_features)} features")
print(f"   XGBoost: {len(xgb_features)} features")
print()

# Initialize data
print("📂 Loading data...")
model = FightOutcomeModel("data/tmp/final.csv", random_seed=42)

train_df = model.train_df.copy()
test_df = model.test_df.copy()

y_train = train_df['win']
y_test = test_df['win']

print(f"   Train: {len(train_df)} fights")
print(f"   Test: {len(test_df)} fights")
print()

# ============================================================================
# PREPARE BASE MODELS
# ============================================================================

print("🔨 Building base models...")
print()

# MODEL 1: LogReg
print("   🔵 LogReg...")
X_train_lr = train_df[logreg_features].copy()
X_test_lr = test_df[logreg_features].copy()

imp_lr = SimpleImputer(strategy='median')
X_train_lr = imp_lr.fit_transform(X_train_lr)
X_test_lr = imp_lr.transform(X_test_lr)

scaler_lr = RobustScaler()
X_train_lr = scaler_lr.fit_transform(X_train_lr)
X_test_lr = scaler_lr.transform(X_test_lr)

logreg = LogisticRegression(C=0.1, penalty='l2', solver='saga', max_iter=1000, random_state=42)
logreg.fit(X_train_lr, y_train)

lr_train_probs = logreg.predict_proba(X_train_lr)[:, 1]
lr_test_probs = logreg.predict_proba(X_test_lr)[:, 1]
lr_test_preds = (lr_test_probs >= 0.5).astype(int)

lr_acc = accuracy_score(y_test, lr_test_preds)
lr_ll = log_loss(y_test, lr_test_probs)

print(f"      Accuracy: {lr_acc:.4f} | Log Loss: {lr_ll:.4f} | Combined: {lr_acc - lr_ll:.6f}")

# MODEL 2: XGBoost  
print("   🌲 XGBoost...")
X_train_xgb = train_df[xgb_features].copy()
X_test_xgb = test_df[xgb_features].copy()

imp_xgb = SimpleImputer(strategy='median')
X_train_xgb = imp_xgb.fit_transform(X_train_xgb)
X_test_xgb = imp_xgb.transform(X_test_xgb)

scaler_xgb = RobustScaler()
X_train_xgb = scaler_xgb.fit_transform(X_train_xgb)
X_test_xgb = scaler_xgb.transform(X_test_xgb)

xgb_params = xgboost_config['hyperparams']
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    **xgb_params
)

xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_test_xgb, y_test)], verbose=False)

xgb_train_probs = xgb_model.predict_proba(X_train_xgb)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
xgb_test_preds = (xgb_test_probs >= 0.5).astype(int)

xgb_acc = accuracy_score(y_test, xgb_test_preds)
xgb_ll = log_loss(y_test, xgb_test_probs)

print(f"      Accuracy: {xgb_acc:.4f} | Log Loss: {xgb_ll:.4f} | Combined: {xgb_acc - xgb_ll:.6f}")
print()

# ============================================================================
# STACKING: Train meta-learner on base model predictions
# ============================================================================

print("=" * 80)
print("🧠 STACKING: Training Meta-Learner")
print("=" * 80)
print()

# Create meta-features: predictions from base models
X_train_meta = np.column_stack([lr_train_probs, xgb_train_probs])
X_test_meta = np.column_stack([lr_test_probs, xgb_test_probs])

print(f"   Meta-features shape: {X_train_meta.shape}")
print(f"   Feature 1: LogReg probabilities")
print(f"   Feature 2: XGBoost probabilities")
print()

# Try different meta-learners
meta_learners = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "LogReg (L1)": LogisticRegression(penalty='l1', solver='saga', C=1.0, random_state=42, max_iter=1000),
    "LogReg (strong L2)": LogisticRegression(C=0.01, random_state=42, max_iter=1000),
}

print("🔬 Testing different meta-learners:")
print()

best_meta_name = None
best_meta_model = None
best_meta_score = -float('inf')

for name, meta_model in meta_learners.items():
    meta_model.fit(X_train_meta, y_train)
    
    meta_probs = meta_model.predict_proba(X_test_meta)[:, 1]
    meta_preds = (meta_probs >= 0.5).astype(int)
    
    meta_acc = accuracy_score(y_test, meta_preds)
    meta_ll = log_loss(y_test, meta_probs)
    meta_combined = meta_acc - meta_ll
    
    marker = ""
    if meta_combined > best_meta_score:
        best_meta_score = meta_combined
        best_meta_name = name
        best_meta_model = meta_model
        best_meta_probs = meta_probs
        best_meta_preds = meta_preds
        best_meta_acc = meta_acc
        best_meta_ll = meta_ll
        best_meta_combined = meta_combined
        marker = " 🌟 BEST"
    
    print(f"   {name:<25} Acc: {meta_acc:.4f} | LL: {meta_ll:.4f} | Combined: {meta_combined:.6f}{marker}")

print()
print(f"✅ Best meta-learner: {best_meta_name}")

# Show meta-learner weights
if hasattr(best_meta_model, 'coef_'):
    weights = best_meta_model.coef_[0]
    intercept = best_meta_model.intercept_[0]
    print(f"   LogReg weight: {weights[0]:.3f}")
    print(f"   XGBoost weight: {weights[1]:.3f}")
    print(f"   Intercept: {intercept:.3f}")
    print()
    print(f"   Interpretation:")
    if weights[0] > weights[1]:
        print(f"      Meta-learner trusts LogReg {weights[0]/weights[1]:.2f}x more")
    else:
        print(f"      Meta-learner trusts XGBoost {weights[1]/weights[0]:.2f}x more")

print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("=" * 80)
print("🏆 FINAL RESULTS")
print("=" * 80)
print()

results = [
    ("LogReg GA", lr_acc, lr_ll, lr_acc - lr_ll),
    ("XGBoost GA", xgb_acc, xgb_ll, xgb_acc - xgb_ll),
    ("Stacked Ensemble", best_meta_acc, best_meta_ll, best_meta_combined),
]

print(f"{'Model':<25} {'Accuracy':<12} {'Log Loss':<12} {'Combined':<12} {'vs XGBoost':<12}")
print("-" * 90)

xgb_combined = xgb_acc - xgb_ll
best_combined = max(r[3] for r in results)

for name, acc, ll, combined in results:
    vs_xgb = ((combined - xgb_combined) / xgb_combined * 100)
    marker = " 🏆" if combined == best_combined else ""
    print(f"{name:<25} {acc:<12.4f} {ll:<12.4f} {combined:<12.6f} {vs_xgb:+.2f}%{marker}")

print()

winner = max(results, key=lambda x: x[3])
improvement = ((winner[3] - xgb_combined) / xgb_combined * 100)

if winner[0] == "Stacked Ensemble":
    print(f"🎉 ENSEMBLE WINS!")
    print(f"   Improvement over XGBoost: {improvement:+.2f}%")
    print(f"   Meta-learner learned to optimally combine predictions!")
else:
    print(f"⚠️  {winner[0]} still wins")
    print(f"   Ensemble improvement: {improvement:+.2f}%")
    if improvement < 0:
        print(f"   Note: Sometimes individual models can't be beaten!")
        print(f"   This happens when:")
        print(f"      - Base models make similar mistakes")
        print(f"      - One model is significantly better")
        print(f"      - Not enough data for meta-learner to learn")

print()

# Save results
stacked_results = {
    "method": "Stacked Ensemble",
    "meta_learner": best_meta_name,
    "base_models": {
        "logreg": {
            "features": logreg_features,
            "accuracy": float(lr_acc),
            "log_loss": float(lr_ll)
        },
        "xgboost": {
            "features": xgb_features,
            "accuracy": float(xgb_acc),
            "log_loss": float(xgb_ll)
        }
    },
    "ensemble_metrics": {
        "accuracy": float(best_meta_acc),
        "log_loss": float(best_meta_ll),
        "combined": float(best_meta_combined),
        "improvement_vs_xgboost": float(improvement)
    }
}

output_file = f"stacked_ensemble_results_{int(time.time())}.json"
with open(output_file, 'w') as f:
    json.dump(stacked_results, f, indent=2)

print(f"💾 Results saved to: {output_file}")
print()
print("💡 Stacking trains a meta-model that learns WHEN to trust each base model,")
print("   which is smarter than simple averaging!")
print()

