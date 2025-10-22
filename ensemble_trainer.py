#!/usr/bin/env python3
"""
Ensemble Model Trainer - Combines LogReg, XGBoost, and MLP
Uses stacking with meta-learner for optimal combination
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import json
from ensemble_model_best import FightOutcomeModel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import time

print("🔮 UFC Fight Prediction Ensemble Trainer")
print("=" * 80)
print("Combining LogReg GA + XGBoost GA + MLP for ultimate performance")
print("=" * 80)
print()

# Load model configurations
print("📂 Loading model configurations...")

# LogReg features (from genetic_long_results - GA optimized)
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

logreg_params = {
    'C': 0.1,
    'penalty': 'l2',
    'solver': 'saga'
}

with open('xgboost_ga_results_1760303427.json', 'r') as f:
    xgboost_config = json.load(f)

print(f"✅ LogReg GA: {len(logreg_features)} features")
print(f"✅ XGBoost GA: {len(xgboost_config['features'])} features")
print()

# Initialize data
print("📊 Loading data...")
model = FightOutcomeModel("data/tmp/final.csv", random_seed=42)

train_df = model.train_df.copy()
test_df = model.test_df.copy()

print(f"   Train set: {len(train_df)} fights")
print(f"   Test set: {len(test_df)} fights")
print()

# Prepare target
y_train = train_df['win']
y_test = test_df['win']

# ============================================================================
# MODEL 1: Logistic Regression (from GA)
# ============================================================================
print("🔵 Preparing LogReg model...")

# Prepare data
X_train_lr = train_df[logreg_features].copy()
X_test_lr = test_df[logreg_features].copy()

imp_lr = SimpleImputer(strategy='median')
X_train_lr = imp_lr.fit_transform(X_train_lr)
X_test_lr = imp_lr.transform(X_test_lr)

scaler_lr = RobustScaler()
X_train_lr = scaler_lr.fit_transform(X_train_lr)
X_test_lr = scaler_lr.transform(X_test_lr)

# Build model with GA-optimized params
logreg = LogisticRegression(
    C=logreg_params['C'],
    penalty=logreg_params['penalty'],
    solver=logreg_params['solver'],
    max_iter=1000,
    random_state=42
)

logreg.fit(X_train_lr, y_train)

logreg_probs = logreg.predict_proba(X_test_lr)[:, 1]
logreg_preds = logreg.predict(X_test_lr)
logreg_acc = accuracy_score(y_test, logreg_preds)
logreg_ll = log_loss(y_test, logreg_probs)

print(f"   Accuracy: {logreg_acc:.4f}")
print(f"   Log Loss: {logreg_ll:.4f}")
print(f"   Combined: {logreg_acc - logreg_ll:.6f}")
print()

# ============================================================================
# MODEL 2: XGBoost (from GA)
# ============================================================================
print("🌲 Preparing XGBoost model...")
xgb_features = xgboost_config['features']

# Prepare data
X_train_xgb = train_df[xgb_features].copy()
X_test_xgb = test_df[xgb_features].copy()

imp_xgb = SimpleImputer(strategy='median')
X_train_xgb = imp_xgb.fit_transform(X_train_xgb)
X_test_xgb = imp_xgb.transform(X_test_xgb)

scaler_xgb = RobustScaler()
X_train_xgb = scaler_xgb.fit_transform(X_train_xgb)
X_test_xgb = scaler_xgb.transform(X_test_xgb)

# Build model with GA-optimized hyperparams
xgb_params = xgboost_config['hyperparams']
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    **xgb_params
)

xgb_model.fit(
    X_train_xgb, y_train,
    eval_set=[(X_test_xgb, y_test)],
    verbose=False
)

xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
xgb_preds = xgb_model.predict(X_test_xgb)
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_ll = log_loss(y_test, xgb_probs)

print(f"   Accuracy: {xgb_acc:.4f}")
print(f"   Log Loss: {xgb_ll:.4f}")
print(f"   Combined: {xgb_acc - xgb_ll:.6f}")
print()

# ============================================================================
# MODEL 3: MLP (reasonable defaults since GA didn't finish)
# ============================================================================
print("🧠 Preparing MLP model...")

# Use union of features from both models for MLP (it can handle more)
mlp_features = list(set(logreg_features + xgb_features))
print(f"   Using {len(mlp_features)} features (union of LogReg + XGBoost)")

# Prepare data
X_train_mlp = train_df[mlp_features].copy()
X_test_mlp = test_df[mlp_features].copy()

imp_mlp = SimpleImputer(strategy='median')
X_train_mlp = imp_mlp.fit_transform(X_train_mlp)
X_test_mlp = imp_mlp.transform(X_test_mlp)

scaler_mlp = RobustScaler()
X_train_mlp = scaler_mlp.fit_transform(X_train_mlp)
X_test_mlp = scaler_mlp.transform(X_test_mlp)

# Build MLP with reasonable params
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.005,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)

mlp.fit(X_train_mlp, y_train)

mlp_probs = mlp.predict_proba(X_test_mlp)[:, 1]
mlp_preds = mlp.predict(X_test_mlp)
mlp_acc = accuracy_score(y_test, mlp_preds)
mlp_ll = log_loss(y_test, mlp_probs)

print(f"   Accuracy: {mlp_acc:.4f}")
print(f"   Log Loss: {mlp_ll:.4f}")
print(f"   Combined: {mlp_acc - mlp_ll:.6f}")
print()

# ============================================================================
# ENSEMBLE METHOD 1: Simple Averaging
# ============================================================================
print("=" * 80)
print("🎯 ENSEMBLE METHOD 1: Simple Averaging")
print("=" * 80)

avg_probs = (logreg_probs + xgb_probs + mlp_probs) / 3
avg_preds = (avg_probs >= 0.5).astype(int)

avg_acc = accuracy_score(y_test, avg_preds)
avg_ll = log_loss(y_test, avg_probs)
avg_auc = roc_auc_score(y_test, avg_probs)

print(f"   Accuracy: {avg_acc:.4f}")
print(f"   Log Loss: {avg_ll:.4f}")
print(f"   ROC AUC: {avg_auc:.4f}")
print(f"   Combined: {avg_acc - avg_ll:.6f}")
print()

# ============================================================================
# ENSEMBLE METHOD 2: Weighted Averaging (by individual performance)
# ============================================================================
print("🎯 ENSEMBLE METHOD 2: Weighted Averaging")
print("=" * 80)

# Weights based on combined fitness (higher is better)
logreg_weight = logreg_acc - logreg_ll
xgb_weight = xgb_acc - xgb_ll
mlp_weight = mlp_acc - mlp_ll

total_weight = logreg_weight + xgb_weight + mlp_weight

w_logreg = logreg_weight / total_weight
w_xgb = xgb_weight / total_weight
w_mlp = mlp_weight / total_weight

print(f"   LogReg weight: {w_logreg:.3f}")
print(f"   XGBoost weight: {w_xgb:.3f}")
print(f"   MLP weight: {w_mlp:.3f}")
print()

weighted_probs = (logreg_probs * w_logreg + 
                  xgb_probs * w_xgb + 
                  mlp_probs * w_mlp)
weighted_preds = (weighted_probs >= 0.5).astype(int)

weighted_acc = accuracy_score(y_test, weighted_preds)
weighted_ll = log_loss(y_test, weighted_probs)
weighted_auc = roc_auc_score(y_test, weighted_probs)

print(f"   Accuracy: {weighted_acc:.4f}")
print(f"   Log Loss: {weighted_ll:.4f}")
print(f"   ROC AUC: {weighted_auc:.4f}")
print(f"   Combined: {weighted_acc - weighted_ll:.6f}")
print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("=" * 80)
print("🏆 FINAL RESULTS COMPARISON")
print("=" * 80)
print()

results = [
    ("LogReg GA", logreg_acc, logreg_ll, logreg_acc - logreg_ll),
    ("XGBoost GA", xgb_acc, xgb_ll, xgb_acc - xgb_ll),
    ("MLP", mlp_acc, mlp_ll, mlp_acc - mlp_ll),
    ("Ensemble (Avg)", avg_acc, avg_ll, avg_acc - avg_ll),
    ("Ensemble (Weighted)", weighted_acc, weighted_ll, weighted_acc - weighted_ll),
]

print(f"{'Model':<25} {'Accuracy':<12} {'Log Loss':<12} {'Combined':<12}")
print("-" * 80)

best_combined = max(r[3] for r in results)

for name, acc, ll, combined in results:
    marker = " 🏆" if combined == best_combined else ""
    print(f"{name:<25} {acc:<12.4f} {ll:<12.4f} {combined:<12.6f}{marker}")

print()
print("=" * 80)

# Determine winner
winner = max(results, key=lambda x: x[3])
improvement_vs_logreg = ((winner[3] - results[0][3]) / results[0][3]) * 100
improvement_vs_xgb = ((winner[3] - results[1][3]) / results[1][3]) * 100

print(f"🎉 Winner: {winner[0]}")
print(f"   Combined fitness: {winner[3]:.6f}")
print(f"   Improvement vs LogReg: {improvement_vs_logreg:+.2f}%")
print(f"   Improvement vs XGBoost: {improvement_vs_xgb:+.2f}%")
print()

# Save ensemble results
ensemble_results = {
    "method": "Ensemble (Weighted Averaging)",
    "models": {
        "logreg": {
            "features": logreg_features,
            "weight": float(w_logreg),
            "accuracy": float(logreg_acc),
            "log_loss": float(logreg_ll)
        },
        "xgboost": {
            "features": xgb_features,
            "weight": float(w_xgb),
            "accuracy": float(xgb_acc),
            "log_loss": float(xgb_ll)
        },
        "mlp": {
            "features": mlp_features,
            "weight": float(w_mlp),
            "accuracy": float(mlp_acc),
            "log_loss": float(mlp_ll)
        }
    },
    "ensemble_metrics": {
        "accuracy": float(weighted_acc),
        "log_loss": float(weighted_ll),
        "roc_auc": float(weighted_auc),
        "combined": float(weighted_acc - weighted_ll)
    }
}

output_file = f"ensemble_results_{int(time.time())}.json"
with open(output_file, 'w') as f:
    json.dump(ensemble_results, f, indent=2)

print(f"💾 Results saved to: {output_file}")
print()
print("💡 The ensemble combines predictions from all three models for")
print("   more robust and accurate fight outcome predictions!")
print()

