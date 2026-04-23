"""Side-by-side feature dump for one fight: training vs predictor_v2.
Shows top N features that differ most."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "app")
import elo_feature
elo_feature.DB_PATH = Path("data/sqlite_db/sqlite_scrapper.db")

from predictor_v2 import PredictorV2
from run_threshold_sweep_both_elos import load_base_both_elos, apply_threshold, TEST_FIRST, TEST_LAST

p2 = PredictorV2(verbose=True)
base = load_base_both_elos()
df = apply_threshold(base, 3)
test = df[(df["DATE"] >= TEST_FIRST) & (df["DATE"] <= TEST_LAST)]

# Pick a specific mismatch — Geoff Neal vs Rafael dos Anjos 2024-10-26
mask = (test["DATE"] == pd.Timestamp("2024-10-26")) & (
    test["jfighter"].isin(["GeoffNeal", "RafaelDosAnjos"]))
rows = test[mask]
if len(rows) == 0:
    print("No matching row"); sys.exit(1)
trow = rows.iloc[0]
print(f"\nFight: {trow['jfighter']} vs {trow['opp_jfighter']} on {trow['DATE'].date()}")

# Build predictor_v2 row for the same
r = p2.predict(trow["jfighter"], trow["opp_jfighter"], event_date=trow["DATE"])
v2_row = r["feature_row"]

# Compare each feature
print("\nTop 30 features by absolute difference:")
print(f"{'feature':<40s}  {'train':>10s}  {'v2':>10s}  {'diff':>10s}")
rows_list = []
for c in p2.feat_cols:
    tv = float(trow.get(c, 0.0))
    vv = float(v2_row.get(c, 0.0))
    rows_list.append((c, tv, vv, vv - tv))
rows_list.sort(key=lambda x: abs(x[3]), reverse=True)
for c, tv, vv, d in rows_list[:30]:
    flag = " ***" if abs(d) > 0.1 else ""
    print(f"{c:<40s}  {tv:>10.4f}  {vv:>10.4f}  {d:>+10.4f}{flag}")

# Also show predictor result vs training-LR result
from sklearn.metrics import log_loss
X_t = np.array([[float(trow.get(c, 0.0)) for c in p2.feat_cols]])
X_t = p2.imputer.transform(X_t)
X_t = p2.scaler.transform(X_t)
p_train = float(p2.lr.predict_proba(X_t)[0, 1])

X_v = np.array([[float(v2_row.get(c, 0.0)) for c in p2.feat_cols]])
X_v = p2.imputer.transform(X_v)
X_v = p2.scaler.transform(X_v)
p_v2 = float(p2.lr.predict_proba(X_v)[0, 1])

print(f"\nTraining-row → p({trow['jfighter']}) = {p_train:.4f}")
print(f"V2 row       → p({trow['jfighter']}) = {p_v2:.4f}")
print(f"Actual winner: {trow['jfighter'] if trow['win'] == 1 else trow['opp_jfighter']}")
