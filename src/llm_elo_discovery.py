import os
import sys
import time
import json
import tempfile
import shutil
import subprocess
import ast

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from openai import OpenAI

# ─────────────── Configuration ───────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(SCRIPT_DIR, "src")
DATA_IN      = os.path.join(SCRIPT_DIR, "data", "interleaved_cleaned.csv")
CODE_PATH    = os.path.join(SRC_DIR, "gpt_elo_best_time_finnish.py")
LOG_PATH     = os.path.join(SCRIPT_DIR, "logs", "iterations.jsonl")
OPENAI_MODEL = "gpt-4o"
RUN_SECONDS  = 3600  # seconds
TEMP_DIR     = tempfile.mkdtemp()

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
sys.path.insert(0, SRC_DIR)
import gpt_elo_best_time_finnish as elo_module

def save_iteration_log(iter_no, metrics, suggestion):
    entry = {
        "iter":       iter_no,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics":    metrics,
        "suggestion": suggestion
    }
    with open(LOG_PATH, "a") as lf:
        lf.write(json.dumps(entry) + "\n")

def load_and_run_elo():
    # 1) Load raw CSV
    df0 = pd.read_csv(DATA_IN, parse_dates=["DATE"], low_memory=False)
    # 2) Core Elo only
    df1 = elo_module.EnhancedElo().process_fights(df0)
    out = os.path.join(TEMP_DIR, "elo_only.csv")
    df1.to_csv(out, index=False)
    return out

def evaluate(tmp_csv):
    # Evaluate on last 548 days (~1.5 years)
    df = pd.read_csv(tmp_csv, parse_dates=["DATE"], low_memory=False)
    latest = df["DATE"].max()
    cutoff = latest - pd.Timedelta(days=548)
    test = df[df["DATE"] >= cutoff].dropna(subset=["precomp_elo","opp_precomp_elo","win"])
    # Binary predictions and logistic probability
    preds = (test["precomp_elo"] > test["opp_precomp_elo"]).astype(int).values
    diff  = (test["precomp_elo"] - test["opp_precomp_elo"]).values
    probs = 1 / (1 + np.exp(-diff/170))
    acc   = accuracy_score(test["win"], preds)
    ll    = log_loss(test["win"], probs)
    return {"accuracy": acc, "log_loss": ll}

def ask_llm(metrics, code_snippet):
    client = OpenAI()
    prompt = f"""
We ran `EnhancedElo.process_fights` on UFC data.
Metrics:
  • accuracy = {metrics['accuracy']:.4f}
  • log loss = {metrics['log_loss']:.4f}

Here is `process_fights` from gpt_elo_best_time_finnish.py:
```python
{code_snippet}
```
Suggest exactly one small change to improve log loss.
Reply with a unified diff modifying only `process_fights`.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"user","content":prompt}],
    )
    return resp.choices[0].message.content

def validate_patch(diff, sample_df):
    added = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and "df[" in ln]
    cols = set()
    for line in added:
        try:
            tree = ast.parse(line)
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                    cols.add(node.slice.value)
        except:
            pass
    missing = [c for c in cols if c not in sample_df.columns]
    if missing:
        raise ValueError(f"Patch requests missing columns: {missing}")

def main():
    with open(CODE_PATH) as cf:
        base_code = cf.read()
    iter_no = 0
    start = time.time()

    while time.time() - start < RUN_SECONDS:
        iter_no += 1
        print(f"--- Iter {iter_no} ---")
        # Run and eval
        try:
            tmp = load_and_run_elo()
            metrics = evaluate(tmp)
            print(f"Acc={metrics['accuracy']:.4f}, LogLoss={metrics['log_loss']:.4f}")
        except Exception as e:
            print(f"Eval error: {e}")
            save_iteration_log(iter_no, {}, {"status":"eval_error","error":str(e)})
            continue
        # Ask LLM
        try:
            diff = ask_llm(metrics, base_code)
        except Exception as e:
            print(f"LLM call failed: {e}")
            break
        # Validate
        try:
            sample = pd.read_csv(DATA_IN, nrows=5)
            validate_patch(diff, sample)
        except Exception as e:
            print(f"Invalid patch: {e}")
            save_iteration_log(iter_no, metrics, {"status":"invalid_patch","error":str(e)})
            continue
        # Apply
        backup = f"{CODE_PATH}.bak{iter_no}"
        shutil.copy(CODE_PATH, backup)
        pf = os.path.join(TEMP_DIR, f"patch{iter_no}.diff")
        with open(pf,"w") as f: f.write(diff)
        try:
            subprocess.run(["patch", CODE_PATH], stdin=open(pf), check=True, text=True)
        except Exception as e:
            print(f"Patch failed: {e}")
            shutil.copy(backup, CODE_PATH)
            save_iteration_log(iter_no, metrics, {"status":"patch_fail","error":str(e)})
            continue
        # Reload
        try:
            importlib.reload(elo_module)
            with open(CODE_PATH) as cf: base_code = cf.read()
        except Exception as e:
            print(f"Reload failed: {e}")
            shutil.copy(backup, CODE_PATH)
            continue
        save_iteration_log(iter_no, metrics, {"status":"success","diff":diff})

    print("Done.")
if __name__ == '__main__':
    main()

