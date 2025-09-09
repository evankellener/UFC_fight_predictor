#!/usr/bin/env python3
import os
import sys
import re
import json
import logging
import shutil
from dotenv import load_dotenv

# --- Load env variables ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# Add src to path
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

import pandas as pd
import openai
import importlib

# Elo module and classes
import gpt_elo_best_time_finnish as elo_mod
from gpt_elo_best_time_finnish import EnhancedElo
from elo_feature_enhancer import EloFeatureEnhancer
from ensemble_model_best import FightOutcomeModel

# --- Config ---
DATA_CLEANED = os.path.join(SCRIPT_DIR, "../data/tmp/interleaved_cleaned.csv")
DATA_WITH_ELO = os.path.join(SCRIPT_DIR, "../data/tmp/interleaved_with_elo.csv")
DATA_FINAL = os.path.join(SCRIPT_DIR, "../data/tmp/final.csv")
ELO_SRC_PATH = os.path.join(SCRIPT_DIR, "src", "gpt_elo_best_time_finnish.py")
ELO_BACKUP = os.path.join(SCRIPT_DIR, "src", "gpt_elo_best_time_finnish_backup.py")
LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "iterations.jsonl")
LLM_LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "llm_interactions.jsonl")
OPENAI_MODEL = "gpt-4o"
ITERATIONS = 3

# Setup OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

# For openai>=1.0.0
client = openai.OpenAI(api_key=openai.api_key)

# Logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def evaluate_elo_module():
    global elo_mod
    elo_mod = importlib.reload(elo_mod)

    # Step 1: Load cleaned data and apply ELO
    logging.info("Loading cleaned data and applying ELO...")
    cleaned_data = pd.read_csv(DATA_CLEANED)
    # Convert relevant columns to numeric, coercing errors to NaN
    for col in ['precomp_boutcount', 'opp_precomp_boutcount', 'precomp_sigstr_perc', 'opp_precomp_sigstr_perc', 'age', 'opp_age']:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    cleaned_data['DATE'] = pd.to_datetime(cleaned_data['DATE'])
    elo_calculator = EnhancedElo(k_factor=32)
    results_with_elo = elo_calculator.process_fights(cleaned_data)
    results_with_elo.to_csv(DATA_WITH_ELO, index=False)

    # Step 2: Use ELO feature enhancer to clean and format
    logging.info("Processing with ELO feature enhancer...")
    editor = EloFeatureEnhancer(DATA_WITH_ELO)
    filtered_data = editor.filter_by_fight_count(min_fights=1)
    
    # Convert ELO columns to numeric
    cols_to_numeric = ['precomp_elo', 'opp_precomp_elo']
    filtered_data[cols_to_numeric] = filtered_data[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
    
    # Duplicate and swap rows
    swapped_data = editor.duplicate_and_swap_rows()
    swapped_data.to_csv(DATA_FINAL, index=False)

    # Step 3: Evaluate ELO accuracy (all fights)
    logging.info("Evaluating ELO accuracy (all fights)...")
    fight_model_all = FightOutcomeModel(DATA_FINAL)
    acc_all = fight_model_all.basic_elo_pred()
    loss_all = fight_model_all.elo_log_loss()
    print(f"All fights - Accuracy: {acc_all:.4f}, Log Loss: {loss_all:.4f}")
    logging.info(f"All fights - Accuracy: {acc_all:.4f}, Log Loss: {loss_all:.4f}")

    # Step 4: Evaluate ELO accuracy (recent 18 months)
    logging.info("Evaluating ELO accuracy (recent 18 months)...")
    df_eval = pd.read_csv(DATA_FINAL)
    df_eval['DATE'] = pd.to_datetime(df_eval['DATE'])
    cutoff = df_eval['DATE'].max() - pd.Timedelta(days=548)
    recent_fights = df_eval[df_eval['DATE'] >= cutoff].copy()

    # Debug prints for date filtering
    print(f"Total fights in all data: {len(df_eval)}")
    print(f"Date range in all data: {df_eval['DATE'].min()} to {df_eval['DATE'].max()}")
    print(f"Recent cutoff date: {cutoff}")
    print(f"Fights in recent 18 months: {len(recent_fights)}")
    if len(recent_fights) > 0:
        print(f"Date range in recent: {recent_fights['DATE'].min()} to {recent_fights['DATE'].max()}")
    else:
        print("No recent fights found.")

    if len(recent_fights) > 0:
        recent_path = os.path.join(SCRIPT_DIR, "../data/tmp/recent_fights.csv")
        recent_fights.to_csv(recent_path, index=False)
        fight_model_recent = FightOutcomeModel(recent_path)
        fight_model_recent.test_df = fight_model_recent.df  # Use all recent fights for evaluation
        acc_recent = fight_model_recent.basic_elo_pred()
        loss_recent = fight_model_recent.elo_log_loss()
        print(f"Recent 18 months - Accuracy: {acc_recent:.4f}, Log Loss: {loss_recent:.4f}")
        logging.info(f"Recent 18 months - Accuracy: {acc_recent:.4f}, Log Loss: {loss_recent:.4f}")
    else:
        acc_recent = None
        loss_recent = None
        print("No recent fights found for evaluation.")
        logging.info("No recent fights found for evaluation.")

    # Get sample data for LLM prompt
    sample_data = cleaned_data.head(3)  # Sample from original data for context
    
    return acc_all, loss_all, acc_recent, loss_recent, sample_data

def build_llm_prompt(code: str, acc_all: float, loss_all: float, acc_recent: float, loss_recent: float, sample_df: pd.DataFrame) -> str:
    # Get column information from the sample data
    cols = sample_df.columns.tolist()
    examples = sample_df.to_dict(orient='records')
    
    metrics_str = (
        f"Performance on ALL fights:\n- Accuracy: {acc_all:.4f}\n- Log Loss: {loss_all:.4f}\n"
        f"Performance on RECENT (last 18 months) fights:\n"
        f"- Accuracy: {acc_recent if acc_recent is not None else 'N/A'}\n"
        f"- Log Loss: {loss_recent if loss_recent is not None else 'N/A'}\n"
    )
    
    return (
        "You are an expert Python engineer specializing in ELO rating systems for combat sports. "
        "Below is the full ELO module code:\n\n"
        f"```python\n{code}\n```\n\n"
        "Current ELO Performance Metrics:\n"
        f"{metrics_str}\n"
        "Available Data Columns from interleaved_cleaned.csv:\n"
        f"{cols}\n\n"
        "Sample Data Rows (first 3 rows):\n"
        f"{examples}\n\n"
        "Your task: Analyze the current ELO implementation and suggest targeted improvements to enhance predictive accuracy and/or reduce log loss. "
        "Consider:\n"
        "- Weight class adjustments\n"
        "- Win streak bonuses\n"
        "- Method of victory multipliers\n"
        "- Time decay factors\n"
        "- Additional fighter statistics integration\n"
        "- K-factor optimization\n\n"
        "Return the full revised ELO module code wrapped in ```python ...```. "
        "Do not change the class name or main method signatures unless absolutely necessary. "
        "If you add new features, explain them in comments. "
        "Ensure the code maintains compatibility with the existing pipeline."
    )

def extract_code_from_response(response) -> str:
    text = response['choices'][0]['message']['content']
    match = re.search(r'```python([\s\S]*?)```', text)
    return match.group(1).strip() if match else text.strip()

def log_iteration(i, acc, loss):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps({'iter':i,'acc':acc,'loss':loss}) + '\n')

def log_llm_interaction(prompt, response):
    # For openai>=1.0.0, response.choices[0].message.content
    content = response.choices[0].message.content
    with open(LLM_LOG_FILE, 'a') as f:
        f.write(json.dumps({'prompt': prompt, 'response': content}) + '\n')

def validate_new_code(new_code: str) -> bool:
    """
    Try to compile the new code to catch syntax errors before writing.
    """
    try:
        compile(new_code, ELO_SRC_PATH, 'exec')
        return True
    except Exception as e:
        logging.error(f"LLM-generated code failed to compile: {e}")
        return False

def main():
    logging.info("Starting pipeline")
    for i in range(1, ITERATIONS+1):
        logging.info(f"=== Iter {i} ===")
        # Backup current ELO code
        shutil.copy(ELO_SRC_PATH, ELO_BACKUP)
        try:
            acc_all, loss_all, acc_recent, loss_recent, sample = evaluate_elo_module()
            logging.info(f"All fights - Acc: {acc_all:.4f}, Loss: {loss_all:.4f}")
            logging.info(f"Recent fights - Acc: {acc_recent}, Loss: {loss_recent}")
            log_iteration(i, acc_all, loss_all)
            with open(ELO_SRC_PATH) as f: code = f.read()
            prompt = build_llm_prompt(code, acc_all, loss_all, acc_recent, loss_recent, sample)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{'role':'user','content':prompt}],
                temperature=0.2
            )
            log_llm_interaction(prompt, resp)
            # Extract content from new API response
            new_code = extract_code_from_response({'choices': [{'message': {'content': resp.choices[0].message.content}}]})
            if validate_new_code(new_code):
                with open(ELO_SRC_PATH, 'w') as f: f.write(new_code + '\n')
                logging.info("Patched Elo module")
            else:
                logging.error("Restoring previous ELO code due to validation failure.")
                shutil.copy(ELO_BACKUP, ELO_SRC_PATH)
                break
        except Exception as e:
            logging.error(f"Error in iteration {i}: {e}")
            logging.error("Restoring previous ELO code.")
            shutil.copy(ELO_BACKUP, ELO_SRC_PATH)
            break
    logging.info("Done")

if __name__=='__main__': main()
