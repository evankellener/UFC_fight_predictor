#!/usr/bin/env python3
"""
Batch prediction script for UFC matchups.

This script reads a CSV file of matchups and runs each one through the enhanced ELO
prediction model, then saves the results back to a new CSV file with confidence levels.
"""

import os
import sys
import pandas as pd
import csv
import io
import contextlib
from predict_with_updated_elo import predict_with_updated_elo

# Create a context manager to suppress output
@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout

def process_matchups(input_file, output_file):
    """
    Process all matchups in the input CSV and write predictions to output CSV.
    
    Args:
        input_file (str): Path to input CSV with matchups
        output_file (str): Path to output CSV where results will be saved
    """
    print(f"Reading matchups from {input_file}")
    
    # Read the input CSV file
    matchups = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:  # Make sure we have at least two fighters
                fighter_a = row[0].strip()
                fighter_b = row[1].strip()
                matchups.append((fighter_a, fighter_b))
    
    # Skip header if present
    if matchups and matchups[0][0] == "Fighter A":
        matchups = matchups[1:]
        
    print(f"Found {len(matchups)} matchups to process")
    
    # Process each matchup and store results
    results = []
    
    for i, (fighter_a, fighter_b) in enumerate(matchups):
        print(f"Processing matchup {i+1}/{len(matchups)}: {fighter_a} vs {fighter_b}")
        
        try:
            # Suppress verbose output from prediction function
            with suppress_stdout():
                # Use the enhanced ELO prediction function
                winner, probability = predict_with_updated_elo(fighter_a, fighter_b)
            
            # Format probability as percentage
            probability_pct = f"{probability:.2%}"
            
            print(f"  Predicted winner: {winner} with {probability_pct} probability")
            
            # Save the result
            results.append({
                "Fighter A": fighter_a,
                "Fighter B": fighter_b,
                "Predicted Winner": winner,
                "Confidence": probability_pct
            })
            
            # Save after each prediction in case of interruption
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_file, index=False)
            
        except Exception as e:
            print(f"  Error predicting {fighter_a} vs {fighter_b}: {str(e)}")
            results.append({
                "Fighter A": fighter_a,
                "Fighter B": fighter_b,
                "Predicted Winner": "ERROR",
                "Confidence": f"Error: {str(e)}"
            })
            
            # Save after each error too
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_file, index=False)
    
    # Final save to CSV
    print(f"\nWriting results to {output_file}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"Done! Processed {len(results)} matchups")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Use command line arguments if provided, otherwise use default paths
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = os.path.join('data', 'tmp', 'UFC_Fight_Night_Results.csv')
        output_file = os.path.join('data', 'tmp', 'UFC_Fight_Night_Predictions.csv')
    
    process_matchups(input_file, output_file)