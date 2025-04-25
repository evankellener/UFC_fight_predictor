#!/usr/bin/env python3
"""
Script to predict the outcome of a UFC fight using the latest ELO ratings.

This script:
1. Gets the most recent ELO ratings for both fighters
2. Applies any decay based on inactivity
3. Calculates win probability using the ELO formula
"""

import pandas as pd
import os
import sys
import argparse
from datetime import datetime, timedelta
import math
from get_fighter_elo import get_fighter_elo

# Set paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'data', 'tmp')

def predict_fight(fighter1, fighter2, verbose=True):
    """
    Predict the outcome of a fight between two fighters.
    
    Args:
        fighter1 (str): Name of the first fighter
        fighter2 (str): Name of the second fighter
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (fighter1_win_prob, fighter2_win_prob, elo_diff)
    """
    # Get the latest ELO ratings
    if verbose:
        print(f"Getting ELO rating for {fighter1}...")
    elo1 = get_fighter_elo(fighter1, verbose=verbose)
    
    if elo1 is None:
        print(f"Error: Could not find ELO rating for {fighter1}")
        return None
    
    if verbose:
        print(f"\nGetting ELO rating for {fighter2}...")
    elo2 = get_fighter_elo(fighter2, verbose=verbose)
    
    if elo2 is None:
        print(f"Error: Could not find ELO rating for {fighter2}")
        return None
    
    # Calculate win probabilities using ELO formula
    elo_diff = elo1 - elo2
    fighter1_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
    fighter2_win_prob = 1 - fighter1_win_prob
    
    if verbose:
        print("\nFight Prediction:")
        print(f"{fighter1} ELO: {elo1:.2f}")
        print(f"{fighter2} ELO: {elo2:.2f}")
        print(f"ELO Difference: {elo_diff:.2f}")
        print(f"{fighter1} win probability: {fighter1_win_prob*100:.1f}%")
        print(f"{fighter2} win probability: {fighter2_win_prob*100:.1f}%")
        
        # Add some context
        if fighter1_win_prob > 0.65:
            print(f"\n{fighter1} is a strong favorite")
        elif fighter1_win_prob > 0.55:
            print(f"\n{fighter1} is a slight favorite")
        elif fighter1_win_prob < 0.35:
            print(f"\n{fighter2} is a strong favorite")
        elif fighter1_win_prob < 0.45:
            print(f"\n{fighter2} is a slight favorite")
        else:
            print("\nThis fight is very close to even")
    
    return (fighter1_win_prob, fighter2_win_prob, elo_diff)

def main():
    parser = argparse.ArgumentParser(description="Predict UFC fight outcome using ELO ratings")
    parser.add_argument('fighter1', help="Name of the first fighter")
    parser.add_argument('fighter2', help="Name of the second fighter")
    parser.add_argument('--quiet', '-q', action='store_true', help="Suppress detailed output")
    
    args = parser.parse_args()
    
    result = predict_fight(args.fighter1, args.fighter2, verbose=not args.quiet)
    
    if result is None:
        sys.exit(1)
    
    fighter1_win_prob, fighter2_win_prob, elo_diff = result
    
    if args.quiet:
        print(f"{fighter1_win_prob:.4f},{fighter2_win_prob:.4f},{elo_diff:.2f}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()