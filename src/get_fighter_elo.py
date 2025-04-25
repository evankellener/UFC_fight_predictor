#!/usr/bin/env python3
"""
Script to get the most recent ELO rating for a fighter.

This utility helps with making predictions by retrieving the most
up-to-date ELO rating, which can be used for future fight predictions.
"""

import pandas as pd
import os
import sys
import argparse
from datetime import datetime

# Set paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'data', 'tmp')

def get_fighter_elo(fighter_name, verbose=True):
    """
    Get the most recent ELO rating for a fighter.
    
    Args:
        fighter_name (str): Name of the fighter to look up
        verbose (bool): Whether to print detailed information
        
    Returns:
        float: Most recent ELO rating, or None if fighter not found
    """
    # Define the dataset path
    dataset_path = os.path.join(DATA_DIR, 'tmp', 'interleaved_with_enhanced_elo.csv')
    
    if not os.path.exists(dataset_path):
        if verbose:
            print(f"Error: Dataset not found at {dataset_path}")
        return None
    
    # Load the dataset
    try:
        df = pd.read_csv(dataset_path, low_memory=False)
        if verbose:
            print(f"Loaded {len(df)} records from dataset")
    except Exception as e:
        if verbose:
            print(f"Error loading dataset: {str(e)}")
        return None
    
    # Look for exact match first
    fighter_matches = df[df['FIGHTER'] == fighter_name]
    
    # If no exact match, try a case-insensitive search
    if len(fighter_matches) == 0:
        fighter_matches = df[df['FIGHTER'].str.lower() == fighter_name.lower()]
    
    # If still no match, try a partial match
    if len(fighter_matches) == 0:
        fighter_matches = df[df['FIGHTER'].str.contains(fighter_name, case=False, na=False)]
        
        # Also check as opponent
        opp_matches = df[df['opp_FIGHTER'].str.contains(fighter_name, case=False, na=False)]
        
        if len(fighter_matches) == 0 and len(opp_matches) == 0:
            if verbose:
                print(f"Fighter '{fighter_name}' not found in dataset")
            return None
        
        # If found only as opponent, convert to fighter perspective
        if len(fighter_matches) == 0 and len(opp_matches) > 0:
            if verbose:
                print(f"Found {fighter_name} only as opponent in {len(opp_matches)} fights")
                
            # Get the most recent opponent fight
            opp_matches['DATE'] = pd.to_datetime(opp_matches['DATE'])
            latest_fight = opp_matches.sort_values('DATE', ascending=False).iloc[0]
            
            # Return the opponent's most recent ELO
            latest_elo = latest_fight['opp_postcomp_elo']
            fight_date = latest_fight['DATE'].strftime('%Y-%m-%d')
            opponent = latest_fight['FIGHTER']
            
            if verbose:
                print(f"Most recent fight: {fight_date} vs {opponent}")
                print(f"Latest ELO: {latest_elo:.2f}")
            
            return latest_elo
    
    # Get the most recent fighter match
    fighter_matches['DATE'] = pd.to_datetime(fighter_matches['DATE'])
    latest_fight = fighter_matches.sort_values('DATE', ascending=False).iloc[0]
    
    # Extract the ELO rating
    latest_elo = latest_fight['postcomp_elo']
    
    if verbose:
        # Show detailed info
        fight_date = latest_fight['DATE'].strftime('%Y-%m-%d')
        opponent = latest_fight['opp_FIGHTER']
        fight_result = 'Win' if latest_fight.get('result', 0) == 1 else 'Loss'
        
        print(f"\nFighter: {fighter_name}")
        print(f"Most recent fight: {fight_date} vs {opponent} ({fight_result})")
        print(f"Latest ELO rating: {latest_elo:.2f}")
        
        # Get fighter's record
        wins = len(fighter_matches[fighter_matches['result'] == 1])
        losses = len(fighter_matches) - wins
        print(f"UFC Record: {wins}-{losses}")
        
        # Get ELO history
        print("\nELO History (last 5 fights):")
        history = fighter_matches.sort_values('DATE', ascending=False).head(5)
        for _, fight in history.iterrows():
            date = pd.to_datetime(fight['DATE']).strftime('%Y-%m-%d')
            opp = fight['opp_FIGHTER']
            pre_elo = fight['precomp_elo']
            post_elo = fight['postcomp_elo']
            result = 'Win' if fight.get('result', 0) == 1 else 'Loss'
            print(f"  {date} vs {opp} ({result}): {pre_elo:.2f} → {post_elo:.2f}")
    
    return latest_elo

def main():
    parser = argparse.ArgumentParser(description="Get the most recent ELO rating for a UFC fighter")
    parser.add_argument('fighter', help="Name of the fighter to look up")
    parser.add_argument('--quiet', '-q', action='store_true', help="Suppress detailed output")
    
    args = parser.parse_args()
    
    elo = get_fighter_elo(args.fighter, verbose=not args.quiet)
    
    if args.quiet and elo is not None:
        print(f"{elo:.2f}")
    
    sys.exit(0 if elo is not None else 1)

if __name__ == "__main__":
    main()