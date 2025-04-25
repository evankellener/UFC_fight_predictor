#!/usr/bin/env python3
"""
Improved fight prediction script that updates ELO ratings before making predictions.

This script combines the consolidated EnhancedElo system with the fight predictor to provide more
accurate predictions by using the most up-to-date fighter ratings. The ELO calculations have been
optimized for model accuracy while ensuring champion fighters maintain appropriate ratings.
"""

import os
import sys
import argparse
import pandas as pd
from model import DNNFightPredictor
from update_elo_for_prediction import update_elo_for_prediction

def predict_with_updated_elo(fighter1, fighter2, recent_results1=None, recent_results2=None):
    """
    Make a fight prediction with up-to-date ELO ratings.
    
    Args:
        fighter1 (str): Name of first fighter
        fighter2 (str): Name of second fighter
        recent_results1 (list): Recent results for fighter1
        recent_results2 (list): Recent results for fighter2
        
    Returns:
        tuple: (predicted winner, probability)
    """
    print(f"Making prediction for {fighter1} vs {fighter2} with updated ELO...")
    
    # Step 1: Update ELO ratings
    updated_elos = update_elo_for_prediction(fighter1, fighter2, recent_results1, recent_results2)
    
    # Step 2: Load the main dataset
    file_path = os.path.join('data', 'tmp', 'interleaved_with_elo.csv')
    predictor = DNNFightPredictor(file_path=file_path)
    
    # Step 3: Get fighter data from the dataset
    fighter1_data = predictor.data[predictor.data['FIGHTER'] == fighter1]
    fighter2_data = predictor.data[predictor.data['FIGHTER'] == fighter2]
    
    if len(fighter1_data) == 0 or len(fighter2_data) == 0:
        print("One or both fighters not found in the dataset, using fallback method...")
        return predictor.predict_fight_winner(fighter1, fighter2)
    
    # Step 4: Get the most recent data for each fighter
    fighter1_data = fighter1_data.sort_values('DATE', ascending=False).iloc[0].copy()
    fighter2_data = fighter2_data.sort_values('DATE', ascending=False).iloc[0].copy()
    
    # Step 5: Override ELO values with updated values
    print(f"Using updated ELO values for prediction:")
    print(f"  {fighter1}: {updated_elos['fighter1']['elo']:.2f} (was {fighter1_data['precomp_elo']:.2f})")
    print(f"  {fighter2}: {updated_elos['fighter2']['elo']:.2f} (was {fighter2_data['precomp_elo']:.2f})")
    
    # Update fighter1 ELO values in the dataset
    fighter1_data['precomp_elo'] = updated_elos['fighter1']['elo']
    fighter1_data['precomp_elo_change_3'] = updated_elos['fighter1']['elo_change_3']
    fighter1_data['precomp_elo_change_5'] = updated_elos['fighter1']['elo_change_5']
    
    # Update fighter2 ELO values in the dataset
    fighter2_data['precomp_elo'] = updated_elos['fighter2']['elo']
    fighter2_data['precomp_elo_change_3'] = updated_elos['fighter2']['elo_change_3']
    fighter2_data['precomp_elo_change_5'] = updated_elos['fighter2']['elo_change_5']
    
    # Step 6: Create a temporary dataset with updated values
    temp_data = predictor.data.copy()
    # Find the most recent row for each fighter and update it
    f1_idx = temp_data[temp_data['FIGHTER'] == fighter1].sort_values('DATE', ascending=False).index[0]
    f2_idx = temp_data[temp_data['FIGHTER'] == fighter2].sort_values('DATE', ascending=False).index[0]
    
    # Update with the new ELO values
    temp_data.loc[f1_idx, 'precomp_elo'] = updated_elos['fighter1']['elo']
    temp_data.loc[f1_idx, 'precomp_elo_change_3'] = updated_elos['fighter1']['elo_change_3']
    temp_data.loc[f1_idx, 'precomp_elo_change_5'] = updated_elos['fighter1']['elo_change_5']
    
    temp_data.loc[f2_idx, 'precomp_elo'] = updated_elos['fighter2']['elo']
    temp_data.loc[f2_idx, 'precomp_elo_change_3'] = updated_elos['fighter2']['elo_change_3']
    temp_data.loc[f2_idx, 'precomp_elo_change_5'] = updated_elos['fighter2']['elo_change_5']
    
    # Save temporary dataset
    temp_file = os.path.join('data', 'tmp', 'temp_prediction_data.csv')
    temp_data.to_csv(temp_file, index=False)
    
    # Step 7: Create a new predictor with the updated dataset
    updated_predictor = DNNFightPredictor(file_path=temp_file)
    
    # Step 8: Make the prediction
    winner, probability = updated_predictor.predict_fight_winner(fighter1, fighter2)
    
    # Clean up temporary file
    try:
        os.remove(temp_file)
    except:
        pass
    
    return winner, probability

def main():
    parser = argparse.ArgumentParser(description='Make UFC fight predictions with updated ELO ratings')
    parser.add_argument('fighter1', type=str, help='Name of first fighter')
    parser.add_argument('fighter2', type=str, help='Name of second fighter')
    parser.add_argument('--add-result', action='store_true', help='Add recent fight results for the fighters')
    args = parser.parse_args()
    
    # Example usage
    fighter1 = args.fighter1
    fighter2 = args.fighter2
    
    recent_results1 = None
    recent_results2 = None
    
    # If the user wants to add recent results
    if args.add_result:
        print("\nAdd recent fights for", fighter1)
        add_more = True
        recent_results1 = []
        
        while add_more:
            opponent = input("Opponent name: ")
            result = input("Result (W/L): ").upper() == "W"
            date = input("Date (YYYY-MM-DD): ")
            method = input("Method (KO/TKO, SUB, DEC): ")
            
            recent_results1.append((opponent, result, date, method))
            
            add_more = input("Add another fight? (y/n): ").lower() == "y"
        
        print("\nAdd recent fights for", fighter2)
        add_more = True
        recent_results2 = []
        
        while add_more:
            opponent = input("Opponent name: ")
            result = input("Result (W/L): ").upper() == "W"
            date = input("Date (YYYY-MM-DD): ")
            method = input("Method (KO/TKO, SUB, DEC): ")
            
            recent_results2.append((opponent, result, date, method))
            
            add_more = input("Add another fight? (y/n): ").lower() == "y"
    
    # Make prediction with updated ELO values
    winner, probability = predict_with_updated_elo(fighter1, fighter2, recent_results1, recent_results2)
    
    print("\n=== FINAL PREDICTION ===")
    print(f"Winner: {winner}")
    print(f"Probability: {probability:.2%}")

if __name__ == "__main__":
    main()