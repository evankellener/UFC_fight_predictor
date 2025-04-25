#!/usr/bin/env python3
"""
Update ELO ratings for prediction purposes.

This script retrieves the latest ELO ratings for fighters and updates them with 
any recent results that may not be in the database yet. It uses the consolidated
EnhancedElo class for all calculations.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from elo_class import FightDataProcessor, EnhancedElo

def update_elo_for_prediction(fighter1, fighter2, recent_results1=None, recent_results2=None):
    """
    Update ELO ratings for two fighters based on their recent results.
    
    Args:
        fighter1 (str): Name of first fighter
        fighter2 (str): Name of second fighter
        recent_results1 (list): Recent results for fighter1 [(opponent, win/loss, date, method), ...]
        recent_results2 (list): Recent results for fighter2 [(opponent, win/loss, date, method), ...]
        
    Returns:
        dict: Updated ELO information for both fighters
    """
    # Load fighter data
    data_path = os.path.join('data', 'tmp', 'interleaved_with_enhanced_elo.csv')
    
    try:
        data = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        # If enhanced ELO file is not found, use the standard ELO file
        data_path = os.path.join('data', 'tmp', 'interleaved_with_elo.csv')
        data = pd.read_csv(data_path, low_memory=False)
    
    # Initialize ELO calculator with optimal parameters
    elo_calculator = EnhancedElo(
        base_elo=1500,
        k_factor=40,
        decay_rate=2
    )
    
    # Get the most recent data for each fighter
    fighter1_data = data[data['FIGHTER'] == fighter1].sort_values('DATE', ascending=False)
    fighter2_data = data[data['FIGHTER'] == fighter2].sort_values('DATE', ascending=False)
    
    # Initialize results
    result = {
        'fighter1': {
            'name': fighter1,
            'elo': 1500,
            'elo_change_3': 0,
            'elo_change_5': 0,
            'last_fight_date': None
        },
        'fighter2': {
            'name': fighter2,
            'elo': 1500,
            'elo_change_3': 0,
            'elo_change_5': 0,
            'last_fight_date': None
        }
    }
    
    # If fighter data exists, get their current ELO
    if len(fighter1_data) > 0:
        latest_f1 = fighter1_data.iloc[0]
        result['fighter1']['elo'] = latest_f1['postcomp_elo']
        result['fighter1']['elo_change_3'] = latest_f1['precomp_elo_change_3']
        result['fighter1']['elo_change_5'] = latest_f1['precomp_elo_change_5']
        result['fighter1']['last_fight_date'] = latest_f1['DATE']
        
    if len(fighter2_data) > 0:
        latest_f2 = fighter2_data.iloc[0]
        result['fighter2']['elo'] = latest_f2['postcomp_elo']
        result['fighter2']['elo_change_3'] = latest_f2['precomp_elo_change_3']
        result['fighter2']['elo_change_5'] = latest_f2['precomp_elo_change_5']
        result['fighter2']['last_fight_date'] = latest_f2['DATE']
    
    # If recent results are provided, update the ELO ratings
    if recent_results1:
        # Process fighter1's recent results
        for fight in recent_results1:
            opponent, win, date, method = fight
            
            # Create synthetic fight data row
            fight_data = {
                'FIGHTER': fighter1,
                'opp_FIGHTER': opponent,
                'DATE': date,
                'BOUT': f"{fighter1} vs {opponent}",
                'result': 1 if win else 0,
                'winner': fighter1 if win else opponent,
                'loser': opponent if win else fighter1
            }
            
            # Add method information if available
            if method.upper() in ['KO', 'TKO']:
                fight_data['ko'] = 1 if win else 0
                fight_data['kod'] = 0 if win else 1
            elif method.upper() == 'SUB':
                fight_data['subw'] = 1 if win else 0
                fight_data['subwd'] = 0 if win else 1
            else:  # Decision
                fight_data['udec'] = 1 if win else 0
                fight_data['udecd'] = 0 if win else 1
            
            # Get opponent ELO if available
            opponent_data = data[data['FIGHTER'] == opponent].sort_values('DATE', ascending=False)
            opponent_elo = 1500
            if len(opponent_data) > 0:
                opponent_elo = opponent_data.iloc[0]['postcomp_elo']
            
            # Initialize ratings if needed
            if fight_data['FIGHTER'] not in elo_calculator.rating_dict:
                elo_calculator.rating_dict[fight_data['FIGHTER']] = result['fighter1']['elo']
                elo_calculator.last_fight_date[fight_data['FIGHTER']] = result['fighter1']['last_fight_date']
            
            if fight_data['opp_FIGHTER'] not in elo_calculator.rating_dict:
                elo_calculator.rating_dict[fight_data['opp_FIGHTER']] = opponent_elo
            
            # Update ELO based on fight result
            elo_calculator.calculate_elo(
                fight_data['winner'],
                fight_data['loser'],
                fight_data['DATE'],
                fight_data
            )
            
            # Store updated ELO
            result['fighter1']['elo'] = elo_calculator.rating_dict[fighter1]
            result['fighter1']['last_fight_date'] = date
    
    # Similarly for fighter2
    if recent_results2:
        # Process fighter2's recent results
        for fight in recent_results2:
            opponent, win, date, method = fight
            
            # Create synthetic fight data row
            fight_data = {
                'FIGHTER': fighter2,
                'opp_FIGHTER': opponent,
                'DATE': date,
                'BOUT': f"{fighter2} vs {opponent}",
                'result': 1 if win else 0,
                'winner': fighter2 if win else opponent,
                'loser': opponent if win else fighter2
            }
            
            # Add method information if available
            if method.upper() in ['KO', 'TKO']:
                fight_data['ko'] = 1 if win else 0
                fight_data['kod'] = 0 if win else 1
            elif method.upper() == 'SUB':
                fight_data['subw'] = 1 if win else 0
                fight_data['subwd'] = 0 if win else 1
            else:  # Decision
                fight_data['udec'] = 1 if win else 0
                fight_data['udecd'] = 0 if win else 1
            
            # Get opponent ELO if available
            opponent_data = data[data['FIGHTER'] == opponent].sort_values('DATE', ascending=False)
            opponent_elo = 1500
            if len(opponent_data) > 0:
                opponent_elo = opponent_data.iloc[0]['postcomp_elo']
            
            # Initialize ratings if needed
            if fight_data['FIGHTER'] not in elo_calculator.rating_dict:
                elo_calculator.rating_dict[fight_data['FIGHTER']] = result['fighter2']['elo']
                elo_calculator.last_fight_date[fight_data['FIGHTER']] = result['fighter2']['last_fight_date']
            
            if fight_data['opp_FIGHTER'] not in elo_calculator.rating_dict:
                elo_calculator.rating_dict[fight_data['opp_FIGHTER']] = opponent_elo
            
            # Update ELO based on fight result
            elo_calculator.calculate_elo(
                fight_data['winner'],
                fight_data['loser'],
                fight_data['DATE'],
                fight_data
            )
            
            # Store updated ELO
            result['fighter2']['elo'] = elo_calculator.rating_dict[fighter2]
            result['fighter2']['last_fight_date'] = date
    
    # Calculate ELO changes over last 3 and 5 fights (if available)
    # This is a simplified version since we don't have access to the full history
    # In a real system, we'd track this more carefully
    
    return result

if __name__ == "__main__":
    # Example usage
    fighter1 = "Max Holloway"
    fighter2 = "Alexander Volkanovski"
    
    # Example recent results
    recent_results1 = [
        ("Justin Gaethje", True, "2024-06-29", "KO")  # Holloway won by KO
    ]
    
    recent_results2 = [
        ("Ilia Topuria", False, "2024-02-17", "KO")  # Volkanovski lost by KO
    ]
    
    updated_elos = update_elo_for_prediction(
        fighter1, 
        fighter2, 
        recent_results1, 
        recent_results2
    )
    
    print("\nUpdated ELO Values:")
    print(f"{fighter1}: {updated_elos['fighter1']['elo']:.2f}")
    print(f"{fighter2}: {updated_elos['fighter2']['elo']:.2f}")