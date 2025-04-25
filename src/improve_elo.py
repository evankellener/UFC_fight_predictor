import pandas as pd
import os
import sys
import argparse
from elo_class import FightDataProcessor, EnhancedElo

def regenerate_dataset(base_path='.', min_fights=0, check_fighter=None):
    """
    Regenerate the UFC fighter dataset with improved ELO calculations.
    
    Args:
        base_path (str): Base path for file locations
        min_fights (int): Minimum number of fights to include in dataset
        check_fighter (str): Optional fighter name to check for inclusion
        
    Returns:
        DataFrame: The processed dataset
    """
    # Define file paths
    input_file = os.path.join(base_path, 'data', 'tmp', 'interleaved_with_elo.csv')
    output_file = os.path.join(base_path, 'data', 'tmp', 'interleaved_with_enhanced_elo.csv')
    
    print(f"Regenerating enhanced ELO dataset (min_fights={min_fights})...")
    
    # Load and process data
    processor = FightDataProcessor(input_file)
    data = processor.load_data()
    data = processor.clean_data()
    
    # Check for specific fighter if requested
    if check_fighter:
        check_fighter_clean = check_fighter.replace(" ", "")
        fighter_in_input = check_fighter_clean in str(data['FIGHTER'].values)
        print(f"{check_fighter} found in input data: {fighter_in_input}")
        
        if not fighter_in_input:
            # Check raw files for the fighter
            raw_file = os.path.join(base_path, 'data', 'tmp', 'interleaved_dup_sawp.csv')
            if os.path.exists(raw_file):
                try:
                    raw_data = pd.read_csv(raw_file, low_memory=False)
                    raw_fighter = raw_data[raw_data['FIGHTER'] == check_fighter]
                    if len(raw_fighter) > 0:
                        print(f"Found {check_fighter} in raw data ({len(raw_fighter)} rows)")
                        print(f"The fighter exists in raw data but not in processed ELO data.")
                        print(f"This script will attempt to ensure inclusion in the final dataset.")
                except Exception as e:
                    print(f"Error checking raw data: {str(e)}")
    
    # Initialize ELO calculator with improved parameters
    elo_calculator = EnhancedElo(
        base_elo=1500,      # Standard starting ELO
        k_factor=40,        # Increased from 32 for more dramatic changes
        decay_rate=2        # Reduced from 5 for less decay penalty
    )
    
    # Process fights with the enhanced ELO calculator
    print("Processing fights with enhanced ELO calculations...")
    processed_data = elo_calculator.process_fights(data, min_fights=min_fights)
        
    # Save processed data
    processed_data.to_csv(output_file, index=False)
    print(f"Enhanced ELO dataset saved to {output_file}")
    
    # Check if specific fighter made it to the final dataset
    if check_fighter:
        check_fighter_clean = check_fighter.replace(" ", "")
        fighter_in_output = check_fighter in processed_data['FIGHTER'].values
        print(f"{check_fighter} found in output data: {fighter_in_output}")
        
        if not fighter_in_output:
            print(f"WARNING: {check_fighter} is still not in the final dataset.")
            print(f"This could indicate additional filtering in other parts of the pipeline.")
    
    # Print some statistics for comparison
    print("\nELO Statistics Comparison:")
    print("-" * 50)
    print(f"Number of fights: {len(processed_data)}")
    print(f"Average ELO change (3 fights): {processed_data['precomp_elo_change_3'].mean():.2f}")
    print(f"Average ELO change (5 fights): {processed_data['precomp_elo_change_5'].mean():.2f}")
    print(f"Max ELO rating: {processed_data['precomp_elo'].max():.2f}")
    print(f"Min ELO rating: {processed_data['precomp_elo'].min():.2f}")
    
    return processed_data

def main():
    """
    Main function for command line usage with argument parsing.
    """
    parser = argparse.ArgumentParser(description='Regenerate UFC fighter dataset with improved ELO calculations')
    parser.add_argument('--min-fights', type=int, default=0, help='Minimum number of fights to include in dataset')
    parser.add_argument('--check-fighter', type=str, help='Check for a specific fighter in the datasets')
    args = parser.parse_args()
    
    regenerate_dataset(min_fights=args.min_fights, check_fighter=args.check_fighter)
    
if __name__ == "__main__":
    main()