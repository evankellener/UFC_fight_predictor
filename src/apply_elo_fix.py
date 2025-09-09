import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_elo_consistency(data_path):
    """
    Analyze the consistency of ELO values across the dataset.
    
    Args:
        data_path (str): Path to the interleaved data with ELO values
        
    Returns:
        pd.DataFrame: DataFrame with ELO consistency metrics for fighters
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Create mapping of fighters to their fights
    fighter_fights = {}
    for fighter in df['FIGHTER'].unique():
        fighter_fights[fighter] = df[df['FIGHTER'] == fighter].sort_values('DATE')
        
    # Analyze ELO consistency
    elo_consistency = []
    for fighter, fights in fighter_fights.items():
        if len(fights) <= 1:
            continue
            
        # Check each pair of consecutive fights
        inconsistencies = 0
        for i in range(len(fights) - 1):
            curr_fight = fights.iloc[i]
            next_fight = fights.iloc[i+1]
            
            # Compare postcomp_elo to next fight's precomp_elo
            if 'postcomp_elo' in curr_fight and 'precomp_elo' in next_fight:
                try:
                    post_elo = float(curr_fight['postcomp_elo'])
                    pre_elo = float(next_fight['precomp_elo'])
                    
                    if not pd.isna(post_elo) and not pd.isna(pre_elo):
                        diff = abs(post_elo - pre_elo)
                        if diff > 1.0:  # Significant difference
                            inconsistencies += 1
                except:
                    pass
                    
        # Calculate consistency percentage
        consistency = 100 * (1 - inconsistencies / (len(fights) - 1)) if len(fights) > 1 else 100
        
        elo_consistency.append({
            'Fighter': fighter,
            'Total Fights': len(fights),
            'Inconsistencies': inconsistencies,
            'Consistency %': consistency,
            'Last Elo': fights.iloc[-1].get('postcomp_elo', 0)
        })
    
    # Convert to DataFrame and sort
    elo_df = pd.DataFrame(elo_consistency)
    elo_df = elo_df.sort_values('Consistency %')
    
    # Print summary
    print(f"Analyzed ELO consistency for {len(elo_df)} fighters")
    print(f"Average consistency: {elo_df['Consistency %'].mean():.2f}%")
    print(f"Fighters with perfect consistency: {len(elo_df[elo_df['Consistency %'] == 100])}")
    print(f"Fighters with poor consistency (<50%): {len(elo_df[elo_df['Consistency %'] < 50])}")
    
    # Plot histogram of consistency
    plt.figure(figsize=(10, 6))
    plt.hist(elo_df['Consistency %'], bins=20)
    plt.title('ELO Consistency Distribution')
    plt.xlabel('Consistency %')
    plt.ylabel('Number of Fighters')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(os.path.dirname(data_path), f'elo_continuity_{timestamp}.csv')
    elo_df.to_csv(plot_path, index=False)
    print(f"Saved consistency data to {plot_path}")
    
    return elo_df

def fix_elo_continuity(data_path, output_path=None):
    """
    Fix ELO continuity issues in the dataset.
    
    Args:
        data_path (str): Path to the interleaved data with ELO values
        output_path (str, optional): Path to save the fixed data
        
    Returns:
        pd.DataFrame: DataFrame with fixed ELO values
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Create a copy to modify
    fixed_df = df.copy()
    
    # Track fixes made
    fixes_made = 0
    fighters_fixed = set()
    
    # Process each fighter
    for fighter in df['FIGHTER'].unique():
        # Get all fights for this fighter in chronological order
        fighter_fights = df[df['FIGHTER'] == fighter].sort_values('DATE')
        
        if len(fighter_fights) <= 1:
            continue
            
        # Check each pair of consecutive fights
        for i in range(len(fighter_fights) - 1):
            current_idx = fighter_fights.index[i]
            next_idx = fighter_fights.index[i+1]
            
            # Get postcomp from current fight and precomp from next fight
            current_fight = fighter_fights.iloc[i]
            next_fight = fighter_fights.iloc[i+1]
            
            # Process ELO columns
            elo_keys = [
                ('postcomp_elo', 'precomp_elo'),
                ('postcomp_elo_prev', 'precomp_elo_prev'),
                ('postcomp_elo_change_3', 'precomp_elo_change_3'),
                ('postcomp_elo_change_5', 'precomp_elo_change_5')
            ]
            
            for post_key, pre_key in elo_keys:
                if post_key in current_fight and pre_key in next_fight:
                    try:
                        post_val = float(current_fight[post_key])
                        pre_val = float(next_fight[pre_key])
                        
                        # Only fix significant differences
                        if not pd.isna(post_val) and not pd.isna(pre_val) and abs(post_val - pre_val) > 1.0:
                            # Update the precomp value in next fight to match postcomp from current fight
                            fixed_df.at[next_idx, pre_key] = post_val
                            fixes_made += 1
                            fighters_fixed.add(fighter)
                    except:
                        # Skip if conversion fails
                        pass
    
    print(f"Made {fixes_made} fixes for {len(fighters_fixed)} fighters")
    
    # Save fixed data if output path provided
    if output_path:
        fixed_df.to_csv(output_path, index=False)
        print(f"Saved fixed data to {output_path}")
    
    return fixed_df

def main():
    """Main function to run the ELO fixes"""
    # Paths
    data_dir = os.path.join(os.getcwd(), 'data/tmp')
    input_path = os.path.join(data_dir, 'interleaved_with_elo.csv')
    
    if not os.path.exists(input_path):
        print(f"ERROR: Could not find input data at {input_path}")
        return
    
    # First analyze ELO consistency
    consistency_df = analyze_elo_consistency(input_path)
    
    # Now fix the continuity issues
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(data_dir, f'interleaved_with_elo_fixed.csv')
    fixed_df = fix_elo_continuity(input_path, output_path)
    
    # Generate a report
    report_path = os.path.join(os.getcwd(), 'elo_fix_report.md')
    with open(report_path, 'w') as f:
        f.write("# ELO Continuity Fix Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Analyzed {len(consistency_df)} fighters\n")
        f.write(f"- Average ELO consistency before fix: {consistency_df['Consistency %'].mean():.2f}%\n")
        f.write(f"- Fighters with perfect consistency: {len(consistency_df[consistency_df['Consistency %'] == 100])}\n")
        f.write(f"- Fighters with poor consistency (<50%): {len(consistency_df[consistency_df['Consistency %'] < 50])}\n\n")
        
        f.write("## Worst Fighters (by consistency)\n\n")
        f.write("| Fighter | Fights | Inconsistencies | Consistency % | Last ELO |\n")
        f.write("|---------|--------|-----------------|---------------|----------|\n")
        
        for _, row in consistency_df.sort_values('Consistency %').head(10).iterrows():
            f.write(f"| {row['Fighter']} | {row['Total Fights']} | {row['Inconsistencies']} | {row['Consistency %']:.1f}% | {row['Last Elo']:.1f} |\n")
            
    print(f"Generated report at {report_path}")
    
    print("\nSuggested next steps:")
    print("1. Use the fixed data file for feature engineering")
    print("2. Update the notebook to read from the fixed file instead")
    print("3. Re-run the model evaluation with the fixed data")

if __name__ == "__main__":
    main()