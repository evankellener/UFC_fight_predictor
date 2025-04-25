import pandas as pd
import os
import numpy as np

def compare_precomp_postcomp_values():
    """
    Compare a fighter's postcomp values from one fight with their precomp values 
    in their next fight to confirm they match.
    """
    # Load the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Try to find the data file based on common patterns
    potential_data_files = [
        os.path.join(project_root, 'data', 'formatted_data', 'final.csv'),
        os.path.join(project_root, 'data', 'tmp', 'final.csv'),
        os.path.join(project_root, 'data', 'tmp', 'interleaved_with_elo.csv'),
        os.path.join(project_root, 'data', 'tmp', 'interleaved_with_elo_fixed.csv')
    ]
    
    data_file = None
    for file_path in potential_data_files:
        if os.path.exists(file_path):
            data_file = file_path
            break
    
    if data_file is None:
        print("Could not find data file. Please specify the correct path.")
        return
    
    print(f"Loading data from: {data_file}")
    data = pd.read_csv(data_file)
    
    # Convert DATE to datetime
    data['DATE'] = pd.to_datetime(data['DATE'])
    
    # Sort by fighter and date
    data = data.sort_values(['FIGHTER', 'DATE'])
    
    # Get list of all fighters
    fighters = data['FIGHTER'].unique()
    
    # Get list of columns to compare (all columns with 'precomp' or 'postcomp')
    precomp_cols = [col for col in data.columns if 'precomp_' in col and not col.startswith('opp_')]
    postcomp_cols = [col for col in data.columns if 'postcomp_' in col and not col.startswith('opp_')]
    
    # Create pairs of corresponding precomp and postcomp columns
    col_pairs = []
    for postcomp_col in postcomp_cols:
        corresponding_precomp = postcomp_col.replace('postcomp_', 'precomp_')
        if corresponding_precomp in precomp_cols:
            col_pairs.append((postcomp_col, corresponding_precomp))
    
    print(f"Found {len(col_pairs)} matching column pairs.")
    
    # Number of fighters to sample for analysis
    sample_size = min(20, len(fighters))
    sampled_fighters = np.random.choice(fighters, sample_size, replace=False)
    
    # Track statistics
    total_comparisons = 0
    matching_values = 0
    mismatch_details = []
    
    print("\nAnalyzing continuity between postcomp and next fight's precomp values...")
    
    # For each fighter, compare their postcomp values from one fight to precomp values in next fight
    for fighter in sampled_fighters:
        print(f"\nAnalyzing fighter: {fighter}")
        
        # Get all fights for this fighter, sorted by date
        fighter_fights = data[data['FIGHTER'] == fighter].sort_values('DATE')
        
        # Skip if fighter has less than 2 fights
        if len(fighter_fights) < 2:
            print(f"  Skipping - has only {len(fighter_fights)} fight")
            continue
            
        # Compare each pair of consecutive fights
        for i in range(len(fighter_fights) - 1):
            current_fight = fighter_fights.iloc[i]
            next_fight = fighter_fights.iloc[i+1]
            
            current_date = current_fight['DATE']
            next_date = next_fight['DATE']
            
            print(f"  Comparing fight on {current_date.date()} to next fight on {next_date.date()}")
            
            # Compare all column pairs
            for postcomp_col, precomp_col in col_pairs:
                postcomp_val = current_fight[postcomp_col]
                next_precomp_val = next_fight[precomp_col]
                
                # If both are nan, consider them matching
                if pd.isna(postcomp_val) and pd.isna(next_precomp_val):
                    matching_values += 1
                    total_comparisons += 1
                    continue
                
                # If only one is nan, they don't match
                if pd.isna(postcomp_val) or pd.isna(next_precomp_val):
                    total_comparisons += 1
                    mismatch_details.append({
                        'fighter': fighter,
                        'date1': current_date,
                        'date2': next_date,
                        'column': (postcomp_col, precomp_col),
                        'val1': postcomp_val,
                        'val2': next_precomp_val,
                        'reason': 'One value is NaN'
                    })
                    continue
                
                # Handle numeric comparison with tolerance for floating point issues
                try:
                    # Convert to float if possible
                    postcomp_float = float(postcomp_val)
                    next_precomp_float = float(next_precomp_val)
                    
                    # Check if equal within tolerance
                    if abs(postcomp_float - next_precomp_float) < 1e-6:
                        matching_values += 1
                    else:
                        mismatch_details.append({
                            'fighter': fighter,
                            'date1': current_date,
                            'date2': next_date,
                            'column': (postcomp_col, precomp_col),
                            'val1': postcomp_val,
                            'val2': next_precomp_val,
                            'reason': 'Values differ', 
                            'diff': postcomp_float - next_precomp_float
                        })
                    
                    total_comparisons += 1
                    
                except (ValueError, TypeError):
                    # Handle non-numeric values with exact comparison
                    if postcomp_val == next_precomp_val:
                        matching_values += 1
                    else:
                        mismatch_details.append({
                            'fighter': fighter,
                            'date1': current_date,
                            'date2': next_date,
                            'column': (postcomp_col, precomp_col),
                            'val1': postcomp_val,
                            'val2': next_precomp_val,
                            'reason': 'Non-numeric values differ'
                        })
                    
                    total_comparisons += 1
    
    # Calculate percentage of matching values
    if total_comparisons > 0:
        match_percentage = (matching_values / total_comparisons) * 100
    else:
        match_percentage = 0
    
    print("\n=== Comparison Results ===")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Matching values: {matching_values}")
    print(f"Match percentage: {match_percentage:.2f}%")
    
    # Display some sample mismatches
    if mismatch_details:
        print("\nSample mismatches:")
        sample_count = min(5, len(mismatch_details))
        for i in range(sample_count):
            mismatch = mismatch_details[i]
            print(f"  Fighter: {mismatch['fighter']}")
            print(f"  Dates: {mismatch['date1'].date()} -> {mismatch['date2'].date()}")
            print(f"  Columns: {mismatch['column'][0]} -> {mismatch['column'][1]}")
            print(f"  Values: {mismatch['val1']} -> {mismatch['val2']}")
            print(f"  Reason: {mismatch['reason']}")
            if 'diff' in mismatch:
                print(f"  Difference: {mismatch['diff']}")
            print()
    else:
        print("\nNo mismatches found!")
    
    # Group mismatches by column pair to see patterns
    if mismatch_details:
        print("\nMismatches by column:")
        col_mismatch_count = {}
        for mismatch in mismatch_details:
            col_pair = f"{mismatch['column'][0]} -> {mismatch['column'][1]}"
            col_mismatch_count[col_pair] = col_mismatch_count.get(col_pair, 0) + 1
        
        # Sort by count
        sorted_cols = sorted(col_mismatch_count.items(), key=lambda x: x[1], reverse=True)
        
        # Print top 10
        for col, count in sorted_cols[:10]:
            print(f"  {col}: {count} mismatches")
    
    return match_percentage, mismatch_details

if __name__ == "__main__":
    print("Comparing precomp and postcomp values for continuity between fights...")
    match_pct, mismatches = compare_precomp_postcomp_values()