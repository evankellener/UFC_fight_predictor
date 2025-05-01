import pandas as pd
import random
import os

def SanityCheck(csv_path='../data/tmp/final.csv', fighter_name=None):
    """
    Validates that a fighter's postcomp stats from one fight match their precomp stats in the next fight.
    
    Args:
        csv_path (str): Path to the CSV file containing fight data
        fighter_name (str, optional): Name of fighter to check. If None, selects a random fighter.
    
    Returns:
        tuple: (bool, dict) - Success flag and dictionary with validation results
    """
    df = pd.read_csv(csv_path)
    # Pick a random fighter if none is provided and make sure they have at least 6 fights
    if fighter_name is None:
        fighter_name = random.choice(df['FIGHTER'].unique())
        while df[df['FIGHTER'] == fighter_name].shape[0] < 6:
            fighter_name = random.choice(df['FIGHTER'].unique())
    # Create a dataframe with only the fights of the selected fighter
    fighter_df = df[df['FIGHTER'] == fighter_name]
    # Sort the dataframe by date
    fighter_df = fighter_df.sort_values(by='DATE')
    # Create a dictionary to store the results
    results = {}
    # Loop through the fights of the selected fighter and compare the postcomp stats of one fight with the precomp stats of the next fight
    for i in range(len(fighter_df) - 1):
        try:
            postcomp_stats = fighter_df.iloc[i][[
                'postcomp_sigstr_pm', 'postcomp_sigstr_pm3',
                'postcomp_tdavg', 'postcomp_tdavg5',
                'postcomp_sapm', 'postcomp_sapm3'
            ]]
            precomp_stats = fighter_df.iloc[i + 1][[
                'precomp_sigstr_pm', 'precomp_sigstr_pm3',
                'precomp_tdavg', 'precomp_tdavg5',
                'precomp_sapm', 'precomp_sapm3'
            ]]
            
            for j, stat in enumerate(postcomp_stats.index):
                precomp_stat = precomp_stats.index[j]
                if pd.notnull(postcomp_stats[stat]) and pd.notnull(precomp_stats[precomp_stat]):
                    if postcomp_stats[stat] != precomp_stats[precomp_stat]:
                        results[f"{fighter_name} - Fight {i+1} vs Fight {i+2}"] = {
                            'stat': stat,
                            'postcomp_stat': postcomp_stats[stat],
                            'precomp_stat': precomp_stats[precomp_stat]
                        }
        except KeyError as e:
            print(f"Missing column during sanity check: {e}")
            continue
    # Check if the results dictionary is empty
    if len(results) == 0:
        return True, results
    else:
        return False, results

# Alias for alternate capitalization
sanity_check = SanityCheck

if __name__ == "__main__":
    # Use absolute path to ensure the file is found regardless of working directory
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tmp', 'final.csv')
    success, results = SanityCheck(csv_path, fighter_name='Alexandre Pantoja')
    
    if success:
        print(f"SANITY CHECK PASSED for fighter: All postcomp stats match the precomp stats in subsequent fights.")
    else:
        print(f"SANITY CHECK FAILED: Some stats don't match between fights.")
        for fight, mismatch in results.items():
            print(f"\n{fight}:")
            print(f"  Stat: {mismatch['stat']}")
            print(f"  Postcomp value: {mismatch['postcomp_stat']}")
            print(f"  Precomp value: {mismatch['precomp_stat']}")