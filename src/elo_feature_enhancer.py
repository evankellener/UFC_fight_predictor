import pandas as pd
import pickle
import os

class EloFeatureEnhancer:
    def __init__(self, df):
        self.df = pd.read_csv(df)
        self.recent_fighter_stats = {}

    def insert_elo_features_in_order(self):
        # Anchors for insertion points
        fighter_anchor = 'precomp_mdecavg5'
        opp_anchor = 'opp_precomp_mdecavg5'

        # Fighter and opponent Elo columns
        fighter_elo_cols = [
            'precomp_elo', 'postcomp_elo',
            'precomp_elo_prev', 'postcomp_elo_prev',
            'precomp_elo_change_3', 'precomp_elo_change_5',
            'postcomp_elo_change_3', 'postcomp_elo_change_5'
        ]
        opp_elo_cols = ['opp_' + col for col in fighter_elo_cols]

        # Get current columns
        cols = list(self.df.columns)

        # Remove existing Elo columns to avoid duplication
        for col in fighter_elo_cols + opp_elo_cols:
            if col in cols:
                cols.remove(col)

        # Find insertion locations
        fighter_idx = cols.index(fighter_anchor) + 1 if fighter_anchor in cols else None
        opp_idx = cols.index(opp_anchor) + 1 if opp_anchor in cols else None

        # Insert fighter Elo columns
        if fighter_idx is not None:
            for i, col in enumerate(fighter_elo_cols):
                cols.insert(fighter_idx + i, col)

        # Insert opponent Elo columns
        if opp_idx is not None:
            for i, col in enumerate(opp_elo_cols):
                cols.insert(opp_idx + i, col)

        # Reorder DataFrame
        self.df = self.df[cols]
        return self.df

    def store_fighter_postcomp_stats(self, output_path='../data/tmp/recent_fighter_stats.pkl'):
        """
        Stores the most recent postcomp stats for each fighter before filtering.
        Also ensures consistency between postcomp stats and precomp stats for the next fight.
        
        Args:
            output_path (str): Path to save the pickle file with fighter stats
            
        Returns:
            dict: Dictionary mapping fighter names to their most recent postcomp stats
        """
        print("Storing postcomp stats from each fighter's most recent fight...")
        self.recent_fighter_stats = {}
        consistent_stats = {}
        
        # Process fighters to ensure consistency
        for fighter in self.df['FIGHTER'].unique():
            # Get all fights for this fighter sorted by date
            fighter_data = self.df[self.df['FIGHTER'] == fighter].copy()
            fighter_data['DATE'] = pd.to_datetime(fighter_data['DATE'])
            fighter_data = fighter_data.sort_values('DATE', ascending=True)
            
            if len(fighter_data) <= 0:
                continue
                
            # Get most recent fight data
            most_recent = fighter_data.iloc[-1].copy()
            consistent_stats[fighter] = most_recent.to_dict()
            
            # If fighter has multiple fights, ensure consistent stats between fights
            if len(fighter_data) > 1:
                # Go through each pair of fights chronologically
                for i in range(len(fighter_data) - 1):
                    current_fight = fighter_data.iloc[i]
                    next_fight = fighter_data.iloc[i+1]
                    
                    # Find all postcomp columns in current fight
                    postcomp_cols = [col for col in current_fight.index if col.startswith('postcomp_')]
                    
                    # Ensure corresponding precomp values match in next fight
                    for postcomp_col in postcomp_cols:
                        precomp_col = postcomp_col.replace('postcomp_', 'precomp_')
                        
                        if precomp_col in next_fight.index:
                            try:
                                # Update only if there's a mismatch and both values are valid numbers
                                postcomp_val = pd.to_numeric(current_fight[postcomp_col], errors='coerce')
                                precomp_val = pd.to_numeric(next_fight[precomp_col], errors='coerce')
                                
                                # Only process if both values are valid
                                if not pd.isna(postcomp_val) and not pd.isna(precomp_val):
                                    # If they differ significantly, use postcomp value from previous fight
                                    if abs(postcomp_val - precomp_val) > 0.01:
                                        # Update the consistent stats for this fighter
                                        if precomp_col in consistent_stats[fighter]:
                                            consistent_stats[fighter][precomp_col] = postcomp_val
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                continue
            
            # Store the corrected most recent stats
            self.recent_fighter_stats[fighter] = consistent_stats[fighter]
        
        # Save recent fighter stats for later use in evaluation
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.recent_fighter_stats, f)
            
        print(f"Saved recent fighter stats for {len(self.recent_fighter_stats)} fighters")
        return self.recent_fighter_stats
    
    def verify_precomp_postcomp_consistency(self):
        """
        Verifies that precomp stats match postcomp stats from previous fights for each fighter.
        This is a sanity check to ensure data consistency.
        
        Returns:
            tuple: (passed, mismatches) where passed is a boolean indicating if the check passed
                  and mismatches is the number of mismatches found
        """
        print("Performing sanity check on precomp/postcomp values...")
        sanity_check_passed = True
        mismatches = 0
        
        for fighter in self.df['FIGHTER'].unique():
            fighter_data = self.df[self.df['FIGHTER'] == fighter].sort_values('DATE')
            if len(fighter_data) <= 1:
                continue
                
            for i in range(1, len(fighter_data)):
                prev_fight = fighter_data.iloc[i-1]
                curr_fight = fighter_data.iloc[i]
                
                # Check core features that should match between postcomp and precomp
                for postcomp_col in [col for col in prev_fight.index if col.startswith('postcomp_')]:
                    precomp_col = postcomp_col.replace('postcomp_', 'precomp_')
                    if precomp_col in curr_fight:
                        try:
                            if abs(float(prev_fight[postcomp_col]) - float(curr_fight[precomp_col])) > 0.01:
                                if mismatches < 5:  # Only show first 5 mismatches to avoid flooding output
                                    print(f"Mismatch for {fighter}: {postcomp_col}={prev_fight[postcomp_col]}, {precomp_col}={curr_fight[precomp_col]}")
                                mismatches += 1
                                sanity_check_passed = False
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            continue
        
        if sanity_check_passed:
            print("Sanity check passed: precomp stats match postcomp stats from previous fights")
        else:
            print(f"WARNING: Found {mismatches} instances where precomp stats don't match postcomp stats from previous fights")
            
        return sanity_check_passed, mismatches

    def duplicate_and_swap_rows(self):
        # Columns to consider for swapping
        left_cols = [col for col in self.df.columns if not col.startswith('opp_') and col != 'result']
        right_cols = [col for col in self.df.columns if col.startswith('opp_')]

        swapped_rows = []

        # Duplicate and swap each row
        for _, row in self.df.iterrows():
            original = row.copy()
            swapped = row.copy()

            for l_col in left_cols:
                r_col = 'opp_' + l_col
                if r_col in self.df.columns:
                    swapped[l_col], swapped[r_col] = original[r_col], original[l_col]

            swapped_rows.append(original)
            swapped_rows.append(swapped)

        result_df = pd.DataFrame(swapped_rows)

        # Drop all "Unnamed" columns
        result_df = result_df.loc[:, ~result_df.columns.str.startswith("Unnamed")]

        # Round float columns to 2 decimal places
        float_cols = result_df.select_dtypes(include=['float32', 'float64']).columns
        result_df[float_cols] = result_df[float_cols].round(2)

        result_df['DATE'] = pd.to_datetime(result_df['DATE'], errors='coerce')

        return result_df
    
    def filter_by_fight_count(self, min_fights=1):
        """
        Filter the dataset to only include fights where both fighters have at least min_fights precomp_boutcount.
        
        For min_fights=1: excludes fights where precomp_boutcount < 1 (i.e., precomp_boutcount = 0)
        For min_fights=2: excludes fights where precomp_boutcount < 2 (i.e., precomp_boutcount = 0 or 1)
        """
        print(f"Filtering by minimum precomp_boutcount: {min_fights}")
        original_size = len(self.df)
        
        # Apply filter - both fighters must have at least min_fights precomp_boutcount
        self.df = self.df[
            (self.df['precomp_boutcount'] >= min_fights) &
            (self.df['opp_precomp_boutcount'] >= min_fights)
        ]
        
        final_size = len(self.df)
        print(f"Filtering complete: {original_size} -> {final_size} rows ({original_size - final_size} removed)")
        
        return self.df
    
    def differential_and_rolling_stats(self, data):
        data['precomp_elo_prev'] = data.groupby('FIGHTER')['precomp_elo'].diff().fillna(0).round(2)
        # if it's the fighter's first fight, the diff is 0 only for precomp_elo_prev, for 
        # postcomp_elo_prev it should be the differnece between the fighter's precomp_elo and the fighter's postcomp_elo(precomp_elo - postcomp_elo)
        # do not just do: data['postcomp_elo_prev'] = data.groupby('FIGHTER')['postcomp_elo'].diff().fillna(0).round(2)
        data['postcomp_elo_prev'] = data['postcomp_elo'] - data['precomp_elo']
        data['opp_precomp_elo_prev'] = data.groupby('opp_FIGHTER')['opp_precomp_elo'].diff().fillna(0).round(2)
        data['opp_postcomp_elo_prev'] = data['opp_postcomp_elo'] - data['opp_precomp_elo']

        #round to 2 decimal places
        data['precomp_elo_prev'] = data['precomp_elo_prev'].round(2)
        data['postcomp_elo_prev'] = data['postcomp_elo_prev'].round(2)
        data['opp_precomp_elo_prev'] = data['opp_precomp_elo_prev'].round(2)
        data['opp_postcomp_elo_prev'] = data['opp_postcomp_elo_prev'].round(2)

        # Calculate rolling averages
        #data['precomp_elo_change_3'] = data.groupby('FIGHTER')['precomp_elo_prev'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)

        # Define a helper function for rolling sums excluding the current row
        data['precomp_elo_change_3'] = (
            data.groupby('FIGHTER')['precomp_elo_prev']
            .transform(lambda x: x.shift(0).rolling(3, min_periods=1).sum().fillna(0).round(2))
        )
        data['precomp_elo_change_5'] = (
            data.groupby('FIGHTER')['precomp_elo_prev']
            .transform(lambda x: x.shift(0).rolling(5, min_periods=1).sum().fillna(0).round(2))
        )
        data['postcomp_elo_change_3'] = (
            data.groupby('FIGHTER')['postcomp_elo_prev']
            .transform(lambda x: x.shift(0).rolling(3, min_periods=1).sum().fillna(0).round(2))
        )
        data['postcomp_elo_change_5'] = (
            data.groupby('FIGHTER')['postcomp_elo_prev']
            .transform(lambda x: x.shift(0).rolling(5, min_periods=1).sum().fillna(0).round(2))
        )

        data['opp_precomp_elo_change_3'] = (
            data.groupby('opp_FIGHTER')['opp_precomp_elo_prev']
            .transform(lambda x: x.shift(0).rolling(3, min_periods=1).sum().fillna(0).round(2))
        )
        data['opp_precomp_elo_change_5'] = (
            data.groupby('opp_FIGHTER')['opp_precomp_elo_prev']
            .transform(lambda x: x.shift(0).rolling(5, min_periods=1).sum().fillna(0).round(2))
        )
        data['opp_postcomp_elo_change_3'] = (
            data.groupby('opp_FIGHTER')['opp_postcomp_elo_prev']
            .transform(lambda x: x.shift(0).rolling(3, min_periods=1).sum().fillna(0).round(2))
        )
        data['opp_postcomp_elo_change_5'] = (
            data.groupby('opp_FIGHTER')['opp_postcomp_elo_prev']
            .transform(lambda x: x.shift(0).rolling(5, min_periods=1).sum().fillna(0).round(2))
        )
        """
        data['precomp_elo_change_5'] = data.groupby('FIGHTER')['precomp_elo_prev'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
        data['postcomp_elo_change_3'] = data.groupby('FIGHTER')['postcomp_elo_prev'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['postcomp_elo_change_5'] = data.groupby('FIGHTER')['postcomp_elo_prev'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_precomp_elo_change_3'] = data.groupby('opp_FIGHTER')['opp_precomp_elo_prev'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_precomp_elo_change_5'] = data.groupby('opp_FIGHTER')['opp_precomp_elo_prev'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_postcomp_elo_change_3'] = data.groupby('opp_FIGHTER')['opp_postcomp_elo_prev'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
        data['opp_postcomp_elo_change_5'] = data.groupby('opp_FIGHTER')['opp_postcomp_elo_prev'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
        #round to 2 decimal places
        data['precomp_elo_change_3'] = data['precomp_elo_change_3'].round(2)
        data['precomp_elo_change_5'] = data['precomp_elo_change_5'].round(2)
        data['postcomp_elo_change_3'] = data['postcomp_elo_change_3'].round(2)
        data['postcomp_elo_change_5'] = data['postcomp_elo_change_5'].round(2)
        data['opp_precomp_elo_change_3'] = data['opp_precomp_elo_change_3'].round(2)
        data['opp_precomp_elo_change_5'] = data['opp_precomp_elo_change_5'].round(2)
        data['opp_postcomp_elo_change_3'] = data['opp_postcomp_elo_change_3'].round(2)
        data['opp_postcomp_elo_change_5'] = data['opp_postcomp_elo_change_5'].round(2)


        """

        data['precomp_strike_elo_prev'] = data.groupby('FIGHTER')['precomp_strike_elo'].diff().fillna(0).round(2)
        # if it's the fighter's first fight, the diff is 0 only for precomp_elo_prev, for 
        # postcomp_elo_prev it should be the differnece between the fighter's precomp_elo and the fighter's postcomp_elo(precomp_elo - postcomp_elo)
        # do not just do: data['postcomp_elo_prev'] = data.groupby('FIGHTER')['postcomp_elo'].diff().fillna(0).round(2)
        data['postcomp_strike_elo_prev'] = data['postcomp_strike_elo'] - data['precomp_strike_elo']
        data['opp_precomp_strike_elo_prev'] = data.groupby('opp_FIGHTER')['opp_precomp_strike_elo'].diff().fillna(0).round(2)
        data['opp_postcomp_strike_elo_prev'] = data['opp_postcomp_strike_elo'] - data['opp_precomp_strike_elo']
        #round to 2 decimal places
        data['precomp_strike_elo_prev'] = data['precomp_strike_elo_prev'].round(2)
        data['postcomp_strike_elo_prev'] = data['postcomp_strike_elo_prev'].round(2)
        data['opp_precomp_strike_elo_prev'] = data['opp_precomp_strike_elo_prev'].round(2)
        data['opp_postcomp_strike_elo_prev'] = data['opp_postcomp_strike_elo_prev'].round(2)
        # Calculate rolling averages for strike Elo
        data['precomp_strike_elo_change_3'] = (
            data.groupby('FIGHTER')['precomp_strike_elo_prev']
            .transform(lambda x: x.shift(0).rolling(3, min_periods=1).sum().fillna(0).round(2))
        )
        data['precomp_strike_elo_change_5'] = (
            data.groupby('FIGHTER')['precomp_strike_elo_prev']
            .transform(lambda x: x.shift(0).rolling(5, min_periods=1).sum().fillna(0).round(2))
        )
        data['postcomp_strike_elo_change_3'] = (
            data.groupby('FIGHTER')['postcomp_strike_elo_prev']
            .transform(lambda x: x.shift(0).rolling(3, min_periods=1).sum().fillna(0).round(2))
        )
        data['postcomp_strike_elo_change_5'] = (
            data.groupby('FIGHTER')['postcomp_strike_elo_prev']
            .transform(lambda x: x.shift(0).rolling(5, min_periods=1).sum().fillna(0).round(2))
        )
        data['opp_precomp_strike_elo_change_3'] = (
            data.groupby('opp_FIGHTER')['opp_precomp_strike_elo_prev']
            .transform(lambda x: x.shift(0).rolling(3, min_periods=1).sum().fillna(0).round(2))
        )
        data['opp_precomp_strike_elo_change_5'] = (
            data.groupby('opp_FIGHTER')['opp_precomp_strike_elo_prev']
            .transform(lambda x: x.shift(0).rolling(5, min_periods=1).sum().fillna(0).round(2))
        )
        data['opp_postcomp_strike_elo_change_3'] = (
            data.groupby('opp_FIGHTER')['opp_postcomp_strike_elo_prev']
            .transform(lambda x: x.shift(0).rolling(3, min_periods=1).sum().fillna(0).round(2))
        )
        data['opp_postcomp_strike_elo_change_5'] = (
            data.groupby('opp_FIGHTER')['opp_postcomp_strike_elo_prev']
            .transform(lambda x: x.shift(0).rolling(5, min_periods=1).sum().fillna(0).round(2))
        )
        #round to 2 decimal places



        #caluculate differences between fighter and opponent precomp and postcomp elo
        data['precomp_elo_diff'] = (data['precomp_elo'] - data['opp_precomp_elo']).round(2)
        data['postcomp_elo_diff'] = (data['postcomp_elo'] - data['opp_postcomp_elo']).round(2)
        
        return data
