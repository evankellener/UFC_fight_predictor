import pandas as pd
import sqlite3

class Cleanup:
    def __init__(self, df):
        self.df = df
    
    def order_by_date(self):

        self.df.columns = self.df.columns.str.replace("\'","")

        # Convert the 'DATE' column to datetime format
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df['opp_DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')

        # Sort the data by 'DATE' in descending order
        self.df = self.df.sort_values(by='DATE', ascending=False)

        self.df.to_csv('../data/tmp/pre_final.csv', index=True)
    """
    def duplicate_and_swap_row(self):
        left_columns = [col for col in self.df.columns if not col.startswith('opp_')]
        right_columns = [col for col in self.df.columns if col.startswith('opp_')]

        # List to hold the original and swapped rows
        interleaved_rows = []

        for index, row in self.df.iterrows():
            # Append the original row
            interleaved_rows.append(row)

            # Create and append the swapped row
            swapped_row = row.copy()
            for l_col, r_col in zip(left_columns, right_columns):
                swapped_row[l_col], swapped_row[r_col] = row[r_col], row[l_col]

            interleaved_rows.append(swapped_row)

        # Create the DataFrame from the list of rows
        interleaved_df = pd.DataFrame(interleaved_rows)
        self.df = interleaved_df

        return interleaved_df
    """
    def results_column(self):
        # Filter the DataFrame to remove fights where either fighter or opponent is having their first fight
        self.df['result'] = self.df['win']
        self.df = self.df.round(2)
        self.df = self.df.rename(columns={'sub':'subw', 'subd':'subwd'})
        interleaved_df = self.df
        return interleaved_df