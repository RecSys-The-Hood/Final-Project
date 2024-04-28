import os
import pandas as pd

def combine_csv_files(directory_path, output_file):
    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    # Initialize an empty DataFrame to store combined data
    combined_df = pd.DataFrame()

    # Flag to keep track of whether column names are read
    column_names_read = False

    # Loop through each CSV file and concatenate it horizontally to the combined DataFrame
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        if(combined_df.size==0) :
            combined_df=pd.DataFrame(df)
            combined_df=combined_df.set_index(keys='zpid')
        else : 
            df=df.set_index(keys='zpid')
            combined_df=pd.concat([combined_df,df])
    # # Write the combined DataFrame to a single CSV file
    print(combined_df)
    combined_df.to_csv(output_file)
# Example usage:
combine_csv_files('./', './combined_data_1_501.csv')
