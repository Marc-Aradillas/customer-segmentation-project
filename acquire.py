import os
import pandas as pd

# ---------------------- ACQUIRE FUNCTION ---------------------------------
def acquire_data():
    '''
    Reads .xlsx file as dataframe and saves xlsx as a .csv. 
    
    In order to use must have 'online_retail.xlsx' in same folder.
    
    'online_retail.xlsx' can be found at : https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset/code?datasetId=1835880
    '''
    
    # Specify the file names
    xlsx_file_name = 'online_retail.xlsx'
    csv_file_name = 'online_retail.csv'

    # Construct the full paths to the files
    xlsx_file_path = os.path.join(os.getcwd(), xlsx_file_name)
    csv_file_path = os.path.join(os.getcwd(), csv_file_name)


    # Check if the CSV file exists
    if os.path.isfile(csv_file_path):
        # If it exists, read the CSV file
        return pd.read_csv(csv_file_path)
    else:
        # If it doesn't exist, read the XLSX file
        try:
            df = pd.read_excel(xlsx_file_path)
        except pd.errors.ParserError as e:
            print(f"Error reading XLSX file: {e}")
            return None

        # Save the DataFrame as a CSV file
        df.to_csv(csv_file_path, index=False)

    return df