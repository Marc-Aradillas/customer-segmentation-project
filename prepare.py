# prepare function for CSD

import acquire as a

def prepare_data(df):
    df = a.acquire_data()
    df.columns = [
        col.lower().replace(' ','_') for col in df.columns
    ]

    # DataFrame with NaNs filled in the 'description' and 'customerid' column
    df = df.copy()
    df['description'].fillna('no description', inplace=True)
    
    df = df.copy()
    df['customerid'].fillna(0, inplace=True)

    # IQR in this function?