# wrangle module

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# custom imports
import acquire as a
import prepare as p


def wrangle_data():
    """
    Orchestrates the data acquisition, cleaning, splitting, and scaling process.

    Returns: tuple or None: A tuple containing train, validation, and test DataFrames if data is acquired successfully, or None IF 
    data acquisition fails.
    """
    df = a.acquire_data()

    if df is not None:
        
        # Clean the data
        df = p.prepare_data(df)

        # df_encoded = one_hot_encode(df, ['invoice_no', 'stock_code', 'description', 'country'])

        # a new_df for k-means
        new_df = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
                                            'InvoiceNo': 'count',
                                            'TotalPrice': 'sum'})
        # Split the data
        train, val, test = train_val_test(df)

        # Scale the data
        mms = MinMaxScaler()
        train_scaled, val_scaled, test_scaled = scale_data(train, val, test, mms)
        
        return train_scaled, val_scaled, test_scaled
    
    else:
        
        # Handle the case where data acquisition failed
        return None


def train_val_test(df, target=None, seed = 42):
    '''
    A function to split dataset into train, val, test
    '''

    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed,
                                       stratify = target)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed,
                                 stratify = target)
    
    return train, val, test


def scale_data(train, val, test, scaler):
    '''
    Scales data
    '''
    # Make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    # Define numeric columns for scaling
    numeric_columns = train.select_dtypes(include='number').columns.tolist()

    # Fit the scaler on the training data for numeric columns
    scaler.fit(train[numeric_columns])

    # Transform the data for each split using the fitted scaler
    train_scaled[numeric_columns] = scaler.transform(train[numeric_columns])
    validate_scaled[numeric_columns] = scaler.transform(val[numeric_columns])
    test_scaled[numeric_columns] = scaler.transform(test[numeric_columns])

    scaled_data = [train_scaled, validate_scaled, test_scaled]

    return scaled_data


def one_hot_encode(df, categorical_columns):
    """
    One-hot encodes the specified categorical columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame
    - categorical_columns: list of str, names of categorical columns to encode

    Returns:
    - df_encoded: pandas DataFrame, the DataFrame with one-hot encoded columns
    """
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    return df_encoded