# wrangle module

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# custom imports
import acquire as a
import prepare as p
from datetime import timedelta


def wrangle_data():
    """
    Orchestrates the data acquisition, cleaning, splitting, and scaling process.

    Returns: tuple or None: A tuple containing train, validation, and test DataFrames if data is acquired successfully, or None IF 
    data acquisition fails.
    """

    ### Retrieve Raw Data ###
    
    df = a.acquire_data()

    ### Prepared Data ###
        
    # Clean the data
    df = p.prepare_data(df)

    if df is not None:

        ### a new_df for k-means approach! ###
        
        # Group the DataFrame 'df' by 'customer_id'
        # For each group, calculate aggregated values using the agg() function
        new_df = df.groupby('customer_id').agg({
            # Calculate the recency (days since last invoice) for each customer
            'invoice_date': lambda x: (df['invoice_date'].max() - x.max()).days,
            
            # Count the number of invoices for each customer
            'invoice_no': 'count',
            
            # Sum the total_price for each customer
            'total_price': 'sum'
        })

        ### Customers Dataframe ###

        # Calculate a reference date for recency calculation
        # The reference date is set as the maximum invoice date in the 'invoice_date_day' column + 1 day
        ref_date = df["invoice_date_day"].max() + timedelta(days=1)
        
        # Group the DataFrame 'df' by 'customer_id'
        # For each group, calculate aggregated values using the agg() function
        df_customers = df.groupby("customer_id").agg({
            # Calculate recency for each customer by finding the days since their last invoice
            "invoice_date_day": lambda x: (ref_date - x.max()).days,
            
            # Count the number of invoices for each customer
            "invoice_no": "count",
            
            # Sum the total_price for each customer
            "total_price": "sum"
        }).rename(columns={
            # Rename the columns for clarity in the resulting DataFrame
            "invoice_date_day": "Recency",
            "invoice_no": "Frequency",
            "total_price": "MonetaryValue"
        })

        ### Train Validation and Test Subsets ###
        
        # Split the data
        train, val, test = train_val_test(df)

        ### Scaled Train, Validation, and Test Subsets ###
        
        # Scale the data using the MinMaxScaler from the SKLearn Library Preprocessing Module
        mms = MinMaxScaler()
        # Assigned subset names
        train_scaled, val_scaled, test_scaled = scale_data(train, val, test, mms)
        
        return train, val, test, train_scaled, val_scaled, test_scaled, new_df, df_customers, df
    
    else:
        
        # Handle the case where data acquisition failed
        return None, None, None, None, None



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