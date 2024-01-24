# prepare function for CSD

# imported libs
import pandas as pd
import numpy as np
from datetime import timedelta

# custome imports
import acquire as a

# DATA PREPARATION FUNCTION

def prepare_data(df):
    
    '''
    This function serves to accomplish cleaning of raw customir segmentation 
    data from kaggle.
    '''
    # import acquisition of raw data
    df = a.acquire_data()

    # make feature names pythonic
    df.columns = [
        col.lower().replace(' ','_') for col in df.columns
    ]

    # renaming feature names for readability
    df.rename(columns={
        'invoiceno': 'invoice_no',
        'stockcode': 'stock_code',
        'invoicedate': 'invoice_date',
        'unitprice': 'unit_price',
        'customerid': 'customer_id'
    }, inplace=True)

    # Reassigning the invoicedate column to be a datetime type
    df.invoice_date = pd.to_datetime(df.invoice_date)
    
    # Sorting rows by the date and then set the index as that date
    # df = df.set_index("invoice_date").sort_index()

    df = handle_missing_values(df, prop_required_column=.25, prop_required_row=0.95)
    
    df['customer_id'].fillna(0, inplace=True)

    # Converting the following features to strings
    
    df['invoice_no'] = df['invoice_no'].astype(str)
    
    df['stock_code'] = df['stock_code'].astype(str)

    df['customer_id'] = df['customer_id'].astype(int)
    
    df['customer_id'] = df['customer_id'].astype(str)

    # need to remove all canceled invoices and its positive counterparts

    negative_quantity = df[df["quantity"] < 0][["customer_id", "stock_code", "quantity"]].sort_values("quantity")

    filtered = df[df["customer_id"].isin(negative_quantity["customer_id"])]
    filtered = filtered[filtered["stock_code"].isin(negative_quantity["stock_code"])]

    
    # Initialize an empty list to store the indices of corresponding positive counterparts
    pos_counters = []
    
    # Iterate over rows in the 'negative_quantity' DataFrame
    for idx, series in negative_quantity.iterrows():
        # Extract relevant information from the current row
        customer = series["customer_id"]
        code = series["stock_code"]
        quantity = -1 * series["quantity"]  # Convert quantity to positive
    
        # Filter rows in 'filtered' DataFrame matching the specified conditions
        counterpart = filtered[(filtered["customer_id"] == customer) & 
                               (filtered["stock_code"] == code) & 
                               (filtered["quantity"] == quantity)]
    
        # Extend the list of indices with the found positive counterparts
        pos_counters.extend(counterpart.index.to_list())
    
    # Create a list of indices to drop, including both negative quantity rows and their positive counterparts
    to_drop = negative_quantity.index.to_list() + pos_counters
    
    # Drop the specified rows from the DataFrame 'df'
    df.drop(to_drop, axis=0, inplace=True)

    # Total price feature addition

    df['total_price'] = df['quantity'] * df['unit_price']

    # extracted datetime features from invoice_date
    
    df["invoice_date_day"] = df["invoice_date"].dt.date
    
    df["invoice_date_time"] = df["invoice_date"].dt.time
    
    df["invoice_year"] = df["invoice_date"].dt.year
    
    df["invoice_month"] = df["invoice_date"].dt.month
    
    df["invoice_month_name"] = df["invoice_date"].dt.month_name()
    
    df["invoice_day"] = df["invoice_date"].dt.day
    
    df["invoice_day_name"] = df["invoice_date"].dt.day_name()
    
    df["invoice_day_of_week"] = df["invoice_date"].dt.day_of_week
    
    df["invoice_week_of_year"] = df["invoice_date"].dt.isocalendar().week
    
    df["invoice_hour"] = df["invoice_date"].dt.hour

    # Outlier columns features

    for col in df.select_dtypes(include='number').columns:
        
        df[f'{col}_outliers'] = identify_outliers(df[col])


    return df




# MISSING VALUES FUNCTION

def missing_values(df):
    '''
    This function serves to identify missing values in the dataset
    '''
    # calculate number of missing value for each attribute
    missing_counts = df.isna().sum()

    # calculate the percent of missing vals in each attribute
    total_rows = len(df)
    missing_percentages = (missing_counts / total_rows) * 100

    # create a summary df
    summary_df = pd.DataFrame({'Missing Values' : missing_counts, 'Percentage Missing (%)': missing_percentages})

    return summary_df




# HANDLE MISSING VALUES FUNCTION

def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
    This function is to calculate the threshold for columns and rows.
    '''
    # create thresholds and totals
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    col_threshold = int(total_rows * prop_required_column)
    row_threshold = int(total_columns * prop_required_row)
    
    # Drop columns with missing values exceeding the threshold
    df = df.dropna(axis=1, thresh=col_threshold)
    
    # Drop rows with missing values exceeding the threshold
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df



# IDENTIFY OUTLIERS FUNCTION

def identify_outliers(col, k=1.5):
    
    q1, q3 = col.quantile([0.25, 0.75])
    
    iqr = q3 - q1
    
    lower_fence = q1 - iqr * k
    upper_fence = q3 + iqr * k
    
    return np.where((col < lower_fence) | (col > upper_fence), 1, 0)


