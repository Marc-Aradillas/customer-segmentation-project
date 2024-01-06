# prepare function for CSD

# imported libs
import pandas as pd

# custome imports
import acquire as a

# DATA PREPARATION FUNCTION

def prepare_data(df):
    
    '''
    This function serves to accomplish cleaning of raw customier segmentation 
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
    df = df.set_index("invoice_date").sort_index()

    df = handle_missing_values(df, prop_required_column=.25, prop_required_row=0.95)
    
    df['customer_id'].fillna(0, inplace=True)

    df['is_return'] = (df['quantity'] < 0).astype(int)

    df['return_unit_price'] = (df['unit_price'] < 0).astype(int)

    # changing all negative values to zeros
    df['quantity'] = df['quantity'].apply(lambda x: max(x, 0))
    
    df['unit_price'] = df['unit_price'].apply(lambda x: max(x, 0))

    
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