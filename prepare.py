# prepare function for CSD

# imported libs
import pandas as pd

# custome imports
import acquire as a

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

    # DataFrame with NaNs filled in the 'description' and 'customerid' columns
    df = df.copy()
    df['description'].fillna('no description', inplace=True)
    
    df = df.copy()
    df['customer_id'].fillna(0, inplace=True)

    # IQR in this function?

    return df