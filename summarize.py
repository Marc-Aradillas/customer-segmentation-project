# add a summary function to populate summary of data.

import env
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def missing_by_col(df): 
    '''
    returns a single series of null values by column name
    '''
    return df.isnull().sum(axis=0)

def missing_by_row(df) -> pd.DataFrame:
    '''
    prints out a report of how many rows have a certain
    number of columns/fields missing both by count and proportion
    
    '''
    count_missing = df.isnull().sum(axis=1)
    percent_missing = round((df.isnull().sum(axis=1) / df.shape[1]) * 100)
    rows_df = pd.DataFrame({
    'num_cols_missing': count_missing,
    'percent_cols_missing': percent_missing
    }).reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns={'index':'num_rows'})
    return rows_df

def get_fences(df, col, k=1.5) -> (float, float):
    '''
    get fences will calculate the upper and lower fence
    based on the inner quartile range of a single Series
    
    return: lower_bound and upper_bound, two floats
    '''
    # df.describe['.75']
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound

def report_outliers(df, k=1.5) -> None:
    '''
    report_outliers will print a subset of each continuous
    series in a dataframe (based on numeric quality and n>20)
    and will print out results of this analysis with the fences
    in places
    '''
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(df[col][(
                df[col] > upper_bound) | (df[col] < lower_bound)])
            print('----------')

def summarize(df, k=1.5) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    
    info
    shape
    outliers
    description
    missing data stats
    
    return: None (prints to console)
    '''
    # print info on the df
    print('Shape of Data: ')
    print(df.shape)
    print('======================\n======================')
    print('Info: ')
    print(df.info())
    print('======================\n======================')
    print('Descriptions:')
    # print the description of the df, transpose, output markdown
    print(df.describe().T.to_markdown())
    print('======================\n======================')
    # lets do that for categorical info as well
    # we will use select_dtypes to look at just Objects
    print(df.select_dtypes('O').describe().T.to_markdown())
    print('======================\n======================')
    print('missing values:')
    print('by column:')
    print(missing_by_col(df).to_markdown())
    print('by row: ')
    print(missing_by_row(df).to_markdown())
    print('======================\n======================')
    print('Outliers: ')
    print(report_outliers(df, k=k))
    print('======================\n======================')


def identify_outliers(col, k=1.5):
    
    q1, q3 = col.quantile([0.25, 0.75])
    
    iqr = q3 - q1
    
    lower_fence = q1 - iqr * k
    upper_fence = q3 + iqr * k
    
    return np.where((col < lower_fence) | (col > upper_fence), 1, 0)

# example use:
'''
for col in df.columns[df.dtypes == 'float64']:
    
    df[f'{col}_outliers'] = identify_outliers(df[col])  
'''