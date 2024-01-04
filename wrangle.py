# wrangle module

import pandas

# custom imports
import acquire as a
import prepare as p


def wrangle_data():
    '''
    Function to acquire and clean data
    '''
    df = a.acquire_data()

    df = p.prepare_data(df)

    return df