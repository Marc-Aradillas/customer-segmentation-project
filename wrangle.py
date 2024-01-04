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


def train_val_test(df, target=None, seed = 42):

    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed,
                                       stratify = target)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed,
                                 stratify = target)
    
    return train, val, test


def scale_data(train, val, test, scaler):

    # make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    columns_to_scale = ['customer_id', 'age', 'annual_income', 'spending_score', 'female', 'male']
    
    # Fit the scaler on the training data for all of the columns
    scaler.fit(train[columns_to_scale])
    
    # Transform the data for each split
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(val[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    scaled_col = [train_scaled, validate_scaled, test_scaled]
    
    return train_scaled, validate_scaled, test_scaled