import pandas as pd


def train_validation_test_split(df):
    '''
    Split the dataset into train, validation and test sets
    while preserving chronological order.

    Train      : 70%
    Validation : 15%
    Test       : 15%
    '''

    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    print('=' * 50)
    print('Time Series Data Split')
    print('=' * 50)
    print(f'Total Records : {n}')
    print(f'Training      : {len(train)}')
    print(f'Validation    : {len(validation)}')
    print(f'Testing       : {len(test)}')
    print('=' * 50)

    return train, validation, test