'''
Auto ARIMA Forecasting Model

This module performs:
1. Automatic parameter selection
2. Model training
3. Forecasting
4. Performance evaluation
'''

import pandas as pd

from pmdarima import auto_arima
from evaluation import evaluate_model

# Find Best ARIMA Model

def build_model(train_df):
    '''
    Automatically selects the best ARIMA model.
    '''
    print('\nSearching Best ARIMA Model...\n')

    model = auto_arima(train_df['Close'], 
        seasonal=False,
        information_criterion='aic',
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        d=1,                   # already confirmed d=1 using the ADF test
        stepwise=True,
        trace=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    print('\n-------------------------------------')
    print('Best ARIMA Model')
    print(model.summary())
    print('-------------------------------------\n')

    return model

# Forecast
def forecast(model, test_df):
    predictions = model.predict(n_periods=len(test_df))
    predictions.index = test_df.index

    return predictions

# Main Wrapper
def run_arima(train_df, validation_df, test_df):

    print('=' * 60)
    print('AUTO ARIMA MODEL')
    print('=' * 60)

    # Combine training and validation datasets
    train_validation = pd.concat([train_df, validation_df])

    # Build model
    model = build_model(train_validation)

    # Forecast
    predictions = forecast(model, test_df)

    # Evaluate
    results = evaluate_model(actual=test_df['Close'], predicted = predictions, model_name = 'Auto ARIMA')

    return results