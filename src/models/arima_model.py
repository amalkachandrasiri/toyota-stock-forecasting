'''
ARIMA Model Implementation

This module performs:
1. Hyperparameter tuning using Grid Search
2. Model training
3. Forecasting
4. Model evaluation
'''

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from itertools import product

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from evaluation import evaluate_model

# ===========
# Grid Search
# ===========

def find_best_arima(train, validation):
    '''
    Performs grid search to determine the optimal
    (p,d,q) parameters using the validation dataset.

    Returns:
        best_order
    '''

    p_values = [0, 1, 2, 3]
    d = 1                   # determined from ADF Test
    q_values = [0, 1, 2, 3]

    best_rmse = float('inf')
    best_order = None

    print('\nSearching for Best ARIMA Model...\n')

    for p, q in product(p_values, q_values):
        order = (p, d, q)
        try:
            model = ARIMA(train['Close'], order = order)
            fitted_model = model.fit()

            predictions = fitted_model.forecast(steps=len(validation))
            rmse = np.sqrt(mean_squared_error(validation['Close'], predictions))

            print(f'Order {order} -> RMSE = {rmse:.3f}')

            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order

        except:
            continue

    print('\n---------------------------------------')
    print('Best ARIMA Order :', best_order)
    print(f'Validation RMSE : {best_rmse:.3f}')
    print('---------------------------------------\n')

    return best_order


# =================
# Train Final Model
# =================

def train_arima(train_validation, order):
    '''
    Train the final ARIMA model using
    Train + Validation data.
    '''

    model = ARIMA(train_validation['Close'], order = order)
    fitted_model = model.fit()

    return fitted_model

# ========
# Forecast
# ========

def forecast_arima(model, test):
    '''
    Forecast future observations.
    '''

    predictions = model.forecast(steps = len(test))
    return predictions


# ============
# Main Wrapper
# ============

def run_arima(train, validation, test):
    '''
    Complete ARIMA Pipeline

    1. Hyperparameter tuning
    2. Retrain on Train + Validation
    3. Forecast Test Set
    4. Evaluate
    '''

    print('=' * 60)
    print('ARIMA MODEL')
    print('=' * 60)

    # Hyperparameter Tuning
    best_order = find_best_arima(train, validation)

    # Combine Train + Validation
    train_validation = pd.concat([train, validation])

    # Train Final Model
    model = train_arima(train_validation, best_order)

    # Forecast
    predictions = forecast_arima(model, test)

    # Evaluation
    results = evaluate_model(test['Close'], predictions, 'ARIMA')
    return results
