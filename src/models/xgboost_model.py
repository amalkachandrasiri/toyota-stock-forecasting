'''
XGBoost Model Implementation

This module performs:

1. Hyperparameter tuning
2. Model training
3. Forecasting
4. Model evaluation
'''

import pandas as pd
import numpy as np

from itertools import product

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from evaluation import evaluate_model

# ================
# Prepare Features
# ================

def prepare_features(df):
    '''
    Separate features and target variable.
    '''

    X = df.drop(columns=['Close'])
    y = df['Close']

    return X, y

# =====================
# Hyperparameter Tuning
# =====================

def find_best_xgboost(train, validation):

    X_train, y_train = prepare_features(train)
    X_valid, y_valid = prepare_features(validation)

    param_grid = {
        'n_estimators' : [100, 200],
        'max_depth'    : [3, 5],
        'learning_rate': [0.01, 0.1]
    }

    best_model = None
    best_params = None
    best_rmse = float('inf')

    print('\nSearching Best XGBoost Parameters...\n')

    for params in ParameterGrid(param_grid):
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **params
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        rmse = np.sqrt(mean_squared_error(y_valid, predictions))

        print(f'{params} --> RMSE = {rmse:.3f}')

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model

    print('\n----------------------------------------')
    print('Best Parameters')
    print(best_params)
    print(f'Validation RMSE : {best_rmse:.3f}')
    print('----------------------------------------\n')

    return best_params

# =================
# Train Final Model
# =================

def train_xgboost(train_validation, params):

    X_train, y_train = prepare_features(train_validation)

    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **params
    )
    model.fit(X_train, y_train)
    return model

# ========
# Forecast
# ========

def forecast_xgboost(model, test):
    X_test, _ = prepare_features(test)
    predictions = model.predict(X_test)
    return predictions


# =======
# Wrapper
# =======

def run_xgboost(train, validation, test):

    print('=' * 60)
    print('XGBOOST MODEL')
    print('=' * 60)

    best_params = find_best_xgboost(train, validation)
    train_validation = pd.concat([train, validation])

    model = train_xgboost(train_validation, best_params)
    predictions = forecast_xgboost(model, test)

    results = evaluate_model(test['Close'], predictions, 'XGBoost')

    return results