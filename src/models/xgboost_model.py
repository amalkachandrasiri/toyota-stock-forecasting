'''
XGBoost Forecasting Model

This module performs:
1. Hyperparameter tuning
2. Model training
3. Forecasting
4. Performance evaluation
'''

import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from evaluation import evaluate_model

# Feature Preparation

def prepare_features_and_target(df):
    X = df.drop(columns=['Target'])
    y = df['Target']

    return X, y

# Hyperparameter Search

def find_best_model(train_df, validation_df):
    X_train, y_train = prepare_features_and_target(train_df)
    X_validation, y_validation = prepare_features_and_target(validation_df)

    parameter_grid = {
        'n_estimators' : [100, 200],
        'max_depth'    : [3, 5],
        'learning_rate': [0.01, 0.1]
    }

    best_model = None
    best_parameters = None
    best_rmse = float('inf')

    print('\nSearching Best XGBoost Parameters...\n')

    for parameters in ParameterGrid(parameter_grid):
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **parameters
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)

        rmse = np.sqrt(mean_squared_error(y_validation, predictions))

        print(f'{parameters} --> RMSE = {rmse:.3f}')

        if rmse < best_rmse:
            best_rmse = rmse
            best_parameters = parameters
            best_model = model

    print('\n----------------------------------------')
    print('Best Parameters')
    print(best_parameters)
    print(f'Validation RMSE : {best_rmse:.3f}')
    print('----------------------------------------\n')

    return best_parameters

# Train Final Model

def train_model(train_validation_df, parameters):
    X_train, y_train = prepare_features_and_target(train_validation_df)

    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **parameters
    )

    model.fit(X_train, y_train)

    return model

# Forecast

def forecast(model, test_df):
    X_test, _ = prepare_features_and_target(test_df)
    predictions = model.predict(X_test)
    return predictions

# Main Wrapper

def run_xgboost(train_df, validation_df, test_df):
    print('=' * 60)
    print('XGBOOST MODEL')
    print('=' * 60)

    best_parameters = find_best_model(train_df, validation_df)

    train_validation_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True
    )

    model = train_model(train_validation_df, best_parameters)
    predictions = forecast(model, test_df)

    results = evaluate_model(actual = test_df['Target'], predicted = predictions, model_name = 'XGBoost')

    return results