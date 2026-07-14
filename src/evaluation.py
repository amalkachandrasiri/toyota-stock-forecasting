import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

def evaluate_model(actual, predicted, model_name):

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print('\n' + '=' * 50)
    print(f'Model : {model_name}')
    print('=' * 50)
    print(f'MAE  : {mae:.4f}')
    print(f'RMSE : {rmse:.4f}')
    print(f'MAPE : {mape:.2f}%')
    print('=' * 50)

    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }