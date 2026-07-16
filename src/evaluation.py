import numpy as np
import matplotlib.pyplot as plt
import config

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

def evaluate_model(actual, predicted, model_name):

    actual    = np.array(actual)
    predicted = np.array(predicted)

    mae  = mean_absolute_error(actual, predicted)
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
        'MAPE': mape,
        'Actual': actual,
        'Predicted': predicted
    }

def plot_model_comparison(arima_result,  xgb_result, lstm_result):

    plt.figure(figsize=(14,6))

    # Actual values
    plt.plot(arima_result['Actual'],    color = 'black',  linewidth = 2, label = 'Actual')

    # Auto ARIMA
    plt.plot(arima_result['Predicted'], color = 'red',    linewidth = 2, label = 'Auto ARIMA')

    # XGBoost
    plt.plot(xgb_result['Predicted'],   color = 'blue',   linewidth = 2, label = 'XGBoost')

    # LSTM
    plt.plot(lstm_result['Predicted'],  color = 'green',  linewidth = 2, label = 'LSTM')

    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Test Observations')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # save figure 
    plt.savefig(config.MODEL_COMPARISON, dpi = 300, bbox_inches = 'tight')
    plt.show()