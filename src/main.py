import pandas as pd

import config
from base_dataset import train_validation_test_split
from models.arima_model import run_arima
from models.xgboost_model import run_xgboost
from models.lstm_model import run_lstm
from models.garch_model import run_garch
from evaluation import plot_model_comparison

xgboost_df = pd.read_csv(config.XGBOOST_DATA_PATH, index_col = 'Date', parse_dates = True)
arima_df   = pd.read_csv(config.ARIMA_DATA_PATH,   index_col = 'Date', parse_dates = True)
lstm_df    = pd.read_csv(config.LSTM_DATA_PATH,    index_col = 'Date', parse_dates = True)
garch_df   = arima_df # use the same dataset 

results = []

# ARIMA
train, validation, test = train_validation_test_split(arima_df) 
arima_result = run_arima(train, validation, test)
results.append(arima_result)

# XGBOOST
train, validation, test = train_validation_test_split(xgboost_df) 
xgb_result = run_xgboost(train, validation, test)
results.append(xgb_result)

# LSTM
train, validation, test = train_validation_test_split(lstm_df)
lstm_result = run_lstm(train, validation, test)
results.append(lstm_result)

plot_model_comparison(arima_result, xgb_result, lstm_result)

# print(results)

# GARCH
# run_garch(garch_df)
