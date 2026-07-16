import pandas as pd

import config
from base_dataset import train_validation_test_split
from models.arima_model import run_arima
from models.xgboost_model import run_xgboost
from models.lstm_model import run_lstm
from models.garch_model import run_garch

xgboost_df = pd.read_csv(config.XGBOOST_DATA_PATH, index_col = 'Date', parse_dates = True)
arima_df   = pd.read_csv(config.ARIMA_DATA_PATH,   index_col = 'Date', parse_dates = True)
lstm_df    = pd.read_csv(config.LSTM_DATA_PATH,    index_col = 'Date', parse_dates = True)
garch_df   = arima_df # use the same dataset 

results = []

# ARIMA
train, validation, test = train_validation_test_split(arima_df) 
results.append(run_arima(train, validation, test))

# XGBOOST
train, validation, test = train_validation_test_split(xgboost_df) 
results.append(run_xgboost(train, validation, test))

# LSTM
train, validation, test = train_validation_test_split(lstm_df)
results.append(run_lstm(train, validation, test))

print(results)

# GARCH
run_garch(garch_df)
