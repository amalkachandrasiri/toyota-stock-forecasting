import pandas as pd

import config
from base_dataset import train_validation_test_split
from models.arima_model import run_arima
from models.xgboost_model import run_xgboost

import os

print(os.getcwd())

toyota_df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col="Date", parse_dates=True)
results = []

print(toyota_df.isna().sum())

# train, validation, test = train_validation_test_split(toyota_df)

results = []

#results.append(run_arima(train, validation, test))
#results.append(run_xgboost(train, validation, test))

#print(results)

