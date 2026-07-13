import yfinance as yf
import config
from statsmodels.tsa.stattools import adfuller

def fetch_data():
    stock = yf.Ticker('7203.T')
    raw_data = stock.history(period='3y')
    raw_data.to_csv(config.RAW_DATA_PATH)

    print('Toyota raw data saved successfully.')

    return raw_data

def describe_data(data):
    print('Shape:', data.shape)
    print('\nColumns')
    print(data.columns)

    print('\nData Types')
    print(data.dtypes)

    print('\nMissing Values')
    print(data.isnull().sum())

    print('\nSummary Statistics')
    print(data.describe())

    print('\nFirst 5 Rows')
    print(data.head())

    print('\nLast 5 Rows')
    print(data.tail())

raw_data = fetch_data()
# describe_data(raw_data)

# delete dividents and stock split 
toyota_df = raw_data.drop(columns=['Dividends','Stock Splits'])

# outlier detection 
Q1 = toyota_df['Close'].quantile(0.25)
Q3 = toyota_df['Close'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = toyota_df[(toyota_df['Close'] < lower) | (toyota_df['Close'] > upper)]

print("Number of outliers:", len(outliers))

# ===================
# Feature Engineering
# ===================

# lag features 
toyota_df['Lag_1'] = toyota_df['Close'].shift(1)
toyota_df['Lag_7'] = toyota_df['Close'].shift(7)
toyota_df['Lag_30'] = toyota_df['Close'].shift(30)

# Rolling Statistics
toyota_df['Rolling_Mean_7'] = toyota_df['Close'].rolling(7).mean()
toyota_df['Rolling_STD_7'] = toyota_df['Close'].rolling(7).std()
toyota_df['Rolling_Mean_30'] = toyota_df['Close'].rolling(30).mean()

# Sesonal Indicators 
toyota_df['Day_of_Week'] = toyota_df.index.dayofweek
toyota_df['Month'] = toyota_df.index.month
toyota_df['Quarter'] = toyota_df.index.quarter

# ==========================
# Stationary Test - ADF Test
# ==========================

# Augmented Dickey-Fuller Test
result = adfuller(toyota_df['Close'])

print("ADF Statistic :", result[0])
print("p-value       :", result[1])
print("Critical Values:")

for key, value in result[4].items():
    print(f"   {key}: {value}")

# first differncing 
toyota_df['Close_Diff'] = toyota_df['Close'].diff()

# rerun ADF test 
result = adfuller(toyota_df['Close_Diff'].dropna())

print("ADF Statistic :", result[0])
print("p-value       :", result[1])

toyota_df.to_csv(config.PROCESSED_DATA_PATH)