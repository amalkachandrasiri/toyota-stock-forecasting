import yfinance as yf
import config
from statsmodels.tsa.stattools import adfuller

def fetch_raw_data():
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

    # outlier detection 
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data[(data['Close'] < lower) | (data['Close'] > upper)]

    print('Number of outliers:', len(outliers))

def data_cleaning(df):
    # delete dividents and stock split 
    df = df.drop(columns=['Dividends','Stock Splits'])

    return df

def create_arima_dataset(df): 
    df = df.drop(columns=['Open','High', 'Low', 'Volume'])

    # Remove rows containing NaN values
    df = df.dropna()
    return df

def create_lstm_dataset(df): 
    df = df.drop(columns=['Open','High', 'Low', 'Volume'])

    # Remove rows containing NaN values
    df = df.dropna()
    
    return df

def create_xgboost_dataset(df):
    # lag features 
    df['Lag_1']  = df['Close'].shift(1)
    df['Lag_7']  = df['Close'].shift(7)
    df['Lag_30'] = df['Close'].shift(30)

    # Rolling Statistics
    df['Rolling_Mean_7']  = df['Close'].rolling(7).mean()
    df['Rolling_STD_7']   = df['Close'].rolling(7).std()
    df['Rolling_Mean_30'] = df['Close'].rolling(30).mean()

    # Sesonal Indicators 
    df['Day_of_Week'] = df.index.dayofweek
    df['Month']       = df.index.month
    df['Quarter']     = df.index.quarter

    # Target (Next Day Close)
    df['Target'] = df['Close'].shift(-1)   

    # Remove rows containing NaN values
    df = df.dropna()

    return df

raw_data = fetch_raw_data()
#describe_data(raw_data)
cleaned_data = data_cleaning(raw_data)

# create arima dataset - contains only date and close
arima_df = create_arima_dataset(cleaned_data)
arima_df.to_csv(config.ARIMA_DATA_PATH) 

# create lstm dataset - contains only date and close
lstm_df = create_lstm_dataset(cleaned_data)
lstm_df.to_csv(config.LSTM_DATA_PATH)

# create xgboost dataset - 
xgb_df = create_xgboost_dataset(cleaned_data)
xgb_df.to_csv(config.XGBOOST_DATA_PATH)

# testing 
'''
print(xgb_df.head())
print(xgb_df.tail())
print(xgb_df.shape)
print(xgb_df.isna().sum())
'''

'''
# ==========================
# Stationary Test - ADF Test
# ==========================

# Augmented Dickey-Fuller Test
result = adfuller(toyota_df['Close'])

print('ADF Statistic :', result[0])
print('p-value       :', result[1])
print('Critical Values:')

for key, value in result[4].items():
    print(f'   {key}: {value}')

# first differncing 
toyota_df['Close_Diff'] = toyota_df['Close'].diff()

# rerun ADF test 
result = adfuller(toyota_df['Close_Diff'].dropna())

print('ADF Statistic :', result[0])
print('p-value       :', result[1])
'''
