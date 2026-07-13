import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import config

# Load data
toyota_raw_df = pd.read_csv(config.RAW_DATA_PATH, index_col='Date', parse_dates=True)
print(toyota_raw_df.head())

# check index 
print(toyota_raw_df.index)

# ===================
# Close Pricing Trend 
# ===================

plt.figure(figsize=(14,6))

plt.plot(toyota_raw_df.index, toyota_raw_df['Close'])

plt.title('Toyota Daily Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price (JPY)')

plt.grid(True)

# save figure 
plt.savefig(config.CLOSE_PRICING_TRENDS, dpi = 300, bbox_inches = 'tight')

plt.show()

# ==============
# trading Volume 
# ==============

plt.figure(figsize=(14,6))

plt.plot(toyota_raw_df.index, toyota_raw_df['Volume'])

plt.title('Toyota Daily Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')

plt.grid(True)

# save figure 
plt.savefig(config.TRADING_VOLUME, dpi = 300, bbox_inches = 'tight')

plt.show()

# =============
# Daily Returns 
# =============
toyota_raw_df['Daily_Return'] = toyota_raw_df['Close'].pct_change()

plt.figure(figsize=(14,6))

plt.plot(toyota_raw_df.index, toyota_raw_df['Daily_Return'])
plt.title('Toyota Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return')

plt.grid(True)

# save figure 
plt.savefig(config.DAILY_RETURNS, dpi = 300, bbox_inches = 'tight')

plt.show()

# =========================
# Distributed Daily Returns
# =========================

plt.figure(figsize=(8,6))

plt.hist(toyota_raw_df['Daily_Return'].dropna(), bins=40)

plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')

# save figure 
plt.savefig(config.DIST_DAILY_RETURNS, dpi = 300, bbox_inches = 'tight')

plt.show()

# =================
# Outlier Detection 
# =================
plt.figure(figsize=(6,6))

plt.boxplot(toyota_raw_df['Close'])
plt.title('Boxplot of Closing Price')

# save figure 
plt.savefig(config.OUTLIERS, dpi = 300, bbox_inches = 'tight')

plt.show()

# =====================
# 30 Day Moving Average 
# =====================

toyota_raw_df['MA30'] = toyota_raw_df['Close'].rolling(30).mean()

plt.figure(figsize=(14,6))

plt.plot(toyota_raw_df.index, toyota_raw_df['Close'], label='Close Price')
plt.plot(toyota_raw_df.index, toyota_raw_df['MA30'], label='30-Day Moving Average')
plt.title('Closing Price with 30-Day Moving Average')

plt.legend()
plt.grid(True)

# save figure 
plt.savefig(config.MOVING_AVG_30_DAYS, dpi = 300, bbox_inches = 'tight')

plt.show()

# =========================
# Time Series Decomposition 
# =========================

decomposition = seasonal_decompose(toyota_raw_df['Close'], model='additive', period=30)

fig = decomposition.plot()
fig.set_size_inches(14,10)

# save figure 
plt.savefig(config.TS_DECOMPOSITION, dpi = 300, bbox_inches = 'tight')

plt.show()