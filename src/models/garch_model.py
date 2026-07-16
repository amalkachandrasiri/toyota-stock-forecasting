'''
GARCH (1,1) Volatility Model

This module performs

1. Daily return calculation
2. Volatility clustering visualization
3. ARCH effect test
4. GARCH(1,1) model estimation
5. Volatility forecasting
6. Parameter interpretation
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

# Calculate Daily Returns
def calculate_returns(df):
    '''
    Computes percentage daily returns.
    '''
    returns = 100 * np.log(df["Close"] / df["Close"].shift(1))
    returns = returns.dropna()

    return returns

# ARCH Effect Test
def arch_test(returns):

    print('\n' + '='*60)
    print('ARCH EFFECT TEST')
    print('='*60)

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns)

    print(f'LM Statistic : {lm_stat:.4f}')
    print(f'P-value      : {lm_pvalue:.6f}')

    if lm_pvalue < 0.05:
        print('\nARCH effects detected.')
        print('GARCH modelling is appropriate.')
    else:
        print('\nNo ARCH effects detected.')
        print('GARCH modelling may not be necessary.')

# Fit GARCH

def build_garch(returns):

    print('\n' + '='*60)
    print('BUILDING GARCH(1,1)')
    print('='*60)

    model = arch_model(returns, mean = 'Constant', vol = 'GARCH', p = 1, q = 1, dist = 'normal')
    fitted_model = model.fit(disp='off')

    print(fitted_model.summary())

    return fitted_model

# Forecast Volatility
def forecast_volatility(model, horizon = 30):

    print('\nForecasting Future Volatility...\n')
 
    forecast = model.forecast(horizon = horizon)
    variance = forecast.variance.iloc[-1]
    volatility = variance ** 0.5

    print(volatility)

    return volatility

# Plot Returns

def plot_returns(returns):

    plt.figure(figsize=(12,5))
    plt.plot(returns)
    plt.title('Daily Stock Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True)

    # save figure 
    plt.savefig(config.DAILY_STK_RETURNS, dpi = 300, bbox_inches = 'tight')

    plt.show()

# Plot Conditional Volatility

def plot_volatility(model):

    plt.figure(figsize=(12,5))
    plt.plot(model.conditional_volatility)
    plt.title('Estimated Conditional Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)

    # save figure 
    plt.savefig(config.EST_CND_VOLATILITY, dpi = 300, bbox_inches = 'tight')

    plt.show()

# Main Wrapper

def run_garch(full_data):

    print('='*60)
    print('GARCH MODEL')
    print('='*60)

    returns = calculate_returns(full_data)    

    plot_returns(returns)

    arch_test(returns)

    model = build_garch(returns)

    print(model.summary())

    plot_volatility(model)

    volatility = forecast_volatility(model)

    return model