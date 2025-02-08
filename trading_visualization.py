import pandas as pd
import numpy as np
import talib
import yfinance as yf
import mplfinance as mpf

# Download historical data for AAPL
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

# Check if 'Close' column exists (for multi-level column handling)
if isinstance(data.columns, pd.MultiIndex):
    data = data['Close']  # Extract the 'Close' price if multi-index

# Calculate SMA (Simple Moving Average)
data['SMA_20'] = talib.SMA(data['Close'].values, timeperiod=20)

# Calculate RSI (Relative Strength Index)
data['RSI_14'] = talib.RSI(data['Close'].values, timeperiod=14)

# Generate Buy/Sell Signals
buy_signal = (data['Close'] > data['SMA_20']) & (data['RSI_14'] < 30)
sell_signal = (data['Close'] < data['SMA_20']) & (data['RSI_14'] > 70)

data['Buy_Signal'] = np.where(buy_signal, data['Close'], np.nan)
data['Sell_Signal'] = np.where(sell_signal, data['Close'], np.nan)

# Visualization with mplfinance
apds = [
    mpf.make_addplot(data['SMA_20'], color='blue'),
    mpf.make_addplot(data['Buy_Signal'], type='scatter', markersize=100, marker='^', color='green'),
    mpf.make_addplot(data['Sell_Signal'], type='scatter', markersize=100, marker='v', color='red')
]

mpf.plot(
    data,
    type='candle',
    style='charles',
    title='AAPL Candlestick Chart with Buy/Sell Signals',
    addplot=apds,
    volume=True,
    figratio=(16, 9),
    figscale=1.2
)

