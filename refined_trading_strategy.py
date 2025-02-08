import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
import yfinance as yf

# Fetch historical data
symbol = 'AAPL'  # Example stock symbol
start_date = '2023-01-01'
end_date = '2024-01-01'

df = yf.download(symbol, start=start_date, end=end_date)

# Ensure close data is a numpy array with the correct shape (1D)
close = df['Close'].to_numpy().flatten()  # Flatten to 1D array
high = df['High'].to_numpy().flatten()
low = df['Low'].to_numpy().flatten()
volume = df['Volume'].to_numpy().flatten()

# Check the shape of the 'close' array
print(f"Shape of 'close' array: {close.shape}")

# Calculate indicators
try:
    df['SMA_50'] = ta.SMA(close, timeperiod=50)
    print("SMA_50 calculation succeeded.")
except Exception as e:
    print(f"Error with SMA_50 calculation: {e}")

df['SMA_200'] = ta.SMA(close, timeperiod=200)
df['RSI'] = ta.RSI(close, timeperiod=14)

# Add MACD
df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

# Add Bollinger Bands
df['upper_band'], df['middle_band'], df['lower_band'] = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# Add Stochastic Oscillator
df['slowk'], df['slowd'] = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)

# Add OBV (On-Balance Volume)
df['OBV'] = ta.OBV(close, volume)

# Add CMF (Chaikin Money Flow)
df['CMF'] = ta.CMF(high, low, close, volume, timeperiod=20)

# Example Backtest Function (Simple Strategy)
def backtest(df, sma_short=50, sma_long=200, rsi_threshold=30, macd_threshold=0):
    # Entry and exit signals
    df['signal'] = 0  # No signal
    df['signal'][(df['RSI'] < rsi_threshold) & (df['MACD'] > macd_threshold)] = 1  # Buy signal
    df['signal'][(df['RSI'] > 100 - rsi_threshold) & (df['MACD'] < -macd_threshold)] = -1  # Sell signal

    # Simulate trades based on signals
    df['position'] = df['signal'].shift(1)  # Position at the next period based on signal
    df['daily_return'] = df['Close'].pct_change() * df['position']

    # Backtest performance
    df['strategy_return'] = df['daily_return'].cumsum()
    df['market_return'] = df['daily_return'].cumsum()

    return df

# Run backtest and plot
df_5min = df.resample('5T').last()  # Resample for 5-minute intervals
df_1hour = df.resample('1H').last()  # Resample for 1-hour intervals
df_1day = df  # Original data, daily intervals

# Backtest for different timeframes
df_5min_backtest = backtest(df_5min)
df_1hour_backtest = backtest(df_1hour)
df_1day_backtest = backtest(df_1day)

# Plot results
plt.figure(figsize=(12, 8))

# Plot 5-minute backtest results
plt.subplot(3, 1, 1)
plt.plot(df_5min_backtest['strategy_return'], label='5min Strategy Return', color='green')
plt.plot(df_5min_backtest['market_return'], label='5min Market Return', color='blue')
plt.title('5-minute Timeframe Backtest')
plt.legend()

# Plot 1-hour backtest results
plt.subplot(3, 1, 2)
plt.plot(df_1hour_backtest['strategy_return'], label='1-hour Strategy Return', color='green')
plt.plot(df_1hour_backtest['market_return'], label='1-hour Market Return', color='blue')
plt.title('1-hour Timeframe Backtest')
plt.legend()

# Plot daily backtest results
plt.subplot(3, 1, 3)
plt.plot(df_1day_backtest['strategy_return'], label='Daily Strategy Return', color='green')
plt.plot(df_1day_backtest['market_return'], label='Daily Market Return', color='blue')
plt.title('Daily Timeframe Backtest')
plt.legend()

plt.tight_layout()
plt.show()



