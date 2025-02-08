import pandas as pd
import ta
import matplotlib.pyplot as plt

# Define strategy parameters
rsi_overbought = 70
rsi_oversold = 30
sma_period = 14
macd_signal = 9

# Sample data (replace this with the real data from IBKR)
data = pd.DataFrame({
    'close': [222.32, 222.37, 222.61, 222.63, 222.71, 222.74, 222.84, 222.56, 222.79, 222.87],
})

# Calculate RSI, SMA, and MACD
data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
data['SMA'] = ta.trend.sma_indicator(data['close'], window=sma_period)
data['MACD'] = ta.trend.macd(data['close'], window_slow=26, window_fast=12, fillna=True)
data['MACD_signal'] = ta.trend.macd_signal(data['close'], window_slow=26, window_fast=12, window_sign=macd_signal, fillna=True)

# Generate Buy/Sell Signals
data['Buy_signal'] = (data['RSI'] < rsi_oversold) & (data['close'] > data['SMA']) & (data['MACD'] > data['MACD_signal'])
data['Sell_signal'] = (data['RSI'] > rsi_overbought) & (data['close'] < data['SMA']) & (data['MACD'] < data['MACD_signal'])

# Visualize Buy/Sell signals on the chart
plt.figure(figsize=(12,6))
plt.plot(data['close'], label='Close Price')
plt.plot(data['SMA'], label='SMA', linestyle='--', color='orange')

# Plot buy and sell signals
plt.plot(data.index[data['Buy_signal']], data['close'][data['Buy_signal']], 'g^', markersize=10, label='Buy Signal')
plt.plot(data.index[data['Sell_signal']], data['close'][data['Sell_signal']], 'rv', markersize=10, label='Sell Signal')

plt.legend(loc='best')
plt.show()

# Print signals
print("Buy signals:\n", data[data['Buy_signal']])
print("Sell signals:\n", data[data['Sell_signal']])
