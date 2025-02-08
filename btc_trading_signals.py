import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load BTCUSDT data
try:
    data = pd.read_csv('C:/Users/gill_/OneDrive/Documents/tradingbot/BTCUSDT_data.csv')
except FileNotFoundError:
    print("Error: The file 'BTCUSDT_data.csv' was not found. Please ensure the file exists in the correct location.")
    exit()

# Verify the data
print("First few rows of the data:")
print(data.head())

print("\nLast few rows of the data:")
print(data.tail())

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Calculate SMA (20) and EMA (10)
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

# Calculate RSI (14)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI_14'] = 100 - (100 / (1 + rs))

# Generate Buy/Sell Signals
data['Buy_Signal'] = np.where((data['EMA_10'] > data['SMA_20']) & (data['EMA_10'].shift(1) <= data['SMA_20'].shift(1)) & (data['RSI_14'] > 30), data['Close'], np.nan)
data['Sell_Signal'] = np.where((data['EMA_10'] < data['SMA_20']) & (data['EMA_10'].shift(1) >= data['SMA_20'].shift(1)) & (data['RSI_14'] < 70), data['Close'], np.nan)

# Candlestick Chart
candlestick = go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlesticks')

# Moving Averages
sma_line = go.Scatter(x=data['Date'], y=data['SMA_20'], line=dict(color='orange', width=2), name='SMA 20')
ema_line = go.Scatter(x=data['Date'], y=data['EMA_10'], line=dict(color='blue', width=2), name='EMA 10')

# Buy/Sell Markers
buy_signal = go.Scatter(x=data['Date'], y=data['Buy_Signal'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal')
sell_signal = go.Scatter(x=data['Date'], y=data['Sell_Signal'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal')

# RSI Plot
rsi = go.Scatter(x=data['Date'], y=data['RSI_14'], line=dict(color='purple', width=2), name='RSI 14')

# Overbought and Oversold Levels
rsi_overbought = go.Scatter(x=data['Date'], y=[70]*len(data), line=dict(color='red', dash='dash'), name='Overbought (70)')
rsi_oversold = go.Scatter(x=data['Date'], y=[30]*len(data), line=dict(color='green', dash='dash'), name='Oversold (30)')

# Create subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

# Candlestick and Moving Averages
fig.add_trace(candlestick, row=1, col=1)
fig.add_trace(sma_line, row=1, col=1)
fig.add_trace(ema_line, row=1, col=1)
fig.add_trace(buy_signal, row=1, col=1)
fig.add_trace(sell_signal, row=1, col=1)

# RSI and Levels
fig.add_trace(rsi, row=2, col=1)
fig.add_trace(rsi_overbought, row=2, col=1)
fig.add_trace(rsi_oversold, row=2, col=1)

# Update Layout
fig.update_layout(
    title='BTCUSDT Trading Strategy with Buy/Sell Signals',
    xaxis_title='Date',
    yaxis_title='Price',
    legend_title='Indicators',
    template='plotly_white',
    height=700
)

# Save the chart as HTML
try:
    fig.write_html('C:/Users/gill_/OneDrive/Documents/tradingbot/chart_with_signals.html')
    print("Chart saved successfully as 'chart_with_signals.html'.")
except Exception as e:
    print(f"Error saving chart: {e}")