import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
import time
import plotly.io as pio

# =================== ENTER YOUR API KEYS HERE ===================
BINANCE_API_KEY = '11qTxu0Caz29CclLX00YQFahEj17s2NqRXjMZMMQ9DNIRT2iF7rPUg0oRNxywdxr'
BINANCE_SECRET_KEY = 'dP1MqhrkviZi9f4b99TSJRmeRaPt9Ya7s6z9P0mHMhAjQ9q1QgsZa2obJbEDzWcd'
# ================================================================

# Initialize Binance client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Parameters
symbol = 'BTCUSDT'
timeframe = '1h'
sma_period = 20
ema_period = 10
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 30

# Set Plotly to render offline
pio.renderers.default = "browser"  # Render in the browser

# Fetch historical data
def fetch_data(symbol, timeframe, limit=100):
    klines = client.get_historical_klines(symbol, timeframe, f"{limit} hours ago UTC")
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    data['close'] = data['close'].astype(float)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data

# Calculate SMA
def calculate_sma(data, period):
    return data['close'].rolling(window=period).mean()

# Calculate EMA
def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

# Calculate RSI
def calculate_rsi(data, period):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Generate trading signals
def generate_signals(data):
    data['SMA_20'] = calculate_sma(data, sma_period)
    data['EMA_10'] = calculate_ema(data, ema_period)
    data['RSI_14'] = calculate_rsi(data, rsi_period)

    data['Buy_Signal'] = np.where(
        (data['close'] > data['SMA_20']) & 
        (data['RSI_14'] < rsi_oversold) & 
        (data['close'] > data['EMA_10']), 1, 0
    )
    data['Sell_Signal'] = np.where(
        (data['close'] < data['SMA_20']) & 
        (data['RSI_14'] > rsi_overbought) & 
        (data['close'] < data['EMA_10']), 1, 0
    )

    return data

# Visualize data
def visualize_data(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data['timestamp'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Candlesticks'
    ), row=1, col=1)

    # SMA and EMA
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['SMA_20'], line=dict(color='blue'), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_10'], line=dict(color='orange'), name='EMA 10'), row=1, col=1)

    # Buy/Sell signals
    buy_signals = data[data['Buy_Signal'] == 1]
    sell_signals = data[data['Sell_Signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals['close'], mode='markers', marker=dict(color='green', size=10), name='Buy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals['close'], mode='markers', marker=dict(color='red', size=10), name='Sell'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['RSI_14'], line=dict(color='purple'), name='RSI 14'), row=2, col=1)
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)

    # Layout
    fig.update_layout(title=f'{symbol} Trading Strategy', xaxis_title='Date', yaxis_title='Price')
    fig.update_xaxes(rangeslider_visible=False)

    # Save the chart as an HTML file and open it
    pio.write_html(fig, file='chart.html', auto_open=True)

# Execute trades
def execute_trade(signal):
    if signal == 'BUY':
        order = client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=0.001  # Adjust quantity as needed
        )
        print("Buy Order Executed:", order)
    elif signal == 'SELL':
        order = client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=0.001  # Adjust quantity as needed
        )
        print("Sell Order Executed:", order)

# Main loop
def main():
    while True:
        data = fetch_data(symbol, timeframe)
        data = generate_signals(data)
        visualize_data(data)

        # Check for latest signal
        latest_signal = 'BUY' if data.iloc[-1]['Buy_Signal'] == 1 else 'SELL' if data.iloc[-1]['Sell_Signal'] == 1 else None
        if latest_signal:
            execute_trade(latest_signal)

        # Wait before next iteration
        time.sleep(3600)  # 1 hour delay

if __name__ == "__main__":
    main()