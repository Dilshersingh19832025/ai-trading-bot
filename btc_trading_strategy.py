import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from binance.client import Client
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize Binance Client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Fetch historical data for BTC/USDT
def fetch_binance_data(symbol, interval, limit=500):
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# Calculate SMA, EMA, and RSI
def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    return df

# Fetch news headlines
def fetch_news():
    url = f'https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    articles = response.json().get('articles', [])[:5]
    return [(article['title'], article['source']['name'], article['url']) for article in articles]

# Plot candlestick chart with indicators
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Candlesticks'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='SMA 20'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA_10'], line=dict(color='orange', width=1), name='EMA 10'
    ))

    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df.index, y=df['RSI_14'], line=dict(color='purple', width=1), name='RSI 14'
    ))
    fig_rsi.add_shape(
        type='line', x0=df.index.min(), x1=df.index.max(), y0=70, y1=70, 
        line=dict(color='red', dash='dash')
    )
    fig_rsi.add_shape(
        type='line', x0=df.index.min(), x1=df.index.max(), y0=30, y1=30, 
        line=dict(color='green', dash='dash')
    )

    # Save as HTML
    plot(fig, filename='btc_trading_strategy.html', auto_open=True)
    plot(fig_rsi, filename='btc_rsi_analysis.html', auto_open=True)

# Main function
def main():
    df = fetch_binance_data('BTCUSDT', '1h')
    df = calculate_indicators(df)
    news = fetch_news()

    print("Latest Business News:")
    for title, source, url in news:
        print(f"- {title} | {source}: {url}")

    plot_chart(df)

if __name__ == "__main__":
    main()



