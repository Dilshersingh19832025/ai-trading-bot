import os
from binance.client import Client
import requests
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
binance_api_key = os.getenv("BINANCE_API_KEY")
binance_api_secret = os.getenv("BINANCE_API_SECRET")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize the Binance Client
binance_client = Client(binance_api_key, binance_api_secret)

# Function to get stock data from Alpha Vantage
def get_alpha_vantage_data(symbol, interval='1min', outputsize='compact'):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": alpha_vantage_api_key,
        "outputsize": outputsize
    }
    response = requests.get(url, params=params)
    data = response.json()
    time_series = data.get(f"Time Series ({interval})", {})
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.astype(float)
    return df

# Function to get crypto data from Binance
def get_binance_data(symbol, interval='1m', limit=100):
    klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = []
    for kline in klines:
        data.append({
            "open_time": kline[0],
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
            "close_time": kline[6],
        })
    df = pd.DataFrame(data)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df

# Fetching data
stock_data = get_alpha_vantage_data('AAPL', '5min')  # Example stock: AAPL (Apple)
crypto_data = get_binance_data('BTCUSDT', '1m', 200)  # Example crypto: BTC/USDT

# Displaying the data
print("Stock Data (Alpha Vantage):")
print(stock_data.tail())

print("\nCrypto Data (Binance):")
print(crypto_data.tail())














































