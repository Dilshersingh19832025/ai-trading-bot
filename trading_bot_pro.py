from ib_insync import IB, Stock
import requests
import time
import pandas as pd

# Function to connect to IBKR
def connect_ibkr():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)  # Correct socket port for paper trading
        if ib.isConnected():
            print("Successfully connected to IBKR")
        else:
            print("Failed to connect to IBKR")
    except Exception as e:
        print(f"Error connecting to IBKR: {e}")
    return ib

# Function to fetch data from IBKR
def get_ibkr_data(ib, symbol):
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='MIDPOINT',
            useRTH=True
        )
        df = pd.DataFrame(bars)
        print(f"Successfully fetched data for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching IBKR data for {symbol}: {e}")
        return None

# Function to fetch data from Binance
def get_binance_data(symbol):
    try:
        url = f'https://api.binance.com/api/v1/klines'
        params = {
            'symbol': symbol,
            'interval': '5m',
            'limit': 10  # Fetch last 10 candles
        }
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code == 200:
            print(f"Successfully fetched Binance data for {symbol}")
            return data
        else:
            print(f"Error fetching Binance data for {symbol}")
            return None
    except Exception as e:
        print(f"Error fetching Binance data for {symbol}: {e}")
        return None

# Main function to run the bot
def run_bot():
    ib = connect_ibkr()

    if ib is None:
        print("IBKR connection failed.")
        return

    # Fetch data for symbols from IBKR and Binance
    symbols_ibkr = ['AAPL', 'TSLA']  # IBKR symbols
    symbols_binance = ['BTCUSDT', 'ETHUSDT']  # Binance symbols

    for symbol in symbols_ibkr:
        df = get_ibkr_data(ib, symbol)
        if df is not None:
            print(df.tail())  # Print the last few rows of the data fetched from IBKR

    for symbol in symbols_binance:
        data = get_binance_data(symbol)
        if data is not None:
            print(data)  # Print the data fetched from Binance

    # Disconnect from IBKR after fetching data
    ib.disconnect()

if __name__ == '__main__':
    run_bot()


