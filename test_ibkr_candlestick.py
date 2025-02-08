import time
from ib_insync import *
import talib
import numpy as np
import pandas as pd

# Connect to IBKR Paper Trading
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Ensure TWS Paper Trading is running

def fetch_market_data(symbol):
    """Fetches delayed market data for the given symbol."""
    contract = Stock(symbol, "SMART", "USD")
    ib.reqMarketDataType(3)  # Request delayed data
    ib.qualifyContracts(contract)
    
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 D',  # Changed from '5 D' to '1 D'
        barSizeSetting='5 mins',  # Changed from '5 min' to '5 mins' (correct IB format)
        whatToShow='TRADES',
        useRTH=True
    )

    if not bars:
        print(f"Failed to fetch data for {symbol}.")
        return None

    df = pd.DataFrame(bars)
    df.set_index("date", inplace=True)
    return df

def identify_candlestick_patterns(df):
    """Identifies candlestick patterns in the last 5 candlesticks."""
    open_prices = np.array(df['open'])
    high_prices = np.array(df['high'])
    low_prices = np.array(df['low'])
    close_prices = np.array(df['close'])

    patterns = {
        "Doji": talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices),
        "Engulfing": talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
        "Hammer": talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
        "Shooting Star": talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices),
        "Morning Star": talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices),
        "Evening Star": talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices),
        "Hanging Man": talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices),
        "Piercing Line": talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)
    }

    detected_patterns = []
    for i in range(-5, 0):  # Check the last 5 candlesticks
        for pattern, values in patterns.items():
            if values[i] != 0:
                detected_patterns.append(f"{pattern} at index {len(df) + i}")

    return detected_patterns

def main():
    print("Connected to IBKR")
    symbol = input("Enter the stock ticker symbol (e.g., 'TSLA'): ").upper()
    
    print(f"Fetching delayed market data for {symbol}...")
    df = fetch_market_data(symbol)

    if df is not None:
        print(f"Latest Market Price for {symbol}: {df['close'].iloc[-1]} USD")

        # Print last 5 candlesticks
        print("\nRecent Candlesticks:")
        print(df.tail(5)[["open", "high", "low", "close"]])

        # Identify candlestick patterns
        patterns_found = identify_candlestick_patterns(df)

        if patterns_found:
            print("\nDetected Candlestick Patterns:")
            for pattern in patterns_found:
                print(f"- {pattern}")
        else:
            print("No significant candlestick patterns detected in the last 5 candlesticks.")

    ib.disconnect()

if __name__ == "__main__":
    main()


