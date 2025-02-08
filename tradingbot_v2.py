import yfinance as yf
import talib
import pandas as pd

def fetch_data(ticker, interval):
    df = yf.download(ticker, period='1d', interval=interval)
    return df

def identify_candlestick_patterns(df):
    if df.empty:
        print("No data to analyze.")
        return
    
    open_prices = df['Open']
    high_prices = df['High']
    low_prices = df['Low']
    close_prices = df['Close']
    
    patterns = {
        "Doji": talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices),
        "Engulfing": talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
        "Hammer": talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
        "Shooting Star": talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
    }
    
    for pattern, result in patterns.items():
        if result[-1] != 0:  # Check if the last value is non-zero (indicating a pattern)
            print(f"{pattern} detected at {df.index[-1]}")

def main():
    ticker = input("Enter the stock ticker symbol (e.g., 'AAPL'): ")
    interval = input("Enter the time interval (e.g., '1m', '5m', '15m'): ")
    
    df = fetch_data(ticker, interval)
    if df.empty:
        print(f"Failed to fetch data for {ticker} with interval {interval}.")
        return
    
    print("Fetched Data:")
    print(df.tail())
    
    identify_candlestick_patterns(df)

if __name__ == "__main__":
    main()


