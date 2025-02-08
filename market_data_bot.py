import pandas as pd
import matplotlib.pyplot as plt
from ib_insync import *
import ta
import numpy as np

# Function to connect to IBKR
def connect_ibkr():
    print("Connecting to IBKR...")
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)
    return ib

# Function to request market data for a specific symbol
def get_market_data(ib, symbol):
    contract = Stock(symbol, 'SMART', 'USD')
    ib.reqMktData(contract, '', False, False)
    ib.sleep(2)  # Give it a moment to fetch data
    data = ib.reqHistoricalData(contract, endDateTime='', durationStr='1 D',
                                barSizeSetting='5 mins', whatToShow='MIDPOINT',
                                useRTH=True, formatDate=1)
    df = util.df(data)
    return df

# Function to calculate and plot indicators
def plot_indicators(df):
    # Add moving average (SMA)
    df['SMA'] = ta.trend.sma_indicator(df['close'], window=15)
    
    # Add RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    
    # Plot data
    plt.figure(figsize=(12, 6))
    
    # Plot closing price
    plt.plot(df['date'], df['close'], label='Close Price', color='blue')

    # Plot SMA
    plt.plot(df['date'], df['SMA'], label='SMA (15)', color='yellow', linestyle='--')
    
    # Highlight buy/sell signals
    buy_signal = df[df['RSI'] < 30]
    sell_signal = df[df['RSI'] > 70]
    
    # Plot buy signals
    plt.scatter(buy_signal['date'], buy_signal['close'], marker='^', color='green', label='Buy Signal', alpha=1)
    
    # Plot sell signals
    plt.scatter(sell_signal['date'], sell_signal['close'], marker='v', color='red', label='Sell Signal', alpha=1)
    
    plt.title('Market Data with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# Main function
def main():
    ib = connect_ibkr()
    print("Requesting market data...")
    df = get_market_data(ib, 'AAPL')
    
    # Print raw data for confirmation
    print("\n--- Raw Market Data ---")
    print(df[['date', 'open', 'high', 'low', 'close', 'volume']])
    
    # Plot the indicators and signals
    plot_indicators(df)
    
    print("\n[✓] Data looks valid. No issues detected.")
    ib.disconnect()

if __name__ == "__main__":
    main()



