from ib_insync import *
import pandas as pd
import numpy as np

# Connect to IBKR API (Paper Trading)
def connect_ibkr():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)  # IBKR paper trading connection
    return ib

# Request market data
def get_market_data(ib, symbol):
    contract = Stock(symbol, 'SMART', 'USD')
    ib.reqMktData(contract, '', False, False)
    ib.sleep(2)  # Allow time for data to arrive
    market_data = ib.reqHistoricalData(
        contract, 
        endDateTime='', 
        durationStr='1 D', 
        barSizeSetting='5 mins', 
        whatToShow='MIDPOINT', 
        useRTH=True
    )
    df = util.df(market_data)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Add RSI and SMA
def add_indicators(df):
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['SMA'] = df['close'].rolling(window=20).mean()
    return df

# Define Strategy Logic (with Risk Management)
def apply_strategy(df):
    buy_signals = []
    sell_signals = []
    
    # Simple Strategy (Buy when RSI < 30 and Close > SMA, Sell when RSI > 70)
    for i in range(1, len(df)):
        if df['RSI'][i] < 30 and df['close'][i] > df['SMA'][i]:
            buy_signals.append(df['close'][i])
            sell_signals.append(np.nan)
        elif df['RSI'][i] > 70 and df['close'][i] < df['SMA'][i]:
            buy_signals.append(np.nan)
            sell_signals.append(df['close'][i])
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    df['buy'] = buy_signals
    df['sell'] = sell_signals
    return df

# Place Orders in Paper Trading
def place_order(ib, symbol, signal, quantity=1):
    contract = Stock(symbol, 'SMART', 'USD')
    if signal == 'buy':
        order = MarketOrder('BUY', quantity)
        ib.placeOrder(contract, order)
    elif signal == 'sell':
        order = MarketOrder('SELL', quantity)
        ib.placeOrder(contract, order)

# Main Function to Run the Strategy
def run_strategy(symbol):
    ib = connect_ibkr()
    df = get_market_data(ib, symbol)
    df = add_indicators(df)
    df = apply_strategy(df)

    for i in range(len(df)):
        if pd.notna(df['buy'][i]):
            print(f"Buy signal at {df['date'][i]}: {df['buy'][i]}")
            place_order(ib, symbol, 'buy')
        elif pd.notna(df['sell'][i]):
            print(f"Sell signal at {df['date'][i]}: {df['sell'][i]}")
            place_order(ib, symbol, 'sell')

    ib.disconnect()

# Run the strategy for a specific symbol
run_strategy('AAPL')


