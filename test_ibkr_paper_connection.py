from ib_insync import *
import time

# Connect to IBKR Paper Trading
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # 7497 is the port for Paper Trading

print("Connected to IBKR")

# Get user input for stock ticker
ticker = input("Enter the stock ticker symbol (e.g., 'TSLA'): ").strip().upper()

# Define contract
contract = Stock(ticker, 'SMART', 'USD')

# Request delayed market data
print(f"Fetching delayed market data for {ticker}...")
ib.reqMarketDataType(3)  # 3 = Delayed Market Data

# Request market data snapshot
market_data = ib.reqMktData(contract)

# Wait for data to be received
time.sleep(2)  

# Check if data is available
if market_data.last:
    print(f"Market price for {ticker}: {market_data.last} USD")
else:
    print(f"Market price for {ticker}: Data not available (Possible subscription issue)")

# Disconnect from IBKR
ib.disconnect()

