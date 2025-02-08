from ib_insync import *

# Connect to IBKR TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Adjust the port if necessary

# Define the stock symbol you want to test with (e.g., TSLA)
stock = Stock('TSLA', 'SMART', 'USD')

# Request Market Data
ib.reqMktData(stock)

# Wait for data (10 seconds)
ib.sleep(10)

# Print the market data
print(stock.marketPrice())

# Disconnect
ib.disconnect()
