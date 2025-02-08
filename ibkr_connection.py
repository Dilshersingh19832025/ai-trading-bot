from ib_insync import *

# Connect to IBKR paper trading account
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # 7497 is the default for paper trading

# Check if connection is successful
if ib.isConnected():
    print("Connected to IBKR!")
else:
    print("Failed to connect to IBKR.")

# Disconnect after checking the connection
ib.disconnect()
