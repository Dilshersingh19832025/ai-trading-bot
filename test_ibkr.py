from ib_insync import IB

ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=1)  # Paper trading port
    print("Connected to IBKR:", ib.isConnected())
    ib.disconnect()
except Exception as e:
    print("IBKR Connection Failed:", str(e))
