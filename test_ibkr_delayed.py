from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)  # Paper Trading port

contract = Stock('TSLA', 'SMART', 'USD')

# Force request for delayed data
ib.reqMarketDataType(3)  # 3 = Delayed
market_data = ib.reqMktData(contract)

ib.sleep(2)  # Give IBKR time to send data

print(f"TSLA Market Price: {market_data.last}")

ib.disconnect()
