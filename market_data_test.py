from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time


class IBKRApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = None

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        print(f"Next valid order ID: {orderId}")

    def error(self, reqId, errorCode, errorString):
        print(f"ERROR {reqId} {errorCode} {errorString}")

    def tickPrice(self, reqId, tickType, price, attrib):
        print(f"Tick Price. Ticker ID: {reqId}, Type: {tickType}, Price: {price}")

    def tickSize(self, reqId, tickType, size):
        print(f"Tick Size. Ticker ID: {reqId}, Type: {tickType}, Size: {size}")


def run_loop():
    app.run()


app = IBKRApp()
app.connect("127.0.0.1", 7497, clientId=0)

api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(2)  # Allow time to connect

print("Requesting market data...")

contract = Contract()
contract.symbol = "AAPL"
contract.secType = "STK"
contract.currency = "USD"
contract.exchange = "SMART"

app.reqMarketDataType(3)  # 3 = Delayed market data, 1 = Live, 2 = Frozen
app.reqMktData(1, contract, "", False, False, [])

time.sleep(10)  # Allow time to receive market data

app.disconnect()





