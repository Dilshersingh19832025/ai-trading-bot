from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time

class TradingApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.order_id = None

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.order_id = orderId
        print(f"Next valid order ID: {self.order_id}")
        self.place_order()

    def place_order(self):
        contract = Contract()
        contract.symbol = "AAPL"  # Stock symbol
        contract.secType = "STK"  # Security type: Stock
        contract.exchange = "SMART"
        contract.currency = "USD"

        order = Order()
        order.action = "BUY"  # Buy order
        order.orderType = "MKT"  # Market order
        order.totalQuantity = 1  # Number of shares
        order.outsideRth = True  # Allow trading outside regular hours
        order.eTradeOnly = False  # Disable 'EtradeOnly' to avoid error
        order.firmQuoteOnly = False  # Disable 'FirmQuoteOnly' to avoid issues

        print("Placing order...")
        self.placeOrder(self.order_id, contract, order)
        time.sleep(3)
        self.disconnect()

    def error(self, reqId, errorCode, errorString):
        print(f"ERROR {reqId} {errorCode} {errorString}")

def run_loop():
    app.run()

app = TradingApp()
app.connect("127.0.0.1", 7497, clientId=0)

api_thread = threading.Thread(target=run_loop)
api_thread.start()
time.sleep(3)
app.disconnect()











