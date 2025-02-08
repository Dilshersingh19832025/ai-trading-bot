import logging
import csv
import os
from datetime import datetime

# Setting up the logger
log_file = 'trade_log.csv'

# Check if the log file exists, if not create it and write headers
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Action', 'Symbol', 'Quantity', 'Price', 'Profit/Loss'])

def log_trade(action, symbol, quantity, price, profit_loss=None):
    """
    Log the details of the trade.
    
    action: 'buy' or 'sell'
    symbol: Stock or asset symbol
    quantity: Number of shares/contracts
    price: Price at which trade was executed
    profit_loss: Profit or loss from the trade (if available)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, action, symbol, quantity, price, profit_loss])

# Example usage: Log a trade
def execute_trade(action, symbol, quantity, price):
    """
    Simulate a trade execution and log it.
    """
    # In a real bot, you would execute the trade here (buy/sell)
    print(f"Executed {action} order for {quantity} shares of {symbol} at ${price}")
    
    # Optionally, calculate profit/loss if selling
    if action == 'sell':
        # Example profit calculation - modify as per your strategy
        profit_loss = (price - previous_buy_price) * quantity
        log_trade(action, symbol, quantity, price, profit_loss)
    else:
        log_trade(action, symbol, quantity, price)

# Example of trade execution
previous_buy_price = 50.0  # Assume this was the last buy price
execute_trade('buy', 'AAPL', 100, 50.0)
execute_trade('sell', 'AAPL', 100, 55.0)
