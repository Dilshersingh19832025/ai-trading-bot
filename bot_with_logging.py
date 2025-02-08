import os
from datetime import datetime

# Log file location
log_file = 'trade_log.txt'

# Function to log the trade details
def log_trade(order_type, symbol, quantity, price):
    """
    Log trade information to a file
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {order_type.capitalize()} order for {quantity} shares of {symbol} at ${price}\n"
    
    # Open the log file and append the trade details
    with open(log_file, 'a') as file:
        file.write(log_message)

# Example of a trading function within your bot
def execute_trade(order_type, symbol, quantity, price):
    """
    Execute a trade (buy/sell) and log the trade.
    """
    # Code to execute the trade (this depends on your bot's setup and trading platform)
    
    # Simulating the trade execution (this part should be where your actual trade logic goes)
    print(f"Executing {order_type} order for {quantity} shares of {symbol} at ${price}")
    
    # Log the trade to the file
    log_trade(order_type, symbol, quantity, price)
    print(f"Executed {order_type} order for {quantity} shares of {symbol} at ${price}")

# Example: Triggering the execution and logging
if __name__ == "__main__":
    # Example trades for testing the logging integration
    execute_trade('buy', 'AAPL', 100, 50.0)
    execute_trade('sell', 'AAPL', 100, 55.0)
