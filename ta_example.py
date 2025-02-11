import talib
import numpy as np

# Sample data: a simple array of prices
prices = np.array([81.59, 81.06, 82.87, 83.00, 83.61, 83.15, 82.84, 83.99, 84.55, 84.36])

# Calculate a 5-period simple moving average (SMA)
sma = talib.SMA(prices, timeperiod=5)

print("5-period SMA:", sma)
