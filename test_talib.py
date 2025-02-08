import talib
import numpy as np

# Sample data for testing
close_prices = np.random.random(100)  # 100 random values as close prices

# Calculate the Simple Moving Average (SMA)
sma = talib.SMA(close_prices, timeperiod=14)

print("SMA values:", sma)
