import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib

# Sample data (you can replace this with your own stock data)
data = {
    'date': pd.date_range('20230101', periods=100),
    'close': np.random.randn(100).cumsum() + 100  # Random stock closing prices
}

# Create a DataFrame
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Calculate SMA using TA-Lib
sma = talib.SMA(df['close'], timeperiod=14)

# Plot the stock prices and the SMA
plt.figure(figsize=(10,6))
plt.plot(df.index, df['close'], label='Stock Price', color='blue')
plt.plot(df.index, sma, label='14-day SMA', color='red')
plt.title('Stock Price and 14-Day SMA')
plt.legend()
plt.show()
