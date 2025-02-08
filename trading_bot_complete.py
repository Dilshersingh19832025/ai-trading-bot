import yfinance as yf
import pandas as pd
import talib
import numpy as np

# Download historical data
symbol = 'AAPL'
print(f"Downloading data for symbol: {symbol}...")
data = yf.download(symbol, start="2020-01-01", end="2025-01-01", progress=False)
print("Data downloaded successfully.")

# Check for multi-level columns and extract relevant data if needed
if isinstance(data.columns, pd.MultiIndex):
    print("Multi-level column index detected. Extracting data for the symbol...")
    data = data.xs(symbol, axis=1, level=1, drop_level=True)
    print("Data extracted successfully for the symbol.")

# Ensure required columns are present
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
if all(col in data.columns for col in required_columns):
    print("All required columns are present in the data.")
else:
    raise ValueError(f"Missing required columns in the data. Found columns: {data.columns}")

# Convert data types
print("Converting data types for required columns...")
for col in required_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
print("Data types converted successfully.")

# Add technical indicators
print("Adding technical indicators...")
data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)
data['EMA_10'] = talib.EMA(data['Close'], timeperiod=10)
print("Technical indicators added successfully.")

# Implement Buy/Sell Signals
data['Buy_Signal'] = np.where(
    (data['Close'] > data['SMA_20']) & (data['RSI_14'] < 30) & (data['Close'] > data['EMA_10']),
    1, 0
)

data['Sell_Signal'] = np.where(
    (data['Close'] < data['SMA_20']) & (data['RSI_14'] > 70) & (data['Close'] < data['EMA_10']),
    -1, 0
)

# Display data shape and preview
print("Shape of data after adding technical features:", data.shape)
print("\nFirst few rows of the prepared data with signals:")
print(data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI_14', 'EMA_10', 'Buy_Signal', 'Sell_Signal']].head())

# Save prepared data
data.to_csv('prepared_data_with_signals.csv')
print("\nPrepared data with buy/sell signals saved to 'prepared_data_with_signals.csv'.")




