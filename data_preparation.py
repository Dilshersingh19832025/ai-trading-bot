import yfinance as yf
import pandas as pd
import numpy as np

# Step 1: Gather market data (Example: Apple stock data)
symbol = 'AAPL'
data = yf.download(symbol, start="2020-01-01", end="2025-01-01", progress=False)

# Step 2: Ensure the data doesn't have NaN values
data.dropna(inplace=True)

# Step 3: Ensure the columns are in the correct format
data['Open'] = data['Open'].astype(float)
data['High'] = data['High'].astype(float)
data['Low'] = data['Low'].astype(float)
data['Close'] = data['Close'].astype(float)
data['Volume'] = data['Volume'].astype(float)

# Step 4: Check the shape of data
print("Shape of data before adding technical features:")
print(data[['Open', 'High', 'Low', 'Close', 'Volume']].shape)

# Step 5: Manually calculate technical indicators
# Accumulation/Distribution Index (ADI)
data['adi'] = ((2 * data['Close'] - data['High'] - data['Low']) / (data['High'] - data['Low'])) * data['Volume']
data['adi'] = data['adi'].cumsum()  # Cumulative sum for ADI

# Simple Moving Average (SMA)
data['sma_14'] = data['Close'].rolling(window=14).mean()

# Exponential Moving Average (EMA)
data['ema_14'] = data['Close'].ewm(span=14, adjust=False).mean()

# Relative Strength Index (RSI)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['rsi_14'] = 100 - (100 / (1 + rs))

# Bollinger Bands
data['sma_20'] = data['Close'].rolling(window=20).mean()
data['std_20'] = data['Close'].rolling(window=20).std()
data['bollinger_hband'] = data['sma_20'] + (2 * data['std_20'])  # Upper Band
data['bollinger_lband'] = data['sma_20'] - (2 * data['std_20'])  # Lower Band

# Step 6: Adding sentiment score (replace with actual sentiment data)
data['sentiment_score'] = np.random.rand(len(data))  # Placeholder for sentiment scores

# Step 7: Calculate target variable (Price Change)
data['price_change'] = data['Close'].pct_change().shift(-1)  # percentage change in closing price (target variable)

# Step 8: Dropping NaN values (if any remaining after calculation)
data.dropna(inplace=True)

# Step 9: Splitting data into training and test sets
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

# Show the prepared data
print("\nShape of data after adding technical features:")
print(data.shape)
print("\nFirst few rows of the prepared data:")
print(data.head())









