import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf

# Load stock data (using Yahoo Finance for demonstration)
ticker = 'AAPL'  # Example ticker symbol
data = yf.download(ticker, start='2000-01-01', end='2020-12-31')

# Ensure the 'Close' column is a pandas Series and convert it to a numpy array for talib
close_prices = data['Close'].to_numpy()

# Flatten the array to avoid dimension errors with talib
close_prices = np.ravel(close_prices)

# Calculate technical indicators (SMA, RSI, etc.)
data['50_SMA'] = talib.SMA(close_prices, timeperiod=50)
data['14_RSI'] = talib.RSI(close_prices, timeperiod=14)

# Calculate price change as the target variable (1 for up, 0 for down)
data['Price_Change'] = data['Close'].pct_change().shift(-1)
data['Target'] = np.where(data['Price_Change'] > 0, 1, 0)

# Drop any missing values
data = data.dropna()

# Feature selection, include 'Price_Change' here
features = ['50_SMA', '14_RSI', 'Price_Change']
X = data[features]
y = data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Initialize balance and stock variables
balance = 10000  # Starting balance in USD
stocks_bought = 0
initial_balance = balance

# Run the backtest
for i in range(len(X_test)):
    # Extract scalar values for price and price change
    # Use .loc to access rows by the index of X_test and ensure scalar values
    close_price = data.loc[X_test.index[i], 'Close'].item()  # Force scalar value
    price_change = X_test.iloc[i]['Price_Change'].item()  # Force scalar value

    # Debugging: Print types and values
    print(f"Close price type: {type(close_price)}, Close price value: {close_price}")
    print(f"Price change type: {type(price_change)}, Price change value: {price_change}")

    # Buy condition: if the model predicts an increase and we have enough balance
    if y_pred[i] == 1 and balance >= close_price:  # Buy condition
        stocks_bought = balance // close_price  # Buy as many shares as possible
        balance -= stocks_bought * close_price  # Deduct price for buying
        print(f"Bought {stocks_bought} shares at price: {close_price}")

    # Sell condition: if the model predicts a decrease and we have bought stocks
    elif y_pred[i] == 0 and stocks_bought > 0:  # Sell condition
        balance += stocks_bought * close_price  # Sell stocks at the current close price
        stocks_bought = 0
        print(f"Sold all shares at price: {close_price}")

# Calculate final balance by selling any remaining shares at the last close price
if stocks_bought > 0:
    balance += stocks_bought * data.loc[X_test.index[-1], 'Close'].item()
    stocks_bought = 0

# Output final balance
print(f"Final Balance after Backtest: ${balance:.2f}")






































