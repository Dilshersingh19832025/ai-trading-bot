import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf

# Load stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2000-01-01', end='2020-12-31')

# Ensure 'Close' column is a numpy array
close_prices = data['Close'].to_numpy().ravel()

# Calculate technical indicators
data['50_SMA'] = talib.SMA(close_prices, timeperiod=50)
data['14_RSI'] = talib.RSI(close_prices, timeperiod=14)

# Target variable
data['Price_Change'] = data['Close'].pct_change().shift(-1)
data['Target'] = np.where(data['Price_Change'] > 0, 1, 0)

# Drop missing values
data = data.dropna()

# Features and target
features = ['50_SMA', '14_RSI', 'Price_Change']
X = data[features]
y = data['Target']

# k-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy score
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Model accuracy
print(f"Cross-Validation Accuracy Scores: {accuracy_scores}")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}")

# Backtesting
balance = 10000
stocks_bought = 0
initial_balance = balance

# We use the most recent model prediction to backtest the last fold
for i in range(len(X_test)):
    close_price = data.loc[X_test.index[i], 'Close'].item()  # Ensure scalar
    price_change = X_test.iloc[i]['Price_Change'].item()     # Ensure scalar

    if y_pred[i] == 1 and balance >= float(close_price):  # Fixed comparison
        stocks_bought = balance // close_price
        balance -= stocks_bought * close_price
        print(f"Bought {stocks_bought} shares at price: {close_price}")

    elif y_pred[i] == 0 and stocks_bought > 0:
        balance += stocks_bought * close_price
        stocks_bought = 0
        print(f"Sold all shares at price: {close_price}")

# Sell remaining stocks
if stocks_bought > 0:
    balance += stocks_bought * data.loc[X_test.index[-1], 'Close'].item()
    stocks_bought = 0

print(f"Final Balance after Backtest: ${balance:.2f}")
