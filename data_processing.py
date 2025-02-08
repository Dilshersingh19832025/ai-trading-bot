import pandas as pd
import numpy as np
import talib

# Load historical data
file_path = "historical_data.csv"
print("Loading data...")

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Ensure 'date' is in datetime format and sort by date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by="date")

# Ensure 'close' column exists
if "close" not in df.columns:
    print("Error: 'close' column missing in data.")
    exit()

print("Checking 'close' column...")
print("'close' column exists.")

# Add technical indicators
print("Adding indicators...")

# Simple Moving Averages (SMA)
df["SMA_50"] = talib.SMA(df["close"], timeperiod=50)
df["SMA_200"] = talib.SMA(df["close"], timeperiod=200)

# Relative Strength Index (RSI)
df["RSI"] = talib.RSI(df["close"], timeperiod=14)

# Moving Average Convergence Divergence (MACD)
df["MACD"], df["MACD_signal"], _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)

# Exponential Moving Averages (EMA)
df["EMA_50"] = talib.EMA(df["close"], timeperiod=50)
df["EMA_200"] = talib.EMA(df["close"], timeperiod=200)

print("Indicators added.")

# Show the first few rows of the updated dataframe
print("Data with indicators:")
print(df.head(60))  # Print first 60 rows to verify calculations

# Save the data with indicators to a new CSV file
output_file = "historical_data_with_indicators.csv"
df.to_csv(output_file, index=False)
print(f"Updated data saved to {output_file}")




