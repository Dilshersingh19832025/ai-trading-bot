import pandas as pd
import matplotlib.pyplot as plt

# Load historical data
file_path = "C:\\Users\\gill_\\OneDrive\\Documents\\tradingbot\\historical_data.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# Check if SMA_50 exists
if "SMA_50" not in df.columns:
    print("❌ Error: SMA_50 column is missing! Run update_historical_data.py first.")
    exit()

# Plot stock price and SMA 50
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["close"], label="Stock Price", color="blue")
plt.plot(df["date"], df["SMA_50"], label="SMA 50", color="orange", linestyle="dashed")

# Chart settings
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price with SMA 50")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

# Show plot
plt.show()

