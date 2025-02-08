import pandas as pd

# Load historical data from CSV
file_path = "C:\\Users\\gill_\\OneDrive\\Documents\\tradingbot\\historical_data.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# Ensure the data is sorted by date
df = df.sort_values(by="date")

# Calculate SMA 50
df["SMA_50"] = df["close"].rolling(window=50).mean()

# Save the updated file
df.to_csv(file_path, index=False)

print("✅ Updated historical_data.csv with SMA_50")
