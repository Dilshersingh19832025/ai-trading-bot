import pandas as pd
import numpy as np

# Generate 300+ days of synthetic stock price data
num_days = 300
start_price = 100

dates = pd.date_range(start="2024-01-01", periods=num_days, freq="D")
prices = start_price + np.cumsum(np.random.normal(0.5, 2, num_days))  # Random price fluctuations

# Create DataFrame
df = pd.DataFrame({"date": dates, "close": prices})

# Save as CSV
df.to_csv("historical_data.csv", index=False)

print("Generated historical_data.csv with 300+ rows.")
