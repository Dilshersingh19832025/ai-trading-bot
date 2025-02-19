import yfinance as yf

# Fetch 1 day of data at a 1-minute interval for a well-known stock (e.g., AAPL)
data = yf.download("AAPL", period="1d", interval="1m")
print(data.head())
