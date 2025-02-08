import requests
import pandas as pd

API_KEY = "Y7FBL7K1XB634HAD"

def fetch_stock_data(symbol, interval):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "datatype": "json",
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Debugging information
        if "Note" in data:
            print("Note: API limit reached. Please wait before retrying.")
            return None
        if "Error Message" in data:
            print(f"Error Message: {data['Error Message']}")
            return None
        if "Time Series" not in data:
            print(f"Unexpected response: {data}")
            return None

        time_series_key = list(data.keys())[1]  # E.g., "Time Series (5min)"
        time_series = data[time_series_key]

        df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.rename(columns=lambda x: x.split(" ")[1].capitalize(), inplace=True)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


if __name__ == "__main__":
    symbol = input("Enter the stock ticker symbol (e.g., 'AAPL'): ").upper()
    interval = input("Enter the time interval (e.g., '1min', '5min', '15min'): ").lower()

    stock_data = fetch_stock_data(symbol, interval)
    if stock_data is not None:
        print(stock_data.head())
        stock_data.to_csv(f"{symbol}_{interval}_data.csv")
        print(f"Data saved to {symbol}_{interval}_data.csv")

