import requests
import pandas as pd
import time

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "Y7FBL7K1XB634HAD"

def fetch_stock_data(symbol, interval="1min", retries=3, timeout=10):
    """
    Fetch stock data from Alpha Vantage API with improved error handling and optimization.

    Parameters:
        symbol (str): Stock ticker symbol (e.g., "AAPL").
        interval (str): Time interval for data ("1min", "5min", "15min", "30min", "60min").
        retries (int): Number of retry attempts if API fails.
        timeout (int): Timeout duration for API requests (in seconds).
        
    Returns:
        pd.DataFrame: DataFrame with Open, High, Low, and Close prices, or None if an error occurs.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise HTTP errors (if any)
            data = response.json()

            # Check for API rate limit
            if "Note" in data:
                print("[ERROR] API limit reached. Retrying in 60 seconds...")
                time.sleep(60)  # Wait before retrying
                continue
            
            # Handle invalid API key or incorrect response
            if "Error Message" in data:
                print(f"[ERROR] {data['Error Message']}")
                return None

            key = f"Time Series ({interval})"
            if key not in data:
                print(f"[ERROR] Unexpected API response: {data}")
                return None

            # Convert JSON data to DataFrame
            df = pd.DataFrame.from_dict(data[key], orient="index")
            df = df.rename(columns={"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"})
            df = df.astype(float)  # Convert values to float
            df.index = pd.to_datetime(df.index)  # Convert index to datetime
            df = df.sort_index()  # Sort by oldest first

            print(f"[INFO] Successfully fetched data for {symbol} ({interval})")
            return df

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Network issue: {e}. Retrying ({attempt+1}/{retries})...")
            time.sleep(5)  # Wait before retrying

    print("[ERROR] Failed to fetch stock data after multiple attempts.")
    return None

# Example Execution
if __name__ == "__main__":
    stock_symbol = "AAPL"  # Change this to any stock symbol
    interval = "5min"  # Adjust interval as needed

    stock_data = fetch_stock_data(stock_symbol, interval)
    
    if stock_data is not None:
        print(stock_data.head())  # Display first few rows
