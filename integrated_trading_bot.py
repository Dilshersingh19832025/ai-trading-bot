import requests
import pandas as pd

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "Y7FBL7K1XB634HAD"

def fetch_stock_data(symbol, interval="1min"):
    """
    Fetch stock data from Alpha Vantage API.
    
    Parameters:
        symbol (str): Stock ticker symbol (e.g., "AAPL").
        interval (str): Time interval for data ("1min", "5min", "15min", "30min", "60min").
        
    Returns:
        pd.DataFrame: DataFrame with Open, High, Low, and Close prices.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"
    
    response = requests.get(url)
    data = response.json()

    # Check for API errors or rate limits
    if "Note" in data:
        print("API limit reached. Try again later.")
        return None
    if "Error Message" in data:
        print(f"Error: {data['Error Message']}")
        return None

    key = f"Time Series ({interval})"
    if key not in data:
        print(f"Unexpected API response: {data}")
        return None

    # Convert JSON data to DataFrame
    df = pd.DataFrame.from_dict(data[key], orient="index")
    df = df.rename(columns={"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"})

    # Convert data types
    df = df.astype(float)

    # Convert index to datetime for time-based analysis
    df.index = pd.to_datetime(df.index)

    # Sort index (oldest first)
    df = df.sort_index()
    
    return df

# Example Execution
if __name__ == "__main__":
    stock_symbol = "AAPL"  # Change this to any stock symbol
    interval = "5min"  # Adjust interval as needed

    stock_data = fetch_stock_data(stock_symbol, interval)
    
    if stock_data is not None:
        print(stock_data.head())  # Display first few rows

