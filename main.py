# main.py
from dotenv import load_dotenv
import os

# Load environment variables from the .env file located in the current directory.
load_dotenv()

# Retrieve API keys from the environment.
alpha_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
binance_api_key = os.getenv("BINANCE_API_KEY")
binance_secret_key = os.getenv("BINANCE_SECRET_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

# For testing, print the keys (or at least confirm they are loaded)
print("Alpha Vantage API Key:", alpha_api_key)
print("Binance API Key:", binance_api_key)
print("Binance Secret Key:", binance_secret_key)
print("News API Key:", news_api_key)

# Continue with your trading bot code here...
# For example, you might print "Hello, world!" or run your strategy.
print("Hello, world!")



