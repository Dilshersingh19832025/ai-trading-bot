from dotenv import load_dotenv 
import os
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
from binance.client import Client

# Load environment variables from .env file
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Initialize sentiment analyzer
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Initialize Binance client
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

def fetch_financial_news(company_name):
    """
    Fetch real-time financial news related to a company using NewsAPI.
    """
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

    if "status" in data and data["status"] == "error":
        print(f"Error fetching news: {data}")
        return []

    articles = data.get("articles", [])
    return [article["title"] + " - " + (article.get("description") or "") for article in articles if article.get("title")]

def fetch_stock_data(symbol):
    """
    Fetch historical stock data using Yahoo Finance.
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")  # Fetch historical data for the last day
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return {}

def fetch_crypto_data(symbol):
    """
    Fetch real-time cryptocurrency data using Binance.
    """
    try:
        ticker = binance_client.get_symbol_ticker(symbol=symbol)
        return ticker
    except Exception as e:
        print(f"Error fetching crypto data: {e}")
        return {}

def analyze_sentiment(news_list):
    """
    Analyze the sentiment of news articles using VADER.
    """
    if not news_list:
        return "Neutral"

    sentiment_scores = [sia.polarity_scores(news)["compound"] for news in news_list]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    if avg_sentiment > 0.05:
        return "Positive"
    elif avg_sentiment < -0.05:
        return "Negative"
    else:
        return "Neutral"

def execute_trade(symbol, action):
    """
    Execute buy/sell trade on Binance based on sentiment analysis.
    """
    try:
        if action == "BUY":
            order = binance_client.order_market_buy(symbol=symbol, quantity=0.001)  # Adjust quantity as needed
        elif action == "SELL":
            order = binance_client.order_market_sell(symbol=symbol, quantity=0.001)
        else:
            print("No trade action taken.")
            return
        print(f"[TRADE] {action} order placed for {symbol}: {order}")
    except Exception as e:
        print(f"Error executing trade: {e}")

if __name__ == "__main__":
    stock_symbols = ["AAPL", "GOOGL", "TSLA"]  # Add more stocks as needed
    crypto_symbols = ["BTCUSDT", "ETHUSDT"]  # Add more cryptos as needed

    for stock_symbol in stock_symbols:
        print(f"🔹 [INFO] Fetching financial news for {stock_symbol}...")
        news_articles = fetch_financial_news(stock_symbol)
        sentiment = analyze_sentiment(news_articles)
        print(f"✅ [INFO] Market Sentiment for {stock_symbol}: {sentiment}")

        if sentiment == "Positive":
            execute_trade(stock_symbol, "BUY")
        elif sentiment == "Negative":
            execute_trade(stock_symbol, "SELL")

    for crypto_symbol in crypto_symbols:
        print(f"🔹 [INFO] Fetching crypto data for {crypto_symbol}...")
        crypto_data = fetch_crypto_data(crypto_symbol)
        if crypto_data:
            print(f"📊 [INFO] Crypto Data for {crypto_symbol}: {crypto_data}")
