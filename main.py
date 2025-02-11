import os
import logging
import yfinance as yf
import pandas as pd
import talib
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_api_keys():
    load_dotenv()
    alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    binance_api_key = os.getenv("BINANCE_API_KEY")
    binance_secret_key = os.getenv("BINANCE_SECRET_KEY")
    news_api_key = os.getenv("NEWS_API_KEY")
    
    logger.info("Loaded API keys successfully.")
    print("Alpha Vantage API Key:", alpha_vantage_api_key)
    print("Binance API Key:", binance_api_key)
    print("Binance Secret Key:", binance_secret_key)
    print("News API Key:", news_api_key)
    
    return alpha_vantage_api_key, binance_api_key, binance_secret_key, news_api_key

def fetch_historical_data(symbol, start_date, end_date):
    logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        logger.error("No historical data fetched!")
    else:
        logger.info("Historical data fetched successfully.")
    return data

def generate_trading_signals(df):
    # Calculate SMAs
    df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['Signal_SMA'] = 0
    df.loc[df['SMA_5'] > df['SMA_20'], 'Signal_SMA'] = 1
    df.loc[df['SMA_5'] < df['SMA_20'], 'Signal_SMA'] = -1

    # Calculate RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['Signal_RSI'] = 0
    # Buy when RSI is low (<30) and sell when RSI is high (>70)
    df.loc[df['RSI'] < 30, 'Signal_RSI'] = 1
    df.loc[df['RSI'] > 70, 'Signal_RSI'] = -1

    # Calculate MACD and its signal line
    macd, macdsignal, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['Signal_MACD'] = 0
    df.loc[df['MACD'] > df['MACD_Signal'], 'Signal_MACD'] = 1
    df.loc[df['MACD'] < df['MACD_Signal'], 'Signal_MACD'] = -1

    # Combine the signals (a simple method: sum the individual signals)
    df['Combined_Signal'] = df[['Signal_SMA', 'Signal_RSI', 'Signal_MACD']].sum(axis=1)
    # Interpret the combined signal:
    # if the sum > 0, we interpret as a buy signal; if < 0, sell; else hold (0)
    df['Trade_Signal'] = 0
    df.loc[df['Combined_Signal'] > 0, 'Trade_Signal'] = 1
    df.loc[df['Combined_Signal'] < 0, 'Trade_Signal'] = -1

    logger.info("Trading signals generated using SMA, RSI, and MACD.")
    return df

def apply_risk_management(df):
    # Placeholder risk management:
    # In a full implementation, you would adjust position sizing, set stop-loss, and take-profit levels here.
    logger.info("Applying risk management rules (placeholder).")
    # For example, you could filter signals further based on recent volatility.
    return df

def run_backtest(symbol, start_date, end_date, initial_capital=10000):
    data = fetch_historical_data(symbol, start_date, end_date)
    if data.empty:
        return initial_capital

    data = generate_trading_signals(data)
    data = apply_risk_management(data)

    capital = initial_capital
    position = 0
    entry_price = 0

    # Simple backtest loop (buy 1 unit when signal turns 1, sell when signal turns -1)
    for i in range(1, len(data)):
        signal = data['Trade_Signal'].iloc[i]
        price = data['Close'].iloc[i]

        if signal == 1 and position == 0:
            # Enter a long position
            position = 1
            entry_price = price
            logger.info(f"Buy at {price:.2f}")
        elif signal == -1 and position == 1:
            # Exit the position
            profit = price - entry_price
            capital += profit
            logger.info(f"Sell at {price:.2f} | Profit/Loss: {profit:.2f}")
            position = 0

    # If still in position at the end, close it at the last price.
    if position == 1:
        profit = data['Close'].iloc[-1] - entry_price
        capital += profit
        logger.info(f"Closing final position at {data['Close'].iloc[-1]:.2f} | Profit/Loss: {profit:.2f}")

    return capital

def main():
    # Load API keys
    load_api_keys()

    logger.info("Starting backtesting module...")
    final_capital = run_backtest("AAPL", "2022-01-01", "2023-01-01", initial_capital=10000)
    logger.info(f"Backtest complete. Final portfolio value for AAPL from 2022-01-01 to 2023-01-01: ${final_capital:.2f}")
    print(f"Final portfolio value for AAPL: ${final_capital:.2f}")
    logger.info("Bot execution complete.")

if __name__ == "__main__":
    main()





