#!/usr/bin/env python
"""
main.py - Trading Bot with Enhanced Data Integrity, Logging, and Error Handling

This script:
  - Loads settings and API credentials from a .env file.
  - Fetches historical data from yfinance.
  - Prepares and cleans the data (including forward-filling missing values and replacing zeros).
  - Calculates a suite of technical indicators.
  - Generates a dummy signal (for demonstration) and simulates order placement.
  - Logs each critical step along with any data integrity issues.
  - Sends a Telegram notification with a summary update.

Future improvements may include real order execution logic and more advanced signal generation.
"""

import os
import time
import logging
from logging.handlers import RotatingFileHandler
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Technical indicators from the ta library
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

# Machine Learning for signal generation (dummy example)
from sklearn.neighbors import KNeighborsClassifier

# Notifications (Telegram and Email)
import telegram_send
import smtplib
from email.mime.text import MIMEText

# ------------------------------------------------------------------------------
# Load Environment Variables
# ------------------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------------------
# Configuration Flags & Global Variables
# ------------------------------------------------------------------------------
USE_TWS = False         # Set to True if using TWS; False to use Binance (or simulation)
TEST_MODE = False        # Set to True for simulation mode
SIMULATE_NOTIFICATIONS = True

# Binance API credentials (example: using multiple variable names from .env)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BINANCE_LATEST_API_KEY = os.getenv("BINANCE_LATEST_API_KEY")
BINANCE_LATEST_SECRET_KEY = os.getenv("BINANCE_LATEST_SECRET_KEY")

# Trading settings
TRADING_PAIRS = ['ETHUSDT', 'BTCUSDT']
TIMEFRAME = '1h'        # Time interval for historical data
SLIPPAGE = 0.001        # 0.1% slippage adjustment
STOP_LOSS = 0.02        # 2% stop loss target
TAKE_PROFIT = 0.04      # 4% take profit target
TRAILING_STOP_PERCENT = 0.01  # 1% trailing stop
RISK_PERCENTAGE = 0.01  # 1% risk per trade
REINVEST_PERCENTAGE = 0.5  # Reinvest 50% of profit
INITIAL_BALANCE = 1000.0
portfolio_balance = INITIAL_BALANCE
portfolio = {pair: {'balance': INITIAL_BALANCE / len(TRADING_PAIRS)} for pair in TRADING_PAIRS}

# ------------------------------------------------------------------------------
# Logging Configuration with Rotating File Handler
# ------------------------------------------------------------------------------
logger = logging.getLogger("TradingBot")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("trading_bot.log", maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ------------------------------------------------------------------------------
# Warning Filters: Suppress known RuntimeWarnings from ta.trend module
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ta\\.trend")

# ------------------------------------------------------------------------------
# Data Preparation and Cleaning Functions
# ------------------------------------------------------------------------------
def prepare_dataframe(df):
    """
    Prepare and clean the DataFrame:
      - Reset index and standardize column names.
      - Convert numeric columns to floats.
      - Forward-fill missing values.
      - Replace zeros with NaN and forward-fill.
      - If the 'volume' column is entirely missing, fill with 1.0.
    Returns the cleaned DataFrame.
    """
    df = df.copy()
    df.reset_index(inplace=True)
    # Standardize columns based on typical yfinance output
    if 'Datetime' in df.columns:
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(method='ffill', inplace=True)
    df.replace(0, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    if df['volume'].isnull().all():
        logger.warning("Volume data is entirely missing. Filling volume with 1.0.")
        df['volume'] = 1.0
    return df

# ------------------------------------------------------------------------------
# Fetch Historical Data from yfinance
# ------------------------------------------------------------------------------
def fetch_historical_data(symbol, interval, period="60d"):
    """
    Fetch historical data for a symbol using yfinance.
    Cleans the data and performs integrity checks.
    Returns a DataFrame or None if data is invalid.
    """
    try:
        yf_symbol = symbol.replace("USDT", "-USD")
        logger.info(f"Fetching data for {yf_symbol} (period={period}, interval={interval})...")
        df = yf.download(yf_symbol, period=period, interval=interval)
        df = prepare_dataframe(df)
        if df.empty:
            logger.error(f"Fetched data for {symbol} is empty.")
            return None
        # Ensure essential columns do not contain NaNs
        if df[['open','high','low','close']].isnull().any().any():
            logger.error(f"Data integrity issue for {symbol}: essential price columns contain NaNs. Sample data:\n{df.head()}")
            return None
        logger.info(f"Data for {symbol} fetched successfully with shape {df.shape}.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch historical data for {symbol}: {e}")
        return None

# ------------------------------------------------------------------------------
# Higher Timeframe Trend Analysis
# ------------------------------------------------------------------------------
def higher_timeframe_trend(symbol):
    """
    Analyze daily data to determine the higher timeframe trend.
    Returns 'bullish', 'bearish', or 'neutral'.
    """
    try:
        yf_symbol = symbol.replace("USDT", "-USD")
        df_daily = yf.download(yf_symbol, period="90d", interval="1d")
        df_daily = prepare_dataframe(df_daily)
        if len(df_daily) < 50:
            return "neutral"
        sma_50 = df_daily['close'].rolling(window=50).mean().iloc[-1]
        if len(df_daily) >= 200:
            sma_200 = df_daily['close'].rolling(window=200).mean().iloc[-1]
        else:
            sma_200 = df_daily['close'].rolling(window=50).mean().iloc[-1]
        return "bullish" if sma_50 > sma_200 else "bearish"
    except Exception as e:
        logger.error(f"Error in higher timeframe trend for {symbol}: {e}")
        return "neutral"

# ------------------------------------------------------------------------------
# Indicator Calculation Functions
# ------------------------------------------------------------------------------
def calculate_supertrend(df, period=10, multiplier=3):
    """
    Calculate the SuperTrend indicator and add it to the DataFrame.
    """
    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period)
    df['atr'] = atr_indicator.average_true_range()
    hl2 = (df['high'] + df['low']) / 2
    df['upperband'] = hl2 + multiplier * df['atr']
    df['lowerband'] = hl2 - multiplier * df['atr']
    final_upperband = [0] * len(df)
    final_lowerband = [0] * len(df)
    supertrend = [True] * len(df)
    for i in range(len(df)):
        if i == 0:
            final_upperband[i] = df['upperband'].iloc[i]
            final_lowerband[i] = df['lowerband'].iloc[i]
            supertrend[i] = True
        else:
            final_upperband[i] = (df['upperband'].iloc[i]
                                  if (df['upperband'].iloc[i] < final_upperband[i-1] or df['close'].iloc[i-1] > final_upperband[i-1])
                                  else final_upperband[i-1])
            final_lowerband[i] = (df['lowerband'].iloc[i]
                                  if (df['lowerband'].iloc[i] > final_lowerband[i-1] or df['close'].iloc[i-1] < final_lowerband[i-1])
                                  else final_lowerband[i-1])
            supertrend[i] = False if df['close'].iloc[i] <= final_upperband[i] else True
    df['supertrend'] = supertrend
    df['final_upperband'] = final_upperband
    df['final_lowerband'] = final_lowerband
    return df

def calculate_bull_support_band(df, window=20, percent=0.005):
    """
    Calculate the Bull Support Band indicator.
    """
    df['rolling_low'] = df['low'].rolling(window=window).min()
    df['bull_support_band'] = df['rolling_low'] * (1 + percent)
    return df

def calculate_trend_filters(df):
    """
    Calculate SMA50, SMA200 and derive a trend filter.
    """
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['trend_prime'] = df['sma_50'] > df['sma_200']
    return df

def apply_indicators(df):
    """
    Apply technical indicators to the DataFrame and return the enriched DataFrame.
    """
    try:
        df = df.copy()
        df.fillna(method='ffill', inplace=True)
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        # MACD
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        # EMA
        ema_fast = EMAIndicator(close=df['close'], window=12)
        ema_slow = EMAIndicator(close=df['close'], window=26)
        df['ema_fast'] = ema_fast.ema_indicator()
        df['ema_slow'] = ema_slow.ema_indicator()
        # SuperTrend and Bull Support Band
        df = calculate_supertrend(df)
        df = calculate_bull_support_band(df)
        # ADX
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx.adx()
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch'] = stoch.stoch()
        df['stoch_signal'] = stoch.stoch_signal()
        # Ichimoku
        ichi = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
        df['ichimoku_conversion_line'] = ichi.ichimoku_conversion_line()
        df['ichimoku_base_line'] = ichi.ichimoku_base_line()
        df['ichimoku_a'] = ichi.ichimoku_a()
        df['ichimoku_b'] = ichi.ichimoku_b()
        # OBV and MFI
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
        df['mfi'] = mfi.money_flow_index()
        # Trend Filters
        df = calculate_trend_filters(df)
        return df
    except Exception as e:
        logger.error(f"Failed to apply indicators: {e}")
        return df

# ------------------------------------------------------------------------------
# Placeholder for Core Bot Functionality (Signal generation, order simulation)
# ------------------------------------------------------------------------------
def run_bot():
    """
    Core function to run the trading bot.
    Iterates over trading pairs, fetches data, applies indicators,
    generates a dummy signal, and simulates order execution.
    """
    global portfolio_balance, portfolio
    for symbol in TRADING_PAIRS:
        logger.info(f"----- Processing {symbol} -----")
        df = fetch_historical_data(symbol, TIMEFRAME)
        if df is None or df.empty:
            logger.error(f"Insufficient data for {symbol}.")
            continue

        df = apply_indicators(df)
        # Dummy signal generation: for demonstration, we use "HOLD"
        signal = "HOLD"  # Replace with your actual signal logic
        logger.info(f"[{symbol}] Computed Signal: {signal}")
        last_close = df['close'].iloc[-1]
        logger.info(f"[{symbol}] Last Close Price: {last_close:.2f}")

        if signal not in ['BUY', 'SELL']:
            logger.info(f"[{symbol}] No trade signal. Skipping {symbol}.")
            continue

        # Set order parameters based on signal
        if signal == 'BUY':
            entry_price = last_close * (1 + SLIPPAGE)
            stop_loss_price = entry_price * (1 - STOP_LOSS)
        else:  # SELL signal
            entry_price = last_close * (1 - SLIPPAGE)
            stop_loss_price = entry_price * (1 + STOP_LOSS)

        asset_balance = portfolio[symbol]['balance']
        # Calculate position size based on risk
        quantity = asset_balance * RISK_PERCENTAGE / abs(entry_price - stop_loss_price)
        if quantity <= 0:
            logger.error(f"[{symbol}] Calculated position size is 0. Skipping trade.")
            continue

        logger.info(f"[{symbol}] Calculated Position Size: {quantity:.4f}")
        try:
            # Simulated order placement (replace with actual order logic when ready)
            logger.info(f"[{symbol}] Simulated {signal} order placed.")
        except Exception as e:
            logger.error(f"Order placement failed for {symbol}: {e}")
            continue

        logger.info(f"[{symbol}] Entry Price: {entry_price:.2f}, Stop Loss Price: {stop_loss_price:.2f}")
        # Dummy exit and profit calculation: exit at a 1% loss for demonstration
        exit_price = entry_price * 0.99
        profit_pct = (exit_price - entry_price) / entry_price
        logger.info(f"[{symbol}] Exit Price: {exit_price:.2f}, Profit Percentage: {profit_pct*100:.2f}%")
        profit_amount = profit_pct * quantity * entry_price
        portfolio[symbol]['balance'] += profit_amount
        portfolio_balance = sum(asset['balance'] for asset in portfolio.values())
        logger.info(f"[{symbol}] Trade Profit: ${profit_amount:.2f}, New {symbol} Balance: ${portfolio[symbol]['balance']:.2f}")
        logger.info(f"Overall Portfolio Balance: ${portfolio_balance:.2f}")

        try:
            telegram_send.send(messages=[f"Trade Update for {symbol}:\nSignal: {signal}\nProfit: {profit_amount:.2f}"])
            logger.info("Telegram notification sent.")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    logger.info("Trading Bot run complete.")

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Trading Bot...")
    run_bot()














