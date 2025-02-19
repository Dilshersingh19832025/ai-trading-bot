#!/usr/bin/env python3
"""
data_fetching.py

This module handles fetching and preparing historical data for symbols using yfinance.
It includes data cleaning steps, caching, retry logic, and an asynchronous wrapper.

Usage Examples:
    1. Using default parameters (ETHUSDT, 1h interval, 60d period):
         python data_fetching.py
    2. Specifying a shorter period for intraday data:
         python data_fetching.py --symbol ETHUSDT --interval 1h --period 15d
    3. Using a daily interval over a longer period:
         python data_fetching.py --symbol ETHUSDT --interval 1d --period 60d
    4. Specifying an explicit date range:
         python data_fetching.py --symbol ETHUSDT --interval 1h --start 2025-01-01 --end 2025-01-15
    5. Setting logging level to DEBUG:
         python data_fetching.py --loglevel DEBUG
"""

import logging
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import os
import time
import pickle
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Modular configuration
config = {
    "cache_dir": os.path.join(os.path.dirname(__file__), "cache"),
    "cache_ttl": 3600,        # Cache time-to-live in seconds
    "max_retries": 3,
    "initial_delay": 1,       # Initial delay for retry (seconds)
    "backoff": 2              # Exponential backoff factor
}

# Ensure cache directory exists
if not os.path.exists(config["cache_dir"]):
    os.makedirs(config["cache_dir"])

# Configure logger for this module
logger = logging.getLogger("TradingBot.data_fetching")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate logging
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def get_cache_key(symbol, interval, period, start, end):
    key_str = f"{symbol}_{interval}_{period}_{start}_{end}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def load_cache(cache_key):
    cache_file = os.path.join(config["cache_dir"], cache_key + ".pkl")
    if os.path.exists(cache_file):
        # Check if cache file is still valid
        if time.time() - os.path.getmtime(cache_file) < config["cache_ttl"]:
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Loaded data from cache with key: {cache_key}")
                return data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
    return None

def save_cache(cache_key, data):
    cache_file = os.path.join(config["cache_dir"], cache_key + ".pkl")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved data to cache with key: {cache_key}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def retry(func):
    """Decorator to retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        retries = config["max_retries"]
        delay = config["initial_delay"]
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed with error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= config["backoff"]
        raise Exception(f"Function {func.__name__} failed after {retries} retries.")
    return wrapper

@retry
def download_data(yf_symbol, period, interval, start, end):
    """Helper function to download data using yfinance with retry logic."""
    if start and end:
        logger.info(f"Downloading data for {yf_symbol} from {start} to {end} with interval {interval}...")
        df = yf.download(yf_symbol, start=start, end=end, interval=interval)
    else:
        logger.info(f"Downloading data for {yf_symbol} for period {period} and interval {interval}...")
        df = yf.download(yf_symbol, period=period, interval=interval)
    if df.empty:
        raise Exception("No data returned")
    return df

def prepare_dataframe(df):
    """
    Prepare and clean the DataFrame:
      - Reset the index and standardize column names.
      - Convert key columns to floats.
      - Combine forward-fill and replace operations.
      - Fill missing volume data with zeros.
    """
    df = df.copy()
    df.reset_index(inplace=True)
    # Standardize columns based on typical yfinance output
    if 'Datetime' in df.columns:
        selected_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    elif 'Date' in df.columns:
        selected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    else:
        logger.error("Unexpected DataFrame format: Missing 'Date' or 'Datetime' column")
        return pd.DataFrame()

    df = df[selected_cols]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    # Combine fill operations
    df = df.fillna(method='ffill').replace(0, np.nan).fillna(method='ffill')
    if df['volume'].isnull().any():
        logger.warning("Volume data contains missing values; filling missing volume with zeros.")
        df['volume'] = df['volume'].fillna(0)
    return df

def fetch_historical_data(symbol, interval, period=None, start=None, end=None):
    """
    Fetch historical data for a symbol from yfinance.
    Incorporates caching, retry logic, and data cleaning.
    """
    try:
        yf_symbol = symbol.replace("USDT", "-USD")
        cache_key = get_cache_key(symbol, interval, period, start, end)
        cached_data = load_cache(cache_key)
        if cached_data is not None:
            return cached_data

        df = download_data(yf_symbol, period, interval, start, end)
        df = prepare_dataframe(df)
        if df.empty or df[['open', 'high', 'low', 'close']].isnull().any().any():
            logger.error(f"Data integrity issue for {symbol}.")
            return None
        save_cache(cache_key, df)
        logger.info(f"Data for {symbol} fetched successfully with shape {df.shape}.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch historical data for {symbol}: {e}")
        return None

async def async_fetch_historical_data(symbol, interval, period=None, start=None, end=None):
    """
    Asynchronous wrapper for fetch_historical_data.
    """
    return await asyncio.to_thread(fetch_historical_data, symbol, interval, period, start, end)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data fetching for the trading bot.")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol (default: ETHUSDT)")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (default: 1h)")
    parser.add_argument("--period", type=str, default="60d", help="Data period (default: 60d)")
    parser.add_argument("--start", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    # Set logging level based on command-line argument
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)

    logger.info("Starting data fetch test from data_fetching.py...")
    data = fetch_historical_data(args.symbol, args.interval, period=args.period, start=args.start, end=args.end)
    if data is not None:
        logger.info("Data fetch successful. Sample data:")
        print(data.head())
    else:
        logger.error("Data fetch failed. Please verify the symbol, period/interval, and date range.")

    # Uncomment below to test the asynchronous version:
    # async def test_async():
    #     data_async = await async_fetch_historical_data(args.symbol, args.interval, period=args.period, start=args.start, end=args.end)
    #     if data_async is not None:
    #         logger.info("Async data fetch successful. Sample data:")
    #         print(data_async.head())
    #     else:
    #         logger.error("Async data fetch failed.")
    #
    # asyncio.run(test_async())


