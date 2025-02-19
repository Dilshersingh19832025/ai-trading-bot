#!/usr/bin/env python3
"""
indicators.py

This module calculates technical indicators for our self-trading AI bot.
It incorporates:
  - Precomputation of common rolling statistics
  - Parameter validation for robust error handling
  - Optional TA‑Lib integration for production performance (with caching)
  - Parallel computation support for multiple symbols (configurable max_workers)
  - Comprehensive logging and profiling to monitor performance

Implemented indicators:
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - Bollinger Bands
  - RSI (Relative Strength Index)
  - SuperTrend (with iterative refinement; vectorized approximation as fallback)

Usage:
    Import this module and call the functions with your DataFrame containing
    the required price columns (e.g., "close", "high", "low").
    Use `compute_all_indicators(data)` for a single symbol and
    `compute_indicators_parallel(data_dict)` for multiple symbols.
"""

import pandas as pd
import numpy as np
import logging
import time
import functools
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Optional: Try to import TA‑Lib for optimized calculations.
try:
    import talib
    TA_LIB_AVAILABLE = True
    logging.info("TA‑Lib is available. Will use TA‑Lib functions where possible.")
except ImportError:
    TA_LIB_AVAILABLE = False
    logging.info("TA‑Lib not available. Falling back to Pandas implementations.")

# Configure module-level logger.
logger = logging.getLogger("TradingBot.indicators")
logger.setLevel(logging.DEBUG)  # Use DEBUG level for profiling; adjust as needed.
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# ---------------------------
# Global Caches for TA‑Lib results
# ---------------------------
_talib_sma_cache = {}
_talib_ema_cache = {}
_talib_rsi_cache = {}

def hash_array(arr: np.ndarray) -> str:
    """Return an MD5 hash of the array's bytes."""
    return hashlib.md5(arr.tobytes()).hexdigest()

# ---------------------------
# Profiling Decorator
# ---------------------------
def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# ---------------------------
# Parameter Validation Helper
# ---------------------------
def validate_params(data: pd.DataFrame, window: int, price_column: str):
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Window must be a positive integer")
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in the provided DataFrame")
    return True

# ---------------------------
# Precomputation Helper Function
# ---------------------------
def precompute_rolling(data: pd.DataFrame, window: int, price_column: str = "close"):
    """Precompute common rolling statistics: SMA, rolling std, and delta."""
    validate_params(data, window, price_column)
    sma = data[price_column].rolling(window=window, min_periods=1).mean()
    rstd = data[price_column].rolling(window=window, min_periods=1).std()
    delta = data[price_column].diff()
    return sma, rstd, delta

# ---------------------------
# Indicator Functions
# ---------------------------

@profile
def calculate_sma(data: pd.DataFrame, window: int = 20, price_column: str = "close") -> pd.Series:
    validate_params(data, window, price_column)
    # Use Pandas if data length is too short or TA‑Lib is unavailable.
    if len(data) < window or not TA_LIB_AVAILABLE:
        return data[price_column].rolling(window=window, min_periods=1).mean()
    try:
        prices = data[price_column].astype(np.float64).values
        key = (hash_array(prices), window)
        if key in _talib_sma_cache:
            logger.debug("Using cached TA‑Lib SMA result")
            return _talib_sma_cache[key]
        sma_vals = talib.SMA(prices, timeperiod=window)
        result = pd.Series(sma_vals, index=data.index)
        _talib_sma_cache[key] = result
        return result
    except Exception as e:
        logger.error(f"TA‑Lib SMA error: {e}. Falling back to Pandas.")
        return data[price_column].rolling(window=window, min_periods=1).mean()

@profile
def calculate_ema(data: pd.DataFrame, window: int = 20, price_column: str = "close") -> pd.Series:
    validate_params(data, window, price_column)
    if len(data) < window or not TA_LIB_AVAILABLE:
        return data[price_column].ewm(span=window, adjust=False).mean()
    try:
        prices = data[price_column].astype(np.float64).values
        key = (hash_array(prices), window)
        if key in _talib_ema_cache:
            logger.debug("Using cached TA‑Lib EMA result")
            return _talib_ema_cache[key]
        ema_vals = talib.EMA(prices, timeperiod=window)
        result = pd.Series(ema_vals, index=data.index)
        _talib_ema_cache[key] = result
        return result
    except Exception as e:
        logger.error(f"TA‑Lib EMA error: {e}. Falling back to Pandas.")
        return data[price_column].ewm(span=window, adjust=False).mean()

@profile
def bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2, price_column: str = "close"):
    """
    Calculate Bollinger Bands.
    Returns:
        sma: Simple Moving Average (centerline)
        upper_band: SMA + (num_std * rolling std)
        lower_band: SMA - (num_std * rolling std)
    """
    validate_params(data, window, price_column)
    sma, rstd, _ = precompute_rolling(data, window, price_column)
    upper_band = sma + num_std * rstd
    lower_band = sma - num_std * rstd
    return sma, upper_band, lower_band

@profile
def rsi(data: pd.DataFrame, window: int = 14, price_column: str = "close") -> pd.Series:
    validate_params(data, window, price_column)
    if len(data) < window or not TA_LIB_AVAILABLE:
        # Fallback vectorized implementation.
        delta = data[price_column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean().replace(to_replace=0, method='ffill')
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values.fillna(0)
    try:
        prices = data[price_column].astype(np.float64).values
        key = (hash_array(prices), window)
        if key in _talib_rsi_cache:
            logger.debug("Using cached TA‑Lib RSI result")
            return _talib_rsi_cache[key]
        rsi_vals = talib.RSI(prices, timeperiod=window)
        result = pd.Series(rsi_vals, index=data.index).fillna(0)
        _talib_rsi_cache[key] = result
        return result
    except Exception as e:
        logger.error(f"TA‑Lib RSI error: {e}. Falling back to Pandas.")
        delta = data[price_column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean().replace(to_replace=0, method='ffill')
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values.fillna(0)

@profile
def supertrend(data: pd.DataFrame, atr_period: int = 10, multiplier: float = 3) -> pd.Series:
    """
    Calculate the SuperTrend indicator using vectorized ATR and an iterative process.
    """
    for col in ['high', 'low', 'close']:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")
    try:
        high = data['high']
        low = data['low']
        close = data['close']
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = high_low.combine(high_close, max).combine(low_close, max)
        atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        supertrend = pd.Series(index=data.index, dtype=np.float64)
        for i in range(len(data)):
            if i < atr_period:
                supertrend.iloc[i] = np.nan
            else:
                supertrend.iloc[i] = upper_band.iloc[i] if close.iloc[i] <= upper_band.iloc[i] else lower_band.iloc[i]
        return supertrend
    except Exception as e:
        logger.error(f"Error calculating SuperTrend: {e}")
        raise

# ---------------------------
# Aggregator: Compute All Indicators
# ---------------------------
@profile
def compute_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a set of indicators and return a DataFrame containing:
      - SMA (window 20)
      - EMA (window 20)
      - Bollinger Bands (window 20, num_std=2)
      - RSI (window 14)
      - SuperTrend (atr_period 10, multiplier 3)
    """
    indicators = pd.DataFrame(index=data.index)
    try:
        indicators['sma_20'] = calculate_sma(data, window=20)
        indicators['ema_20'] = calculate_ema(data, window=20)
        sma_bb, upper_bb, lower_bb = bollinger_bands(data, window=20)
        indicators['bb_sma'] = sma_bb
        indicators['bb_upper'] = upper_bb
        indicators['bb_lower'] = lower_bb
        indicators['rsi_14'] = rsi(data, window=14)
        indicators['supertrend'] = supertrend(data, atr_period=10, multiplier=3)
    except Exception as e:
        logger.error(f"Error computing all indicators: {e}")
        raise
    return indicators

# ---------------------------
# Parallel Processing for Multiple Symbols
# ---------------------------
def compute_indicators_for_symbol(data: pd.DataFrame) -> pd.DataFrame:
    return compute_all_indicators(data)

def compute_indicators_parallel(data_dict: dict, max_workers: int = 4) -> dict:
    """
    Compute indicators for multiple symbols concurrently.
    
    Parameters:
        data_dict: dict mapping symbol -> DataFrame
        max_workers: Maximum number of parallel threads
    Returns:
        dict mapping symbol -> indicators DataFrame
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_indicators_for_symbol, df): symbol
                   for symbol, df in data_dict.items()}
        for future in futures:
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    return results

# ---------------------------
# Main Testing Block
# ---------------------------
if __name__ == "__main__":
    try:
        # Create a sample DataFrame for testing indicators
        data = pd.DataFrame({
            "close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "high":  [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
            "low":   [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
        })
        indicators_df = compute_all_indicators(data)
        print("Computed Indicators:\n", indicators_df)
    except Exception as e:
        logger.error(f"Error during testing: {e}")
