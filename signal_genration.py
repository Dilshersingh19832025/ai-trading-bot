#!/usr/bin/env python3
"""
signal_generation.py

This module generates trading signals for our self-trading AI bot.
It uses technical indicators computed in indicators.py and
implements multiple strategies:
    1. SMA/EMA Crossover
    2. Bollinger Bands with RSI
    3. SuperTrend-based Signal

Signals are then aggregated into a final trading signal.

Usage:
    Import and call generate_signals(data) with a DataFrame containing 
    'close', 'high', and 'low' columns.
    
Requirements:
    - indicators.py must be in the same directory.
    - The data_fetching and indicators modules should be properly integrated.
"""

import pandas as pd
import numpy as np
import logging
from indicators import compute_all_indicators  # Import our indicator aggregator

# Configure logger for this module
logger = logging.getLogger("TradingBot.signal_generation")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# ---------------------------
# Strategy 1: SMA/EMA Crossover Signal
# ---------------------------
def sma_ema_crossover_signal(data: pd.DataFrame, price_column: str = "close") -> pd.Series:
    """
    Generates signals based on the EMA crossover:
      - Buy signal (1): if price crosses above EMA from previous row.
      - Sell signal (-1): if price crosses below EMA.
      - Hold (0): otherwise.
    """
    # We assume that indicators_df already contains 'ema_20'
    signal = pd.Series(0, index=data.index)
    # Ensure we have enough data
    if len(data) < 2:
        return signal

    # Compare current close vs previous close relative to EMA
    prev_price = data[price_column].shift(1)
    prev_ema = data['ema_20'].shift(1)
    curr_price = data[price_column]
    curr_ema = data['ema_20']
    
    # Buy signal: previous price <= previous EMA and current price > current EMA
    buy_mask = (prev_price <= prev_ema) & (curr_price > curr_ema)
    # Sell signal: previous price >= previous EMA and current price < current EMA
    sell_mask = (prev_price >= prev_ema) & (curr_price < curr_ema)
    
    signal[buy_mask] = 1
    signal[sell_mask] = -1
    return signal

# ---------------------------
# Strategy 2: Bollinger Bands and RSI Signal
# ---------------------------
def bb_rsi_signal(data: pd.DataFrame, price_column: str = "close", rsi_buy: int = 30, rsi_sell: int = 70) -> pd.Series:
    """
    Generates signals based on Bollinger Bands and RSI:
      - Buy signal (1): if price touches or is below the lower band and RSI < rsi_buy.
      - Sell signal (-1): if price touches or is above the upper band and RSI > rsi_sell.
      - Hold (0): otherwise.
    Assumes that indicators_df contains 'bb_upper', 'bb_lower', and 'rsi_14'.
    """
    signal = pd.Series(0, index=data.index)
    if len(data) < 1:
        return signal

    price = data[price_column]
    lower_band = data['bb_lower']
    upper_band = data['bb_upper']
    rsi_val = data['rsi_14']

    buy_mask = (price <= lower_band) & (rsi_val < rsi_buy)
    sell_mask = (price >= upper_band) & (rsi_val > rsi_sell)
    
    signal[buy_mask] = 1
    signal[sell_mask] = -1
    return signal

# ---------------------------
# Strategy 3: SuperTrend Signal
# ---------------------------
def supertrend_signal(data: pd.DataFrame, price_column: str = "close") -> pd.Series:
    """
    Generates signals based on the SuperTrend indicator:
      - Buy signal (1): if the price is above SuperTrend.
      - Sell signal (-1): if the price is below SuperTrend.
    Assumes that indicators_df contains 'supertrend'.
    """
    signal = pd.Series(0, index=data.index)
    if len(data) < 1:
        return signal
    price = data[price_column]
    st = data['supertrend']
    # Only generate signals when SuperTrend is available (non-NaN)
    signal[price > st] = 1
    signal[price < st] = -1
    return signal

# ---------------------------
# Signal Aggregation
# ---------------------------
def aggregate_signals(*signals) -> pd.Series:
    """
    Aggregates multiple signal Series by taking a majority vote.
    Returns:
      1 for buy, -1 for sell, and 0 for hold.
    If there is a tie or insufficient data, returns 0.
    """
    # Convert signals to DataFrame for aggregation.
    signals_df = pd.concat(signals, axis=1)
    # Sum signals for each row.
    total = signals_df.sum(axis=1)
    # If the total is positive -> buy; negative -> sell; zero -> hold.
    aggregated = total.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return aggregated

# ---------------------------
# Main Signal Generation Function
# ---------------------------
def generate_signals(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates final trading signals by:
      1. Computing all indicators on the market_data.
      2. Generating individual strategy signals.
      3. Aggregating them into a final signal column.
    
    Returns a DataFrame with the original market data, all indicators, and a 'signal' column.
    """
    try:
        # Compute all indicators (this function is defined in indicators.py)
        indicators_df = compute_all_indicators(market_data)
        # Merge indicators with the original data
        data = market_data.join(indicators_df)
        
        # Generate signals from different strategies
        signal1 = sma_ema_crossover_signal(data)
        signal2 = bb_rsi_signal(data)
        signal3 = supertrend_signal(data)
        
        # Aggregate signals (here we use a simple majority vote)
        aggregated_signal = aggregate_signals(signal1, signal2, signal3)
        data['signal'] = aggregated_signal
        return data
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise

# ---------------------------
# Main Testing Block
# ---------------------------
if __name__ == "__main__":
    try:
        # Sample market data for testing (use realistic sample data for production)
        market_data = pd.DataFrame({
            "close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "high":  [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
            "low":   [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
        })
        # Generate signals
        signals_df = generate_signals(market_data)
        print("Generated Signals:\n", signals_df)
    except Exception as e:
        logger.error(f"Error during testing: {e}")
