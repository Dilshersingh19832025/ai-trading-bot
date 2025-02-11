#!/usr/bin/env python
"""
import_library.py

This script imports the key libraries from our AI Trading Bot project and prints
their versions. Run it with:

    python import_library.py

to confirm that each library is properly installed.
"""

def print_version(module, name=None):
    """Helper function to print the version of a module."""
    name = name or module.__name__
    try:
        version = module.__version__
    except AttributeError:
        version = "No __version__ attribute"
    print(f"{name} version: {version}")

def main():
    # Import libraries
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        import seaborn as sns
        import sklearn
        import tensorflow as tf
        import ccxt
        import alpaca_trade_api as tradeapi
        import yfinance as yf
        import backtrader as bt
        import talib
        import schedule
        import ta
        import requests
        import dotenv
    except ImportError as e:
        print(f"Error importing a module: {e}")
        return

    # Print versions (or confirmation if version not available)
    print_version(np, "NumPy")
    print_version(pd, "Pandas")
    print_version(matplotlib, "Matplotlib")
    print_version(sns, "Seaborn")
    print_version(sklearn, "scikit-learn")
    print_version(tf, "TensorFlow")
    print_version(ccxt, "CCXT")
    print_version(tradeapi, "Alpaca Trade API")
    print_version(yf, "yfinance")
    print_version(bt, "Backtrader")
    
    # TA-Lib may not have a __version__ attribute so handle that separately.
    try:
        print_version(talib, "TA-Lib")
    except Exception as e:
        print("TA-Lib version: Could not determine version")

    try:
        print_version(schedule, "Schedule")
    except Exception as e:
        print("Schedule: Version information not available")
        
    print_version(ta, "ta")
    print_version(requests, "Requests")
    print_version(dotenv, "python-dotenv")

if __name__ == "__main__":
    main()
