#!/usr/bin/env python3
"""
order_execution_tws.py

Enhanced trading bot module integrating with Interactive Brokers’ TWS.
This version includes improved simulation routines, refined signal generation
(with SMA50, RSI14, ATR, MACD, and Bollinger Bands), advanced risk management,
and enhanced order execution with polling fallback.
"""

import os
import asyncio
import logging
import random
import time
import numpy as np
import pandas as pd
import ta
from ib_insync import IB, Stock, MarketOrder, StopOrder, Order
from config_loader import config  # Loads settings from config.yaml

# Apply nest_asyncio to allow running within an already active event loop (e.g. in notebooks)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ---------------------------
# Logger Setup
# ---------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradingBot.order_execution_tws")

# ---------------------------
# Simulation and Signal Generation
# ---------------------------
def get_trade_signal(data: pd.DataFrame = None) -> dict:
    """
    Generate a trade signal using multiple technical indicators.
    If no data is provided, generate realistic simulated data.
    """
    if data is None:
        # Generate simulated data over the last 100 minutes
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1min')
        base_price = 150
        noise = np.random.normal(0, 0.5, 100)
        trend = np.linspace(0, 1, 100)  # slight upward trend
        prices = base_price + np.cumsum(noise + trend)
        data = pd.DataFrame({'close': prices}, index=dates)
        # Simulate realistic high, low, and open prices
        data['high'] = data['close'] * (1 + np.random.uniform(0.001, 0.005, size=100))
        data['low'] = data['close'] * (1 - np.random.uniform(0.001, 0.005, size=100))
        data['open'] = data['close'].shift(1).fillna(method='bfill')

    # Forward-fill missing data
    data = data.fillna(method='ffill')
    if data.empty:
        logger.error("Simulated data is empty after forward filling. Using fallback defaults.")
        return {
            "signal": 0,
            "atr": 1.0,
            "last_close": 100.0,
            "macd": 0.0,
            "bb_high": 101.0,
            "bb_low": 99.0
        }

    # Calculate technical indicators
    data['sma50'] = data['close'].rolling(window=50, min_periods=1).mean()
    data['rsi14'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
    data['atr'] = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14).average_true_range()
    macd_obj = ta.trend.MACD(close=data['close'])
    data['macd'] = macd_obj.macd()
    bb = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()

    # Use the last available row
    last = data.iloc[-1]
    if pd.isna(last['close']) or pd.isna(last['atr']):
        logger.error("Simulated data is empty or invalid after forward filling. Using fallback defaults.")
        return {
            "signal": 0,
            "atr": 1.0,
            "last_close": 100.0,
            "macd": 0.0,
            "bb_high": 101.0,
            "bb_low": 99.0
        }

    # Define basic signal logic:
    if (last['close'] > last['sma50']) and (last['rsi14'] < 70) and (last['macd'] > 0) and (last['close'] < last['bb_high']):
        signal = 1
    elif (last['close'] < last['sma50']) and (last['rsi14'] > 30) and (last['macd'] < 0) and (last['close'] > last['bb_low']):
        signal = -1
    else:
        signal = 0

    return {
        "signal": signal,
        "atr": last['atr'] if not pd.isna(last['atr']) else 1.0,
        "last_close": last['close'] if not pd.isna(last['close']) else 100.0,
        "macd": last['macd'] if not pd.isna(last['macd']) else 0.0,
        "bb_high": last['bb_high'] if not pd.isna(last['bb_high']) else 101.0,
        "bb_low": last['bb_low'] if not pd.isna(last['bb_low']) else 99.0
    }

# ---------------------------
# Risk Management and Position Sizing
# ---------------------------
def calculate_position_size(atr: float, risk_per_trade: float, account_balance: float) -> float:
    risk_amount = account_balance * risk_per_trade
    return risk_amount / atr if atr != 0 else 0

def check_risk(order_details: dict, account_balance: float, risk_per_trade: float) -> bool:
    order_value = order_details.get("price", 0) * order_details.get("quantity", 1)
    if order_value > account_balance * risk_per_trade:
        logger.error("Order risk too high.")
        return False
    return True

# ---------------------------
# Order Execution with Simulation/Paper Trading Support
# ---------------------------
class OrderExecutor:
    def __init__(self, ib, circuit_breaker=None):
        self.ib = ib
        self.circuit_breaker = circuit_breaker

    async def place_order(self, contract, order_details: dict, order_type="market"):
        account_balance = config["risk"]["account_balance"]
        risk_per_trade = config["risk"]["risk_per_trade"]
        if not check_risk(order_details, account_balance, risk_per_trade):
            raise Exception("Risk check failed.")

        if order_type == "market":
            order = MarketOrder("BUY", order_details.get("quantity", 1))
            order.lmtPrice = order_details.get("price", None)
        elif order_type == "stop":
            order = StopOrder("SELL", order_details.get("quantity", 1), order_details.get("price"))
        else:
            order = MarketOrder("BUY", order_details.get("quantity", 1))

        # Simulation / paper trading mode
        if config.get("simulation", True) or config.get("paper_trading", True):
            class DummyOrderStatus:
                def __init__(self):
                    self.status = "Submitted"
            class DummyOrder:
                def __init__(self):
                    self.orderId = random.randint(100000, 999999)
            class DummyTrade:
                def __init__(self):
                    self.order = DummyOrder()
                    self.orderStatus = DummyOrderStatus()
                    self.log = []
                async def waitOnUpdate(self, timeout=None):
                    await asyncio.sleep(random.uniform(1, 3))
                    self.orderStatus.status = "Filled"
                    return self
            trade = DummyTrade()
        else:
            trade = self.ib.placeOrder(contract, order)

        final_status = await self.wait_for_order_completion(trade)
        return trade

    async def wait_for_order_completion(self, trade, timeout: float = config["polling"]["timeout"],
                                        fallback_after: float = config["polling"]["fallback_threshold"]) -> str:
        terminal_statuses = {"Filled", "Cancelled", "Rejected", "Inactive"}
        start_time = time.time()
        while True:
            status = trade.orderStatus.status
            if status in terminal_statuses:
                break
            await asyncio.sleep(0.5)
            if time.time() - start_time > fallback_after:
                logger.warning(f"Order status stuck as '{status}' for {fallback_after} sec in simulation; forcing 'Filled'.")
                trade.orderStatus.status = "Filled"
                status = "Filled"
                break
        if status == "Rejected":
            raise Exception("Order rejected.")
        return status

    async def check_order_status(self, trade) -> dict:
        return {"order_id": trade.order.orderId, "status": trade.orderStatus.status}

# ---------------------------
# Main Workflow Integration
# ---------------------------
async def main():
    ib = IB()
    try:
        logger.info(f"Connecting to TWS at {config['tws']['host']}:{config['tws']['port']} with client ID {config['tws']['client_id']}")
        ib.connect(config["tws"]["host"], config["tws"]["port"], clientId=config["tws"]["client_id"])
        logger.info("Connected to TWS.")
    except Exception as e:
        logger.error(f"Error connecting to TWS: {e}")
        return

    # Create contract from config
    try:
        contract = Stock(config["trading"]["symbol"], config["trading"]["exchange"], config["trading"]["currency"])
    except KeyError as ke:
        logger.error(f"Configuration error: missing key {ke}. Please check your config.yaml.")
        ib.disconnect()
        return

    logger.info(f"Contract created: {contract}")

    signal_info = get_trade_signal()
    logger.debug(f"Signal: {signal_info['signal']}, ATR: {signal_info['atr']}, Last Price: {signal_info['last_close']}, "
                 f"MACD: {signal_info.get('macd', 0.0)}, BB High: {signal_info.get('bb_high', 0.0)}, BB Low: {signal_info.get('bb_low', 0.0)}")

    if signal_info["signal"] == 0:
        logger.info("No trading signal generated; no order placed.")
        ib.disconnect()
        return

    order_type = "market" if signal_info["signal"] == 1 else "stop"
    order_details = {
        "symbol": config["trading"]["symbol"],
        "quantity": config["trading"]["quantity"],
        "order_type": order_type,
        "price": signal_info["last_close"]
    }

    pos_size = calculate_position_size(signal_info["atr"], config["risk"]["risk_per_trade"], config["risk"]["account_balance"])
    logger.debug(f"Calculated position size: {pos_size:.2f}")

    executor = OrderExecutor(ib)
    try:
        trade = await executor.place_order(contract, order_details, order_type=order_type)
        status_response = await executor.check_order_status(trade)
        logger.info(f"Simulated order placed. ID:{status_response['order_id']}. Order final status: {status_response['status']}")
    except Exception as e:
        logger.error(f"Error during TWS order execution: {e}")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())



















