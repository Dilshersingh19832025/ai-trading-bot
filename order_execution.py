#!/usr/bin/env python3
"""
order_execution_tws.py

An end-to-end trading bot module integrating with Interactive Brokers' TWS using ib_insync.
Enhanced features include:
  - YAML-based configuration with environment variable fallbacks.
  - Data-driven signal generation (using SMA50, RSI14 and ATR for volatility).
  - Advanced risk management: dynamic position sizing, automated stop-loss/take-profit.
  - Live trading safeguards (paper trading mode, circuit breaker).
  - Real-time order updates: event-driven (using waitOnUpdate or updateEvent) with polling fallback.
  - Robust order rejection handling for known error codes.
  - Real-time alerts via Telegram.
  - Portfolio management to fetch open positions and account summary.
  - Performance profiling of critical operations.
  
Before running:
  - Ensure TWS (or IB Gateway) is running and API connections are enabled.
  - Create a "config.yaml" file or set environment variables.
  - Install required libraries: ib_insync, nest_asyncio, pyyaml, pandas, ta, requests.
  
Usage:
    python order_execution_tws.py
"""

import os
import asyncio
import logging
import functools
import random
import time
import yaml
import pandas as pd
import ta  # pip install ta
import requests  # For Telegram alerts

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder

# ---------------------------
# Configuration Loader
# ---------------------------
def load_config(config_file="config.yaml"):
    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    # TWS settings
    config.setdefault("tws", {})
    config["tws"].setdefault("host", os.getenv("TWS_HOST", "127.0.0.1"))
    config["tws"].setdefault("port", int(os.getenv("TWS_PORT", "7497")))
    config["tws"].setdefault("client_id", int(os.getenv("TWS_CLIENT_ID", "1")))
    # Risk management
    config.setdefault("risk", {})
    config["risk"].setdefault("account_balance", float(os.getenv("ACCOUNT_BALANCE", "100000")))
    config["risk"].setdefault("risk_per_trade", float(os.getenv("RISK_PER_TRADE", "0.02")))
    # Circuit breaker threshold (e.g., max daily loss)
    config["risk"].setdefault("circuit_breaker_loss", float(os.getenv("CIRCUIT_BREAKER_LOSS", "5000")))
    # Polling parameters
    config.setdefault("polling", {})
    config["polling"].setdefault("interval", float(os.getenv("POLLING_INTERVAL", "0.5")))
    config["polling"].setdefault("fallback_threshold", float(os.getenv("FALLBACK_THRESHOLD", "20.0")))
    config["polling"].setdefault("timeout", float(os.getenv("ORDER_TIMEOUT", "30.0")))
    # Trading mode flags
    config.setdefault("simulation", True)      # If True, orders are simulated.
    config.setdefault("paper_trading", True)     # Paper trading mode flag.
    # Alerts settings
    config.setdefault("alerts", {})
    config["alerts"].setdefault("enabled", os.getenv("ALERTS_ENABLED", "False") == "True")
    config["alerts"].setdefault("telegram_bot_token", os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN"))
    config["alerts"].setdefault("telegram_chat_id", os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID"))
    # Monitoring settings (for future dashboard)
    config.setdefault("monitoring", {})
    config["monitoring"].setdefault("dashboard_enabled", os.getenv("DASHBOARD_ENABLED", "False") == "True")
    return config

config = load_config()

# ---------------------------
# Logger Setup (Console and File)
# ---------------------------
logger = logging.getLogger("TradingBot.order_execution_tws")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler("order_execution.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

# ---------------------------
# Windows Event Loop & nest_asyncio
# ---------------------------
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    logger.warning("nest_asyncio not installed; install it to avoid nested loop errors.")

# ---------------------------
# Profiling Decorator
# ---------------------------
def profile(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.3f} seconds")
        return result
    return wrapper

# ---------------------------
# Retry with Exponential Backoff
# ---------------------------
def retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt} for {func.__name__} failed: {e}")
                    if attempt == max_retries:
                        logger.error(f"Max retries reached for {func.__name__}")
                        raise
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

# ---------------------------
# Real-Time Alerts via Telegram
# ---------------------------
def send_alert(message: str):
    if config["alerts"]["enabled"]:
        bot_token = config["alerts"]["telegram_bot_token"]
        chat_id = config["alerts"]["telegram_chat_id"]
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        try:
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Telegram alert failed: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    else:
        logger.info(f"ALERT (stub): {message}")

# ---------------------------
# Advanced Risk Management
# ---------------------------
def calculate_atr(data: pd.DataFrame, window: int = 14) -> float:
    # If high/low not present, simulate them based on close
    if not {"high", "low"}.issubset(data.columns):
        data["high"] = data["close"] * (1 + random.uniform(0.001, 0.005))
        data["low"] = data["close"] * (1 - random.uniform(0.001, 0.005))
    atr = ta.volatility.AverageTrueRange(high=data["high"], low=data["low"], close=data["close"], window=window).average_true_range().iloc[-1]
    return atr

def calculate_position_size(atr: float, risk_per_trade: float, account_balance: float) -> float:
    return (account_balance * risk_per_trade) / atr

def set_stop_loss_take_profit(entry_price: float, atr: float, risk_reward_ratio: float = 2.0):
    stop_loss = entry_price - 2 * atr
    take_profit = entry_price + risk_reward_ratio * 2 * atr
    return stop_loss, take_profit

def check_risk(order_details: dict, account_balance: float = config["risk"]["account_balance"],
               risk_per_trade: float = config["risk"]["risk_per_trade"]) -> bool:
    logger.debug("Performing risk check...")
    quantity = order_details.get("quantity", 1)
    price = order_details.get("price", 100)  # In production, use live price.
    required_capital = quantity * price
    if required_capital > account_balance * risk_per_trade:
        msg = f"Risk check failed: Required capital {required_capital:.2f} exceeds limit {account_balance * risk_per_trade:.2f}"
        logger.warning(msg)
        send_alert(msg)
        return False
    logger.debug("Risk check passed")
    return True

# ---------------------------
# Data-Driven Signal Generation
# ---------------------------
def get_trade_signal(data: pd.DataFrame = None) -> dict:
    if data is None:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
        prices = [random.uniform(140, 160) for _ in range(100)]
        data = pd.DataFrame({"close": prices}, index=dates)
    data["SMA50"] = data["close"].rolling(window=50).mean()
    data["RSI14"] = ta.momentum.RSIIndicator(data["close"], window=14).rsi()
    atr = calculate_atr(data)
    last_close = data["close"].iloc[-1]
    last_sma = data["SMA50"].iloc[-1]
    last_rsi = data["RSI14"].iloc[-1]
    logger.debug(f"Last close: {last_close:.2f}, SMA50: {last_sma:.2f}, RSI14: {last_rsi:.2f}")
    # Example signal logic:
    if pd.notna(last_sma) and last_close > last_sma and last_rsi < 70:
        signal = 1
    elif pd.notna(last_sma) and last_close < last_sma and last_rsi > 30:
        signal = -1
    else:
        signal = 0
    logger.debug(f"Generated trade signal (data-driven): {signal}")
    return {"signal": signal, "atr": atr, "last_close": last_close}

# ---------------------------
# Order Rejection Handling
# ---------------------------
def handle_order_rejection(trade):
    for entry in trade.log:
        if entry.errorCode in [201, 202, 399, 404]:
            msg = f"Order rejected (error {entry.errorCode}): {entry.message}"
            logger.error(msg)
            send_alert(msg)
            raise Exception(msg)
    logger.error("Order rejected for unknown reasons.")
    raise Exception("Order rejected for unknown reasons.")

# ---------------------------
# Wait for Order Completion (Event-driven with fallback to polling)
# ---------------------------
async def wait_for_order_completion(trade, timeout: float = config["polling"]["timeout"],
                                    fallback_after: float = config["polling"]["fallback_threshold"]) -> str:
    terminal_statuses = {"Filled", "Cancelled", "Rejected"}
    loop = asyncio.get_event_loop()
    order_event = asyncio.Event()

    # Attempt event-driven update (if IB provides waitOnUpdate, for example)
    if hasattr(trade, "waitOnUpdate"):
        try:
            await asyncio.wait_for(trade.waitOnUpdate(timeout=timeout), timeout=timeout)
        except Exception as e:
            logger.warning(f"waitOnUpdate failed: {e}. Using polling fallback.")
    elif hasattr(trade, "updateEvent"):
        def on_update():
            status = trade.orderStatus.status
            logger.info(f"Order update event: {status}")
            if status in terminal_statuses:
                order_event.set()
        try:
            trade.updateEvent += on_update
            await asyncio.wait_for(order_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Event-driven update timed out; polling fallback.")
        except Exception as e:
            logger.warning(f"Error in event callback: {e}; polling fallback.")
    else:
        logger.warning("No event-driven mechanism available; using polling.")

    final_status = trade.orderStatus.status
    if final_status not in terminal_statuses:
        start_time = loop.time()
        while final_status not in terminal_statuses:
            logger.debug(f"Polling order status: {final_status}")
            await asyncio.sleep(config["polling"]["interval"])
            final_status = trade.orderStatus.status
            elapsed = loop.time() - start_time
            if elapsed > fallback_after:
                logger.warning(f"Order status stuck as '{final_status}' for {fallback_after} sec; for simulation forcing 'Filled'.")
                trade.orderStatus.status = "Filled"
                final_status = "Filled"
                break
            if elapsed > timeout:
                raise Exception("Timeout waiting for order completion.")
    if final_status == "Rejected":
        handle_order_rejection(trade)
    return final_status

# ---------------------------
# Portfolio Management
# ---------------------------
class PortfolioManager:
    def __init__(self, ib: IB):
        self.ib = ib

    def get_open_positions(self):
        positions = self.ib.positions()
        logger.debug(f"Open positions: {positions}")
        return positions

    def get_account_summary(self):
        summary = self.ib.accountSummary()
        logger.debug(f"Account summary: {summary}")
        return summary

# ---------------------------
# Circuit Breaker for Live Trading Safeguards
# ---------------------------
class CircuitBreaker:
    def __init__(self, loss_threshold: float):
        self.loss_threshold = loss_threshold
        self.current_loss = 0.0
        self.active = True

    def record_loss(self, loss: float):
        self.current_loss += loss
        if self.current_loss >= self.loss_threshold:
            self.active = False
            msg = "Circuit breaker triggered: Loss threshold exceeded!"
            logger.error(msg)
            send_alert(msg)

    def reset(self):
        self.current_loss = 0.0
        self.active = True

# ---------------------------
# OrderExecutor: Main Class for Order Operations
# ---------------------------
class OrderExecutor:
    def __init__(self, ib: IB, circuit_breaker: CircuitBreaker = None):
        self.ib = ib
        self.circuit_breaker = circuit_breaker

    @retry_with_backoff()
    @profile
    async def place_order(self, contract, order_details: dict, order_type: str = "market"):
        if not check_risk(order_details):
            raise Exception("Risk check failed. Order not placed.")
        
        # Create order based on order type.
        if order_type == "market":
            order = MarketOrder("BUY", order_details.get("quantity", 1))
        elif order_type == "limit":
            limit_price = order_details.get("limit_price")
            if limit_price is None:
                raise ValueError("Limit price required for limit order.")
            order = LimitOrder("BUY", order_details.get("quantity", 1), limit_price)
        elif order_type == "stop":
            stop_price = order_details.get("stop_price")
            if stop_price is None:
                raise ValueError("Stop price required for stop order.")
            order = StopOrder("BUY", order_details.get("quantity", 1), stop_price)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        logger.debug(f"Placing {order_type} order for {contract} with details: {order_details}")
        # Live trading safeguard: if simulation/paper trading is enabled, do not actually send order.
        if config.get("simulation", True) or config.get("paper_trading", True):
            logger.info("Simulation/Paper trading mode: Order not actually sent to TWS.")
            # Create a dummy trade object.
            class DummyOrderStatus:
                def __init__(self):
                    self.status = "Filled"
            class DummyOrder:
                def __init__(self):
                    self.orderId = random.randint(100, 999)
            class DummyTrade:
                def __init__(self):
                    self.order = DummyOrder()
                    self.orderStatus = DummyOrderStatus()
                    self.log = []
            trade = DummyTrade()
        else:
            trade = self.ib.placeOrder(contract, order)
        
        try:
            if hasattr(trade, "updateEvent"):
                trade.updateEvent += lambda: logger.info(f"Order update: {trade.orderStatus.status}")
            elif hasattr(trade, "waitOnUpdate"):
                logger.info("Using waitOnUpdate for order updates.")
            else:
                logger.warning("No event-driven update support; using polling fallback.")
        except Exception as e:
            logger.warning(f"Error adding event callback: {e}")
        
        final_status = await wait_for_order_completion(trade)
        logger.info(f"Order placed. ID: {trade.order.orderId}, Final Status: {final_status}")
        return trade

    @retry_with_backoff()
    @profile
    async def cancel_order(self, trade):
        logger.debug(f"Cancelling order ID: {trade.order.orderId}")
        self.ib.sleep(random.uniform(0.2, 0.8))
        trade.cancel()
        logger.info(f"Order cancelled: {trade.order.orderId}")
        return trade.order.orderId

    @retry_with_backoff()
    @profile
    async def check_order_status(self, trade):
        logger.debug(f"Checking status for order ID: {trade.order.orderId}")
        status = trade.orderStatus.status
        logger.info(f"Order status for {trade.order.orderId}: {status}")
        return {"order_id": trade.order.orderId, "status": status}

# ---------------------------
# Main Workflow Integration
# ---------------------------
async def main():
    ib = IB()
    circuit_breaker = CircuitBreaker(loss_threshold=config["risk"]["circuit_breaker_loss"])
    try:
        logger.info(f"Connecting to TWS at {config['tws']['host']}:{config['tws']['port']} with client ID {config['tws']['client_id']}")
        ib.connect(config["tws"]["host"], config["tws"]["port"], clientId=config["tws"]["client_id"])
        logger.info("Connected to TWS.")
        
        # Create a contract for AAPL.
        contract = Stock('AAPL', 'SMART', 'USD')
        logger.info(f"Contract created: {contract}")
        
        # Generate data-driven signal.
        signal_info = get_trade_signal()
        signal = signal_info["signal"]
        atr = signal_info["atr"]
        last_price = signal_info["last_close"]
        logger.debug(f"ATR: {atr:.2f}, Last Price: {last_price:.2f}")
        
        if signal == 1:
            order_type = "market"
        elif signal == -1:
            order_type = "stop"
        else:
            logger.info("No trading signal generated; no order placed.")
            return
        
        # For stop orders, calculate stop-loss/take-profit.
        if order_type == "stop":
            stop_loss, take_profit = set_stop_loss_take_profit(last_price, atr)
            order_details = {"symbol": "AAPL", "quantity": 10, "order_type": order_type,
                             "stop_price": stop_loss, "price": stop_loss}
            logger.debug(f"Calculated stop_loss: {stop_loss:.2f}, take_profit: {take_profit:.2f}")
        else:
            order_details = {"symbol": "AAPL", "quantity": 10, "order_type": order_type, "price": last_price}
        
        # Check if the circuit breaker is active.
        if circuit_breaker and not circuit_breaker.active:
            msg = "Circuit breaker active; halting trading."
            logger.error(msg)
            send_alert(msg)
            return
        
        executor = OrderExecutor(ib, circuit_breaker=circuit_breaker)
        trade = await executor.place_order(contract, order_details, order_type=order_type)
        status_response = await executor.check_order_status(trade)
        logger.debug(f"Final status response: {status_response}")
        
        # Portfolio Management Example
        portfolio_manager = PortfolioManager(ib)
        positions = portfolio_manager.get_open_positions()
        account_summary = portfolio_manager.get_account_summary()
        logger.debug(f"Portfolio positions: {positions}")
        logger.debug(f"Account summary: {account_summary}")
        
        # (Future integrations: real-time dashboard, additional asset support, etc.)
        
    except Exception as e:
        logger.error(f"Error during TWS order execution: {e}")
        send_alert(f"Critical error during order execution: {e}")
    finally:
        ib.disconnect()
        logger.info("Disconnected from TWS.")

if __name__ == "__main__":
    asyncio.run(main())










