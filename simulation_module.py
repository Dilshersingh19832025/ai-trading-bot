#!/usr/bin/env python
"""
simulation_module.py

This module simulates a limit order book with enhanced features:
 - Multiple order types: 'limit', 'stop', and 'iceberg'
 - Improved order matching with a deeper order book
 - High-frequency tick data simulation
 - Realistic latency with jitter (variable delays)
 - Event-driven simulation (predictable events and sentiment-driven random shocks)
 - Basic unit-test functions to verify module behavior

Run with:
    python simulation_module.py
"""

import numpy as np
import pandas as pd
import random
import time
import logging
from datetime import datetime
from collections import deque

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Simulation Configuration
# ---------------------------
# Order arrival (Poisson mean orders per second)
ORDER_ARRIVAL_RATE = 5  

# Latency configuration (simulate jitter)
LATENCY_MIN = 0.005  # seconds
LATENCY_MAX = 0.05   # seconds

# Slippage factor for market impact (relative)
SLIPPAGE_FACTOR = 0.001  

# Define a base price for simulation purposes
BASE_PRICE = 100.0

# Define event templates (predictable and random sentiment events)
EVENT_TEMPLATES = [
    {"type": "earnings", "impact": 0.02, "description": "Earnings report released"},
    {"type": "interest_rate", "impact": -0.01, "description": "Interest rate change"},
    {"type": "geopolitical", "impact": -0.03, "description": "Geopolitical shock"},
    {"type": "merger", "impact": 0.03, "description": "Merger announcement"},
]

# ---------------------------
# Order and LOB Definitions
# ---------------------------
class Order:
    def __init__(self, order_id, side, price, quantity, order_type='limit', **kwargs):
        """
        order_type can be 'limit', 'stop', or 'iceberg'.
        For iceberg orders, provide additional parameter:
            hidden_qty: quantity hidden from the public book.
        """
        self.order_id = order_id
        self.side = side  # 'bid' or 'ask'
        self.price = price
        self.quantity = quantity  # total quantity
        self.order_type = order_type
        self.timestamp = time.time()
        self.hidden_qty = kwargs.get('hidden_qty', 0)  # for iceberg orders

    def __repr__(self):
        return (f"Order(id={self.order_id}, side={self.side}, type={self.order_type}, "
                f"price={self.price:.2f}, qty={self.quantity}, hidden={self.hidden_qty})")

class LimitOrderBook:
    def __init__(self, depth=10):
        # Use separate deques for bids and asks; orders will be sorted by price.
        self.bids = deque()  # sorted descending by price
        self.asks = deque()  # sorted ascending by price
        self.depth = depth

    def add_order(self, order: Order):
        if order.side == 'bid':
            self._insert_order(self.bids, order, reverse=True)
        else:
            self._insert_order(self.asks, order, reverse=False)
        logging.debug(f"Added {order}")

    def _insert_order(self, book: deque, order: Order, reverse=False):
        """
        Insert order into the deque keeping the order sorted by price.
        For bids, higher prices come first (reverse=True). For asks, lower prices first.
        """
        inserted = False
        for idx, existing in enumerate(book):
            if (reverse and order.price > existing.price) or (not reverse and order.price < existing.price):
                book.insert(idx, order)
                inserted = True
                break
        if not inserted:
            book.append(order)
        # Optionally enforce maximum depth:
        if len(book) > self.depth:
            book.pop()

    def match_orders(self):
        """
        Check if best bid and best ask can be matched.
        For simplicity, match orders when best bid >= best ask.
        Execute at midpoint price.
        """
        if self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            if best_bid.price >= best_ask.price:
                # Determine trade quantity (minimum of available quantities)
                trade_qty = min(best_bid.quantity, best_ask.quantity)
                trade_price = (best_bid.price + best_ask.price) / 2
                # Reduce quantities
                best_bid.quantity -= trade_qty
                best_ask.quantity -= trade_qty
                logging.info(f"Trade executed: {trade_qty} @ {trade_price:.2f} between Bid {best_bid.order_id} and Ask {best_ask.order_id}")
                # Remove orders if fully filled
                if best_bid.quantity <= 0:
                    self.bids.popleft()
                if best_ask.quantity <= 0:
                    self.asks.popleft()
                return trade_price, trade_qty
        return None

# ---------------------------
# Simulation Functions
# ---------------------------
def simulate_latency():
    """Simulate realistic network latency with jitter."""
    delay = random.uniform(LATENCY_MIN, LATENCY_MAX)
    time.sleep(delay)
    return delay

def calculate_slippage(base_price, quantity):
    """Calculate slippage as a function of base price and quantity."""
    slippage = base_price * SLIPPAGE_FACTOR * np.log1p(quantity)
    return slippage

def generate_market_event():
    """Randomly generate a market event with a certain probability."""
    # 10% chance for an event each second
    if random.random() < 0.1:
        event = random.choice(EVENT_TEMPLATES).copy()
        # Randomize impact a bit
        event['impact'] *= random.uniform(0.8, 1.2)
        event['timestamp'] = time.time()
        logging.info(f"Market event triggered: {event['description']} with impact {event['impact']:.2%}")
        return event
    return None

def validate_simulation_data(df: pd.DataFrame):
    """Basic data validation: check if DataFrame is empty or has nulls."""
    if df.empty or df.isnull().sum().sum() > 0:
        logging.error("Simulated data is empty or contains nulls. Using fallback defaults.")
        return False
    return True

def simulate_tick_data():
    """
    Simulate a tick: generate a random price movement based on a small volatility.
    For high-frequency simulation, returns a price delta.
    """
    tick_volatility = 0.05  # adjust for tick-level movements
    return np.random.randn() * tick_volatility

# ---------------------------
# Main Simulation Routine
# ---------------------------
def run_simulation(duration=10):
    """
    Run the simulation for the specified duration (in seconds).
    """
    lob = LimitOrderBook(depth=20)  # deeper book now
    start_time = time.time()
    simulated_trades = []  # list to hold trade details
    order_id = 1

    # Main simulation loop
    while time.time() - start_time < duration:
        # Simulate order arrivals: use tick-level simulation (very high frequency)
        num_orders = np.random.poisson(ORDER_ARRIVAL_RATE)
        for _ in range(num_orders):
            # Randomly choose order type
            order_type = random.choices(
                ['limit', 'stop', 'iceberg'],
                weights=[0.7, 0.2, 0.1],
                k=1
            )[0]
            side = random.choice(['bid', 'ask'])
            # Base price plus tick-level variation
            price = BASE_PRICE + simulate_tick_data() + random.uniform(-1, 1)
            quantity = random.randint(1, 100)
            # For iceberg orders, set a hidden quantity (e.g., 30% hidden)
            extra = {}
            if order_type == 'iceberg':
                extra['hidden_qty'] = int(quantity * 0.3)
            order = Order(order_id, side, price, quantity, order_type, **extra)
            lob.add_order(order)
            order_id += 1

        # Attempt to match orders
        trade = lob.match_orders()
        if trade:
            trade_price, trade_qty = trade
            latency = simulate_latency()
            slippage = calculate_slippage(trade_price, trade_qty)
            # Adjust executed price with slippage: random direction (buy/sell spread effect)
            executed_price = trade_price + (slippage if random.choice([True, False]) else -slippage)
            simulated_trades.append({
                "timestamp": datetime.now(),
                "price": executed_price,
                "quantity": trade_qty,
                "latency": latency,
                "slippage": slippage
            })

        # Process a market event if any
        event = generate_market_event()
        if event:
            # Apply event impact to all simulated trades so far (as a simple adjustment)
            for trade_entry in simulated_trades:
                trade_entry["price"] *= (1 + event["impact"])

        # Short sleep to mimic real-time tick processing
        time.sleep(0.01)

    # Convert to DataFrame and validate
    df = pd.DataFrame(simulated_trades)
    if not validate_simulation_data(df):
        df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "price": [BASE_PRICE],
            "quantity": [0],
            "latency": [np.mean([LATENCY_MIN, LATENCY_MAX])],
            "slippage": [0]
        })
    return df

# ---------------------------
# Unit Testing Functions
# ---------------------------
def unit_test_lob():
    """Basic unit test for the LOB order insertion and matching."""
    logging.info("Running unit tests for LOB...")
    test_lob = LimitOrderBook(depth=5)
    # Create orders
    orders = [
        Order(1, 'bid', 100, 50),
        Order(2, 'bid', 101, 40),
        Order(3, 'ask', 102, 30),
        Order(4, 'ask', 99, 20),  # deliberately low ask to force a match
    ]
    for order in orders:
        test_lob.add_order(order)
    trade = test_lob.match_orders()
    assert trade is not None, "Test failed: No trade executed when there should be a match."
    logging.info("LOB unit test passed.")

def unit_test_event():
    """Test that market events are generated correctly."""
    events = [generate_market_event() for _ in range(100)]
    event_count = sum(1 for e in events if e is not None)
    logging.info(f"Unit test: Generated {event_count} events out of 100 ticks.")
    assert event_count > 0, "Event generator not producing any events."
    logging.info("Event unit test passed.")

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    logging.info("Starting Phase 1 Simulation Enhancements...")
    # Run unit tests
    unit_test_lob()
    unit_test_event()

    # Run the simulation
    simulation_df = run_simulation(duration=10)
    logging.info("Simulation complete. Sample of simulated trade data:")
    logging.info(simulation_df.head())
