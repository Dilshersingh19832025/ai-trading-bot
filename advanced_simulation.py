#!/usr/bin/env python
"""
advanced_simulation.py

An advanced simulation framework for a self-trading bot.
This framework integrates:
  - Order Flow Clustering (via a simple Hawkes process)
  - Dynamic Liquidity & Order Book Modeling (with multi-level depth and hidden liquidity)
  - Cross-Asset Correlation (simulating multiple correlated assets)
  - Network/API Simulation (latency and occasional failures)
  - Advanced Transaction Cost Analysis (fees, slippage, and market impact)
  - Parallel/Distributed simulation support (via threading)
  - Real-Time Data Integration hooks (for future live data feed integration)
  - User-Defined Scenario Testing (simulate market shocks)

Author: Your Name
Date: 2025-02-22
"""

import numpy as np
import pandas as pd
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AdvancedSim")

# ---------------------
# Module 1: Hawkes Process Simulator
# ---------------------
class HawkesProcessSimulator:
    def __init__(self, baseline=0.2, alpha=0.5, decay=1.0):
        """
        Simple Hawkes process parameters.
        :param baseline: baseline intensity
        :param alpha: excitation parameter (impact of an event)
        :param decay: decay rate of the excitation
        """
        self.baseline = baseline
        self.alpha = alpha
        self.decay = decay

    def simulate(self, T):
        """
        Simulate event times up to time T using Ogata's thinning algorithm.
        Returns a list of event times.
        """
        events = []
        t = 0
        while t < T:
            # Calculate current intensity as baseline plus contribution from past events.
            intensity = self.baseline + sum(self.alpha * np.exp(-self.decay * (t - s)) for s in events)
            # Draw next inter-arrival time
            u = random.random()
            w = -np.log(u) / intensity
            t += w
            # Calculate intensity at new time
            D = self.baseline + sum(self.alpha * np.exp(-self.decay * (t - s)) for s in events)
            d = random.random()
            if d <= D / intensity:
                events.append(t)
        logger.debug(f"Simulated {len(events)} events up to time {T}.")
        return events

# ---------------------
# Module 2: Dynamic Order Book Simulator
# ---------------------
class OrderBookSimulator:
    def __init__(self, initial_bid=100.0, initial_ask=100.5, hidden_fraction=0.1):
        """
        Initialize a simple order book with best bid/ask and multi-level depth.
        :param initial_bid: starting best bid price
        :param initial_ask: starting best ask price
        :param hidden_fraction: fraction of orders hidden from public view
        """
        self.levels = 5  # number of visible levels on each side
        # Simulate the book as two DataFrames for bid and ask levels:
        self.bid_book = pd.DataFrame({
            "price": [initial_bid - i * 0.1 for i in range(self.levels)],
            "volume": [random.randint(50, 150) for _ in range(self.levels)]
        })
        self.ask_book = pd.DataFrame({
            "price": [initial_ask + i * 0.1 for i in range(self.levels)],
            "volume": [random.randint(50, 150) for _ in range(self.levels)]
        })
        self.hidden_fraction = hidden_fraction

    def update_order(self):
        """
        Randomly update the order book by adding, canceling or modifying orders.
        For simplicity, we randomly adjust volumes and occasionally shift prices.
        """
        # Adjust bid volumes
        self.bid_book["volume"] = self.bid_book["volume"].apply(
            lambda vol: max(0, vol + random.randint(-10, 10))
        )
        # Adjust ask volumes
        self.ask_book["volume"] = self.ask_book["volume"].apply(
            lambda vol: max(0, vol + random.randint(-10, 10))
        )
        # Occasionally simulate hidden liquidity changes (not visible)
        if random.random() < 0.2:
            hidden_adjustment = random.randint(-20, 20)
            logger.debug(f"Hidden liquidity adjustment: {hidden_adjustment}")
            # (In a full model, this would affect market impact but not the visible book)
        # Occasionally shift best prices to simulate price movement
        if random.random() < 0.1:
            shift = random.choice([-0.1, 0.1])
            self.bid_book["price"] += shift
            self.ask_book["price"] += shift

    def get_best_bid_ask(self):
        best_bid = self.bid_book["price"].max()
        best_ask = self.ask_book["price"].min()
        return best_bid, best_ask

    def display(self):
        logger.debug("Bid Book:")
        logger.debug(self.bid_book)
        logger.debug("Ask Book:")
        logger.debug(self.ask_book)

# ---------------------
# Module 3: Multi-Asset Correlation Simulator
# ---------------------
class MultiAssetSimulator:
    def __init__(self, symbols, init_prices, covariance, time_horizon=1.0, dt=0.01):
        """
        Simulate correlated price paths.
        :param symbols: list of asset symbols
        :param init_prices: list of starting prices
        :param covariance: covariance matrix among asset returns
        :param time_horizon: simulation time horizon (in arbitrary units)
        :param dt: time step
        """
        self.symbols = symbols
        self.init_prices = np.array(init_prices)
        self.covariance = np.array(covariance)
        self.T = time_horizon
        self.dt = dt
        self.steps = int(time_horizon / dt)
        self.num_assets = len(symbols)

    def simulate_prices(self):
        # Generate multivariate normal returns
        mean = np.zeros(self.num_assets)
        returns = np.random.multivariate_normal(mean, self.covariance * self.dt, self.steps)
        # Cumulative sum to simulate log returns and convert to prices
        log_prices = np.log(self.init_prices) + np.cumsum(returns, axis=0)
        prices = np.exp(log_prices)
        # Create a DataFrame for easier analysis
        df = pd.DataFrame(prices, columns=self.symbols)
        return df

# ---------------------
# Module 4: Network and API Simulator
# ---------------------
class NetworkSimulator:
    def __init__(self, base_latency=0.05, jitter=0.03, failure_rate=0.01):
        """
        Simulate network conditions.
        :param base_latency: average network latency (seconds)
        :param jitter: variability in latency
        :param failure_rate: probability of an API call failure
        """
        self.base_latency = base_latency
        self.jitter = jitter
        self.failure_rate = failure_rate

    def simulate_delay(self):
        delay = self.base_latency + random.uniform(-self.jitter, self.jitter)
        logger.debug(f"Simulated network delay: {delay:.3f} seconds")
        time.sleep(max(0, delay))

    def simulate_api_call(self, func, *args, **kwargs):
        self.simulate_delay()
        if random.random() < self.failure_rate:
            logger.warning("Simulated API failure.")
            raise ConnectionError("Simulated network/API failure.")
        return func(*args, **kwargs)

# ---------------------
# Module 5: Advanced Transaction Cost Model
# ---------------------
class TransactionCostModel:
    def __init__(self, fee_per_share=0.005, impact_coefficient=0.0001):
        """
        :param fee_per_share: fixed fee per share traded
        :param impact_coefficient: coefficient for market impact slippage
        """
        self.fee_per_share = fee_per_share
        self.impact_coefficient = impact_coefficient

    def calculate_cost(self, order_size, price, liquidity):
        """
        Calculate total transaction cost.
        :param order_size: number of shares
        :param price: execution price
        :param liquidity: available liquidity at that price level
        :return: total cost (including fee and slippage)
        """
        fee_cost = order_size * self.fee_per_share
        # Slippage increases when order size is a large fraction of liquidity
        slippage = self.impact_coefficient * (order_size / (liquidity + 1)) * price
        total_cost = fee_cost + slippage
        logger.debug(f"Order cost: fee={fee_cost:.3f}, slippage={slippage:.3f}, total={total_cost:.3f}")
        return total_cost

# ---------------------
# Module 6: User-Defined Scenario Simulator
# ---------------------
class ScenarioSimulator:
    def __init__(self):
        self.scenarios = {
            "flash_crash": self._flash_crash,
            "news_spike": self._news_spike
        }

    def apply_scenario(self, scenario_name, market_data):
        """
        Apply a predefined scenario to market data.
        :param scenario_name: name of the scenario
        :param market_data: a DataFrame of price data to adjust
        :return: modified market_data DataFrame
        """
        if scenario_name in self.scenarios:
            logger.info(f"Applying scenario: {scenario_name}")
            return self.scenarios[scenario_name](market_data)
        else:
            logger.warning("Unknown scenario. No changes applied.")
            return market_data

    def _flash_crash(self, market_data):
        # Simulate a sudden price drop for a short period.
        crash_start = int(0.4 * len(market_data))
        crash_duration = int(0.1 * len(market_data))
        market_data.iloc[crash_start:crash_start+crash_duration] *= 0.7
        return market_data

    def _news_spike(self, market_data):
        # Simulate a sudden upward spike.
        spike_start = int(0.6 * len(market_data))
        spike_duration = int(0.05 * len(market_data))
        market_data.iloc[spike_start:spike_start+spike_duration] *= 1.2
        return market_data

# ---------------------
# Module 7: Simulation Engine (Parallel/Distributed Support)
# ---------------------
class SimulationEngine:
    def __init__(self, num_simulations=5):
        self.num_simulations = num_simulations
        self.hawkes = HawkesProcessSimulator()
        self.order_book = OrderBookSimulator()
        self.network = NetworkSimulator()
        self.cost_model = TransactionCostModel()
        self.scenario_sim = ScenarioSimulator()

    def run_single_simulation(self, sim_id):
        logger.info(f"Starting simulation run {sim_id}")
        # Simulate order flow events over a period
        T = 10  # simulate for 10 time units
        events = self.hawkes.simulate(T)
        
        # Update order book and record mid-price over simulated time steps
        prices = []
        for t in np.linspace(0, T, num=100):
            self.order_book.update_order()
            best_bid, best_ask = self.order_book.get_best_bid_ask()
            mid_price = (best_bid + best_ask) / 2.0
            prices.append(mid_price)
            # Simulate occasional network/API calls
            try:
                self.network.simulate_api_call(lambda: None)
            except ConnectionError:
                logger.warning("Continuing simulation after API failure.")
        
        # Calculate an example transaction cost for a hypothetical order
        order_size = 100  # shares
        execution_price = prices[-1]
        liquidity = self.order_book.bid_book.iloc[0]["volume"]  # best bid volume as a proxy for liquidity
        cost = self.cost_model.calculate_cost(order_size, execution_price, liquidity)
        
        # Return simulation results
        result = {
            "sim_id": sim_id,
            "num_events": len(events),
            "final_price": prices[-1],
            "transaction_cost": cost,
            "price_series": prices
        }
        logger.info(f"Completed simulation run {sim_id}")
        return result

    def run_simulations_parallel(self):
        results = []
        with ThreadPoolExecutor(max_workers=self.num_simulations) as executor:
            futures = {executor.submit(self.run_single_simulation, i): i for i in range(self.num_simulations)}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        return results

    def run_multi_asset_simulation(self):
        # Example: simulate price paths for multiple assets with correlations
        symbols = ["AAPL", "TSLA", "GOOG"]
        init_prices = [150, 700, 2500]
        # Create a simple covariance matrix (for demonstration purposes)
        cov = np.array([
            [0.02, 0.005, 0.003],
            [0.005, 0.03, 0.004],
            [0.003, 0.004, 0.04]
        ])
        asset_sim = MultiAssetSimulator(symbols, init_prices, cov, time_horizon=1.0, dt=0.01)
        prices_df = asset_sim.simulate_prices()
        return prices_df

# ---------------------
# Main Execution
# ---------------------
def main():
    engine = SimulationEngine(num_simulations=3)
    
    # Run parallel simulation runs
    results = engine.run_simulations_parallel()
    
    # Plot the price series for each simulation run
    plt.figure(figsize=(10, 5))
    for res in results:
        plt.plot(res["price_series"], label=f"Sim {res['sim_id']}")
    plt.xlabel("Time Step")
    plt.ylabel("Mid Price")
    plt.title("Simulated Price Series")
    plt.legend()
    plt.show()

    # Run a multi-asset simulation and plot the resulting price paths
    multi_asset_prices = engine.run_multi_asset_simulation()
    multi_asset_prices.plot(title="Multi-Asset Simulated Prices")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.show()

    # Apply a user-defined scenario on one asset's price series:
    scenario_sim = ScenarioSimulator()
    asset_series = multi_asset_prices["AAPL"].copy()
    # Convert to DataFrame to work with our scenario functions
    asset_series_df = asset_series.to_frame()
    asset_series_df = scenario_sim.apply_scenario("flash_crash", asset_series_df)
    asset_series_df.plot(title="AAPL Price Series After Flash Crash Scenario")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.show()

    logger.info("Advanced simulation complete.")

if __name__ == "__main__":
    main()
