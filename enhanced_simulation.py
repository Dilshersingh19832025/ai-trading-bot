"""
File: enhanced_simulation.py

This simulation framework is designed as a next‐generation self‐trading bot
testbed. It incorporates:
  • Advanced synthetic price generation via GBM.
  • A multi‐level limit order book with event‐driven updates.
  • A Hawkes process for clustered order arrivals.
  • A network simulator with latency and random API failures.
  • Multiple agent types (market maker, momentum, noise, RL‐agent).
  • A rudimentary risk model that tracks price history and computes intraday VaR.
  • A transaction cost model with fees and market impact.
  
This code is a detailed “blueprint”—each module is more fully implemented than a stub,
but you’ll likely need to further refine models and parameters before live trading.
"""

import numpy as np
import random
import time
import logging
from typing import Dict, List, Any

# -------------------- Logging Configuration ---------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedSimulation")

# -------------------- Simulation Configuration ---------------------------
class SimulationConfig:
    """Central configuration for the simulation environment."""
    SIMULATION_TIME = 3600      # seconds (e.g., one trading hour)
    TIME_STEP = 1.0             # simulation time step in seconds
    NUM_LEVELS = 5              # track top 5 levels on each side
    HAWKES_BASE_INTENSITY = 0.2
    HAWKES_ALPHA = 0.5
    HAWKES_DECAY = 0.8
    API_FAILURE_RATE = 0.01     # probability of simulated API failure per check
    MAX_HIDDEN_LIQUIDITY = 20   # parameter for hidden liquidity simulation

# -------------------- Synthetic Data Generation ---------------------------
class SyntheticDataGenerator:
    """
    Generates synthetic price paths using a Geometric Brownian Motion (GBM) model.
    In production you might use GANs or diffusion models.
    """
    def __init__(self, method: str = "GBM"):
        self.method = method

    def generate_price_series(self, length: int = 1000, dt: float = 1/252,
                              mu: float = 0.1, sigma: float = 0.2) -> List[float]:
        logger.debug(f"Generating synthetic price series using {self.method} method (GBM).")
        prices = [100.0]
        for i in range(1, length):
            drift = (mu - 0.5 * sigma ** 2) * dt
            shock = sigma * np.sqrt(dt) * np.random.normal()
            new_price = prices[-1] * np.exp(drift + shock)
            prices.append(new_price)
        return prices

# -------------------- Multi-Level Order Book ---------------------------
class MultiLevelOrderBook:
    """
    Simulates a multi-level limit order book (LOB) with adjustable liquidity.
    Supports new orders, cancellations, and trades.
    """
    def __init__(self, symbol: str, num_levels: int):
        self.symbol = symbol
        self.num_levels = num_levels
        self.bids: List[Dict[str, float]] = []  # list of dicts with keys "price", "size"
        self.asks: List[Dict[str, float]] = []
        self.mid_price = 100.0

    def initialize_book(self, initial_price: float):
        self.mid_price = initial_price
        self.bids = []
        self.asks = []
        for i in range(self.num_levels):
            bid_price = initial_price - 0.01 * (i + 1)
            ask_price = initial_price + 0.01 * (i + 1)
            self.bids.append({"price": bid_price, "size": 100})
            self.asks.append({"price": ask_price, "size": 100})
        # Ensure bids are sorted descending, asks ascending
        self.bids.sort(key=lambda x: -x["price"])
        self.asks.sort(key=lambda x: x["price"])
        logger.debug(f"[LOB] Initialized for {self.symbol} with mid_price: {initial_price:.2f}")

    def update_for_event(self, event: Dict[str, Any]):
        """
        Process an event that changes the order book.
        Event types: "new_order", "cancel_order", "trade".
        """
        event_type = event.get("type")
        if event_type == "new_order":
            side = event.get("side")
            price = event.get("price")
            size = event.get("size")
            logger.debug(f"[LOB] New order: side={side}, price={price:.3f}, size={size}")
            if side == "buy":
                # Try to find an existing level near the price.
                matched = False
                for level in self.bids:
                    if abs(level["price"] - price) < 0.001:
                        level["size"] += size
                        matched = True
                        break
                if not matched:
                    self.bids.append({"price": price, "size": size})
                    self.bids.sort(key=lambda x: -x["price"])
            else:
                matched = False
                for level in self.asks:
                    if abs(level["price"] - price) < 0.001:
                        level["size"] += size
                        matched = True
                        break
                if not matched:
                    self.asks.append({"price": price, "size": size})
                    self.asks.sort(key=lambda x: x["price"])
        elif event_type == "cancel_order":
            side = event.get("side")
            price = event.get("price")
            size = event.get("size")
            logger.debug(f"[LOB] Cancel order: side={side}, price={price:.3f}, size={size}")
            if side == "buy":
                for level in self.bids:
                    if abs(level["price"] - price) < 0.001:
                        level["size"] = max(0, level["size"] - size)
            else:
                for level in self.asks:
                    if abs(level["price"] - price) < 0.001:
                        level["size"] = max(0, level["size"] - size)
        elif event_type == "trade":
            # The aggressor side is specified; remove quantity from the opposite side.
            side = event.get("side")
            size = event.get("size")
            logger.debug(f"[LOB] Trade event: aggressor side={side}, size executed={size}")
            if side == "buy":
                remaining = size
                for level in self.asks:
                    if remaining <= level["size"]:
                        level["size"] -= remaining
                        remaining = 0
                        break
                    else:
                        remaining -= level["size"]
                        level["size"] = 0
            else:
                remaining = size
                for level in self.bids:
                    if remaining <= level["size"]:
                        level["size"] -= remaining
                        remaining = 0
                        break
                    else:
                        remaining -= level["size"]
                        level["size"] = 0
        # Clean up any levels that have zero size.
        self.bids = [level for level in self.bids if level["size"] > 0]
        self.asks = [level for level in self.asks if level["size"] > 0]
        self._recalc_mid_price()

    def _recalc_mid_price(self):
        if self.bids and self.asks:
            best_bid = self.bids[0]["price"]
            best_ask = self.asks[0]["price"]
            self.mid_price = (best_bid + best_ask) / 2.0
        # Else, leave mid_price unchanged.
        logger.debug(f"[LOB] Updated mid_price: {self.mid_price:.3f}")

# -------------------- Hawkes Process for Order Arrivals ---------------------------
class HawkesOrderArrivals:
    """
    Implements a Hawkes process for order arrivals.
    It stores event times and computes the current intensity as:
       intensity(t) = base_intensity + sum(alpha * exp(-decay*(t-t_i)))
    """
    def __init__(self, base_intensity: float, alpha: float, decay: float):
        self.base_intensity = base_intensity
        self.alpha = alpha
        self.decay = decay
        self.event_times: List[float] = []

    def intensity_at(self, current_time: float) -> float:
        return self.base_intensity + sum(self.alpha * np.exp(-self.decay * (current_time - t))
                                         for t in self.event_times if current_time > t)

    def get_next_arrival_time(self, current_time: float) -> float:
        intensity = self.intensity_at(current_time)
        next_time = current_time + np.random.exponential(1.0 / intensity)
        return next_time

    def trigger_event(self, event_time: float):
        self.event_times.append(event_time)
        intensity = self.intensity_at(event_time)
        logger.debug(f"[Hawkes] Event at t={event_time:.2f}; new intensity={intensity:.2f}")

# -------------------- Network Simulation ---------------------------
class NetworkSimulator:
    """
    Simulates network conditions including latency (with jitter)
    and occasional API failures.
    """
    def __init__(self, average_latency: float = 0.05, failure_rate: float = 0.01):
        self.average_latency = average_latency
        self.failure_rate = failure_rate

    def simulate_network_delay(self):
        # Introduce jitter: delay between 50% and 150% of average_latency.
        delay = np.random.uniform(self.average_latency * 0.5, self.average_latency * 1.5)
        time.sleep(delay)
        logger.debug(f"[Network] Simulated delay: {delay:.3f} seconds")

    def check_for_api_failure(self):
        if random.random() < self.failure_rate:
            logger.warning("[Network] Simulated API failure!")
            raise ConnectionError("Simulated API failure.")

# -------------------- Transaction Cost Model ---------------------------
class TransactionCostModel:
    """
    Computes the transaction cost for an order.
    Cost = (fee_per_share * size) + (impact_coefficient * size^2 * price)
    """
    def __init__(self, fee_per_share: float = 0.005, impact_coefficient: float = 0.0001):
        self.fee_per_share = fee_per_share
        self.impact_coefficient = impact_coefficient

    def compute_cost(self, size: float, price: float) -> float:
        fee = size * self.fee_per_share
        impact = self.impact_coefficient * (size ** 2) * price
        total_cost = fee + impact
        logger.debug(f"[Cost] Order cost: fee={fee:.3f}, impact={impact:.3f}, total={total_cost:.3f}")
        return total_cost

# -------------------- Agent-Based Modeling ---------------------------
class BaseAgent:
    """
    Abstract base class for market agents.
    """
    def __init__(self, name: str, lob: MultiLevelOrderBook):
        self.name = name
        self.lob = lob

    def step(self, current_time: float):
        raise NotImplementedError

class MarketMakerAgent(BaseAgent):
    def step(self, current_time: float):
        mid = self.lob.mid_price
        bid_price = mid - 0.005
        ask_price = mid + 0.005
        bid_size = random.randint(10, 50)
        ask_size = random.randint(10, 50)
        bid_event = {"type": "new_order", "side": "buy", "price": bid_price, "size": bid_size}
        ask_event = {"type": "new_order", "side": "sell", "price": ask_price, "size": ask_size}
        self.lob.update_for_event(bid_event)
        self.lob.update_for_event(ask_event)
        logger.debug(f"[{self.name}] Market-making: bid {bid_price:.3f} x {bid_size}, ask {ask_price:.3f} x {ask_size}")

class MomentumAgent(BaseAgent):
    def __init__(self, name: str, lob: MultiLevelOrderBook):
        super().__init__(name, lob)
        self.last_mid = lob.mid_price

    def step(self, current_time: float):
        current_mid = self.lob.mid_price
        if current_mid > self.last_mid:
            side = "buy"
            size = random.randint(5, 20)
            logger.debug(f"[{self.name}] Momentum bullish at t={current_time:.2f}")
        elif current_mid < self.last_mid:
            side = "sell"
            size = random.randint(5, 20)
            logger.debug(f"[{self.name}] Momentum bearish at t={current_time:.2f}")
        else:
            # No change, do nothing.
            self.last_mid = current_mid
            return
        event = {"type": "new_order", "side": side, "price": current_mid, "size": size}
        self.lob.update_for_event(event)
        self.last_mid = current_mid

class NoiseAgent(BaseAgent):
    def step(self, current_time: float):
        if random.random() < 0.3:
            side = "buy" if random.random() < 0.5 else "sell"
            price = self.lob.mid_price + np.random.normal(0, 0.01)
            size = random.randint(1, 10)
            event = {"type": "new_order", "side": side, "price": price, "size": size}
            self.lob.update_for_event(event)
            logger.debug(f"[{self.name}] Noise order: {side} {price:.3f} x {size}")

class ReinforcementLearningAgent(BaseAgent):
    """
    A stub for an RL-based agent. In practice, you would load and use a trained policy.
    """
    def __init__(self, name: str, lob: MultiLevelOrderBook, policy_model: Any = None):
        super().__init__(name, lob)
        self.policy_model = policy_model

    def step(self, current_time: float):
        # For now, randomly decide to buy or sell based on a simulated policy.
        side = "buy" if random.random() < 0.5 else "sell"
        price = self.lob.mid_price + (0.005 if side=="buy" else -0.005)
        size = random.randint(5, 15)
        event = {"type": "new_order", "side": side, "price": price, "size": size}
        self.lob.update_for_event(event)
        logger.debug(f"[{self.name}] RL agent places {side} order at {price:.3f} x {size}")

# -------------------- Risk Management ---------------------------
class RiskModel:
    """
    Tracks price history and computes a simple intraday VaR (Value at Risk)
    using historical simulation.
    """
    def __init__(self):
        self.positions: Dict[str, float] = {}
        self.price_history: List[float] = []

    def update_positions(self, agent_name: str, quantity: float):
        self.positions[agent_name] = self.positions.get(agent_name, 0.0) + quantity

    def record_price(self, price: float):
        self.price_history.append(price)
        # Keep a fixed-length history (e.g., last 252 time steps)
        if len(self.price_history) > 252:
            self.price_history.pop(0)

    def compute_intraday_var(self, confidence: float = 0.95) -> float:
        if len(self.price_history) < 2:
            return 0.0
        returns = np.diff(self.price_history) / self.price_history[:-1]
        var = np.percentile(returns, (1 - confidence) * 100)
        # Scale VaR by 100 for reporting (this is arbitrary and can be refined)
        return var * 100

    def run_stress_test(self, scenario: Dict[str, Any]):
        logger.debug(f"[Risk] Running stress test with scenario: {scenario}")
        shock = scenario.get("shock", -0.1)
        base_var = self.compute_intraday_var()
        stressed_var = base_var * (1 + shock)
        logger.debug(f"[Risk] Stress test VaR: {stressed_var:.2f}")
        return stressed_var

# -------------------- Main Simulation Class ---------------------------
class EnhancedSimulation:
    """
    Orchestrates the overall simulation: synthetic data generation, order book updates,
    agent actions, risk monitoring, and network simulation.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.network_sim = NetworkSimulator(
            average_latency=0.05, failure_rate=config.API_FAILURE_RATE
        )
        self.data_gen = SyntheticDataGenerator(method="GBM")
        self.hawkes_model = HawkesOrderArrivals(
            base_intensity=config.HAWKES_BASE_INTENSITY,
            alpha=config.HAWKES_ALPHA,
            decay=config.HAWKES_DECAY
        )
        self.lob = MultiLevelOrderBook(symbol="FAKE", num_levels=config.NUM_LEVELS)
        self.risk_model = RiskModel()
        self.transaction_cost_model = TransactionCostModel()
        # Instantiate a list of agents with different strategies.
        self.agents = [
            MarketMakerAgent("MarketMaker_1", self.lob),
            MomentumAgent("Momentum_1", self.lob),
            NoiseAgent("Noise_1", self.lob),
            ReinforcementLearningAgent("RL_Agent_1", self.lob, policy_model=None)
        ]

    def run_simulation(self):
        # 1. Generate synthetic price path.
        price_series = self.data_gen.generate_price_series(length=1000)
        initial_price = price_series[0]
        self.lob.initialize_book(initial_price)

        current_time = 0.0
        end_time = self.config.SIMULATION_TIME
        next_order_arrival = self.hawkes_model.get_next_arrival_time(current_time)

        while current_time < end_time:
            current_time += self.config.TIME_STEP

            # Update price from synthetic series (simple linear mapping).
            index = min(int((current_time / end_time) * (len(price_series) - 1)), len(price_series)-1)
            new_price = price_series[index]
            self.lob.mid_price = new_price
            self.risk_model.record_price(new_price)

            # Let agents act.
            for agent in self.agents:
                agent.step(current_time)

            # Hawkes-based order arrival events.
            if current_time >= next_order_arrival:
                self.hawkes_model.trigger_event(current_time)
                next_order_arrival = self.hawkes_model.get_next_arrival_time(current_time)
                # Create a random new order event.
                event = {
                    "type": "new_order",
                    "side": "buy" if random.random() < 0.5 else "sell",
                    "price": new_price + np.random.normal(0, 0.01),
                    "size": random.randint(5, 20)
                }
                self.lob.update_for_event(event)

            # Simulate network conditions.
            try:
                self.network_sim.simulate_network_delay()
                self.network_sim.check_for_api_failure()
            except ConnectionError:
                logger.warning("[Simulation] Skipping order execution due to network failure.")

            # Every 60 seconds, compute and log risk metrics.
            if int(current_time) % 60 == 0:
                var = self.risk_model.compute_intraday_var()
                logger.debug(f"[Risk] Intraday VaR at t={current_time:.0f} s: {var:.2f}")
                # Also compute transaction cost for a sample order (for demonstration).
                sample_cost = self.transaction_cost_model.compute_cost(size=20, price=new_price)
                logger.debug(f"[Cost] Sample transaction cost at t={current_time:.0f} s: {sample_cost:.3f}")

        logger.info("Enhanced simulation complete.")

# -------------------- Main Execution ---------------------------
if __name__ == "__main__":
    config = SimulationConfig()
    simulation = EnhancedSimulation(config)
    simulation.run_simulation()



