"""
File: enhanced_simulation_detailed.py

Enhanced simulation framework with advanced market microstructure modeling,
detailed synthetic data generation (including regime switching),
enhanced order book dynamics with order queue management,
a detailed Hawkes process for order arrivals,
improved network simulation,
multiple advanced agent strategies,
and advanced risk and transaction cost modeling.

This blueprint replaces many of the simple stubs with more detailed implementations.
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
logger = logging.getLogger("EnhancedSimulationDetailed")

# -------------------- Simulation Configuration ---------------------------
class SimulationConfig:
    """Central configuration for the simulation environment."""
    SIMULATION_TIME = 3600      # seconds (e.g., one trading hour)
    TIME_STEP = 1.0             # simulation time step in seconds
    NUM_LEVELS = 5              # number of levels in the order book
    HAWKES_BASE_INTENSITY = 0.2
    HAWKES_ALPHA = 0.5
    HAWKES_DECAY = 0.8
    API_FAILURE_RATE = 0.01     # probability of simulated API failure per check
    MAX_HIDDEN_LIQUIDITY = 20   # parameter for hidden liquidity simulation

    # Synthetic data generation parameters (regime switching)
    REGIME_SWITCH_PROB = 0.01    # per time step probability of switching regime
    REGIMES = [
        {"mu": 0.1, "sigma": 0.2},   # bullish/normal regime
        {"mu": -0.05, "sigma": 0.3}  # bearish/high-volatility regime
    ]

# -------------------- Synthetic Data Generation ---------------------------
class SyntheticDataGenerator:
    """
    Generates synthetic price paths using a Geometric Brownian Motion (GBM)
    model with regime switching. In production you might use GANs or diffusion models.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate_price_series(self, length: int = 1000, dt: float = 1/252) -> List[float]:
        logger.debug("Generating synthetic price series with regime switching using GBM.")
        prices = [100.0]
        current_regime = random.choice(self.config.REGIMES)
        for i in range(1, length):
            # Possibly switch regime based on probability.
            if random.random() < self.config.REGIME_SWITCH_PROB:
                current_regime = random.choice(self.config.REGIMES)
                logger.debug(f"Regime switched at step {i}: {current_regime}")
            mu = current_regime["mu"]
            sigma = current_regime["sigma"]
            drift = (mu - 0.5 * sigma ** 2) * dt
            shock = sigma * np.sqrt(dt) * np.random.normal()
            new_price = prices[-1] * np.exp(drift + shock)
            prices.append(new_price)
        return prices

# -------------------- Multi-Level Order Book with Order Queue ---------------------------
class MultiLevelOrderBook:
    """
    Simulates a multi-level limit order book (LOB) with detailed order queue management.
    Each level maintains a FIFO queue of orders, where each order is a dictionary with:
      - price: the order price
      - quantity: the order quantity
      - timestamp: when the order was placed
    """
    def __init__(self, symbol: str, num_levels: int):
        self.symbol = symbol
        self.num_levels = num_levels
        self.bids: List[List[Dict[str, Any]]] = []  # sorted descending by price
        self.asks: List[List[Dict[str, Any]]] = []  # sorted ascending by price
        self.mid_price = 100.0

    def initialize_book(self, initial_price: float):
        self.mid_price = initial_price
        self.bids = []
        self.asks = []
        current_time = time.time()
        for i in range(self.num_levels):
            bid_price = initial_price - 0.01 * (i + 1)
            ask_price = initial_price + 0.01 * (i + 1)
            self.bids.append([{"price": bid_price, "quantity": 100, "timestamp": current_time}])
            self.asks.append([{"price": ask_price, "quantity": 100, "timestamp": current_time}])
        # Sort bids descending, asks ascending
        self.bids.sort(key=lambda lvl: -lvl[0]["price"])
        self.asks.sort(key=lambda lvl: lvl[0]["price"])
        logger.debug(f"[LOB] Initialized for {self.symbol} with mid_price: {initial_price:.2f}")

    def _aggregate_level(self, orders: List[Dict[str, Any]]) -> Dict[str, float]:
        total_qty = sum(order["quantity"] for order in orders)
        price = orders[0]["price"] if orders else 0
        return {"price": price, "quantity": total_qty}

    def _recalc_mid_price(self):
        if self.bids and self.asks and self.bids[0] and self.asks[0]:
            best_bid = self.bids[0][0]["price"]
            best_ask = self.asks[0][0]["price"]
            self.mid_price = (best_bid + best_ask) / 2.0
        logger.debug(f"[LOB] Updated mid_price: {self.mid_price:.3f}")

    def update_for_event(self, event: Dict[str, Any]):
        """
        Process an event that changes the order book.
        Supported event types: "new_order", "cancel_order", "trade".
        """
        event_type = event.get("type")
        side = event.get("side")
        price = event.get("price")
        size = event.get("size")
        current_time = time.time()

        if event_type == "new_order":
            logger.debug(f"[LOB] New order: side={side}, price={price:.3f}, size={size}")
            if side == "buy":
                level_found = False
                for level in self.bids:
                    if abs(level[0]["price"] - price) < 0.001:
                        level.append({"price": price, "quantity": size, "timestamp": current_time})
                        level_found = True
                        break
                if not level_found:
                    self.bids.append([{"price": price, "quantity": size, "timestamp": current_time}])
                    self.bids.sort(key=lambda lvl: -lvl[0]["price"])
            else:  # sell order
                level_found = False
                for level in self.asks:
                    if abs(level[0]["price"] - price) < 0.001:
                        level.append({"price": price, "quantity": size, "timestamp": current_time})
                        level_found = True
                        break
                if not level_found:
                    self.asks.append([{"price": price, "quantity": size, "timestamp": current_time}])
                    self.asks.sort(key=lambda lvl: lvl[0]["price"])

        elif event_type == "cancel_order":
            logger.debug(f"[LOB] Cancel order: side={side}, price={price:.3f}, size={size}")
            levels = self.bids if side == "buy" else self.asks
            for level in levels:
                if abs(level[0]["price"] - price) < 0.001:
                    remaining = size
                    new_level = []
                    for order in level:
                        if remaining <= 0:
                            new_level.append(order)
                        else:
                            if order["quantity"] <= remaining:
                                remaining -= order["quantity"]
                            else:
                                order["quantity"] -= remaining
                                remaining = 0
                                new_level.append(order)
                    level.clear()
                    level.extend(new_level)
                    break

        elif event_type == "trade":
            logger.debug(f"[LOB] Trade event: aggressor={side}, size executed={size}")
            # If aggressor is buy, then consume liquidity from asks (and vice versa)
            levels = self.asks if side == "buy" else self.bids
            remaining = size
            for level in levels:
                if remaining <= 0:
                    break
                agg = self._aggregate_level(level)
                if remaining < agg["quantity"]:
                    sub_remaining = remaining
                    new_level = []
                    for order in level:
                        if sub_remaining <= 0:
                            new_level.append(order)
                        else:
                            if order["quantity"] <= sub_remaining:
                                sub_remaining -= order["quantity"]
                            else:
                                order["quantity"] -= sub_remaining
                                sub_remaining = 0
                                new_level.append(order)
                    level.clear()
                    level.extend(new_level)
                    remaining = 0
                else:
                    remaining -= agg["quantity"]
                    level.clear()
            # Remove empty levels
            self.bids = [lvl for lvl in self.bids if lvl]
            self.asks = [lvl for lvl in self.asks if lvl]

        self._recalc_mid_price()

# -------------------- Hawkes Process for Order Arrivals ---------------------------
class HawkesOrderArrivals:
    """
    Implements a Hawkes process for order arrivals.
    Intensity(t) = base_intensity + sum(alpha * exp(-decay*(t-t_i)))
    """
    def __init__(self, base_intensity: float, alpha: float, decay: float):
        self.base_intensity = base_intensity
        self.alpha = alpha
        self.decay = decay
        self.event_times: List[float] = []

    def prune_events(self, current_time: float, window: float = 100.0):
        original_count = len(self.event_times)
        self.event_times = [t for t in self.event_times if current_time - t <= window]
        pruned = original_count - len(self.event_times)
        if pruned > 0:
            logger.debug(f"[Hawkes] Pruned {pruned} events older than {window} seconds.")

    def intensity_at(self, current_time: float) -> float:
        self.prune_events(current_time)
        intensity = self.base_intensity + sum(self.alpha * np.exp(-self.decay * (current_time - t))
                                               for t in self.event_times)
        return intensity

    def get_next_arrival_time(self, current_time: float) -> float:
        intensity = self.intensity_at(current_time)
        next_time = current_time + np.random.exponential(1.0 / intensity)
        logger.debug(f"[Hawkes] t={current_time:.2f}, intensity={intensity:.2f}, next event at {next_time:.2f}")
        return next_time

    def trigger_event(self, event_time: float):
        self.event_times.append(event_time)
        intensity = self.intensity_at(event_time)
        logger.debug(f"[Hawkes] Event at t={event_time:.2f}; new intensity={intensity:.2f}")

# -------------------- Network Simulation ---------------------------
class NetworkSimulator:
    """
    Simulates network conditions including latency (with jitter)
    and occasional API call failures.
    """
    def __init__(self, average_latency: float = 0.05, failure_rate: float = 0.01):
        self.average_latency = average_latency
        self.failure_rate = failure_rate

    def simulate_network_delay(self):
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
    Cost = (fee_per_share * size) + (impact_coefficient * (size / liquidity_factor)^2 * price)
    """
    def __init__(self, fee_per_share: float = 0.005, impact_coefficient: float = 0.0001, liquidity_factor: float = 100.0):
        self.fee_per_share = fee_per_share
        self.impact_coefficient = impact_coefficient
        self.liquidity_factor = liquidity_factor

    def compute_cost(self, size: float, price: float) -> float:
        fee = size * self.fee_per_share
        impact = self.impact_coefficient * ((size / self.liquidity_factor) ** 2) * price
        total_cost = fee + impact
        logger.debug(f"[Cost] Computed cost: fee={fee:.3f}, impact={impact:.3f}, total={total_cost:.3f}")
        return total_cost

# -------------------- Agent-Based Modeling ---------------------------
class BaseAgent:
    """
    Abstract base class for market agents.
    """
    def __init__(self, name: str, lob: MultiLevelOrderBook, risk_model: 'RiskModel' = None):
        self.name = name
        self.lob = lob
        self.risk_model = risk_model

    def step(self, current_time: float):
        raise NotImplementedError

class MarketMakerAgent(BaseAgent):
    def step(self, current_time: float):
        mid = self.lob.mid_price
        # Calculate dynamic spread based on volatility (using risk model price history)
        if self.risk_model and len(self.risk_model.price_history) >= 20:
            recent_prices = self.risk_model.price_history[-20:]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
        else:
            volatility = 0.005
        spread = 0.005 + volatility
        bid_price = mid - spread
        ask_price = mid + spread
        bid_size = random.randint(20, 50)
        ask_size = random.randint(20, 50)
        bid_event = {"type": "new_order", "side": "buy", "price": bid_price, "size": bid_size}
        ask_event = {"type": "new_order", "side": "sell", "price": ask_price, "size": ask_size}
        self.lob.update_for_event(bid_event)
        self.lob.update_for_event(ask_event)
        logger.debug(f"[{self.name}] Market-making: bid {bid_price:.3f} x {bid_size}, ask {ask_price:.3f} x {ask_size}")

class MomentumAgent(BaseAgent):
    def __init__(self, name: str, lob: MultiLevelOrderBook, risk_model: 'RiskModel'):
        super().__init__(name, lob, risk_model)
        self.price_window: List[float] = []

    def step(self, current_time: float):
        current_mid = self.lob.mid_price
        self.price_window.append(current_mid)
        if len(self.price_window) > 20:
            self.price_window.pop(0)
        sma = np.mean(self.price_window)
        if current_mid > sma * 1.001:
            side = "buy"
            size = random.randint(10, 25)
            logger.debug(f"[{self.name}] Momentum bullish at t={current_time:.2f}: mid {current_mid:.3f} > SMA {sma:.3f}")
        elif current_mid < sma * 0.999:
            side = "sell"
            size = random.randint(10, 25)
            logger.debug(f"[{self.name}] Momentum bearish at t={current_time:.2f}: mid {current_mid:.3f} < SMA {sma:.3f}")
        else:
            return
        event = {"type": "new_order", "side": side, "price": current_mid, "size": size}
        self.lob.update_for_event(event)

class NoiseAgent(BaseAgent):
    def step(self, current_time: float):
        if random.random() < 0.3:
            side = "buy" if random.random() < 0.5 else "sell"
            price = self.lob.mid_price + np.random.normal(0, 0.02)
            size = random.randint(5, 15)
            event = {"type": "new_order", "side": side, "price": price, "size": size}
            self.lob.update_for_event(event)
            logger.debug(f"[{self.name}] Noise order: {side} {price:.3f} x {size}")

class ReinforcementLearningAgent(BaseAgent):
    """
    A simple RL-based agent. Here the "policy" is a basic rule:
    if current price > SMA then sell, if below then buy.
    In production, replace with a trained neural network policy.
    """
    def __init__(self, name: str, lob: MultiLevelOrderBook, risk_model: 'RiskModel'):
        super().__init__(name, lob, risk_model)
        self.price_window: List[float] = []

    def step(self, current_time: float):
        current_mid = self.lob.mid_price
        self.price_window.append(current_mid)
        if len(self.price_window) > 50:
            self.price_window.pop(0)
        sma = np.mean(self.price_window)
        if current_mid > sma * 1.001:
            side = "sell"
            size = random.randint(10, 20)
            logger.debug(f"[{self.name}] RL agent: Selling (mid {current_mid:.3f} > SMA {sma:.3f})")
        elif current_mid < sma * 0.999:
            side = "buy"
            size = random.randint(10, 20)
            logger.debug(f"[{self.name}] RL agent: Buying (mid {current_mid:.3f} < SMA {sma:.3f})")
        else:
            return
        event = {"type": "new_order", "side": side, "price": current_mid, "size": size}
        self.lob.update_for_event(event)

# -------------------- Risk Management ---------------------------
class RiskModel:
    """
    Tracks price history and computes a simple historical intraday VaR,
    as well as a basic liquidity risk measure.
    """
    def __init__(self):
        self.price_history: List[float] = []

    def record_price(self, price: float):
        self.price_history.append(price)
        if len(self.price_history) > 252:
            self.price_history.pop(0)

    def compute_intraday_var(self, confidence: float = 0.95) -> float:
        if len(self.price_history) < 2:
            return 0.0
        returns = np.diff(self.price_history) / self.price_history[:-1]
        var = np.percentile(returns, (1 - confidence) * 100)
        return var * 100  # expressed as percentage

    def compute_liquidity_risk(self, lob: MultiLevelOrderBook, side: str) -> float:
        """
        Computes available volume on the specified side ("buy" or "sell").
        """
        levels = lob.bids if side == "buy" else lob.asks
        total_volume = sum(sum(order["quantity"] for order in level) for level in levels)
        return total_volume

    def run_stress_test(self, shock: float = -0.1) -> float:
        base_var = self.compute_intraday_var()
        stressed_var = base_var * (1 + shock)
        logger.debug(f"[Risk] Stress test VaR: {stressed_var:.2f}")
        return stressed_var

# -------------------- Main Simulation Class ---------------------------
class EnhancedSimulationDetailed:
    """
    Orchestrates the overall simulation: synthetic data generation, LOB updates,
    agent actions, risk monitoring, network simulation, and event triggering.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.network_sim = NetworkSimulator(
            average_latency=0.05, failure_rate=config.API_FAILURE_RATE
        )
        self.data_gen = SyntheticDataGenerator(config)
        self.hawkes_model = HawkesOrderArrivals(
            base_intensity=config.HAWKES_BASE_INTENSITY,
            alpha=config.HAWKES_ALPHA,
            decay=config.HAWKES_DECAY
        )
        self.lob = MultiLevelOrderBook(symbol="FAKE", num_levels=config.NUM_LEVELS)
        self.risk_model = RiskModel()
        self.transaction_cost_model = TransactionCostModel()
        self.agents = [
            MarketMakerAgent("MarketMaker_1", self.lob, self.risk_model),
            MomentumAgent("Momentum_1", self.lob, self.risk_model),
            NoiseAgent("Noise_1", self.lob),
            ReinforcementLearningAgent("RL_Agent_1", self.lob, self.risk_model)
        ]

    def run_simulation(self):
        # Generate synthetic price series.
        price_series = self.data_gen.generate_price_series(length=1000)
        initial_price = price_series[0]
        self.lob.initialize_book(initial_price)

        current_time = 0.0
        end_time = self.config.SIMULATION_TIME
        next_hawkes_time = self.hawkes_model.get_next_arrival_time(current_time)

        while current_time < end_time:
            current_time += self.config.TIME_STEP

            # Map simulation time to synthetic price series.
            idx = min(int((current_time / end_time) * (len(price_series) - 1)), len(price_series) - 1)
            new_price = price_series[idx]
            self.lob.mid_price = new_price
            self.risk_model.record_price(new_price)

            # Let all agents take actions.
            for agent in self.agents:
                agent.step(current_time)

            # Trigger a Hawkes process event if it's time.
            if current_time >= next_hawkes_time:
                self.hawkes_model.trigger_event(current_time)
                next_hawkes_time = self.hawkes_model.get_next_arrival_time(current_time)
                # Create an additional random order event.
                event = {
                    "type": "new_order",
                    "side": "buy" if random.random() < 0.5 else "sell",
                    "price": new_price + np.random.normal(0, 0.01),
                    "size": random.randint(5, 20)
                }
                self.lob.update_for_event(event)

            # Simulate network delay and potential API failures.
            try:
                self.network_sim.simulate_network_delay()
                self.network_sim.check_for_api_failure()
            except ConnectionError:
                logger.warning("[Simulation] Skipping network-dependent actions due to API failure.")

            # Every 60 seconds, compute and log risk and cost metrics.
            if int(current_time) % 60 == 0:
                var = self.risk_model.compute_intraday_var()
                liq_buy = self.risk_model.compute_liquidity_risk(self.lob, "buy")
                liq_sell = self.risk_model.compute_liquidity_risk(self.lob, "sell")
                logger.debug(f"[Risk] t={current_time:.0f}s: VaR={var:.2f}%, Liquidity - Buy: {liq_buy}, Sell: {liq_sell}")
                sample_cost = self.transaction_cost_model.compute_cost(size=20, price=new_price)
                logger.debug(f"[Cost] Sample transaction cost at t={current_time:.0f}s: {sample_cost:.3f}")

        logger.info("Enhanced Detailed Simulation complete.")

# -------------------- Main Execution ---------------------------
if __name__ == "__main__":
    config = SimulationConfig()
    simulation = EnhancedSimulationDetailed(config)
    simulation.run_simulation()
