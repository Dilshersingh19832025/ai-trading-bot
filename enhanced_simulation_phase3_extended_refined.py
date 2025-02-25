# File: enhanced_simulation_phase3_extended_refined.py
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import norm
import optuna
import random
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ----------------------------
# Helper functions and metrics
# ----------------------------

def sharpe_ratio(returns, risk_free_rate=0.0):
    if np.std(returns) == 0:
        return 0
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

def sortino_ratio(returns, risk_free_rate=0.0):
    downside = np.array([r for r in returns if r < risk_free_rate])
    if np.std(downside) == 0:
        return 0
    return (np.mean(returns) - risk_free_rate) / np.std(downside)

def calmar_ratio(annual_return, max_drawdown):
    if max_drawdown == 0:
        return 0
    return annual_return / abs(max_drawdown)

def conditional_var(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    return abs(sorted_returns[index])

# A composite objective function: weighted sum of several metrics.
def composite_objective(metrics, weights):
    # metrics: dict containing return, drawdown, sharpe, sortino, calmar, cvar
    # weights: dict of weights for each metric
    obj = 0.0
    for key in weights:
        obj += weights[key] * metrics.get(key, 0)
    return obj

# ----------------------------
# Data Quality & Feature Engineering
# ----------------------------
class DataQualityValidator:
    def __init__(self):
        pass

    def validate(self, data: pd.DataFrame) -> bool:
        # Simple validation: check for missing values and anomalies (e.g., extreme outliers)
        if data.isnull().values.any():
            logging.warning("Data contains missing values.")
            return False
        # Check for extreme anomalies in 'price'
        if (data['price'] > data['price'].mean() * 3).sum() > 0:
            logging.warning("Anomaly detected in price values.")
            return False
        logging.info("Data quality validation passed with no anomalies.")
        return True

class FeatureEngineer:
    def __init__(self):
        pass

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # Simulate technical indicator: moving average
        data['ma_10'] = data['price'].rolling(window=10, min_periods=1).mean()
        # Simulate market microstructure feature: bid-ask spread (randomly generated for demo)
        data['bid_ask_spread'] = np.random.uniform(0.01, 0.05, len(data))
        # Simulate alternative data: sentiment score (random for demo)
        data['sentiment'] = np.random.uniform(-1, 1, len(data))
        return data

# ----------------------------
# Regime Detection & Dynamic Strategy Switching
# ----------------------------
class RegimeDetector:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=self.n_regimes, random_state=42)

    def detect_regime(self, data: pd.DataFrame) -> str:
        # Use features (e.g., returns and volatility) to cluster regimes
        data = data.copy()
        data['return'] = data['price'].pct_change().fillna(0)
        data['volatility'] = data['return'].rolling(window=10, min_periods=1).std().fillna(0)
        features = data[['return', 'volatility']].values
        # Fit KMeans on recent window (for demo, use last 100 points)
        window = features[-100:] if len(features) >= 100 else features
        clusters = self.model.fit_predict(window)
        current_cluster = clusters[-1]
        regime_map = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
        regime = regime_map.get(current_cluster, "Sideways")
        logging.info(f"Regime Detection: Current regime = {regime}")
        return regime

# ----------------------------
# Strategy Optimization (Hybrid: Optuna + Simple Genetic Algorithm)
# ----------------------------
class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame, regime: str):
        self.data = data
        self.regime = regime
        # Weights for our composite objective: lower is better (minimization)
        self.weights = {
            'return': -0.3,  # we want higher return (thus negative weight)
            'drawdown': 0.3,
            'sharpe': -0.2,
            'sortino': -0.1,
            'calmar': -0.1,
            'cvar': 0.3,
        }
        # Define regime-specific adjustments:
        if regime == "Bullish":
            self.regime_multiplier = 0.9  # more aggressive
        elif regime == "Bearish":
            self.regime_multiplier = 1.1  # more conservative
        else:
            self.regime_multiplier = 1.0

    def backtest_strategy(self, short_window, long_window):
        # For demo, simulate strategy performance using rolling window signals.
        data = self.data.copy()
        data['short_ma'] = data['price'].rolling(window=short_window, min_periods=1).mean()
        data['long_ma'] = data['price'].rolling(window=long_window, min_periods=1).mean()
        # Signal: 1 if short_ma > long_ma, else -1.
        data['signal'] = np.where(data['short_ma'] > data['long_ma'], 1, -1)
        data['strategy_return'] = data['signal'].shift(1) * data['price'].pct_change().fillna(0)
        cumulative_return = np.prod(1 + data['strategy_return']) - 1
        max_drawdown = data['strategy_return'].rolling(window=100, min_periods=1).sum().min()  # simplified
        sr = sharpe_ratio(data['strategy_return'])
        so = sortino_ratio(data['strategy_return'])
        # For annualized return assume data frequency is daily and 252 trading days:
        annual_return = (1 + cumulative_return)**(252/len(data)) - 1 if len(data) > 0 else 0
        calmar = calmar_ratio(annual_return, max_drawdown)
        cvar = conditional_var(data['strategy_return'])
        # Volatility (annualized standard deviation)
        volatility = data['strategy_return'].std() * np.sqrt(252)
        metrics = {
            'return': cumulative_return,
            'drawdown': abs(max_drawdown),
            'sharpe': sr,
            'sortino': so,
            'calmar': calmar,
            'cvar': cvar,
            'volatility': volatility,
        }
        return metrics

    def objective(self, trial: optuna.Trial):
        # Suggest strategy parameters
        short_window = trial.suggest_int('short_window', 3, 20)
        long_window = trial.suggest_int('long_window', short_window+1, 50)
        metrics = self.backtest_strategy(short_window, long_window)
        # Compute composite objective (apply regime multiplier)
        obj = self.regime_multiplier * composite_objective(metrics, self.weights)
        logging.info(f"Trial params: short_window={short_window}, long_window={long_window} | "
                     f"Return={metrics['return']*100:.2f}%, Drawdown=-{metrics['drawdown']*100:.2f}%, "
                     f"Sharpe={metrics['sharpe']:.2f}, Sortino={metrics['sortino']:.2f}, "
                     f"Calmar={metrics['calmar']:.2f}, CVaR={metrics['cvar']:.4f}, "
                     f"Volatility={metrics['volatility']:.2f} | Objective={obj:.4f}")
        return obj

    def optimize(self, n_trials=30):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        best_params = study.best_trial.params
        best_obj = study.best_trial.value
        logging.info(f"Optimization complete. Best parameters: {best_params} with objective {best_obj:.4f}")
        return best_params, best_obj

    def genetic_optimization(self, population_size=20, generations=5):
        # A very basic genetic algorithm for further exploration.
        def random_individual():
            return {
                'short_window': random.randint(3, 20),
                'long_window': random.randint(5, 50)
            }
        # Initialize population
        population = [random_individual() for _ in range(population_size)]
        for gen in range(generations):
            scored = []
            for individual in population:
                # Ensure long_window > short_window
                if individual['long_window'] <= individual['short_window']:
                    individual['long_window'] = individual['short_window'] + 1
                metrics = self.backtest_strategy(individual['short_window'], individual['long_window'])
                score = self.regime_multiplier * composite_objective(metrics, self.weights)
                scored.append((score, individual))
            scored.sort(key=lambda x: x[0])
            logging.info(f"Generation {gen} best score: {scored[0][0]:.4f}")
            # Select top 50%
            survivors = [ind for score, ind in scored[:population_size//2]]
            # Create offspring by crossover/mutation
            offspring = []
            while len(offspring) < population_size - len(survivors):
                parent1, parent2 = random.sample(survivors, 2)
                child = {
                    'short_window': random.choice([parent1['short_window'], parent2['short_window']]),
                    'long_window': random.choice([parent1['long_window'], parent2['long_window']])
                }
                # Mutation: small random adjustment
                if random.random() < 0.3:
                    child['short_window'] += random.randint(-1, 1)
                    child['short_window'] = max(3, child['short_window'])
                if random.random() < 0.3:
                    child['long_window'] += random.randint(-2, 2)
                    child['long_window'] = max(child['short_window']+1, child['long_window'])
                offspring.append(child)
            population = survivors + offspring
        # Return best individual
        best_individual = min(population, key=lambda ind: self.regime_multiplier * 
                              composite_objective(self.backtest_strategy(ind['short_window'], ind['long_window']), self.weights))
        best_obj = self.regime_multiplier * composite_objective(self.backtest_strategy(best_individual['short_window'], best_individual['long_window']), self.weights)
        logging.info(f"Genetic Optimization complete. Best parameters: {best_individual} with objective {best_obj:.4f}")
        return best_individual, best_obj

# ----------------------------
# Risk Management Module
# ----------------------------
class RiskManager:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def forecast_volatility(self):
        # For demonstration, use rolling standard deviation as volatility forecast.
        returns = self.data['price'].pct_change().fillna(0)
        volatility = returns.rolling(window=10, min_periods=1).std().iloc[-1] * np.sqrt(252)
        logging.info(f"Volatility Forecast: {volatility*100:.2f}%")
        return volatility

    def dynamic_position_sizing(self, base_size=100):
        # Adjust position size based on recent volatility (lower volatility -> larger positions)
        vol = self.forecast_volatility()
        # For demo: position size scales inversely with volatility
        position_size = base_size * (0.15 / vol) if vol > 0 else base_size
        return position_size

    def adjust_stop_take(self):
        # Adjust stop-loss and take-profit dynamically using volatility forecast.
        vol = self.forecast_volatility()
        stop_loss = 100 - vol * 100  # simplified
        take_profit = 100 + vol * 100
        return stop_loss, take_profit

# ----------------------------
# Main Simulation Class
# ----------------------------
class EnhancedSimulation:
    def __init__(self):
        self.data = None
        self.validator = DataQualityValidator()
        self.feature_engineer = FeatureEngineer()
        self.regime_detector = RegimeDetector(n_regimes=3)
        self.risk_manager = None
        self.strategy_optimizer = None

    def load_data(self):
        # For demonstration, generate synthetic price data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.datetime.now(), periods=300, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
        self.data = pd.DataFrame({'date': dates, 'price': prices})
        self.data.set_index('date', inplace=True)
        # Apply feature engineering
        self.data = self.feature_engineer.add_features(self.data)

    def run(self):
        logging.info("Starting Enhanced Simulation Phase 3 Extended Refined: Robustness, Adaptation & Optimization")
        self.load_data()
        if not self.validator.validate(self.data):
            logging.error("Data validation failed. Exiting simulation.")
            return

        # Regime Detection
        regime = self.regime_detector.detect_regime(self.data)
        # Initialize Risk Manager
        self.risk_manager = RiskManager(self.data)
        position_size = self.risk_manager.dynamic_position_sizing()
        stop_loss, take_profit = self.risk_manager.adjust_stop_take()
        logging.info(f"Risk management parameters: Position size = {position_size:.2f}, Stop-loss = {stop_loss:.2f}, Take-profit = {take_profit:.2f}")

        # Strategy Optimization
        self.strategy_optimizer = StrategyOptimizer(self.data, regime)
        # First use Optuna optimization
        best_params_optuna, obj_optuna = self.strategy_optimizer.optimize(n_trials=30)
        # Then run a simple genetic optimization for further refinement
        best_params_genetic, obj_genetic = self.strategy_optimizer.genetic_optimization(population_size=20, generations=5)

        # Choose the best parameters (for demo, pick the one with lower objective)
        if obj_optuna < obj_genetic:
            best_params = best_params_optuna
            best_obj = obj_optuna
            method = "Optuna"
        else:
            best_params = best_params_genetic
            best_obj = obj_genetic
            method = "Genetic"
        logging.info(f"Selected best parameters from {method} optimization: {best_params} with objective {best_obj:.4f}")

        # Run final simulation with optimized strategy parameters
        final_metrics = self.strategy_optimizer.backtest_strategy(best_params['short_window'], best_params['long_window'])
        logging.info(f"Final simulation - Params: {best_params} | "
                     f"Return={final_metrics['return']*100:.2f}%, Drawdown=-{final_metrics['drawdown']*100:.2f}%, "
                     f"Volatility={final_metrics['volatility']:.2f}, Sharpe={final_metrics['sharpe']:.2f}, "
                     f"Sortino={final_metrics['sortino']:.2f}, Calmar={final_metrics['calmar']:.2f}, "
                     f"CVaR={final_metrics['cvar']:.4f}")
        logging.info("Enhanced Phase 3 Extended Refined simulation completed successfully.")


if __name__ == '__main__':
    simulation = EnhancedSimulation()
    simulation.run()
