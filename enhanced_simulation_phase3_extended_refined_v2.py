# File: enhanced_simulation_phase3_extended_refined_v2.py
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
    std = np.std(returns)
    if std == 0:
        return 0
    return (np.mean(returns) - risk_free_rate) / std

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

def composite_objective(metrics, weights):
    """
    Compute a composite objective as a weighted sum.
    Lower objective is better.
    Metrics keys include: 'return', 'drawdown', 'sharpe', 'sortino', 'calmar', 'cvar'
    (For returns and sharpe, a negative weight means that higher is better.)
    """
    obj = 0.0
    for key in weights:
        obj += weights[key] * metrics.get(key, 0)
    return obj

# ----------------------------
# Data Quality & Feature Engineering
# ----------------------------
class DataQualityValidator:
    def validate(self, data: pd.DataFrame) -> bool:
        if data.isnull().values.any():
            logging.warning("Data contains missing values.")
            return False
        # Check for extreme outliers in 'price'
        if (data['price'] > data['price'].mean() * 3).sum() > 0:
            logging.warning("Anomaly detected in price values.")
            return False
        logging.info("Data quality validation passed with no anomalies.")
        return True

class FeatureEngineer:
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data['ma_10'] = data['price'].rolling(window=10, min_periods=1).mean()
        # Simulated market microstructure: bid-ask spread
        data['bid_ask_spread'] = np.random.uniform(0.01, 0.05, len(data))
        # Simulated alternative data: sentiment (e.g., from social media/news)
        data['sentiment'] = np.random.uniform(-1, 1, len(data))
        return data

# ----------------------------
# Regime Detection & Adaptive Strategy Switching
# ----------------------------
class RegimeDetector:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=self.n_regimes, random_state=42)

    def detect_regime(self, data: pd.DataFrame) -> str:
        df = data.copy()
        df['return'] = df['price'].pct_change().fillna(0)
        df['volatility'] = df['return'].rolling(window=10, min_periods=1).std().fillna(0)
        features = df[['return', 'volatility']].values
        window = features[-100:] if len(features) >= 100 else features
        clusters = self.model.fit_predict(window)
        current_cluster = clusters[-1]
        regime_map = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
        regime = regime_map.get(current_cluster, "Sideways")
        logging.info(f"Regime Detection: Current regime = {regime}")
        return regime

# ----------------------------
# Strategy Optimization (Hybrid: Optuna + Genetic Algorithm)
# ----------------------------
class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame, regime: str):
        self.data = data
        self.regime = regime
        # Weights for composite objective.
        # (Negative weights for metrics where higher is better.)
        self.weights = {
            'return': -0.3,
            'drawdown': 0.3,
            'sharpe': -0.2,
            'sortino': -0.1,
            'calmar': -0.1,
            'cvar': 0.3,
        }
        # Regime multiplier for adjusting aggressiveness
        if regime == "Bullish":
            self.regime_multiplier = 0.9  # more aggressive
        elif regime == "Bearish":
            self.regime_multiplier = 1.1  # more conservative
        else:
            self.regime_multiplier = 1.0

    def backtest_strategy(self, short_window, long_window):
        df = self.data.copy()
        df['short_ma'] = df['price'].rolling(window=short_window, min_periods=1).mean()
        df['long_ma'] = df['price'].rolling(window=long_window, min_periods=1).mean()
        df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
        df['strategy_return'] = df['signal'].shift(1) * df['price'].pct_change().fillna(0)
        cumulative_return = np.prod(1 + df['strategy_return']) - 1
        # Simplified max drawdown estimation over a rolling window:
        max_drawdown = df['strategy_return'].rolling(window=100, min_periods=1).sum().min()
        sr = sharpe_ratio(df['strategy_return'])
        so = sortino_ratio(df['strategy_return'])
        annual_return = (1 + cumulative_return)**(252/len(df)) - 1 if len(df) > 0 else 0
        calmar = calmar_ratio(annual_return, max_drawdown)
        cvar = conditional_var(df['strategy_return'])
        volatility = df['strategy_return'].std() * np.sqrt(252)
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
        short_window = trial.suggest_int('short_window', 3, 20)
        long_window = trial.suggest_int('long_window', short_window+1, 50)
        metrics = self.backtest_strategy(short_window, long_window)
        obj = self.regime_multiplier * composite_objective(metrics, self.weights)
        logging.info(f"Trial params: short_window={short_window}, long_window={long_window} | "
                     f"Return={metrics['return']*100:.2f}%, Drawdown={-metrics['drawdown']*100:.2f}%, "
                     f"Sharpe={metrics['sharpe']:.2f}, Sortino={metrics['sortino']:.2f}, "
                     f"Calmar={metrics['calmar']:.2f}, CVaR={metrics['cvar']:.4f}, "
                     f"Volatility={metrics['volatility']:.2f} | Objective={obj:.4f}")
        return obj

    def optimize(self, n_trials=30):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        best_params = study.best_trial.params
        best_obj = study.best_trial.value
        logging.info(f"Optuna Optimization complete. Best parameters: {best_params} with objective {best_obj:.4f}")
        return best_params, best_obj

    def genetic_optimization(self, population_size=20, generations=5):
        def random_individual():
            return {
                'short_window': random.randint(3, 20),
                'long_window': random.randint(5, 50)
            }
        population = [random_individual() for _ in range(population_size)]
        for gen in range(generations):
            scored = []
            for individual in population:
                if individual['long_window'] <= individual['short_window']:
                    individual['long_window'] = individual['short_window'] + 1
                metrics = self.backtest_strategy(individual['short_window'], individual['long_window'])
                score = self.regime_multiplier * composite_objective(metrics, self.weights)
                scored.append((score, individual))
            scored.sort(key=lambda x: x[0])
            logging.info(f"Generation {gen} best score: {scored[0][0]:.4f}")
            survivors = [ind for score, ind in scored[:population_size//2]]
            offspring = []
            while len(offspring) < population_size - len(survivors):
                parent1, parent2 = random.sample(survivors, 2)
                child = {
                    'short_window': random.choice([parent1['short_window'], parent2['short_window']]),
                    'long_window': random.choice([parent1['long_window'], parent2['long_window']])
                }
                if random.random() < 0.3:
                    child['short_window'] += random.randint(-1, 1)
                    child['short_window'] = max(3, child['short_window'])
                if random.random() < 0.3:
                    child['long_window'] += random.randint(-2, 2)
                    child['long_window'] = max(child['short_window']+1, child['long_window'])
                offspring.append(child)
            population = survivors + offspring
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
        returns = self.data['price'].pct_change().fillna(0)
        vol = returns.rolling(window=10, min_periods=1).std().iloc[-1] * np.sqrt(252)
        logging.info(f"Volatility Forecast: {vol*100:.2f}%")
        return vol

    def dynamic_position_sizing(self, base_size=100):
        vol = self.forecast_volatility()
        position_size = base_size * (0.15 / vol) if vol > 0 else base_size
        return position_size

    def adjust_stop_take(self):
        vol = self.forecast_volatility()
        stop_loss = 100 - vol * 100  # simplified example
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
        np.random.seed(42)
        dates = pd.date_range(end=datetime.datetime.now(), periods=300, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
        self.data = pd.DataFrame({'date': dates, 'price': prices})
        self.data.set_index('date', inplace=True)
        self.data = self.feature_engineer.add_features(self.data)

    def run(self):
        logging.info("Starting Enhanced Simulation Phase 3 Extended Refined v2: Robustness, Adaptation & Optimization")
        self.load_data()
        if not self.validator.validate(self.data):
            logging.error("Data validation failed. Exiting simulation.")
            return

        regime = self.regime_detector.detect_regime(self.data)
        self.risk_manager = RiskManager(self.data)
        position_size = self.risk_manager.dynamic_position_sizing()
        stop_loss, take_profit = self.risk_manager.adjust_stop_take()
        logging.info(f"Risk management parameters: Position size = {position_size:.2f}, Stop-loss = {stop_loss:.2f}, Take-profit = {take_profit:.2f}")

        self.strategy_optimizer = StrategyOptimizer(self.data, regime)
        best_params_optuna, obj_optuna = self.strategy_optimizer.optimize(n_trials=30)
        best_params_genetic, obj_genetic = self.strategy_optimizer.genetic_optimization(population_size=20, generations=5)

        # Choose the best parameters based on the lower objective value
        if obj_optuna < obj_genetic:
            best_params = best_params_optuna
            best_obj = obj_optuna
            method = "Optuna"
        else:
            best_params = best_params_genetic
            best_obj = obj_genetic
            method = "Genetic"
        logging.info(f"Selected best parameters from {method} optimization: {best_params} with objective {best_obj:.4f}")

        final_metrics = self.strategy_optimizer.backtest_strategy(best_params['short_window'], best_params['long_window'])
        logging.info(f"Final simulation - Params: {best_params} | "
                     f"Return={final_metrics['return']*100:.2f}%, Drawdown={-final_metrics['drawdown']*100:.2f}%, "
                     f"Volatility={final_metrics['volatility']:.2f}, Sharpe={final_metrics['sharpe']:.2f}, "
                     f"Sortino={final_metrics['sortino']:.2f}, Calmar={final_metrics['calmar']:.2f}, "
                     f"CVaR={final_metrics['cvar']:.4f}")
        logging.info("Enhanced Phase 3 Extended Refined v2 simulation completed successfully.")

if __name__ == '__main__':
    simulation = EnhancedSimulation()
    simulation.run()
