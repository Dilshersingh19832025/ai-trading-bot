import matplotlib.pyplot as plt
import numpy as np

# Simulated trading data (Replace with real trading data integration)
num_trades = 400
win_rate = np.random.uniform(60, 75, num_trades).tolist()
avg_return = np.random.uniform(-0.5, 0.5, num_trades).tolist()
cumulative_profit = np.cumsum(avg_return)
drawdowns = np.maximum.accumulate(cumulative_profit) - cumulative_profit

# Plot performance metrics
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Win Rate Over Time
axs[0].plot(win_rate, marker='o', linestyle='-', color='green', markersize=4, label='Win Rate (%)')
axs[0].set_title("Trading Performance: Win Rate Over Time")
axs[0].set_xlabel("Trades")
axs[0].set_ylabel("Win Rate (%)")
axs[0].legend()
axs[0].grid(True)

# Average Return Over Time
axs[1].plot(avg_return, marker='x', linestyle='-', color='blue', markersize=4, label='Average Return (%)')
axs[1].set_title("Trading Performance: Average Return Over Time")
axs[1].set_xlabel("Trades")
axs[1].set_ylabel("Average Return (%)")
axs[1].legend()
axs[1].grid(True)

# Cumulative Profit & Drawdown Over Time
axs[2].plot(cumulative_profit, color='black', label='Cumulative Profit')
axs[2].plot(drawdowns, color='red', linestyle='--', label='Drawdown')
axs[2].set_title("Cumulative Profit & Drawdown Over Time")
axs[2].set_xlabel("Trades")
axs[2].set_ylabel("Profit / Drawdown")
axs[2].legend()
axs[2].grid(True)

# Show the updated visualization
plt.tight_layout()
plt.show()








