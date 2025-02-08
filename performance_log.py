import pandas as pd
import matplotlib.pyplot as plt
import os

data_file = "C:\\Users\\gill_\\OneDrive\\Documents\\tradingbot\\trade_performance.csv"

def log_trade(total_trades, wins, losses, avg_return, drawdown):
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    cumulative_profit = avg_return * total_trades
    
    new_data = {
        "Total Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Win Rate (%)": win_rate,
        "Average Return (%)": avg_return,
        "Cumulative Profit (%)": cumulative_profit,
        "Drawdown (%)": drawdown
    }
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        df = df.append(new_data, ignore_index=True)
    else:
        df = pd.DataFrame([new_data])
    
    df.to_csv(data_file, index=False)
    plot_performance(df)

def plot_performance(df):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["Win Rate (%)"], marker='o', linestyle='-', color='green', label='Win Rate (%)')
    plt.xlabel("Trades")
    plt.ylabel("Win Rate (%)")
    plt.title("Trading Performance: Win Rate Over Time")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["Average Return (%)"], marker='x', linestyle='-', color='blue', label='Average Return (%)')
    plt.xlabel("Trades")
    plt.ylabel("Average Return (%)")
    plt.title("Trading Performance: Average Return Over Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage: Update these values from your bot
total_trades = 3
wins = 2
losses = 1
avg_return = 2.43
drawdown = 0.00

log_trade(total_trades, wins, losses, avg_return, drawdown)


