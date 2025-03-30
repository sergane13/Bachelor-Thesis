import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

from backtesting_engine import indicators

def plot_cumulative_returns_with_drawdown(total_returns, initial_investment=1000):
    decimal_returns = [r / 100 for r in total_returns]
    cumulative_returns = np.cumprod([1 + r for r in decimal_returns]) * initial_investment

    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    
    max_drawdown = np.min(drawdowns)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    axs[0].plot(cumulative_returns, marker='o', linestyle='-', color='blue')
    axs[0].set_ylabel("Investment Value ($)")
    axs[0].set_title(f"Cumulative Returns on ${initial_investment} Investment")
    axs[0].grid(True)

    axs[1].fill_between(range(len(drawdowns)), drawdowns, color='red', alpha=0.6)
    axs[1].axhline(max_drawdown, color='black', linestyle='--', label=f"Max Drawdown: {max_drawdown*100:.2f}%")
    axs[1].set_ylabel("Drawdown (%)")
    axs[1].set_xlabel("Time Steps")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_chunk_with_emas(chunk, short_period, long_period, price_col='Close', output_folder='plots', filename='chunk_with_emas.png'):
    os.makedirs(output_folder, exist_ok=True)

    short_dema = indicators.double_exponential_moving_average(chunk[price_col], short_period)
    long_dema = indicators.double_exponential_moving_average(chunk[price_col], long_period)

    plt.figure(figsize=(14, 6))
    plt.plot(chunk.index, chunk[price_col], label='Price', linewidth=1.5)
    plt.plot(chunk.index, short_dema, label=f'Short EMA ({short_period})', linestyle='--')
    plt.plot(chunk.index, long_dema, label=f'Long EMA ({long_period})', linestyle='--')
    
    plt.title(f'{price_col} with EMAs ({short_period}, {long_period})')
    plt.xlabel('Date')
    plt.ylabel('Price [$]')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    path = os.path.join(output_folder, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Plot saved to {path}')
    
def clear_folder(parent_folder):
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)