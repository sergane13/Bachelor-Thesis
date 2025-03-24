from genetic import genetic_algorithm, constants
from my_backtesting import backtest
import numpy as np
import pandas as pd
from input import input_data
import matplotlib.pyplot as plt
import multiprocessing as mp

TRAINING_CHUNKS = 8
TEST_CHUNKS = 5

def run_generation_parallel(train_data, is_short):
    return genetic_algorithm.run_generation(train_data, is_short)

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


# TODO: Save individuals and in what market regimes they were trained and deploy them as things evolve
# TODO: play with different fitness functions to see the output.
if __name__ == "__main__":
    
    total_combined_returns = []
    
    n = len(input_data.BTC_CHUNKS)
    for i in range(0, n - TRAINING_CHUNKS - TEST_CHUNKS, TEST_CHUNKS):
        print(" ")
        print("##############################################")
        print("Iteration: ",  i, " / ", n - TRAINING_CHUNKS - TEST_CHUNKS)
        
        train_data = pd.concat(input_data.BTC_CHUNKS[i : i + TRAINING_CHUNKS])
        emas_data = pd.concat(input_data.BTC_CHUNKS[i + TRAINING_CHUNKS - 3 : i + TRAINING_CHUNKS])
        test_data = pd.concat(input_data.BTC_CHUNKS[i + TRAINING_CHUNKS : i + TRAINING_CHUNKS + TEST_CHUNKS])

        with mp.Pool(processes=2) as pool:
            results = pool.starmap(run_generation_parallel, [
                (train_data, False),
                (train_data, True)
            ])

        population_long, train_max_fitness_long = results[0]
        population_short, train_max_fitness_short = results[1]
        
        regimes = backtest.get_training_regime_range(train_data)

        individuals_performance_long = backtest.runBackTest(population_long[0:constants.TOP_PICK], test_data, False, emas_data)
        individuals_performance_short = backtest.runBackTest(population_short[0:constants.TOP_PICK], test_data, True, emas_data)

        returns_long = np.array([ind[4] for ind in individuals_performance_long])
        drawdowns_long = np.array([ind[5] for ind in individuals_performance_long]) 
        
        returns_short = np.array([ind[4] for ind in individuals_performance_short])
        drawdowns_short = np.array([ind[5] for ind in individuals_performance_short]) 
        
        average_return_long = np.sum(returns_long * (1 / constants.TOP_PICK))
        average_drawdown_long = np.sum(drawdowns_long * (1 / constants.TOP_PICK))
        
        average_return_short= np.sum(returns_short * (1 / constants.TOP_PICK))
        average_drawdown_short= np.sum(drawdowns_short * (1 / constants.TOP_PICK))
        
        average_return_total = (train_max_fitness_long * average_return_long + train_max_fitness_short * average_return_short ) / (train_max_fitness_long + train_max_fitness_short)
        average_drawdown_total = (train_max_fitness_long * average_drawdown_long + train_max_fitness_short * average_drawdown_short) / (train_max_fitness_long + train_max_fitness_short)
        
        print("Market regime: ", regimes)
        print("Buy & Hold return: ", (individuals_performance_long[0][6] + individuals_performance_long[1][6]) / 2, "%")        
        print("Strategy Performance LONG: ", average_return_long, "% | Max drawdown: ",average_drawdown_long, "%")
        print("Strategy Performance SHORT: ", average_return_short, "% | Max drawdown: ",average_drawdown_short, "%")
        print("Combined Strategy: Return: ", average_return_total, "% | Max drawdown: ", average_drawdown_total, "%")
        
        total_combined_returns.append(average_return_total)
    
    plot_cumulative_returns_with_drawdown(total_combined_returns)