from genetic import genetic_algorithm, constants
from my_backtesting import backtest
import numpy as np
import pandas as pd
from input import input_data
import matplotlib.pyplot as plt

TRAINING_CHUNKS = 10
TEST_CHUNKS = 5

def plot_cumulative_returns(total_returns, initial_investment=1000):
    decimal_returns = [r / 100 for r in total_returns]
    cumulative_returns = np.cumprod([1 + r for r in decimal_returns]) * initial_investment
    
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_returns, marker='o', linestyle='-', color='b', label="Cumulative Investment Value")
    plt.xlabel("Time Steps")
    plt.ylabel("Investment Value ($)")
    plt.title(f"Cumulative Returns on ${initial_investment} Investment")
    plt.legend()
    plt.grid(True)
    plt.show()

# DONE: Have different individuals for short and long. Develop 2 populations in parallel for both short and long
# TODO: Let the algo decide which moving average is profitable for the fiven period (SMA, EMA, WMA)
# TODO: play with different fitness functions to see the output.
# TODO: have 2 coeficients that represent the bias of the market (trending, ranging) their sum is 1 - like probabilities
if __name__ == "__main__":
    
    total_returns = []
    total_drawdowns = []
    
    n = len(input_data.BTC_CHUNKS)
    for i in range(0, n - TRAINING_CHUNKS - TEST_CHUNKS, TEST_CHUNKS):
        print(" ")
        print("##############################################")
        print("Iteration: ",  i, " / ", n - TRAINING_CHUNKS - TEST_CHUNKS)
        
        train_data = pd.concat(input_data.BTC_CHUNKS[i : i+TRAINING_CHUNKS])
        test_data = pd.concat(input_data.BTC_CHUNKS[i+TRAINING_CHUNKS-2 : i+TRAINING_CHUNKS+TEST_CHUNKS])
        
        population_long = genetic_algorithm.run_generation(train_data, False)
        individuals_performance_long = backtest.runBackTest(population_long[0:constants.TOP_PICK], test_data, False)
        
        population_short = genetic_algorithm.run_generation(train_data, True)
        individuals_performance_short = backtest.runBackTest(population_short[0:constants.TOP_PICK], test_data, True)

        returns_long = np.array([ind[4] for ind in individuals_performance_long])
        drawdowns_long = np.array([ind[5] for ind in individuals_performance_long]) 
        
        returns_short = np.array([ind[4] for ind in individuals_performance_short])
        drawdowns_short = np.array([ind[5] for ind in individuals_performance_short]) 
        
        average_return_long = np.sum(returns_long * 0.25)
        average_drawdown_long = np.sum(drawdowns_long * 0.25)
        
        average_return_short= np.sum(returns_short * 0.25)
        average_drawdown_short= np.sum(drawdowns_short * 0.25)
        
        print("Buy & Hold return: ", individuals_performance_long[0][6], "%")        
        print("Strategy Performance LONG: ", average_return_long, "% | Max drawdown: ",average_drawdown_long, "%")
        print("Strategy Performance SHORT: ", average_return_short, "% | Max drawdown: ",average_drawdown_short, "%")
        
        total_returns.append(average_return_long)
        total_drawdowns.append(average_return_short)
    
    plot_cumulative_returns(total_returns)
    plot_cumulative_returns(total_drawdowns)