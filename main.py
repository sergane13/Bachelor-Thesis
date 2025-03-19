from genetic import genetic_algorithm, constants
from my_backtesting import backtest
import numpy as np
import pandas as pd
from input import input_data
import matplotlib.pyplot as plt

TRAINING_CHUNKS = 15

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

# TODO: Adding enough data so averages can be computes easily. 500 data points + 100 points lost due to avergaing
if __name__ == "__main__":
    
    total_returns = []
    total_drawdowns = []
    
    n = len(input_data.TQQQ_CHUNKS)
    for i in range(n - TRAINING_CHUNKS):
        print("Chunk tested: ",  i + TRAINING_CHUNKS, " / ", n)
        
        fitness_scores_generation = []

        train_data = pd.concat(input_data.BTC_CHUNKS[i : i+TRAINING_CHUNKS])
        test_data = pd.concat([input_data.BTC_CHUNKS[i+TRAINING_CHUNKS - 1 : i+TRAINING_CHUNKS]])
            
        population = genetic_algorithm.initial_population
        
        for index in range(constants.GENERATIONS):
            print(i, " - Generation: ", index)
            individuals_performance = backtest.runBackTest(population, train_data)
            individuals_score = genetic_algorithm.fitness_function(individuals_performance)
            
            fitness_scores = np.array([ind[-1] for ind in individuals_score], dtype=np.float32)
            fitness_scores_generation.append(np.sum(fitness_scores) / len(fitness_scores))
            
            population = genetic_algorithm.create_new_generation(individuals_score)
                
        print(fitness_scores_generation)
        print('')
            
        print("Buy and hold: ")
        total_return, max_drawdown = (backtest.calculate_buy_and_hold_metrics(test_data))
        print("Total Return: ", float(total_return * 100), '%')
        print("Max Drawdown: ", float(max_drawdown * 100), '%')
        
        print('')
        
        individuals_performance = backtest.runBackTest(population[0:5], test_data)

        returns = np.array([ind[4] for ind in individuals_performance])
        drawdowns = np.array([ind[5] for ind in individuals_performance]) 
        
        average_return = np.sum(returns * 0.2)
        average_drawdown = np.sum(drawdowns * 0.2)
        
        print("Strategy Performance; ", average_return, average_drawdown)
        
        total_returns.append(average_return)
        total_drawdowns.append(average_drawdown)
    
    plot_cumulative_returns(total_returns)