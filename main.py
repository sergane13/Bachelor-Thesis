import numpy as np
import pandas as pd
import multiprocessing as mp

from src.genetic_algorithm import genetic_algorithm, constants
from src.backtesting_engine import run_backtest
from src.input_data import input_data
from src.input_data import plots

# matplotlib.use('Agg')

TRAINING_CHUNKS = 10
TEST_CHUNKS = 5

def run_generation_parallel(train_data, is_short):
    return genetic_algorithm.run_generation(train_data, is_short)

# TODO: Save individuals and in what market regimes they were trained and deploy them as things evolve
# TODO: Add market regime into trading strategy so strategy adapts live to it.
# TODO: Add money management and if losses are ncreasing cut trades, or if a loss was hit
if __name__ == "__main__":
    current_input_data = input_data.DOT_CHUNKS
    total_combined_returns = []
    
    plots.clear_folder('algo_plots')

    n = len(current_input_data)
    for i in range(0, n - TRAINING_CHUNKS - TEST_CHUNKS, TEST_CHUNKS):
        print(" ")
        print("##############################################")
        print("Iteration: ",  i, " / ", n - TRAINING_CHUNKS - TEST_CHUNKS)
        
        train_data = pd.concat(current_input_data[i : i + TRAINING_CHUNKS])
        emas_data = pd.concat(current_input_data[i + TRAINING_CHUNKS - 3 : i + TRAINING_CHUNKS])
        test_data = pd.concat(current_input_data[i + TRAINING_CHUNKS : i + TRAINING_CHUNKS + TEST_CHUNKS])

        # ----------- Training -----------
        
        with mp.Pool(processes=2) as pool:
            results = pool.starmap(run_generation_parallel, [
                (train_data, False),
                (train_data, True)
            ])

        population_long, _ = results[0]
        population_short, _ = results[1]
        
        # ----------- End Training -----------           
        # ----------- Testing -----------
        
        individuals_performance_long = run_backtest.run_backtest(population_long[0:constants.TOP_PICK], test_data, False, emas_data)
        individuals_performance_short = run_backtest.run_backtest(population_short[0:constants.TOP_PICK], test_data, True, emas_data)
        
        print(individuals_performance_long[0][10])
        
        returns_long = np.array([ind[4] for ind in individuals_performance_long])
        drawdowns_long = np.array([ind[5] for ind in individuals_performance_long]) 
        
        returns_short = np.array([ind[4] for ind in individuals_performance_short])
        drawdowns_short = np.array([ind[5] for ind in individuals_performance_short]) 
        
        average_return_long = np.sum(returns_long * (1 / constants.TOP_PICK))
        average_drawdown_long = np.sum(drawdowns_long * (1 / constants.TOP_PICK))
        
        average_return_short= np.sum(returns_short * (1 / constants.TOP_PICK))
        average_drawdown_short= np.sum(drawdowns_short * (1 / constants.TOP_PICK))

        # ----------- End Testing -----------
        # ----------- Plots -----------
        
        # for ind in individuals_performance_long:
        #     plots.plot_chunk_with_emas(train_data, ind[0], ind[1], price_col='Close', output_folder=f'algo_plots/plots_{i}', filename=f'train_{i}_generation_{ind[0]}_{ind[1]}_long')
        #     plots.plot_chunk_with_emas(test_data, ind[0], ind[1], price_col='Close', output_folder=f'algo_plots/plots_{i}', filename=f'test_{i}_generation_{ind[0]}_{ind[1]}_long')

        # for ind in individuals_performance_short:
        #     plots.plot_chunk_with_emas(train_data, ind[0], ind[1], price_col='Close', output_folder=f'algo_plots/plots_{i}', filename=f'train_{i}_generation_{ind[0]}_{ind[1]}_short')
        #     plots.plot_chunk_with_emas(test_data, ind[0], ind[1], price_col='Close', output_folder=f'algo_plots/plots_{i}', filename=f'test_{i}_generation_{ind[0]}_{ind[1]}_short')

        # ----------- End Plots -----------
                
        average_return_total = 0.5 * average_return_long + 0.5 * average_return_short
        
        print("Buy & Hold return: ", individuals_performance_long[0][6], "%")        
        print("Strategy Performance LONG: ", average_return_long, "% | Max drawdown: ",average_drawdown_long, "%")
        print("Strategy Performance SHORT: ", average_return_short, "% | Max drawdown: ",average_drawdown_short, "%")
        print("Combined Strategy: Return: ", average_return_total, "%")
        
        total_combined_returns.append(average_return_total)
    
    plots.plot_cumulative_returns_with_drawdown(total_combined_returns)