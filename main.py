import numpy as np
import pandas as pd
import multiprocessing as mp
import json
import os

from src.trading_genetic_algo import genetic_algorithm, constants
from src.backtesting_engine import run_backtest
from src.input_data import input_data
from src.input_data import plots
from src.market_memory import memory
from src import var_types
from src.meta_genetic_algo import generate_offsrings as meta_genetic
from src.meta_genetic_algo.genetic_algorithm import MetaGeneticAlgorithm
from src.meta_genetic_algo import constants
from src.trading_genetic_algo.genetic_algorithm import Genetic_Algorithm 

# matplotlib.use('Agg')

TRAINING_CHUNKS = 10
TEST_CHUNKS = 8

def save_backtest_result(index, average_return_long, average_drawdown_long, returns_long, drawdowns_long,
                         average_return_short, average_drawdown_short, returns_short, drawdowns_short,
                         filename="backtest_log.json"):
    new_result = {
        "long": {
            "average_return": float(average_return_long),
            "average_drawdown": float(average_drawdown_long),
            "returns": [float(x) for x in returns_long.tolist()],
            "drawdowns": [float(x) for x in drawdowns_long.tolist()]
        },
        "short": {
            "average_return": float(average_return_short),
            "average_drawdown": float(average_drawdown_short),
            "returns": [float(x) for x in returns_short.tolist()],
            "drawdowns": [float(x) for x in drawdowns_short.tolist()]
        }
    }

    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}

    all_results[str(index)] = new_result

    with open(filename, "w") as f:
        json.dump(all_results, f, indent=4)


def run_generation_parallel(method, is_short):
    return method.run_generation(is_short)


# TODO: Save individuals and in what market regimes they were trained and deploy them as things evolve
# TODO: Add a meta genetic algorithm to find the best parameters for the final algo
if __name__ == "__main__":
    current_input_data = input_data.HOT_CHUNKS
    total_combined_returns = []
    
    plots.clear_folder('algo_plots')

    # n = len(current_input_data)
    # for i in range(3, n - TRAINING_CHUNKS - TEST_CHUNKS, TEST_CHUNKS):
    #     print(" ")
    #     print("##############################################")
    #     print("Iteration: ",  i, " / ", n - TRAINING_CHUNKS - TEST_CHUNKS)

    #     train_data_ema_warmup = pd.concat(current_input_data[i - 3 : i])
    #     train_data = pd.concat(current_input_data[i : i + TRAINING_CHUNKS])
    #     test_emas_data_warmup = pd.concat(current_input_data[i + TRAINING_CHUNKS - 3 : i + TRAINING_CHUNKS])
    #     test_data = pd.concat(current_input_data[i + TRAINING_CHUNKS : i + TRAINING_CHUNKS + TEST_CHUNKS])

    #     meta_ga = MetaGeneticAlgorithm(train_data, train_data_ema_warmup)
    #     best_params = meta_ga.run()

    #     print("\nBest discovered meta-individual:")
    #     print(best_params)


    n = len(current_input_data)
    for i in range(3, n - TRAINING_CHUNKS - TEST_CHUNKS, TEST_CHUNKS):
        constant = (
        constants.Genetic_Algorithm
            .Builder()
            .set_population_size(100)
            .set_top_pick(2)
            .set_wheel_of_fortune(78)
            .set_random_individual(20)
            .set_mutation_probability(0.001)
            .set_crossover_rate(0.01)
            .set_no_improvement(5)
            .set_min_improvement(0.001)
            .build()
        )
        
        print(" ")
        print("##############################################")
        print("Iteration: ",  i, " / ", n - TRAINING_CHUNKS - TEST_CHUNKS)
        
        train_data_ema_warmup = pd.concat(current_input_data[i - 3 : i])
        train_data = pd.concat(current_input_data[i : i + TRAINING_CHUNKS])
        test_emas_data_warmup = pd.concat(current_input_data[i + TRAINING_CHUNKS - 3 : i + TRAINING_CHUNKS])
        test_data = pd.concat(current_input_data[i + TRAINING_CHUNKS : i + TRAINING_CHUNKS + TEST_CHUNKS])

        plots.validate_warmup_main_no_internal_duplicates("train_data", train_data_ema_warmup, train_data)
        plots.validate_warmup_main_no_internal_duplicates("test_data", test_emas_data_warmup, test_data)
        
    #     ----------- Training -----------
        
        genetic_algo = Genetic_Algorithm(constant, train_data, train_data_ema_warmup)

        with mp.Pool(processes=2) as pool:
            results = pool.starmap(run_generation_parallel, [
                (genetic_algo, False),
                (genetic_algo, True)
            ])

        population_long, fitness_long = results[0]
        population_short, fitness_short = results[1]
        
    #     ----------- End Training -----------     
    #     -------------- Memory --------------
        
    #     for ind, fit in zip(population_long, fitness_long):
    #         memory.memory.add_long_individual(ind, fit)
        
    #     for ind, fit in zip(population_short, fitness_short):
    #         memory.memory.add_short_individual(ind, fit)
        
    #     top_long_individuals = memory.memory.get_top_long()
    #     top_short_individuals = memory.memory.get_top_short()
        
    #     ------------ End --------------
    #     ----------- Testing -----------
        
        individuals_performance_long = run_backtest.run_backtest(population_long[0:constants.TOP_PICK], test_data, False, test_emas_data_warmup)
        individuals_performance_short = run_backtest.run_backtest(population_short[0:constants.TOP_PICK], test_data, True, test_emas_data_warmup)
        
        print(individuals_performance_long[0][10])
        
        returns_long = np.array([ind[4] for ind in individuals_performance_long])
        drawdowns_long = np.array([ind[5] for ind in individuals_performance_long]) 
        
        returns_short = np.array([ind[4] for ind in individuals_performance_short])
        drawdowns_short = np.array([ind[5] for ind in individuals_performance_short]) 
        
        average_return_long = np.sum(returns_long * (1 / constants.TOP_PICK))
        average_drawdown_long = np.sum(drawdowns_long * (1 / constants.TOP_PICK))
        
        average_return_short= np.sum(returns_short * (1 / constants.TOP_PICK))
        average_drawdown_short= np.sum(drawdowns_short * (1 / constants.TOP_PICK))

        save_backtest_result(
            index=i,
            average_return_long=average_return_long,
            average_drawdown_long=average_drawdown_long,
            returns_long=returns_long,
            drawdowns_long=drawdowns_long,
            average_return_short=average_return_short,
            average_drawdown_short=average_drawdown_short,
            returns_short=returns_short,
            drawdowns_short=drawdowns_short
        )
        
    #     ----------- End Testing -----------
    #     ----------- Plots -----------
        
        for ind in individuals_performance_long:
            plots.plot_chunk_with_emas(pd.concat([train_data_ema_warmup, train_data]), ind[0], ind[1], price_col='Close', output_folder=f'algo_plots/plots_{i}', filename=f'train_{i}_generation_{ind[0]}_{ind[1]}_long')
            plots.plot_chunk_with_emas(pd.concat([test_emas_data_warmup, test_data]), ind[0], ind[1], price_col='Close', output_folder=f'algo_plots/plots_{i}', filename=f'test_{i}_generation_{ind[0]}_{ind[1]}_long')

        for ind in individuals_performance_short:
            plots.plot_chunk_with_emas(pd.concat([train_data_ema_warmup, train_data]), ind[0], ind[1], price_col='Close', output_folder=f'algo_plots/plots_{i}', filename=f'train_{i}_generation_{ind[0]}_{ind[1]}_short')
            plots.plot_chunk_with_emas(pd.concat([test_emas_data_warmup, test_data]), ind[0], ind[1], price_col='Close', output_folder=f'algo_plots/plots_{i}', filename=f'test_{i}_generation_{ind[0]}_{ind[1]}_short')

    #     ----------- End Plots -----------
                
        average_return_total = 0.5 * average_return_long + 0.5 * average_return_short
        
        print("Buy & Hold return: ", individuals_performance_long[0][6], "%")        
        print("Strategy Performance LONG: ", average_return_long, "% | Max drawdown: ",average_drawdown_long, "%")
        print("Strategy Performance SHORT: ", average_return_short, "% | Max drawdown: ",average_drawdown_short, "%")
        print("Combined Strategy: Return: ", average_return_total, "%")
        
        total_combined_returns.append(average_return_total)
    
    plots.plot_cumulative_returns_with_drawdown(total_combined_returns)