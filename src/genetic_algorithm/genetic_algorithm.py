from . import generate_offsprings
from . import constants
from src import var_types
from src.backtesting_engine import run_backtest
import random
import copy
import numpy as np

random.seed(13)

def fitness_function(individuals):
    ind_return = individuals[var_types.RETURN] 
    ind_drawdown = individuals[var_types.MAX_DRAWDOWN]
        
    drawdown_penalty = (np.abs(ind_drawdown) + 1e-1) ** 0.8
    score = ind_return / drawdown_penalty
    final_score = np.maximum(0, score)
    
    short_ma = individuals[var_types.SHORT_MA]
    long_ma = individuals[var_types.LONG_MA]

    invalid_logic = (short_ma >= long_ma) | ((long_ma - short_ma) < 10)
    final_score[invalid_logic] = 0
    
    scored_individuals = np.zeros(len(individuals), dtype=constants.INDIVIDUAL_TYPE_SCORE)
    scored_individuals[var_types.SHORT_MA] = individuals[var_types.SHORT_MA]
    scored_individuals[var_types.LONG_MA] = individuals[var_types.LONG_MA]
    scored_individuals[var_types.STOP_LOSS_MULTIPLIER] = individuals[var_types.STOP_LOSS_MULTIPLIER]
    scored_individuals[var_types.POSITION_SIZE] = individuals[var_types.POSITION_SIZE]
    scored_individuals[var_types.SCORE] = final_score

    sorted_individuals = np.sort(scored_individuals, order=var_types.SCORE)[::-1]
    return sorted_individuals

def wheel_of_fortune(individuals):
    fitness_scores = np.array([ind[-1] for ind in individuals], dtype=np.float32)
    
    if (np.sum(fitness_scores) == 0):
        return  generate_offsprings.generatePopulation(constants.WHEEL_OF_FORTUNE)
        
    selection_probabilities = fitness_scores / np.sum(fitness_scores)
    
    selected_indices = np.random.choice(len(individuals), size=constants.WHEEL_OF_FORTUNE, p=selection_probabilities)
    selected_individuals_wheel = individuals[selected_indices]
            
    selected_individuals = np.zeros(len(selected_individuals_wheel), dtype=constants.INDIVIDUAL_TYPE)
    selected_individuals[var_types.SHORT_MA] = selected_individuals_wheel[var_types.SHORT_MA]
    selected_individuals[var_types.LONG_MA] = selected_individuals_wheel[var_types.LONG_MA]
    selected_individuals[var_types.STOP_LOSS_MULTIPLIER] = selected_individuals_wheel[var_types.STOP_LOSS_MULTIPLIER]
    selected_individuals[var_types.POSITION_SIZE] = selected_individuals_wheel[var_types.POSITION_SIZE]

    return selected_individuals

def crossover(individuals):
    individuals_list = list(individuals)  
    random.shuffle(individuals_list)
    
    new_population = []

    for index in range(0, len(individuals_list), 2):
        parent1 = list(copy.deepcopy(individuals_list[index]))
        parent2 = list(copy.deepcopy(individuals_list[index + 1]))
    
        for _, genes in constants.CROSSOVER_GROUPS.items():
            if random.random() < constants.CROSSOVER_RATE:
                for gene in genes:
                    parent1[gene], parent2[gene] = parent2[gene], parent1[gene]

        new_population.append(tuple(parent1))
        new_population.append(tuple(parent2))

    return np.array(new_population, dtype=constants.INDIVIDUAL_TYPE)

def mutation(individuals):
    individuals_list = list(individuals)  
    mutated_individuals = []
    
    for individual in individuals_list:
        mutated_individual = list(copy.deepcopy(individual))

        for index, _ in enumerate(mutated_individual):
            if random.random() < constants.MUTATION_PROBABILITY:            
                if index == 0:
                    mutated_individual[0] = generate_offsprings.generate_Short_MA()
                elif index == 1:
                    mutated_individual[1] = generate_offsprings.generate_Long_MA()
                elif index == 2:
                    mutated_individual[2] = generate_offsprings.generate_Stop_Loss_Multiplier()
                elif index == 3:
                    mutated_individual[3] = generate_offsprings.generate_Position_Size()

        mutated_individuals.append(tuple(mutated_individual))
    
    return np.array(mutated_individuals, dtype=constants.INDIVIDUAL_TYPE)

def return_top_individuals(individuals):
    top_individuals = individuals[0:constants.TOP_PICK]

    selected_individuals = np.zeros(len(top_individuals), dtype=constants.INDIVIDUAL_TYPE)
    selected_individuals[var_types.SHORT_MA] = top_individuals[var_types.SHORT_MA]
    selected_individuals[var_types.LONG_MA] = top_individuals[var_types.LONG_MA]
    selected_individuals[var_types.STOP_LOSS_MULTIPLIER] = top_individuals[var_types.STOP_LOSS_MULTIPLIER]
    selected_individuals[var_types.POSITION_SIZE] = top_individuals[var_types.POSITION_SIZE]
    
    return selected_individuals
    
def create_new_generation(individuals):
    top_individuals = return_top_individuals(individuals)
    remaining_individuals = individuals[constants.TOP_PICK:-constants.RANDOM_INDIVIDUALS]
    selected_individuals_by_wheel_of_fortune = wheel_of_fortune(remaining_individuals)
    random_individuals = generate_offsprings.generatePopulation(constants.RANDOM_INDIVIDUALS)
    
    assert len(individuals) - len(top_individuals) == len(selected_individuals_by_wheel_of_fortune) + len(random_individuals)
    offsprings = np.concatenate((selected_individuals_by_wheel_of_fortune, random_individuals), axis=0)
    offsprings = crossover(offsprings)
    offsprings = mutation(offsprings)

    return np.concatenate((top_individuals, offsprings), axis=0)

##
## This method runs the genetic algotihm until fitness function has no improvement ##
##
def run_generation(train_data, isShort = False):
    fitness_scores_generation = []
    offsprings = generate_offsprings.generatePopulation(constants.POPULATION_SIZE)
    
    index = 0
    while True:
        print(f"Generation: {index}")
        
        individuals_performance = run_backtest.run_backtest(offsprings, train_data, isShort)
        individuals_score = fitness_function(individuals_performance)
        
        fitness_scores = np.array([ind[-1] for ind in individuals_score], dtype=np.float32)
        fitness_scores_generation.append(np.sum(fitness_scores[0:constants.TOP_PICK]))
        
        print("Fitness score: ", fitness_scores_generation[-1])
        
        if len(fitness_scores_generation) > constants.NO_IMPROVEMENET:
            recent_scores = np.array(fitness_scores_generation[-constants.NO_IMPROVEMENET:])
            improvement = np.max(recent_scores) - np.min(recent_scores)

            if improvement < constants.MIN_IMPROVEMENT:
                print("Plateu")
                break
    
        offsprings = create_new_generation(individuals_score)
        index += 1
    
    print('')
    
    if isShort:
        print("Short only offsprings (best ones): ")
    else:
        print("Long only offsprings (best ones): ")
    
    print(offsprings[0], offsprings[1])
    print("")
    
    return offsprings, fitness_scores_generation[-1]