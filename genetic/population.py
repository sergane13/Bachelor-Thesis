import random
import numpy as np

from shared import var_types
from . import constants

def generate_random_sample(var_type, is_float=False, decimals=0):
    lower = constants.GENE_RANGES[var_type][var_types.LOWER]
    upper = constants.GENE_RANGES[var_type][var_types.UPPER]

    if is_float:
        return round(random.uniform(lower, upper), decimals)
    return random.randint(lower, upper)

def generate_Short_MA():
    return generate_random_sample(var_types.SHORT_MA)

def generate_Long_MA():
    return generate_random_sample(var_types.LONG_MA)

def generate_Stop_Loss():
    return generate_random_sample(var_types.STOP_LOSS, is_float=True, decimals=1)

def generate_Position_Size():
    return generate_random_sample(var_types.POSITION_SIZE, is_float=True, decimals=3)

def generatePopulation(population = 100):
    created_population = []
    seen_individuals = set()
    
    while len(created_population) < population:
        individual = (
            generate_Short_MA(),
            generate_Long_MA(),
            generate_Stop_Loss(),
            generate_Position_Size(),
        )
        
        if individual not in seen_individuals:
            seen_individuals.add(individual)
            created_population.append(individual)

    individual_dtype = np.dtype([
        (var_types.SHORT_MA, np.int32),
        (var_types.LONG_MA, np.int32),
        (var_types.STOP_LOSS, np.float32),
        (var_types.POSITION_SIZE, np.float32)
    ])
    
    return  np.array(created_population, dtype=individual_dtype)
