import random
import numpy as np

from src import var_types
from . import constants

random.seed(42)

def generate_random_sample(var_type, is_float=False, decimals=0):
    lower = constants.GENE_RANGES[var_type][var_types.LOWER]
    upper = constants.GENE_RANGES[var_type][var_types.UPPER]

    if is_float:
        return round(random.uniform(lower, upper), decimals)
    return random.randint(lower, upper)


def generate_meta_individual():
    while True:
        pop_size = generate_random_sample(var_types.POPULATION)
        randoms = generate_random_sample(var_types.WHEEL_OF_FORTUNE)

        if randoms >= pop_size - 2:
            continue

        top = generate_random_sample(var_types.TOP_PICK)
        if top >= pop_size - randoms:
            continue

        wheel = pop_size - top - randoms
        if wheel < 2 or (wheel + randoms) % 2 != 0 or pop_size % 2 != 0:
            continue

        mutation_prob = generate_random_sample(var_types.MUTATION_PROBABILITY, is_float=True, decimals=4)
        crossover_rate = generate_random_sample(var_types.CROSSOVER_RATE, is_float=True, decimals=3)
        no_improvement = generate_random_sample(var_types.NO_IMPROVEMENET)
        min_improvement = generate_random_sample(var_types.MIN_IMPROVEMENT, is_float=True, decimals=4)

        return (
            pop_size,
            top,
            wheel,
            randoms,
            mutation_prob,
            crossover_rate,
            no_improvement,
            min_improvement
        )


def generate_meta_population(size=40):
    population = []
    seen = set()

    while len(population) < size:
        indiv = generate_meta_individual()
        if indiv not in seen:
            seen.add(indiv)
            population.append(indiv)

    return np.array(population, dtype=constants.META_INDIVIDUAL_TYPE)
