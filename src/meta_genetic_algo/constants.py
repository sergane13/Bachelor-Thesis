class Genetic_Algorithm:
    def __init__(self, POPULATION_SIZE, TOP_PICK, WHEEL_OF_FORTUNE, RANDOM_INDIVIDUALS, MUTATION_PROBABILITY, CROSSOVER_RATE, NO_IMPROVEMENET, MIN_IMPROVEMENT):
        self.POPULATION_SIZE = POPULATION_SIZE
        self.TOP_PICK = TOP_PICK
        self.WHEEL_OF_FORTUNE = WHEEL_OF_FORTUNE
        self.RANDOM_INDIVIDUALS = RANDOM_INDIVIDUALS
        self.MUTATION_PROBABILITY = MUTATION_PROBABILITY
        self.CROSSOVER_RATE = CROSSOVER_RATE
        self.NO_IMPROVEMENET = NO_IMPROVEMENET
        self.MIN_IMPROVEMENT = MIN_IMPROVEMENT

    class Builder:
        def __init__(self):
            self._POPULATION_SIZE = 100
            self._TOP_PICK = 10
            self._WHEEL_OF_FORTUNE = 80
            self._RANDOM_INDIVIDUALS = 10
            self._MUTATION_PROBABILITY = 0.1
            self._CROSSOVER_RATE = 0.7
            self._NO_IMPROVEMENET = 10
            self._MIN_IMPROVEMENT = 0.001

        def set_population_size(self, size):
            self._POPULATION_SIZE = size
            return self

        def set_top_pick(self, top):
            self._TOP_PICK = top
            return self

        def set_wheel_of_fortune(self, flag):
            self._WHEEL_OF_FORTUNE = flag
            return self
        
        def set_random_individual(self, ind):
            self._RANDOM_INDIVIDUALS = ind
            return self

        def set_mutation_probability(self, prob):
            self._MUTATION_PROBABILITY = prob
            return self

        def set_crossover_rate(self, rate):
            self._CROSSOVER_RATE = rate
            return self

        def set_no_improvement(self, gen):
            self._NO_IMPROVEMENET = gen
            return self

        def set_min_improvement(self, min_imp):
            self._MIN_IMPROVEMENT = min_imp
            return self

        def build(self):
            return Genetic_Algorithm(
                self._POPULATION_SIZE,
                self._TOP_PICK,
                self._WHEEL_OF_FORTUNE,
                self._RANDOM_INDIVIDUALS,
                self._MUTATION_PROBABILITY,
                self._CROSSOVER_RATE,
                self._NO_IMPROVEMENET,
                self._MIN_IMPROVEMENT
            )

POPULATION_SIZE = 40
TOP_PICK = 2
WHEEL_OF_FORTUNE = 18
RANDOM_INDIVIDUALS = 20

assert POPULATION_SIZE == TOP_PICK + WHEEL_OF_FORTUNE + RANDOM_INDIVIDUALS, "(1) Invalid Population Sum"
assert POPULATION_SIZE % 2 == 0, "(2) Population size must be even"
assert (RANDOM_INDIVIDUALS + WHEEL_OF_FORTUNE) % 2 == 0, "(3) Must be even"

NO_IMPROVEMENET = 5
MIN_IMPROVEMENT = 0.001

MUTATION_PROBABILITY = 0.0005 # 0.05%
CROSSOVER_RATE = 0.02 # 2%

from src import var_types

GENE_RANGES = {
    var_types.POPULATION: {
        var_types.LOWER: 50, 
        var_types.UPPER: 300
    },
    var_types.TOP_PICK: {
        var_types.LOWER: 1, 
        var_types.UPPER: 20
    },
    var_types.WHEEL_OF_FORTUNE: {
        var_types.LOWER: 2, 
        var_types.UPPER: 280
    },
    var_types.MUTATION_PROBABILITY: {
        var_types.LOWER: 0.0001, 
        var_types.UPPER: 0.1
    },
    var_types.CROSSOVER_RATE: {
        var_types.LOWER: 0.001, 
        var_types.UPPER: 0.5
    },
    var_types.NO_IMPROVEMENET: {
        var_types.LOWER: 2, 
        var_types.UPPER: 10
    },
    var_types.MIN_IMPROVEMENT: {
        var_types.LOWER: 0.0001, 
        var_types.UPPER: 0.01
    },
}

import numpy as np

META_INDIVIDUAL_TYPE = np.dtype([
    (var_types.POPULATION, np.int32),
    (var_types.TOP_PICK, np.int32),
    (var_types.WHEEL_OF_FORTUNE, np.int32),
    (var_types.RANDOM_INDIVIDUALS, np.int32),
    (var_types.MUTATION_PROBABILITY, np.float32),
    (var_types.CROSSOVER_RATE, np.float32),
    (var_types.NO_IMPROVEMENET, np.int32),
    (var_types.MIN_IMPROVEMENT, np.float32),
])



