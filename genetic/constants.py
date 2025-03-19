POPULATION_SIZE = 200
TOP_PICK = 4
WHEEL_OF_FORTUNE = 176
RANDOM_INDIVIDUALS = 20
assert POPULATION_SIZE == TOP_PICK + WHEEL_OF_FORTUNE + RANDOM_INDIVIDUALS, "(1) Invalid Population Sum"
assert POPULATION_SIZE % 2 == 0, "(2) Population size must be even"
assert (RANDOM_INDIVIDUALS + WHEEL_OF_FORTUNE) % 2 == 0, "(3) Must be even"

GENERATIONS = 12

MUTATION_PROBABILITY = 0.005 # 0.5%
CROSSOVER_RATE = 0.02 # 2%

TOTAL_RETURN_COEFICIENT = 1
SHARP_RATIO_COEFICIENT = 0
MAX_DRAWDOWN_COEFICIENT = 0
WINRATE_COEFICIENT = 0
MIN_SCORE = -100

# ================================== #

from shared import var_types

lower = var_types.LOWER
upper = var_types.UPPER

GENE_RANGES = {
    var_types.SHORT_MA: {
        lower: 1,
        upper: 150
    },
    var_types.LONG_MA: {
        lower: 30,
        upper: 300,
    },
    var_types.STOP_LOSS: {
        lower: 0.05,
        upper: 20
    }, 
    var_types.POSITION_SIZE: {
        lower: 0.001,
        upper: 1  
    }
}

CROSSOVER_GROUPS = {
    1: [0, 1],
    2: [2, 3],
}

# ================================== #

import numpy as np

INDIVIDUAL_TYPE = np.dtype([
    (var_types.SHORT_MA, np.int32),
    (var_types.LONG_MA, np.int32),
    (var_types.STOP_LOSS, np.float32),
    (var_types.POSITION_SIZE, np.float32)
])

INDIVIDUAL_TYPE_SCORE = np.dtype([
    (var_types.SHORT_MA, np.int32),
    (var_types.LONG_MA, np.int32),
    (var_types.STOP_LOSS, np.float32),
    (var_types.POSITION_SIZE, np.float32),
    (var_types.SCORE, np.float32)
])