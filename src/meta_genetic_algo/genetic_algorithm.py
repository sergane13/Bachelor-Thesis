import random
import numpy as np
import multiprocessing as mp
from . import constants
from src.meta_genetic_algo.generate_offsrings import generate_random_sample, generate_meta_population
from src.trading_genetic_algo.genetic_algorithm import Genetic_Algorithm 

def evaluate_meta_individual(ind, train_data, train_data_ema_warmup):
    (
        pop_size,
        top,
        wheel,
        randoms,
        mutation_prob,
        crossover_rate,
        no_improvement,
        min_improvement
    ) = ind

    constant = (
        constants.Genetic_Algorithm
        .Builder()
        .set_population_size(pop_size)
        .set_top_pick(top)
        .set_wheel_of_fortune(wheel)
        .set_random_individual(randoms)
        .set_mutation_probability(mutation_prob)
        .set_crossover_rate(crossover_rate)
        .set_no_improvement(no_improvement)
        .set_min_improvement(min_improvement)
        .build()
    )

    genetic_algo = Genetic_Algorithm(constant, train_data, train_data_ema_warmup)

    _, fitness_long = genetic_algo.run_generation(False)
    _, fitness_short = genetic_algo.run_generation(True)

    return (
        ind,
        fitness_long,
        fitness_short
    )

class MetaGeneticAlgorithm:
    def __init__(self, train_data, train_data_ema_warmup,
                population_size=8,
                generations=6,
                top_k=2,
                mutation_rate=0.005):
        self.train_data = train_data
        self.train_data_ema_warmup = train_data_ema_warmup
        self.population_size = population_size
        self.generations = generations
        self.top_k = top_k
        self.mutation_rate = mutation_rate

    def run(self):
        population = generate_meta_population(self.population_size)

        for generation in range(self.generations):
            print(f"\n=== Meta Generation {generation} ===")

            args = [(ind, self.train_data, self.train_data_ema_warmup) for ind in population]
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.starmap(evaluate_meta_individual, args)

            scored = []
            for ind, fitness_long, fitness_short in results:
                fitness_long = float(np.mean(fitness_long[:4]))
                fitness_short = float(np.mean(fitness_short[:4]))
                score = fitness_long + fitness_short
                scored.append((score, ind, fitness_long, fitness_short))

            scored.sort(reverse=True, key=lambda x: x[0])
            best_score, best_ind, best_long, best_short = scored[0]
            print(f"Best individual: {best_ind}")
            print(f"Fitness long: {best_long:.4f} | short: {best_short:.4f} | total: {best_score:.4f}")

            top_individuals = [ind for _, ind, _, _ in scored[:self.top_k]]

            new_population = top_individuals.copy()
            while len(new_population) < self.population_size:
                p1 = random.choice(top_individuals)
                p2 = random.choice(top_individuals)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        return best_ind

    def crossover(self, p1, p2):
        cut = random.randint(1, len(p1) - 2)
        return p1[:cut] + p2[cut:]

    def mutate(self, individual):
        if random.random() > self.mutation_rate:
            return individual

        idx = random.randint(0, len(individual) - 1)
        is_float = idx in [4, 5, 7]
        decimals = 4 if idx in [4, 7] else (3 if idx == 5 else 0)
        mutated = generate_random_sample(idx, is_float=is_float, decimals=decimals)

        individual = list(individual)
        individual[idx] = mutated
        return tuple(individual)
