import numpy as np

rng = np.random.default_rng(42)

def initialize_population(pop_size, bounds, initial_individual=None):
    population = []
    if initial_individual is not None:
        population.append(initial_individual)
        pop_size -= 1
    bounds_arr = np.array(bounds)
    for _ in range(pop_size):
        individual = np.array([rng.uniform(bounds_arr[i][0], bounds_arr[i][1]) for i in range(len(bounds_arr))])
        population.append(individual)
    return population

def select_parents(population, fitnesses, num_parents, tournament_size=3):
    parents = []
    for _ in range(num_parents):
        contenders_indices = rng.choice(len(population), size=tournament_size, replace=False)
        contenders_fitness = [fitnesses[i] for i in contenders_indices]
        winner_index = contenders_indices[np.argmax(contenders_fitness)]
        parents.append(population[winner_index])
    return parents

def crossover(parent1, parent2, bounds):
    alpha = rng.uniform(0, 1)
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = (1 - alpha) * parent1 + alpha * parent2
    bounds_arr = np.array(bounds)
    offspring1 = np.clip(offspring1, bounds_arr[:, 0], bounds_arr[:, 1])
    offspring2 = np.clip(offspring2, bounds_arr[:, 0], bounds_arr[:, 1])
    return offspring1, offspring2

def mutate(individual, bounds, current_mutation_rate, mutation_scale):
    bounds_arr = np.array(bounds)
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if rng.random() < current_mutation_rate:
            mutated_individual[i] += rng.normal(0, mutation_scale * (bounds_arr[i][1] - bounds_arr[i][0]))
            mutated_individual[i] = np.clip(mutated_individual[i], bounds_arr[i][0], bounds_arr[i][1])
    return mutated_individual
