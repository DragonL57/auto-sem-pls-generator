"""
Genetic algorithm implementation for SEM parameter optimization.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Any
import numpy.random as rng


class GeneticAlgorithm:
    """Genetic algorithm implementation for parameter optimization."""
    
    def __init__(self, 
                 population_size: int = 100,
                 num_generations: int = 200,
                 crossover_rate: float = 0.8,
                 base_mutation_rate: float = 0.15,
                 mutation_scale: float = 0.08,
                 elitism_count: int = 5,
                 stagnation_threshold: int = 7,
                 mutation_increase_factor: float = 1.3,
                 mutation_decrease_factor: float = 0.8,
                 max_mutation_rate: float = 0.3,
                 min_mutation_rate: float = 0.05,
                 random_seed: Optional[int] = None):
        """
        Initialize genetic algorithm parameters.
        
        Args:
            population_size: Size of the population
            num_generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            base_mutation_rate: Base mutation rate
            mutation_scale: Scale of mutation
            elitism_count: Number of elite individuals to preserve
            stagnation_threshold: Generations without improvement before increasing mutation
            mutation_increase_factor: Factor to increase mutation rate
            mutation_decrease_factor: Factor to decrease mutation rate
            max_mutation_rate: Maximum mutation rate
            min_mutation_rate: Minimum mutation rate
            random_seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.base_mutation_rate = base_mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism_count = elitism_count
        self.stagnation_threshold = stagnation_threshold
        self.mutation_increase_factor = mutation_increase_factor
        self.mutation_decrease_factor = mutation_decrease_factor
        self.max_mutation_rate = max_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        
        self.rng = np.random.default_rng(random_seed)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate genetic algorithm parameters."""
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("crossover_rate must be between 0 and 1")
        if not (0 <= self.base_mutation_rate <= 1):
            raise ValueError("base_mutation_rate must be between 0 and 1")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.num_generations <= 0:
            raise ValueError("num_generations must be positive")
        if self.elitism_count >= self.population_size:
            raise ValueError("elitism_count must be less than population_size")
    
    def initialize_population(self, bounds: List[Tuple[float, float]], 
                            initial_individual: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Initialize the population with random individuals.
        
        Args:
            bounds: List of (min, max) bounds for each parameter
            initial_individual: Optional initial individual to include
            
        Returns:
            List of individuals (each individual is a numpy array)
        """
        population = []
        
        # Add initial individual if provided
        if initial_individual is not None:
            population.append(initial_individual.copy())
        
        # Generate random population
        bounds_array = np.array(bounds)
        remaining_size = self.population_size - len(population)
        
        for _ in range(remaining_size):
            individual = np.array([
                self.rng.uniform(bounds_array[i][0], bounds_array[i][1])
                for i in range(len(bounds_array))
            ])
            population.append(individual)
        
        return population
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitnesses: List[float], 
                           tournament_size: int = 3) -> np.ndarray:
        """
        Select an individual using tournament selection.
        
        Args:
            population: Current population
            fitnesses: Fitness values for each individual
            tournament_size: Size of tournament
            
        Returns:
            Selected individual
        """
        # Select tournament contenders
        contender_indices = self.rng.choice(len(population), size=tournament_size, replace=False)
        contender_fitnesses = [fitnesses[i] for i in contender_indices]
        
        # Select winner (highest fitness)
        winner_index = contender_indices[np.argmax(contender_fitnesses)]
        return population[winner_index]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, 
                  bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            bounds: Parameter bounds
            
        Returns:
            Two offspring
        """
        # Simulated binary crossover
        alpha = self.rng.uniform(0, 1)
        
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = (1 - alpha) * parent1 + alpha * parent2
        
        # Ensure bounds are respected
        bounds_array = np.array(bounds)
        offspring1 = np.clip(offspring1, bounds_array[:, 0], bounds_array[:, 1])
        offspring2 = np.clip(offspring2, bounds_array[:, 0], bounds_array[:, 1])
        
        return offspring1, offspring2
    
    def mutate(self, individual: np.ndarray, bounds: List[Tuple[float, float]], 
               current_mutation_rate: float) -> np.ndarray:
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            bounds: Parameter bounds
            current_mutation_rate: Current mutation rate
            
        Returns:
            Mutated individual
        """
        mutated_individual = individual.copy()
        bounds_array = np.array(bounds)
        
        for i in range(len(mutated_individual)):
            if self.rng.random() < current_mutation_rate:
                # Gaussian mutation
                mutation_magnitude = self.mutation_scale * (bounds_array[i][1] - bounds_array[i][0])
                mutated_individual[i] += self.rng.normal(0, mutation_magnitude)
                
                # Ensure bounds are respected
                mutated_individual[i] = np.clip(
                    mutated_individual[i], bounds_array[i][0], bounds_array[i][1]
                )
        
        return mutated_individual
    
    def evolve_generation(self, population: List[np.ndarray], 
                         fitnesses: List[float], 
                         bounds: List[Tuple[float, float]],
                         current_mutation_rate: float) -> List[np.ndarray]:
        """
        Evolve the population for one generation.
        
        Args:
            population: Current population
            fitnesses: Fitness values
            bounds: Parameter bounds
            current_mutation_rate: Current mutation rate
            
        Returns:
            New population
        """
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        # Elitism: preserve best individuals
        new_population = [population[i].copy() for i in sorted_indices[:self.elitism_count]]
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)
            
            # Crossover
            if self.rng.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover(parent1, parent2, bounds)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            offspring1 = self.mutate(offspring1, bounds, current_mutation_rate)
            offspring2 = self.mutate(offspring2, bounds, current_mutation_rate)
            
            # Add to population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        return new_population
    
    def optimize(self, fitness_function: Callable[[np.ndarray], Tuple[float, str]], 
                 bounds: List[Tuple[float, float]], 
                 initial_individual: Optional[np.ndarray] = None,
                 verbose: bool = True) -> dict:
        """
        Run the genetic algorithm optimization.
        
        Args:
            fitness_function: Function that evaluates fitness (returns (score, reason))
            bounds: Parameter bounds
            initial_individual: Optional initial individual
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results
        """
        # Initialize population
        population = self.initialize_population(bounds, initial_individual)
        
        # Track best solution
        best_individual = None
        best_fitness = -np.inf
        best_reason = ""
        
        # Adaptive mutation rate
        adaptive_mutation_rate = self.base_mutation_rate
        stagnation_counter = 0
        
        # Generation evolution
        for generation in range(self.num_generations):
            # Evaluate population
            fitnesses = []
            reasons = []
            
            for individual in population:
                fitness, reason = fitness_function(individual)
                fitnesses.append(fitness)
                reasons.append(reason)
            
            # Update best solution
            current_best_fitness = max(fitnesses)
            current_best_idx = np.argmax(fitnesses)
            current_best_reason = reasons[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
                best_reason = current_best_reason
                stagnation_counter = 0
                adaptive_mutation_rate = max(
                    self.min_mutation_rate, 
                    adaptive_mutation_rate * self.mutation_decrease_factor
                )
            else:
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_threshold:
                    adaptive_mutation_rate = min(
                        self.max_mutation_rate,
                        adaptive_mutation_rate * self.mutation_increase_factor
                    )
                    stagnation_counter = 0
            
            # Logging
            if verbose and (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.num_generations}: "
                      f"Best = {best_fitness:.3f}, "
                      f"Current = {current_best_fitness:.3f}, "
                      f"Mutation = {adaptive_mutation_rate:.3f}")
            
            # Evolve next generation
            population = self.evolve_generation(
                population, fitnesses, bounds, adaptive_mutation_rate
            )
        
        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'best_reason': best_reason,
            'final_population': population,
            'final_fitnesses': fitnesses
        }