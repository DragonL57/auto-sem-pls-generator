"""
Genetic optimizer for SEM parameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .genetic_algorithm import GeneticAlgorithm
from ..utils.data_generator_utils import generate_items_from_latent, create_latent_correlation_matrix
from ..utils.math_utils import nearest_positive_definite
from ..core.exceptions import OptimizationError, DataGenerationError


class GeneticOptimizer:
    """Genetic algorithm optimizer for SEM parameters."""
    
    def __init__(self, config):
        """
        Initialize the genetic optimizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.genetic_algorithm = GeneticAlgorithm(
            population_size=config.genetic_config.population_size,
            num_generations=config.genetic_config.num_generations,
            crossover_rate=config.genetic_config.crossover_rate,
            base_mutation_rate=config.genetic_config.base_mutation_rate,
            mutation_scale=config.genetic_config.mutation_scale,
            elitism_count=config.genetic_config.elitism_count,
            stagnation_threshold=config.genetic_config.stagnation_threshold,
            mutation_increase_factor=config.genetic_config.mutation_increase_factor,
            mutation_decrease_factor=config.genetic_config.mutation_decrease_factor,
            max_mutation_rate=config.genetic_config.max_mutation_rate,
            min_mutation_rate=config.genetic_config.min_mutation_rate,
            random_seed=42
        )
        
        self.rng = np.random.default_rng(42)
    
    def evaluate_parameters(self, parameters: np.ndarray, rng_seed: int) -> Tuple[float, str]:
        """
        Evaluate parameters for genetic algorithm fitness function.
        
        Args:
            parameters: Array of parameters to evaluate
            rng_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (fitness_score, reason)
        """
        try:
            rng = np.random.default_rng(rng_seed)
            
            # Extract parameters
            latent_cor_values = parameters[:self.config.n_latent_cor_values]
            error_strength = parameters[self.config.n_latent_cor_values]
            loading_strength = parameters[self.config.n_latent_cor_values + 1]
            
            # Create latent correlation matrix
            latent_cor_matrix = create_latent_correlation_matrix(
                latent_cor_values, self.config.n_latent_factors
            )
            
            # Ensure positive definite matrix
            try:
                latent_cor_matrix_adjusted, diag_gt_one_latent = nearest_positive_definite(latent_cor_matrix)
            except RuntimeError:
                return -1_000_000, "Latent PD Error"
            
            # Check for Heywood cases
            if diag_gt_one_latent:
                return -1_000_000, "Heywood (Latent Diag > 1)"
            
            # Generate latent factors
            try:
                latent_samples = rng.multivariate_normal(
                    mean=np.zeros(self.config.n_latent_factors),
                    cov=latent_cor_matrix_adjusted,
                    size=self.config.num_observations
                )
                latent_df = pd.DataFrame(
                    latent_samples,
                    columns=[f"{name}_latent" for name in self.config.latent_factor_names]
                )
            except np.linalg.LinAlgError:
                return -1_000_000, "Latent MVN Error"
            
            # Generate observed items
            generated_factors = {}
            for factor_name in self.config.latent_factor_names:
                config = self.config.factors_config[factor_name]
                item_names = config.original_items
                num_items = len(item_names)
                
                latent_factor = latent_df[f"{factor_name}_latent"].values
                
                factor_data = generate_items_from_latent(
                    latent_factor=latent_factor,
                    num_items=num_items,
                    loading_strength=loading_strength,
                    error_strength=error_strength,
                    rng=rng
                )
                factor_data.columns = item_names
                generated_factors[factor_name] = factor_data
            
            # Combine all factors
            data_for_analysis = pd.concat(generated_factors.values(), axis=1)
            
            # Calculate fitness score
            fitness_score, reason = self._calculate_fitness(
                data_for_analysis, 
                latent_cor_matrix_adjusted,
                loading_strength,
                error_strength
            )
            
            return fitness_score, reason
            
        except Exception as e:
            return -1_000_000, f"Evaluation Error: {str(e)}"
    
    def _calculate_fitness(self, data: pd.DataFrame, latent_cor_matrix: np.ndarray,
                          loading_strength: float, error_strength: float) -> Tuple[float, str]:
        """
        Calculate fitness score based on statistical validation.
        
        Args:
            data: Generated data
            latent_cor_matrix: Latent correlation matrix
            loading_strength: Loading strength parameter
            error_strength: Error strength parameter
            
        Returns:
            Tuple of (fitness_score, reason)
        """
        from ..validation.statistical_validator import StatisticalValidator
        
        validator = StatisticalValidator(self.config)
        validation_results = validator.validate_data_quality(data)
        
        # Calculate fitness score based on validation results
        score = 0.0
        reasons = []
        
        # Cronbach's Alpha scores
        if 'cronbach_alpha' in validation_results:
            alpha_scores = validation_results['cronbach_alpha']
            avg_alpha = np.mean(list(alpha_scores.values()))
            score += avg_alpha * 30  # Weight: 30 points
            reasons.append(f"Avg Alpha: {avg_alpha:.3f}")
        
        # Factor analysis results
        if 'factor_analysis' in validation_results:
            fa_results = validation_results['factor_analysis']
            if 'cross_loadings' in fa_results:
                max_cross_loading = fa_results['cross_loadings']
                if max_cross_loading < 0.4:
                    score += 20  # Weight: 20 points
                    reasons.append(f"Good factor structure (max cross-loading: {max_cross_loading:.3f})")
        
        # Correlation matrix validity
        if 'correlation_validity' in validation_results:
            corr_validity = validation_results['correlation_validity']
            if corr_validity.get('is_valid', False):
                score += 15  # Weight: 15 points
                reasons.append("Valid correlation matrix")
        
        # Parameter penalties
        if loading_strength < 0.5:
            score -= 10
            reasons.append("Low loading strength")
        
        if error_strength > 0.5:
            score -= 10
            reasons.append("High error strength")
        
        # Diversity bonus
        if 'item_diversity' in validation_results:
            diversity = validation_results['item_diversity']
            if diversity > 0.8:
                score += 10
                reasons.append("Good item diversity")
        
        # Latent correlation matrix evaluation
        latent_correlations = latent_cor_matrix[np.triu_indices_from(latent_cor_matrix, k=1)]
        if np.all(np.abs(latent_correlations) < 0.9):
            score += 15
            reasons.append("Reasonable latent correlations")
        
        reason_str = "; ".join(reasons) if reasons else "No specific reason"
        
        return score, reason_str
    
    def optimize(self, num_processes: int = 1) -> Dict[str, Any]:
        """
        Run the genetic algorithm optimization.
        
        Args:
            num_processes: Number of processes to use
            
        Returns:
            Optimization results
        """
        print(f"Starting genetic algorithm optimization with {num_processes} processes")
        
        def fitness_function(params):
            return self.evaluate_parameters(params, self.rng.integers(0, 1000000))
        
        if num_processes == 1:
            # Single process optimization
            results = self.genetic_algorithm.optimize(
                fitness_function=fitness_function,
                bounds=self.config.bounds_list,
                verbose=True
            )
        else:
            # Multi-process optimization
            results = self._optimize_parallel(fitness_function, num_processes)
        
        return {
            'best_parameters': results['best_individual'],
            'best_score': results['best_fitness'],
            'best_reason': results['best_reason'],
            'final_population': results.get('final_population', []),
            'final_fitnesses': results.get('final_fitnesses', [])
        }
    
    def _optimize_parallel(self, fitness_function, num_processes: int) -> Dict[str, Any]:
        """
        Run optimization with parallel processing.
        
        Args:
            fitness_function: Fitness evaluation function
            num_processes: Number of processes
            
        Returns:
            Optimization results
        """
        # Initialize population
        population = self.genetic_algorithm.initialize_population(self.config.bounds_list)
        
        best_individual = None
        best_fitness = -np.inf
        best_reason = ""
        
        adaptive_mutation_rate = self.genetic_algorithm.base_mutation_rate
        stagnation_counter = 0
        
        for generation in range(self.genetic_algorithm.num_generations):
            # Evaluate population in parallel
            fitnesses = []
            reasons = []
            
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit all tasks
                futures = []
                for i, individual in enumerate(population):
                    future = executor.submit(fitness_function, individual)
                    futures.append((future, i))
                
                # Collect results
                for future, idx in futures:
                    try:
                        fitness, reason = future.result()
                        fitnesses.append(fitness)
                        reasons.append(reason)
                    except Exception as e:
                        fitnesses.append(-1_000_000)
                        reasons.append(f"Error: {str(e)}")
            
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
                    self.genetic_algorithm.min_mutation_rate,
                    adaptive_mutation_rate * self.genetic_algorithm.mutation_decrease_factor
                )
            else:
                stagnation_counter += 1
                if stagnation_counter >= self.genetic_algorithm.stagnation_threshold:
                    adaptive_mutation_rate = min(
                        self.genetic_algorithm.max_mutation_rate,
                        adaptive_mutation_rate * self.genetic_algorithm.mutation_increase_factor
                    )
                    stagnation_counter = 0
            
            # Print progress
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.genetic_algorithm.num_generations}: "
                      f"Best = {best_fitness:.3f}, "
                      f"Current = {current_best_fitness:.3f}, "
                      f"Mutation = {adaptive_mutation_rate:.3f}")
            
            # Evolve population
            population = self.genetic_algorithm.evolve_generation(
                population, fitnesses, self.config.bounds_list, adaptive_mutation_rate
            )
        
        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'best_reason': best_reason,
            'final_population': population,
            'final_fitnesses': fitnesses
        }
    
    def generate_data(self, parameters: np.ndarray) -> pd.DataFrame:
        """
        Generate data using the optimized parameters.
        
        Args:
            parameters: Optimized parameters
            
        Returns:
            Generated data as DataFrame
        """
        fitness_score, reason = self.evaluate_parameters(parameters, 42)
        
        if fitness_score <= 0:
            raise DataGenerationError(f"Cannot generate data with these parameters: {reason}")
        
        # Generate the actual data (this is a simplified version)
        # In a full implementation, we'd extract the data from the evaluation process
        latent_cor_values = parameters[:self.config.n_latent_cor_values]
        error_strength = parameters[self.config.n_latent_cor_values]
        loading_strength = parameters[self.config.n_latent_cor_values + 1]
        
        # Create latent correlation matrix
        latent_cor_matrix = create_latent_correlation_matrix(
            latent_cor_values, self.config.n_latent_factors
        )
        
        # Ensure positive definite
        latent_cor_matrix_adjusted, _ = nearest_positive_definite(latent_cor_matrix)
        
        # Generate latent factors
        latent_samples = self.rng.multivariate_normal(
            mean=np.zeros(self.config.n_latent_factors),
            cov=latent_cor_matrix_adjusted,
            size=self.config.num_observations
        )
        latent_df = pd.DataFrame(
            latent_samples,
            columns=[f"{name}_latent" for name in self.config.latent_factor_names]
        )
        
        # Generate observed items
        generated_factors = {}
        for factor_name in self.config.latent_factor_names:
            config = self.config.factors_config[factor_name]
            item_names = config.original_items
            num_items = len(item_names)
            
            latent_factor = latent_df[f"{factor_name}_latent"].values
            
            factor_data = generate_items_from_latent(
                latent_factor=latent_factor,
                num_items=num_items,
                loading_strength=loading_strength,
                error_strength=error_strength,
                rng=self.rng
            )
            factor_data.columns = item_names
            generated_factors[factor_name] = factor_data
        
        # Combine all factors
        data_for_analysis = pd.concat(generated_factors.values(), axis=1)
        
        # Create composite scores
        composite_scores = pd.DataFrame({
            f"{fac}_composite": data_for_analysis[
                [col for col in data_for_analysis.columns if col.startswith(fac)]
            ].mean(axis=1)
            for fac in self.config.factors_config.keys()
        })
        
        # Add interaction terms if needed
        for model in self.config.regression_models:
            for var in model.independent:
                if 'x' in var:  # Interaction term
                    parts = var.split('x')
                    if len(parts) == 2 and all(p in composite_scores.columns for p in parts):
                        composite_scores[var] = composite_scores[parts[0]] * composite_scores[parts[1]]
        
        # Combine observed items and composite scores
        final_data = pd.concat([data_for_analysis, composite_scores], axis=1)
        
        return final_data