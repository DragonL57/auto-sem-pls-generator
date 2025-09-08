"""
Configuration management and validation for SEM/PLS data generation.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from .exceptions import ConfigurationError


@dataclass
class FactorConfig:
    """Configuration for a single latent factor."""
    name: str
    original_items: List[str]
    num_items: int = field(init=False)
    
    def __post_init__(self):
        self.num_items = len(self.original_items)
        if self.num_items == 0:
            raise ConfigurationError(f"Factor {self.name} must have at least one item")


@dataclass
class RegressionModel:
    """Configuration for a regression model."""
    dependent: str
    independent: List[str]
    order: List[str]
    
    def __post_init__(self):
        if not self.independent:
            raise ConfigurationError(f"Regression model for {self.dependent} must have independent variables")
        if set(self.independent) != set(self.order):
            raise ConfigurationError(f"Independent variables and order must match for {self.dependent}")


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm parameters."""
    population_size: int = 100
    num_generations: int = 200
    crossover_rate: float = 0.8
    base_mutation_rate: float = 0.15
    mutation_scale: float = 0.08
    elitism_count: int = 5
    stagnation_threshold: int = 7
    mutation_increase_factor: float = 1.3
    mutation_decrease_factor: float = 0.8
    max_mutation_rate: float = 0.3
    min_mutation_rate: float = 0.05
    
    def __post_init__(self):
        if not (0 <= self.crossover_rate <= 1):
            raise ConfigurationError("crossover_rate must be between 0 and 1")
        if not (0 <= self.base_mutation_rate <= 1):
            raise ConfigurationError("base_mutation_rate must be between 0 and 1")
        if self.population_size <= 0:
            raise ConfigurationError("population_size must be positive")
        if self.num_generations <= 0:
            raise ConfigurationError("num_generations must be positive")


@dataclass
class ParameterBounds:
    """Configuration for parameter bounds."""
    latent_cor_values: Tuple[float, float] = (0.01, 0.95)
    loading_strength: Tuple[float, float] = (0.45, 0.65)
    error_strength: Tuple[float, float] = (0.35, 0.55)
    
    def __post_init__(self):
        # Validate all bounds are valid ranges
        for name, (lower, upper) in self.__dict__.items():
            if lower >= upper:
                raise ConfigurationError(f"{name} lower bound must be less than upper bound")
            if not (-1 <= lower <= 1 and -1 <= upper <= 1):
                raise ConfigurationError(f"{name} bounds must be between -1 and 1")


class ConfigManager:
    """Manages and validates configuration for SEM/PLS data generation."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict
        self.factors_config: Dict[str, FactorConfig] = {}
        self.regression_models: List[RegressionModel] = []
        self.genetic_config: GeneticAlgorithmConfig = GeneticAlgorithmConfig()
        self.param_bounds: ParameterBounds = ParameterBounds()
        self.latent_correlation_matrix: Optional[np.ndarray] = None
        self.num_observations: int = 367
        
        self._parse_config()
        self._validate_config()
    
    def _parse_config(self):
        """Parse the configuration dictionary."""
        # Parse factors configuration
        factors_raw = self.config_dict.get('factors_config', {})
        for name, config in factors_raw.items():
            self.factors_config[name] = FactorConfig(
                name=name,
                original_items=config['original_items']
            )
        
        # Parse regression models
        regression_models_raw = self.config_dict.get('regression_models', [])
        for model in regression_models_raw:
            self.regression_models.append(RegressionModel(
                dependent=model['dependent'],
                independent=model['independent'],
                order=model['order']
            ))
        
        # Parse genetic algorithm configuration
        ga_config_raw = self.config_dict.get('genetic_algorithm_config', {})
        if ga_config_raw:
            self.genetic_config = GeneticAlgorithmConfig(**ga_config_raw)
        
        # Parse parameter bounds
        bounds_raw = self.config_dict.get('parameter_bounds', {})
        if bounds_raw:
            self.param_bounds = ParameterBounds(**bounds_raw)
        
        # Parse latent correlation matrix
        latent_cor_raw = self.config_dict.get('latent_correlation_matrix')
        if latent_cor_raw is not None:
            self.latent_correlation_matrix = np.array(latent_cor_raw)
        
        # Parse number of observations
        self.num_observations = self.config_dict.get('num_observations', 367)
    
    def _validate_config(self):
        """Validate the parsed configuration."""
        if not self.factors_config:
            raise ConfigurationError("At least one factor must be configured")
        
        if not self.regression_models:
            raise ConfigurationError("At least one regression model must be configured")
        
        # Validate that all regression model variables exist
        all_factors = set(self.factors_config.keys())
        all_composite_vars = set(f"{factor}_composite" for factor in all_factors)
        
        for model in self.regression_models:
            if model.dependent not in all_composite_vars:
                raise ConfigurationError(f"Dependent variable {model.dependent} not found in factors")
            
            for var in model.independent:
                # Check if it's a composite variable or interaction term
                if var not in all_composite_vars:
                    # Check if it's an interaction term
                    if 'x' not in var:
                        raise ConfigurationError(f"Independent variable {var} not found in factors")
                    else:
                        # Validate interaction term components
                        parts = var.split('x')
                        if len(parts) != 2:
                            raise ConfigurationError(f"Invalid interaction term format: {var}")
                        if not all(part in all_composite_vars for part in parts):
                            raise ConfigurationError(f"Interaction term components not found: {var}")
        
        # Validate latent correlation matrix dimensions
        if self.latent_correlation_matrix is not None:
            expected_shape = (len(self.factors_config), len(self.factors_config))
            if self.latent_correlation_matrix.shape != expected_shape:
                raise ConfigurationError(
                    f"Latent correlation matrix must be {expected_shape}, got {self.latent_correlation_matrix.shape}"
                )
    
    @property
    def latent_factor_names(self) -> List[str]:
        """Get list of latent factor names."""
        return list(self.factors_config.keys())
    
    @property
    def n_latent_factors(self) -> int:
        """Get number of latent factors."""
        return len(self.factors_config)
    
    @property
    def n_latent_cor_values(self) -> int:
        """Get number of latent correlation values (upper triangle)."""
        return self.n_latent_factors * (self.n_latent_factors - 1) // 2
    
    @property
    def bounds_list(self) -> List[Tuple[float, float]]:
        """Get parameter bounds as a list for genetic algorithm."""
        bounds = []
        for _ in range(self.n_latent_cor_values):
            bounds.append(self.param_bounds.latent_cor_values)
        bounds.append(self.param_bounds.error_strength)
        bounds.append(self.param_bounds.loading_strength)
        return bounds
    
    def update_latent_correlation_matrix(self, new_matrix: np.ndarray):
        """Update the latent correlation matrix with validation."""
        if new_matrix.shape != (self.n_latent_factors, self.n_latent_factors):
            raise ConfigurationError(f"Matrix must be {self.n_latent_factors}x{self.n_latent_factors}")
        
        # Check if matrix is symmetric
        if not np.allclose(new_matrix, new_matrix.T):
            raise ConfigurationError("Correlation matrix must be symmetric")
        
        # Check diagonal is 1.0
        if not np.allclose(np.diag(new_matrix), 1.0):
            raise ConfigurationError("Diagonal elements must be 1.0")
        
        self.latent_correlation_matrix = new_matrix
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration back to dictionary format."""
        return {
            'factors_config': {
                name: {'original_items': config.original_items}
                for name, config in self.factors_config.items()
            },
            'regression_models': [
                {
                    'dependent': model.dependent,
                    'independent': model.independent,
                    'order': model.order
                }
                for model in self.regression_models
            ],
            'latent_correlation_matrix': self.latent_correlation_matrix.tolist() if self.latent_correlation_matrix is not None else None,
            'num_observations': self.num_observations,
            'genetic_algorithm_config': {
                'population_size': self.genetic_config.population_size,
                'num_generations': self.genetic_config.num_generations,
                'crossover_rate': self.genetic_config.crossover_rate,
                'base_mutation_rate': self.genetic_config.base_mutation_rate,
                'mutation_scale': self.genetic_config.mutation_scale,
                'elitism_count': self.genetic_config.elitism_count,
                'stagnation_threshold': self.genetic_config.stagnation_threshold,
                'mutation_increase_factor': self.genetic_config.mutation_increase_factor,
                'mutation_decrease_factor': self.genetic_config.mutation_decrease_factor,
                'max_mutation_rate': self.genetic_config.max_mutation_rate,
                'min_mutation_rate': self.genetic_config.min_mutation_rate
            },
            'parameter_bounds': {
                'latent_cor_values': self.param_bounds.latent_cor_values,
                'loading_strength': self.param_bounds.loading_strength,
                'error_strength': self.param_bounds.error_strength
            }
        }