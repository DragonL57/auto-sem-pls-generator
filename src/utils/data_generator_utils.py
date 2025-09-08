"""
Data generation utilities for SEM/PLS synthetic data.
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_items_from_latent(latent_factor: np.ndarray, 
                             num_items: int, 
                             mean_val: float = 3.0, 
                             sd_val: float = 1.0, 
                             loading_strength: float = 0.6, 
                             error_strength: float = 0.4, 
                             rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """
    Generate Likert-scale items from latent factor.
    
    Args:
        latent_factor: Latent factor values
        num_items: Number of items to generate
        mean_val: Mean value for Likert scale
        sd_val: Standard deviation for Likert scale
        loading_strength: Strength of factor loading
        error_strength: Strength of error term
        rng: Random number generator
        
    Returns:
        DataFrame with generated items
    """
    rng = rng or np.random.default_rng()
    num_obs_current = len(latent_factor)
    item_data = np.empty((num_obs_current, num_items), dtype=float)
    
    # Ensure minimum standard deviation
    if sd_val < np.finfo(float).eps:
        sd_val = 0.1
    
    for i in range(num_items):
        # Generate item values using factor analysis model
        item_values = (
            loading_strength * latent_factor + 
            error_strength * rng.normal(loc=0, scale=1, size=num_obs_current)
        )
        
        # Standardize and scale to desired mean and SD
        if np.std(item_values, ddof=1) < np.finfo(float).eps:
            item_values_scaled = np.full(num_obs_current, mean_val)
        else:
            item_values_scaled = (
                (item_values - np.mean(item_values)) / np.std(item_values, ddof=1) * sd_val + mean_val
            )
        
        # Round and clip to Likert scale (1-5)
        item_data[:, i] = np.clip(np.round(item_values_scaled), 1, 5)
    
    return pd.DataFrame(item_data)


def create_latent_correlation_matrix(correlation_values: np.ndarray, 
                                   n_factors: int) -> np.ndarray:
    """
    Create latent correlation matrix from upper triangle values.
    
    Args:
        correlation_values: Array of correlation values (upper triangle)
        n_factors: Number of factors
        
    Returns:
        Symmetric correlation matrix
    """
    correlation_matrix = np.eye(n_factors)
    
    k = 0
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            correlation_matrix[i, j] = correlation_values[k]
            correlation_matrix[j, i] = correlation_values[k]
            k += 1
    
    return correlation_matrix


def create_composite_scores(data: pd.DataFrame, 
                          factors_config: dict) -> pd.DataFrame:
    """
    Create composite scores for each factor.
    
    Args:
        data: Data containing observed items
        factors_config: Factor configuration dictionary
        
    Returns:
        DataFrame with composite scores
    """
    composite_scores = pd.DataFrame()
    
    for factor_name, config in factors_config.items():
        item_names = config.original_items
        # Calculate mean of items for each factor
        composite_scores[f"{factor_name}_composite"] = data[item_names].mean(axis=1)
    
    return composite_scores


def create_interaction_terms(composite_scores: pd.DataFrame, 
                           regression_models: list) -> pd.DataFrame:
    """
    Create interaction terms for regression models.
    
    Args:
        composite_scores: DataFrame with composite scores
        regression_models: List of regression model configurations
        
    Returns:
        DataFrame with interaction terms added
    """
    result_scores = composite_scores.copy()
    
    for model in regression_models:
        for var in model.independent:
            if 'x' in var:  # Interaction term
                parts = var.split('x')
                if len(parts) == 2:
                    var1, var2 = parts
                    if var1 in result_scores.columns and var2 in result_scores.columns:
                        result_scores[var] = result_scores[var1] * result_scores[var2]
    
    return result_scores


def validate_likert_data(data: pd.DataFrame, min_val: int = 1, max_val: int = 5) -> dict:
    """
    Validate Likert-scale data.
    
    Args:
        data: Data to validate
        min_val: Minimum valid value
        max_val: Maximum valid value
        
    Returns:
        Validation results dictionary
    """
    results = {
        'is_valid': True,
        'invalid_values': [],
        'missing_values': data.isnull().sum().sum(),
        'value_range': (data.min().min(), data.max().max()),
        'warnings': []
    }
    
    # Check for values outside valid range
    for col in data.columns:
        invalid_mask = (data[col] < min_val) | (data[col] > max_val)
        if invalid_mask.any():
            results['invalid_values'].extend([(col, idx, val) for idx, val in data[col][invalid_mask].items()])
            results['is_valid'] = False
    
    # Check for missing values
    if results['missing_values'] > 0:
        results['warnings'].append(f"Found {results['missing_values']} missing values")
    
    # Check value range
    if results['value_range'][0] < min_val or results['value_range'][1] > max_val:
        results['warnings'].append(f"Value range {results['value_range']} exceeds expected range [{min_val}, {max_val}]")
    
    return results