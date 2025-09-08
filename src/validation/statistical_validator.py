"""
Statistical validation for SEM/PLS data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

from ..core.exceptions import ValidationError
from ..utils.math_utils import is_positive_definite, nearest_positive_definite


class StatisticalValidator:
    """Handles statistical validation for SEM/PLS data."""
    
    def __init__(self, config):
        """
        Initialize statistical validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate overall data quality.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'data_quality': {},
            'cronbach_alpha': {},
            'factor_analysis': {},
            'correlation_validity': {},
            'regression_validation': {},
            'item_diversity': 0.0
        }
        
        try:
            # Basic data quality checks
            results['data_quality'] = self._check_data_quality(data)
            
            # Cronbach's Alpha
            results['cronbach_alpha'] = self._calculate_cronbach_alpha(data)
            
            # Factor analysis
            results['factor_analysis'] = self._perform_factor_analysis(data)
            
            # Correlation matrix validity
            results['correlation_validity'] = self._validate_correlation_matrix(data)
            
            # Regression validation
            results['regression_validation'] = self._validate_regression_models(data)
            
            # Item diversity
            results['item_diversity'] = self._calculate_item_diversity(data)
            
        except Exception as e:
            raise ValidationError(f"Statistical validation failed: {str(e)}")
        
        return results
    
    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check basic data quality metrics."""
        quality_results = {
            'sample_size': len(data),
            'missing_values': data.isnull().sum().sum(),
            'complete_cases': len(data.dropna()),
            'duplicate_rows': data.duplicated().sum(),
            'value_range': (data.min().min(), data.max().max()),
            'valid_likert': self._check_likert_validity(data),
            'outliers': self._detect_outliers(data)
        }
        
        return quality_results
    
    def _check_likert_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check if data follows Likert scale assumptions."""
        # Get only the original items (not composite scores)
        item_columns = []
        for factor_config in self.config.factors_config.values():
            item_columns.extend(factor_config.original_items)
        
        item_data = data[item_columns]
        
        validity_results = {
            'all_values_valid': True,
            'invalid_count': 0,
            'value_distribution': {}
        }
        
        for col in item_data.columns:
            # Check if values are within valid range (1-5)
            invalid_count = ((item_data[col] < 1) | (item_data[col] > 5)).sum()
            if invalid_count > 0:
                validity_results['all_values_valid'] = False
                validity_results['invalid_count'] += invalid_count
            
            # Value distribution
            validity_results['value_distribution'][col] = item_data[col].value_counts().to_dict()
        
        return validity_results
    
    def _detect_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in the data."""
        # Get only the original items
        item_columns = []
        for factor_config in self.config.factors_config.values():
            item_columns.extend(factor_config.original_items)
        
        item_data = data[item_columns]
        
        outlier_results = {
            'method': method,
            'outliers_by_item': {},
            'total_outliers': 0
        }
        
        for col in item_data.columns:
            if method == 'iqr':
                Q1 = item_data[col].quantile(0.25)
                Q3 = item_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = item_data[(item_data[col] < lower_bound) | (item_data[col] > upper_bound)]
                outlier_results['outliers_by_item'][col] = len(outliers)
                outlier_results['total_outliers'] += len(outliers)
        
        return outlier_results
    
    def _calculate_cronbach_alpha(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Cronbach's Alpha for each factor."""
        alpha_results = {}
        
        for factor_name, factor_config in self.config.factors_config.items():
            item_names = factor_config.original_items
            factor_data = data[item_names]
            
            # Calculate Cronbach's Alpha
            alpha = self._cronbach_alpha(factor_data)
            alpha_results[factor_name] = alpha
        
        return alpha_results
    
    def _cronbach_alpha(self, df: pd.DataFrame) -> float:
        """Calculate Cronbach's Alpha for a set of items."""
        # Calculate covariance matrix
        cov_matrix = df.cov()
        
        # Calculate variances
        variances = np.diag(cov_matrix)
        total_variance = np.sum(variances)
        
        if total_variance == 0:
            return 0.0
        
        # Calculate Cronbach's Alpha
        n_items = len(df.columns)
        alpha = (n_items / (n_items - 1)) * (1 - np.sum(variances) / total_variance)
        
        return max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
    
    def _perform_factor_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Exploratory Factor Analysis (EFA)."""
        # Get only the original items
        item_columns = []
        for factor_config in self.config.factors_config.values():
            item_columns.extend(factor_config.original_items)
        
        item_data = data[item_columns].dropna()
        
        if len(item_data) < self.config.n_latent_factors:
            return {'error': 'Insufficient sample size for factor analysis'}
        
        try:
            # KMO and Bartlett's test
            kmo_results = self._calculate_kmo(item_data)
            bartlett_results = self._calculate_bartlett(item_data)
            
            # Factor analysis with promax rotation
            fa = FactorAnalyzer(
                n_factors=self.config.n_latent_factors,
                rotation='promax',
                method='principal',
                use_smc=True
            )
            
            fa.fit(item_data)
            loadings = fa.loadings_
            
            # Calculate cross-loadings
            cross_loadings = self._calculate_cross_loadings(loadings)
            
            # Calculate factor correlations
            factor_correlations = fa.phi_ if hasattr(fa, 'phi_') else None
            
            return {
                'kmo': kmo_results,
                'bartlett': bartlett_results,
                'loadings': loadings.tolist(),
                'factor_correlations': factor_correlations.tolist() if factor_correlations is not None else None,
                'cross_loadings': cross_loadings,
                'model_fit': fa.get_factor_variance()
            }
            
        except Exception as e:
            return {'error': f'Factor analysis failed: {str(e)}'}
    
    def _calculate_kmo(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Kaiser-Meyer-Olkin (KMO) test."""
        try:
            kmo_per_variable, kmo_overall = calculate_kmo(data)
            return {
                'kmo_overall': kmo_overall,
                'kmo_per_variable': kmo_per_variable.tolist()
            }
        except Exception:
            return {'kmo_overall': 0.0, 'kmo_per_variable': []}
    
    def _calculate_bartlett(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bartlett's test of sphericity."""
        try:
            chi_square_value, p_value = calculate_bartlett_sphericity(data)
            return {
                'chi_square': chi_square_value,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
        except Exception:
            return {'chi_square': 0.0, 'p_value': 1.0, 'is_significant': False}
    
    def _calculate_cross_loadings(self, loadings: np.ndarray) -> float:
        """Calculate maximum cross-loading as a measure of factor structure quality."""
        n_items, n_factors = loadings.shape
        
        max_cross_loading = 0.0
        
        for i in range(n_items):
            # Get loadings for this item across all factors
            item_loadings = np.abs(loadings[i, :])
            
            # Sort in descending order
            sorted_loadings = np.sort(item_loadings)[::-1]
            
            # If there are at least 2 factors, calculate cross-loading
            if len(sorted_loadings) >= 2:
                cross_loading = sorted_loadings[1]  # Second highest loading
                max_cross_loading = max(max_cross_loading, cross_loading)
        
        return max_cross_loading
    
    def _validate_correlation_matrix(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the correlation matrix."""
        # Get only the original items
        item_columns = []
        for factor_config in self.config.factors_config.values():
            item_columns.extend(factor_config.original_items)
        
        item_data = data[item_columns]
        correlation_matrix = item_data.corr()
        
        validation_results = {
            'is_symmetric': np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-8),
            'is_positive_definite': is_positive_definite(correlation_matrix.values),
            'condition_number': np.linalg.cond(correlation_matrix.values),
            'max_correlation': np.max(np.abs(correlation_matrix.values - np.eye(correlation_matrix.shape[0]))),
            'has_extreme_correlations': False
        }
        
        # Check for extreme correlations (>0.9)
        off_diagonal = correlation_matrix.values - np.eye(correlation_matrix.shape[0])
        validation_results['has_extreme_correlations'] = np.any(np.abs(off_diagonal) > 0.9)
        
        # Overall validity
        validation_results['is_valid'] = (
            validation_results['is_symmetric'] and
            validation_results['is_positive_definite'] and
            not validation_results['has_extreme_correlations'] and
            validation_results['condition_number'] < 1000
        )
        
        return validation_results
    
    def _validate_regression_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate regression models."""
        regression_results = {}
        
        # Create composite scores
        composite_scores = pd.DataFrame()
        for factor_name, factor_config in self.config.factors_config.items():
            item_names = factor_config.original_items
            composite_scores[f"{factor_name}_composite"] = data[item_names].mean(axis=1)
        
        # Add interaction terms
        for model in self.config.regression_models:
            for var in model.independent:
                if 'x' in var:
                    parts = var.split('x')
                    if len(parts) == 2:
                        var1, var2 = parts
                        if var1 in composite_scores.columns and var2 in composite_scores.columns:
                            composite_scores[var] = composite_scores[var1] * composite_scores[var2]
        
        # Run each regression model
        for model in self.config.regression_models:
            dependent_var = model.dependent
            independent_vars = model.independent
            
            if dependent_var not in composite_scores.columns:
                continue
            
            # Prepare data
            X = composite_scores[independent_vars]
            y = composite_scores[dependent_var]
            
            # Add constant
            X = sm.add_constant(X)
            
            try:
                # Fit regression
                model_fit = sm.OLS(y, X).fit()
                
                regression_results[dependent_var] = {
                    'r_squared': model_fit.rsquared,
                    'adj_r_squared': model_fit.rsquared_adj,
                    'f_statistic': model_fit.fvalue,
                    'f_pvalue': model_fit.f_pvalue,
                    'coefficients': model_fit.params.to_dict(),
                    'p_values': model_fit.pvalues.to_dict(),
                    'is_significant': model_fit.f_pvalue < 0.05,
                    'model_summary': str(model_fit.summary())
                }
            except Exception as e:
                regression_results[dependent_var] = {'error': str(e)}
        
        return regression_results
    
    def _calculate_item_diversity(self, data: pd.DataFrame) -> float:
        """Calculate item diversity as a measure of data quality."""
        # Get only the original items
        item_columns = []
        for factor_config in self.config.factors_config.values():
            item_columns.extend(factor_config.original_items)
        
        item_data = data[item_columns]
        
        # Calculate coefficient of variation for each item
        cv_values = []
        for col in item_data.columns:
            mean_val = item_data[col].mean()
            std_val = item_data[col].std()
            if mean_val != 0:
                cv = std_val / mean_val
                cv_values.append(cv)
        
        # Return average coefficient of variation
        return np.mean(cv_values) if cv_values else 0.0