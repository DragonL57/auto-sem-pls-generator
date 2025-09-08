"""
Main data validator class.
"""

import pandas as pd
from typing import Dict, List, Any, Optional

from .statistical_validator import StatisticalValidator
from ..core.exceptions import ValidationError


class DataValidator:
    """Main validator for SEM/PLS data quality and statistical validity."""
    
    def __init__(self, config):
        """
        Initialize data validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.statistical_validator = StatisticalValidator(config)
    
    def validate_all(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data validation.
        
        Args:
            data: Data to validate
            
        Returns:
            Complete validation results
        """
        validation_results = {
            'overall_validity': False,
            'statistical_validation': {},
            'data_quality_checks': {},
            'recommendations': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Perform statistical validation
            statistical_results = self.statistical_validator.validate_data_quality(data)
            validation_results['statistical_validation'] = statistical_results
            
            # Check data quality
            quality_results = self._check_data_quality_requirements(data)
            validation_results['data_quality_checks'] = quality_results
            
            # Generate recommendations
            recommendations = self._generate_recommendations(statistical_results, quality_results)
            validation_results['recommendations'] = recommendations
            
            # Generate warnings
            warnings = self._generate_warnings(statistical_results, quality_results)
            validation_results['warnings'] = warnings
            
            # Check for errors
            errors = self._check_for_errors(statistical_results, quality_results)
            validation_results['errors'] = errors
            
            # Determine overall validity
            validation_results['overall_validity'] = (
                len(errors) == 0 and len(warnings) <= 3
            )
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['overall_validity'] = False
        
        return validation_results
    
    def _check_data_quality_requirements(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check basic data quality requirements."""
        quality_results = {
            'sample_size_adequate': False,
            'missing_data_acceptable': False,
            'no_extreme_outliers': False,
            'valid_likert_scale': False,
            'sufficient_variance': False
        }
        
        # Sample size check
        sample_size = len(data)
        quality_results['sample_size_adequate'] = sample_size >= 100
        
        # Missing data check
        missing_percentage = (data.isnull().sum().sum() / (sample_size * len(data.columns))) * 100
        quality_results['missing_data_acceptable'] = missing_percentage < 5
        
        # Outlier check (using IQR method)
        outlier_count = 0
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count += len(outliers)
        
        outlier_percentage = (outlier_count / (sample_size * len(data.select_dtypes(include=[np.number]).columns))) * 100
        quality_results['no_extreme_outliers'] = outlier_percentage < 5
        
        # Likert scale validity
        item_columns = []
        for factor_config in self.config.factors_config.values():
            item_columns.extend(factor_config.original_items)
        
        item_data = data[item_columns]
        valid_range = ((item_data >= 1) & (item_data <= 5)).all().all()
        quality_results['valid_likert_scale'] = valid_range
        
        # Variance check
        variance_check = True
        for col in item_data.columns:
            if item_data[col].var() < 0.1:  # Very low variance
                variance_check = False
                break
        quality_results['sufficient_variance'] = variance_check
        
        return quality_results
    
    def _generate_recommendations(self, statistical_results: Dict[str, Any], 
                                quality_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Cronbach's Alpha recommendations
        if 'cronbach_alpha' in statistical_results:
            for factor, alpha in statistical_results['cronbach_alpha'].items():
                if alpha < 0.7:
                    recommendations.append(f"Consider removing items from {factor} to improve Cronbach's Alpha (current: {alpha:.3f})")
                elif alpha > 0.95:
                    recommendations.append(f"Check for redundant items in {factor} (Cronbach's Alpha: {alpha:.3f})")
        
        # Factor analysis recommendations
        if 'factor_analysis' in statistical_results:
            fa_results = statistical_results['factor_analysis']
            if 'cross_loadings' in fa_results and fa_results['cross_loadings'] > 0.4:
                recommendations.append("Consider removing items with high cross-loadings to improve factor structure")
            
            if 'kmo' in fa_results and fa_results['kmo']['kmo_overall'] < 0.6:
                recommendations.append("KMO value is low - consider removing items or increasing sample size")
        
        # Sample size recommendations
        if not quality_results['sample_size_adequate']:
            recommendations.append("Increase sample size for more reliable results")
        
        # Missing data recommendations
        if not quality_results['missing_data_acceptable']:
            recommendations.append("Address missing data issues (imputation or removal)")
        
        # Outlier recommendations
        if not quality_results['no_extreme_outliers']:
            recommendations.append("Investigate and handle extreme outliers")
        
        return recommendations
    
    def _generate_warnings(self, statistical_results: Dict[str, Any], 
                          quality_results: Dict[str, Any]) -> List[str]:
        """Generate warnings based on validation results."""
        warnings = []
        
        # Cronbach's Alpha warnings
        if 'cronbach_alpha' in statistical_results:
            for factor, alpha in statistical_results['cronbach_alpha'].items():
                if 0.7 <= alpha < 0.8:
                    warnings.append(f"{factor} has acceptable but marginal reliability (α = {alpha:.3f})")
        
        # Factor analysis warnings
        if 'factor_analysis' in statistical_results:
            fa_results = statistical_results['factor_analysis']
            if 'cross_loadings' in fa_results and 0.3 < fa_results['cross_loadings'] <= 0.4:
                warnings.append("Some items have moderate cross-loadings - review factor structure")
        
        # Correlation matrix warnings
        if 'correlation_validity' in statistical_results:
            corr_results = statistical_results['correlation_validity']
            if not corr_results['is_positive_definite']:
                warnings.append("Correlation matrix is not positive definite - results may be unstable")
            
            if corr_results['condition_number'] > 100:
                warnings.append("High multicollinearity detected - may affect regression results")
        
        # Variance warnings
        if not quality_results['sufficient_variance']:
            warnings.append("Some items have very low variance - consider removing them")
        
        return warnings
    
    def _check_for_errors(self, statistical_results: Dict[str, Any], 
                          quality_results: Dict[str, Any]) -> List[str]:
        """Check for critical errors in validation results."""
        errors = []
        
        # Critical Cronbach's Alpha errors
        if 'cronbach_alpha' in statistical_results:
            for factor, alpha in statistical_results['cronbach_alpha'].items():
                if alpha < 0.6:
                    errors.append(f"{factor} has poor reliability (α = {alpha:.3f})")
        
        # Factor analysis errors
        if 'factor_analysis' in statistical_results:
            fa_results = statistical_results['factor_analysis']
            if 'error' in fa_results:
                errors.append(f"Factor analysis failed: {fa_results['error']}")
        
        # Sample size errors
        if quality_results['sample_size_adequate'] and len(data) < 50:
            errors.append("Sample size is too small for reliable analysis")
        
        # Missing data errors
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_percentage > 20:
            errors.append("Too much missing data for reliable analysis")
        
        # Likert scale errors
        if not quality_results['valid_likert_scale']:
            errors.append("Data contains invalid Likert scale values")
        
        return errors
    
    def validate_regression_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate regression models specifically.
        
        Args:
            data: Data containing composite scores
            
        Returns:
            Regression validation results
        """
        return self.statistical_validator._validate_regression_models(data)
    
    def validate_factor_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate factor structure specifically.
        
        Args:
            data: Data for factor analysis
            
        Returns:
            Factor structure validation results
        """
        return self.statistical_validator._perform_factor_analysis(data)
    
    def get_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of validation results.
        
        Args:
            validation_results: Complete validation results
            
        Returns:
            Validation summary
        """
        summary = {
            'overall_validity': validation_results['overall_validity'],
            'num_recommendations': len(validation_results['recommendations']),
            'num_warnings': len(validation_results['warnings']),
            'num_errors': len(validation_results['errors']),
            'key_issues': [],
            'action_items': []
        }
        
        # Extract key issues
        if validation_results['errors']:
            summary['key_issues'].extend(validation_results['errors'][:3])
        
        if validation_results['warnings']:
            summary['key_issues'].extend(validation_results['warnings'][:2])
        
        # Extract action items
        if validation_results['recommendations']:
            summary['action_items'].extend(validation_results['recommendations'][:3])
        
        return summary