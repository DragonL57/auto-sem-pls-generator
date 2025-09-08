"""
Main SEM/PLS data generator class.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .config_manager import ConfigManager
from .exceptions import SEMDataGenerationError, DataGenerationError
from ..optimization.genetic_optimizer import GeneticOptimizer
from ..validation.data_validator import DataValidator
from ..export.results_exporter import ResultsExporter
from ..utils.math_utils import nearest_positive_definite


class SEMDataGenerator:
    """Main orchestrator for SEM/PLS synthetic data generation."""
    
    def __init__(self, config_dict: Dict[str, Any], output_dir: Optional[str] = None):
        """
        Initialize the SEM data generator.
        
        Args:
            config_dict: Configuration dictionary
            output_dir: Output directory for results (default: auto/output)
        """
        self.config = ConfigManager(config_dict)
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "..", "..", "output")
        self.logger = self._setup_logging()
        
        # Initialize components
        self.genetic_optimizer = GeneticOptimizer(self.config)
        self.data_validator = DataValidator(self.config)
        self.results_exporter = ResultsExporter(self.config, self.output_dir)
        
        # Results storage
        self.best_parameters: Optional[np.ndarray] = None
        self.best_score: float = -np.inf
        self.best_reason: str = ""
        self.generated_data: Optional[pd.DataFrame] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger = logging.getLogger('SEMDataGenerator')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        log_file = os.path.join(self.output_dir, 'sem_generator.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_optimization(self, num_processes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the genetic algorithm optimization.
        
        Args:
            num_processes: Number of processes to use (default: CPU count - 1)
            
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info("Starting genetic algorithm optimization")
        start_time = time.time()
        
        try:
            # Determine number of processes
            if num_processes is None:
                num_processes = max(1, multiprocessing.cpu_count() - 1)
            
            self.logger.info(f"Using {num_processes} processes for optimization")
            
            # Run optimization
            optimization_results = self.genetic_optimizer.optimize(
                num_processes=num_processes
            )
            
            # Store results
            self.best_parameters = optimization_results['best_parameters']
            self.best_score = optimization_results['best_score']
            self.best_reason = optimization_results['best_reason']
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Best score: {self.best_score:.3f}, Reason: {self.best_reason}")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise SEMDataGenerationError(f"Optimization failed: {str(e)}")
    
    def generate_data(self, parameters: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate synthetic data using the best parameters.
        
        Args:
            parameters: Parameters to use (default: best parameters from optimization)
            
        Returns:
            Generated data as DataFrame
        """
        if parameters is None:
            parameters = self.best_parameters
            
        if parameters is None:
            raise DataGenerationError("No parameters available for data generation")
        
        self.logger.info("Generating synthetic data")
        
        try:
            # Generate data using the genetic optimizer
            generated_data = self.genetic_optimizer.generate_data(parameters)
            self.generated_data = generated_data
            
            self.logger.info(f"Generated data shape: {generated_data.shape}")
            return generated_data
            
        except Exception as e:
            self.logger.error(f"Data generation failed: {str(e)}")
            raise DataGenerationError(f"Data generation failed: {str(e)}")
    
    def validate_data(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate the generated data.
        
        Args:
            data: Data to validate (default: generated data)
            
        Returns:
            Validation results dictionary
        """
        if data is None:
            data = self.generated_data
            
        if data is None:
            raise DataGenerationError("No data available for validation")
        
        self.logger.info("Validating generated data")
        
        try:
            validation_results = self.data_validator.validate_all(data)
            self.logger.info("Data validation completed")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise DataGenerationError(f"Data validation failed: {str(e)}")
    
    def export_results(self, data: Optional[pd.DataFrame] = None, 
                      validation_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Export results to Excel file.
        
        Args:
            data: Data to export (default: generated data)
            validation_results: Validation results to include
            
        Returns:
            Path to exported Excel file
        """
        if data is None:
            data = self.generated_data
            
        if data is None:
            raise DataGenerationError("No data available for export")
        
        self.logger.info("Exporting results")
        
        try:
            export_path = self.results_exporter.export_excel(
                data=data,
                validation_results=validation_results,
                parameters=self.best_parameters,
                score=self.best_score,
                reason=self.best_reason
            )
            
            self.logger.info(f"Results exported to: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Results export failed: {str(e)}")
            raise DataGenerationError(f"Results export failed: {str(e)}")
    
    def run_full_pipeline(self, num_processes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline: optimization -> data generation -> validation -> export.
        
        Args:
            num_processes: Number of processes for optimization
            
        Returns:
            Complete results dictionary
        """
        self.logger.info("Starting full SEM data generation pipeline")
        
        try:
            # Step 1: Optimization
            optimization_results = self.run_optimization(num_processes)
            
            # Step 2: Data generation
            generated_data = self.generate_data()
            
            # Step 3: Validation
            validation_results = self.validate_data()
            
            # Step 4: Export
            export_path = self.export_results(generated_data, validation_results)
            
            # Compile final results
            final_results = {
                'optimization': optimization_results,
                'generated_data': generated_data,
                'validation': validation_results,
                'export_path': export_path,
                'config': self.config.to_dict()
            }
            
            self.logger.info("Full pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise SEMDataGenerationError(f"Pipeline execution failed: {str(e)}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration and reinitialize components."""
        self.logger.info("Updating configuration")
        self.config = ConfigManager(new_config)
        self.genetic_optimizer = GeneticOptimizer(self.config)
        self.data_validator = DataValidator(self.config)
        self.results_exporter = ResultsExporter(self.config, self.output_dir)
        self.logger.info("Configuration updated successfully")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state and results."""
        return {
            'config_summary': {
                'num_factors': self.config.n_latent_factors,
                'factor_names': self.config.latent_factor_names,
                'num_observations': self.config.num_observations,
                'num_regression_models': len(self.config.regression_models)
            },
            'optimization_summary': {
                'best_score': self.best_score,
                'best_reason': self.best_reason,
                'has_parameters': self.best_parameters is not None
            },
            'data_summary': {
                'has_data': self.generated_data is not None,
                'data_shape': self.generated_data.shape if self.generated_data is not None else None
            },
            'output_directory': self.output_dir
        }