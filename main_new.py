"""
New main entry point for the refactored SEM/PLS data generator.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data_generator import SEMDataGenerator
from src.core.config_manager import ConfigManager
from src.core.exceptions import SEMDataGenerationError


def load_config_from_old_format(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from the old format and convert to new format.
    
    Args:
        config_path: Path to the old config.py file
        
    Returns:
        Configuration dictionary in new format
    """
    try:
        # Execute the old config file to get variables
        config_globals = {}
        with open(config_path, 'r', encoding='utf-8') as f:
            exec(f.read(), config_globals)
        
        # Convert to new format
        new_config = {
            'factors_config': config_globals.get('factors_config', {}),
            'regression_models': config_globals.get('regression_models', []),
            'latent_correlation_matrix': config_globals.get('latent_correlation_matrix'),
            'num_observations': config_globals.get('num_observations', 367),
            'genetic_algorithm_config': {
                'population_size': config_globals.get('population_size', 100),
                'num_generations': config_globals.get('num_generations', 200),
                'crossover_rate': config_globals.get('crossover_rate', 0.8),
                'base_mutation_rate': config_globals.get('base_mutation_rate', 0.15),
                'mutation_scale': config_globals.get('mutation_scale', 0.08),
                'elitism_count': config_globals.get('elitism_count', 5),
                'stagnation_threshold': config_globals.get('stagnation_threshold', 7),
                'mutation_increase_factor': config_globals.get('mutation_increase_factor', 1.3),
                'mutation_decrease_factor': config_globals.get('mutation_decrease_factor', 0.8),
                'max_mutation_rate': config_globals.get('max_mutation_rate', 0.3),
                'min_mutation_rate': config_globals.get('min_mutation_rate', 0.05)
            },
            'parameter_bounds': {
                'latent_cor_values': config_globals.get('param_bounds', {}).get('latent_cor_values', [0.01, 0.95]),
                'loading_strength': config_globals.get('param_bounds', {}).get('loading_strength', [0.45, 0.65]),
                'error_strength': config_globals.get('param_bounds', {}).get('error_strength', [0.35, 0.55])
            }
        }
        
        return new_config
        
    except Exception as e:
        raise SEMDataGenerationError(f"Failed to load configuration: {str(e)}")


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sem_generator.log', encoding='utf-8')
        ]
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SEM/PLS Synthetic Data Generator')
    parser.add_argument('--config', '-c', 
                       default='config.py',
                       help='Path to configuration file (default: config.py)')
    parser.add_argument('--output', '-o', 
                       help='Output directory (default: auto/output)')
    parser.add_argument('--processes', '-p', 
                       type=int, 
                       help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('--log-level', '-l',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    parser.add_argument('--validation-only', '-v',
                       action='store_true',
                       help='Only run validation (skip optimization)')
    parser.add_argument('--summary', '-s',
                       action='store_true',
                       help='Print summary and exit')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config_dict = load_config_from_old_format(args.config)
        
        # Initialize data generator
        generator = SEMDataGenerator(config_dict, args.output)
        
        # Print summary if requested
        if args.summary:
            summary = generator.get_summary()
            print("\n" + "="*50)
            print("SEM/PLS Data Generator Summary")
            print("="*50)
            print(f"Number of Factors: {summary['config_summary']['num_factors']}")
            print(f"Factor Names: {', '.join(summary['config_summary']['factor_names'])}")
            print(f"Sample Size: {summary['config_summary']['num_observations']}")
            print(f"Regression Models: {summary['config_summary']['num_regression_models']}")
            print(f"Output Directory: {summary['output_directory']}")
            print("="*50)
            return
        
        # Validation only mode
        if args.validation_only:
            logger.info("Running validation only mode")
            
            # Use default parameters for validation
            default_params = [0.5] * generator.config.n_latent_cor_values + [0.4, 0.6]
            
            try:
                data = generator.generate_data(default_params)
                validation_results = generator.validate_data(data)
                
                print("\nValidation Results:")
                print(f"Overall Validity: {validation_results.get('overall_validity', False)}")
                print(f"Recommendations: {len(validation_results.get('recommendations', []))}")
                print(f"Warnings: {len(validation_results.get('warnings', []))}")
                print(f"Errors: {len(validation_results.get('errors', []))}")
                
                if validation_results.get('recommendations'):
                    print("\nRecommendations:")
                    for rec in validation_results['recommendations']:
                        print(f"  • {rec}")
                
                if validation_results.get('warnings'):
                    print("\nWarnings:")
                    for warning in validation_results['warnings']:
                        print(f"  ⚠ {warning}")
                
                if validation_results.get('errors'):
                    print("\nErrors:")
                    for error in validation_results['errors']:
                        print(f"  ❌ {error}")
                
            except Exception as e:
                logger.error(f"Validation failed: {str(e)}")
                return
        
        # Full pipeline
        else:
            logger.info("Starting full SEM data generation pipeline")
            
            # Run full pipeline
            results = generator.run_full_pipeline(num_processes=args.processes)
            
            # Print summary
            print("\n" + "="*50)
            print("SEM/PLS Data Generation Complete")
            print("="*50)
            print(f"Best Score: {results['optimization']['best_score']:.3f}")
            print(f"Reason: {results['optimization']['best_reason']}")
            print(f"Generated Data Shape: {results['generated_data'].shape}")
            print(f"Export Path: {results['export_path']}")
            print(f"Overall Validity: {results['validation']['overall_validity']}")
            print("="*50)
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()