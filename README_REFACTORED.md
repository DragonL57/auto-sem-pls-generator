# SEM/PLS Synthetic Data Generator (Refactored)

A comprehensive, modular system for generating synthetic survey data for Structural Equation Modeling (SEM) and Partial Least Squares (PLS) analysis, featuring genetic algorithm optimization and robust statistical validation.

## 🚀 What's New in the Refactored Version

### ✅ Major Improvements

1. **Modular Architecture**: Clean separation of concerns with dedicated modules
2. **Robust Error Handling**: Comprehensive exception handling with custom error types
3. **Advanced Validation**: Multi-level statistical validation with detailed reporting
4. **Professional Exports**: Styled Excel reports with multiple sheets and formatting
5. **Type Safety**: Full type hints throughout the codebase
6. **CLI Interface**: Command-line interface with multiple options
7. **Configuration Management**: Pydantic-based configuration validation
8. **Performance Optimized**: Efficient algorithms and parallel processing

## 📁 Project Structure

```
auto/
├── src/
│   ├── core/                    # Core modules
│   │   ├── data_generator.py    # Main orchestrator
│   │   ├── config_manager.py    # Configuration management
│   │   └── exceptions.py        # Custom exceptions
│   ├── optimization/            # Optimization modules
│   │   ├── genetic_optimizer.py # GA optimization
│   │   └── genetic_algorithm.py # GA implementation
│   ├── validation/              # Validation modules
│   │   ├── data_validator.py    # Main validator
│   │   └── statistical_validator.py # Statistical validation
│   ├── export/                  # Export modules
│   │   └── results_exporter.py  # Excel/JSON export
│   └── utils/                   # Utility modules
│       ├── data_generator_utils.py # Data generation utilities
│       └── math_utils.py        # Mathematical utilities
├── main_new.py                  # New CLI entry point
├── config.py                    # Configuration file (old format compatible)
├── requirements.txt             # Dependencies
└── README_REFACTORED.md         # This file
```

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the refactored version**:
```bash
python main_new.py --help
```

## 🎯 Usage

### Basic Usage

```bash
# Run full pipeline with default settings
python main_new.py

# Use custom configuration
python main_new.py --config my_config.py

# Specify output directory
python main_new.py --output ./results

# Use multiple processes for optimization
python main_new.py --processes 4

# Run validation only
python main_new.py --validation-only

# Print summary and exit
python main_new.py --summary
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config, -c` | Configuration file path | `config.py` |
| `--output, -o` | Output directory | `auto/output` |
| `--processes, -p` | Number of processes | CPU count - 1 |
| `--log-level, -l` | Logging level | `INFO` |
| `--validation-only, -v` | Run validation only | False |
| `--summary, -s` | Print summary and exit | False |

### Programmatic Usage

```python
from src.core.data_generator import SEMDataGenerator

# Load configuration
config_dict = load_config_from_old_format('config.py')

# Initialize generator
generator = SEMDataGenerator(config_dict, './output')

# Run full pipeline
results = generator.run_full_pipeline()

# Access results
print(f"Best score: {results['optimization']['best_score']}")
print(f"Data shape: {results['generated_data'].shape}")
print(f"Export path: {results['export_path']}")
```

## 🔧 Configuration

The refactored version maintains backward compatibility with the old `config.py` format while adding validation and new features:

### Old Format (Still Supported)

```python
factors_config = {
    "PI": {"original_items": ["PI1", "PI2", "PI3", "PI4", "PI5"]},
    "PA": {"original_items": ["PA1", "PA2", "PA3", "PA4", "PA5"]},
    # ... more factors
}

regression_models = [
    {"dependent": "PA_composite", "independent": ["PI_composite"], "order": ["PI_composite"]},
    # ... more models
]
```

### New Features

- **Configuration Validation**: Automatic validation of configuration parameters
- **Parameter Bounds**: Flexible parameter boundary definitions
- **GA Configuration**: Separate genetic algorithm parameter management
- **Error Handling**: Graceful handling of configuration errors

## 📊 Validation Features

### Statistical Validation

- **Cronbach's Alpha**: Reliability analysis for each factor
- **Factor Analysis**: EFA with Promax rotation
- **KMO & Bartlett's Test**: Factorability assessment
- **Cross-loading Analysis**: Factor structure validation
- **Regression Validation**: Model fit and significance testing

### Data Quality Checks

- **Sample Size Adequacy**: Minimum sample requirements
- **Missing Data Analysis**: Missing value assessment
- **Outlier Detection**: IQR-based outlier identification
- **Likert Scale Validation**: Range and distribution checks
- **Variance Analysis**: Sufficient variance detection

## 📈 Export Features

### Excel Export

The refactored version generates comprehensive Excel reports with:

- **Generated Data Sheet**: Raw synthetic data with proper formatting
- **Validation Results Sheet**: Statistical validation outcomes
- **Configuration Sheet**: Model configuration summary
- **Optimization Parameters Sheet**: GA optimization results
- **Summary Sheet**: Analysis overview and statistics

### Additional Export Formats

- **JSON Export**: Machine-readable results format
- **Validation Report**: Detailed text-based validation report

## 🎮 Optimization Features

### Genetic Algorithm Improvements

- **Adaptive Mutation Rate**: Dynamic mutation adjustment
- **Tournament Selection**: Robust parent selection
- **Elitism Preservation**: Best individuals carried forward
- **Parallel Processing**: Multi-process optimization
- **Convergence Detection**: Stagnation monitoring

### Performance Optimizations

- **Vectorized Operations**: Efficient numerical computations
- **Memory Management**: Optimized data structures
- **Caching**: Strategic caching of expensive operations
- **Parallel Processing**: Multi-core utilization

## 🔍 Error Handling

### Custom Exceptions

- `SEMDataGenerationError`: Base exception class
- `ConfigurationError`: Configuration-related errors
- `OptimizationError`: Optimization failures
- `ValidationError`: Validation issues
- `DataGenerationError`: Data generation problems
- `ExportError`: Export failures

### Graceful Degradation

- **Non-critical Errors**: Continue processing with warnings
- **Error Recovery**: Automatic recovery from transient issues
- **Detailed Logging**: Comprehensive error logging
- **User-friendly Messages**: Clear error descriptions

## 🧪 Testing

### Validation Examples

```python
# Test data generation
data = generator.generate_data(parameters)

# Validate data
validation_results = generator.validate_data(data)

# Check overall validity
if validation_results['overall_validity']:
    print("Data validation passed!")
else:
    print("Data validation failed")
    print("Issues:", validation_results['errors'])
```

## 📚 API Reference

### Core Classes

- `SEMDataGenerator`: Main orchestrator class
- `ConfigManager`: Configuration management and validation
- `GeneticOptimizer`: Genetic algorithm optimization
- `DataValidator`: Comprehensive data validation
- `ResultsExporter`: Multi-format export functionality

### Utility Functions

- `generate_items_from_latent()`: Likert-scale item generation
- `nearest_positive_definite()`: Matrix positive definiteness
- `create_latent_correlation_matrix()`: Correlation matrix creation

## 🔄 Migration from Original Version

### For Users

1. **No changes needed**: Your existing `config.py` files work unchanged
2. **New CLI**: Use `main_new.py` instead of `main.py`
3. **Enhanced features**: Access new validation and export features
4. **Better error handling**: More informative error messages

### For Developers

1. **Modular structure**: Easier to extend and maintain
2. **Type hints**: Better IDE support and code clarity
3. **Comprehensive docs**: Detailed docstrings and examples
4. **Testable architecture**: Modular design facilitates testing

## 🎯 Performance Comparison

| Feature | Original | Refactored | Improvement |
|---------|----------|------------|-------------|
| Code Structure | Monolithic | Modular | ✅ Maintainability |
| Error Handling | Basic | Comprehensive | ✅ Reliability |
| Validation | Limited | Advanced | ✅ Accuracy |
| Export | Basic | Professional | ✅ Usability |
| Performance | Single-threaded | Multi-process | ✅ Speed |
| Documentation | Minimal | Comprehensive | ✅ Developer Experience |

## 🚀 Future Enhancements

Planned improvements for future versions:

- **GUI Interface**: Web-based user interface
- **Advanced Models**: Support for complex SEM models
- **Real-time Validation**: Live validation feedback
- **Cloud Integration**: Cloud-based processing
- **Advanced Analytics**: More sophisticated statistical analysis
- **Database Integration**: Direct database connectivity

## 🤝 Contributing

The refactored version is designed to be extensible and maintainable. Key areas for contribution:

- **New Validation Methods**: Additional statistical tests
- **Export Formats**: Support for more file formats
- **Optimization Algorithms**: Alternative optimization methods
- **User Interface**: GUI or web interface improvements
- **Documentation**: Enhanced documentation and examples

## 📄 License

This project continues under the same license as the original version.

## 🙏 Acknowledgments

- **Original Authors**: For the foundational work on SEM/PLS data generation
- **Statistical Community**: For the validation methodologies and best practices
- **Open Source Contributors**: For the libraries and tools that make this project possible

---

**Note**: This refactored version maintains full backward compatibility while providing significant improvements in code quality, features, and user experience.