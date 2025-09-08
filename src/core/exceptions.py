"""
Custom exceptions for the SEM/PLS data generation system.
"""


class SEMDataGenerationError(Exception):
    """Base exception for SEM data generation errors."""
    pass


class ConfigurationError(SEMDataGenerationError):
    """Exception raised for configuration-related errors."""
    pass


class OptimizationError(SEMDataGenerationError):
    """Exception raised for optimization-related errors."""
    pass


class ValidationError(SEMDataGenerationError):
    """Exception raised for validation-related errors."""
    pass


class DataGenerationError(SEMDataGenerationError):
    """Exception raised for data generation errors."""
    pass


class ExportError(SEMDataGenerationError):
    """Exception raised for export-related errors."""
    pass