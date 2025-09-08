"""
Mathematical utilities for matrix operations and statistical functions.
"""

import numpy as np
from typing import Tuple


def is_positive_definite(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is positive definite.
    
    Args:
        matrix: Input matrix
        
    Returns:
        True if matrix is positive definite
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(matrix: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Find the nearest positive definite matrix to the input matrix.
    
    Uses Higham's 1988 algorithm to find the nearest positive definite matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Tuple of (nearest_positive_definite_matrix, has_diagonal_gt_one)
        
    Raises:
        RuntimeError: If unable to find positive definite matrix after 1000 iterations
    """
    # Ensure matrix is symmetric
    symmetric_matrix = (matrix + matrix.T) / 2
    
    # Compute nearest positive definite matrix
    try:
        # SVD decomposition
        U, s, Vt = np.linalg.svd(symmetric_matrix)
        
        # Create positive semi-definite matrix
        H = Vt.T @ np.diag(s) @ Vt
        
        # Average with original
        A2 = (symmetric_matrix + H) / 2
        
        # Ensure symmetry again
        A3 = (A2 + A2.T) / 2
        
        # Check for diagonal elements > 1.0 (Heywood cases)
        diag_gt_one = np.any(np.diag(A3) > 1.0 + np.finfo(float).eps)
        
        if is_positive_definite(A3):
            return A3, diag_gt_one
        
        # If not positive definite, apply iterative correction
        spacing = np.spacing(np.linalg.norm(A3))
        identity = np.eye(A3.shape[0])
        k = 1
        
        while not is_positive_definite(A3):
            min_eig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += identity * (-min_eig * k**2 + spacing)
            k += 1
            
            if k > 1000:
                raise RuntimeError("Unable to find positive definite matrix after 1000 iterations")
            
            # Check diagonal elements again
            diag_gt_one = np.any(np.diag(A3) > 1.0 + np.finfo(float).eps)
        
        return A3, diag_gt_one
        
    except np.linalg.LinAlgError:
        # Fallback to simpler method if SVD fails
        return _simple_nearest_pd(matrix)


def _simple_nearest_pd(matrix: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Simple fallback method for finding nearest positive definite matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Tuple of (nearest_positive_definite_matrix, has_diagonal_gt_one)
    """
    # Ensure symmetry
    symmetric_matrix = (matrix + matrix.T) / 2
    
    # Add small positive values to diagonal if needed
    diag_gt_one = np.any(np.diag(symmetric_matrix) > 1.0 + np.finfo(float).eps)
    
    if is_positive_definite(symmetric_matrix):
        return symmetric_matrix, diag_gt_one
    
    # Add regularization
    n = symmetric_matrix.shape[0]
    reg_matrix = symmetric_matrix + np.eye(n) * 1e-6
    
    if is_positive_definite(reg_matrix):
        return reg_matrix, diag_gt_one
    
    # If still not positive definite, use eigenvalue correction
    eigvals, eigvecs = np.linalg.eigh(symmetric_matrix)
    eigvals = np.maximum(eigvals, 1e-6)
    corrected_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Ensure symmetry
    corrected_matrix = (corrected_matrix + corrected_matrix.T) / 2
    
    # Check diagonal again
    diag_gt_one = np.any(np.diag(corrected_matrix) > 1.0 + np.finfo(float).eps)
    
    return corrected_matrix, diag_gt_one


def correlation_from_covariance(covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.
    
    Args:
        covariance_matrix: Input covariance matrix
        
    Returns:
        Correlation matrix
    """
    # Calculate standard deviations
    std_devs = np.sqrt(np.diag(covariance_matrix))
    
    # Avoid division by zero
    std_devs[std_devs == 0] = 1e-10
    
    # Calculate correlation matrix
    D = np.diag(1 / std_devs)
    correlation_matrix = D @ covariance_matrix @ D
    
    return correlation_matrix


def covariance_from_correlation(correlation_matrix: np.ndarray, 
                               standard_deviations: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to covariance matrix.
    
    Args:
        correlation_matrix: Input correlation matrix
        standard_deviations: Array of standard deviations
        
    Returns:
        Covariance matrix
    """
    D = np.diag(standard_deviations)
    covariance_matrix = D @ correlation_matrix @ D
    
    return covariance_matrix


def matrix_condition_number(matrix: np.ndarray) -> float:
    """
    Calculate the condition number of a matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Condition number
    """
    try:
        return np.linalg.cond(matrix)
    except np.linalg.LinAlgError:
        return float('inf')


def matrix_rank(matrix: np.ndarray, tol: Optional[float] = None) -> int:
    """
    Calculate the rank of a matrix.
    
    Args:
        matrix: Input matrix
        tol: Tolerance for rank calculation
        
    Returns:
        Matrix rank
    """
    return np.linalg.matrix_rank(matrix, tol=tol)


def is_symmetric(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if a matrix is symmetric.
    
    Args:
        matrix: Input matrix
        tol: Tolerance for symmetry check
        
    Returns:
        True if matrix is symmetric
    """
    return np.allclose(matrix, matrix.T, atol=tol)


def normalize_matrix(matrix: np.ndarray, norm_type: str = 'frobenius') -> np.ndarray:
    """
    Normalize a matrix.
    
    Args:
        matrix: Input matrix
        norm_type: Type of normalization ('frobenius', 'max', 'trace')
        
    Returns:
        Normalized matrix
    """
    if norm_type == 'frobenius':
        norm = np.linalg.norm(matrix, 'fro')
    elif norm_type == 'max':
        norm = np.max(np.abs(matrix))
    elif norm_type == 'trace':
        norm = np.trace(matrix)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
    
    if norm == 0:
        return matrix
    
    return matrix / norm


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Calculate Mahalanobis distance.
    
    Args:
        x: Data point
        mean: Mean vector
        cov: Covariance matrix
        
    Returns:
        Mahalanobis distance
    """
    try:
        inv_cov = np.linalg.inv(cov)
        diff = x - mean
        return np.sqrt(diff.T @ inv_cov @ diff)
    except np.linalg.LinAlgError:
        return float('inf')


def pairwise_distances(X: np.ndarray, Y: Optional[np.ndarray] = None, 
                      metric: str = 'euclidean') -> np.ndarray:
    """
    Calculate pairwise distances between points.
    
    Args:
        X: First set of points
        Y: Second set of points (optional)
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        
    Returns:
        Distance matrix
    """
    if Y is None:
        Y = X
    
    if metric == 'euclidean':
        return np.sqrt(((X[:, np.newaxis] - Y) ** 2).sum(axis=2))
    elif metric == 'manhattan':
        return np.abs(X[:, np.newaxis] - Y).sum(axis=2)
    elif metric == 'cosine':
        # Normalize vectors
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        return 1 - (X_norm @ Y_norm.T)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax function.
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    x_stable = x - np.max(x, axis=axis, keepdims=True)
    
    # Compute softmax
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)