"""
Network analysis utilities.

Provides functions for analyzing network properties and
validating degree distributions.
"""

from typing import Tuple, Optional
import numpy as np


def degree_distribution(degrees: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical degree distribution.

    Args:
        degrees: Array of node degrees.

    Returns:
        Tuple of (unique_degrees, normalized_counts).

    Example:
        >>> degrees = np.array([1, 2, 2, 3, 3, 3])
        >>> k, p = degree_distribution(degrees)
        >>> print(dict(zip(k, p)))  # {1: 0.167, 2: 0.333, 3: 0.5}
    """
    unique, counts = np.unique(degrees, return_counts=True)
    return unique, counts / counts.sum()


def clustering_coefficient(degrees: np.ndarray) -> float:
    """
    Estimate clustering coefficient from degree sequence.

    For scale-free networks, C ~ k^(-1) where k is mean degree.
    This provides an approximation without the full graph.

    Args:
        degrees: Array of node degrees.

    Returns:
        Estimated clustering coefficient.
    """
    mean_k = np.mean(degrees)
    if mean_k <= 1:
        return 0.0
    return 1.0 / mean_k


def power_law_fit(
    degrees: np.ndarray,
    k_min: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Fit power-law distribution to degree sequence.

    Uses maximum likelihood estimation (MLE) for the exponent.
    P(k) ~ k^(-α)

    Args:
        degrees: Array of node degrees.
        k_min: Minimum degree to include in fit (default: min(degrees)).

    Returns:
        Tuple of (exponent, standard_error).

    Example:
        >>> degrees = NetworkGenerator.barabasi_albert(10000, m=3).degrees
        >>> alpha, se = power_law_fit(degrees)
        >>> print(f"α = {alpha:.2f} ± {se:.2f}")  # ~3.0
    """
    if k_min is None:
        k_min = int(np.min(degrees))

    # Filter to tail
    k = degrees[degrees >= k_min].astype(np.float64)
    n = len(k)

    if n < 10:
        return float('nan'), float('nan')

    # MLE estimator
    log_ratio_sum = np.sum(np.log(k / k_min))
    alpha = 1 + n / log_ratio_sum

    # Standard error
    se = (alpha - 1) / np.sqrt(n)

    return float(alpha), float(se)


def assortativity(degrees: np.ndarray) -> float:
    """
    Estimate degree assortativity from degree sequence.

    Assortativity measures whether high-degree nodes connect to
    other high-degree nodes (positive) or low-degree nodes (negative).

    Scale-free networks typically have negative assortativity.

    Args:
        degrees: Array of node degrees.

    Returns:
        Estimated assortativity coefficient in [-1, 1].
    """
    # For a proper calculation we need edge information
    # This is a rough estimate based on degree variance
    var_k = np.var(degrees)
    mean_k = np.mean(degrees)

    if var_k < 1e-10:
        return 0.0

    # High variance -> likely disassortative for BA networks
    cv = np.sqrt(var_k) / mean_k
    return -np.tanh(cv - 1)  # Maps to negative for scale-free


def check_graphicality(degrees: np.ndarray) -> bool:
    """
    Check if degree sequence can form a valid simple graph.

    Uses the Erdős–Gallai theorem.

    Args:
        degrees: Array of node degrees.

    Returns:
        True if degree sequence is graphical.
    """
    d = np.sort(degrees)[::-1]
    n = len(d)

    # Sum must be even
    if np.sum(d) % 2 != 0:
        return False

    # Erdős–Gallai inequalities
    cumsum = np.cumsum(d)
    for k in range(1, n + 1):
        left = cumsum[k - 1]
        right = k * (k - 1) + np.sum(np.minimum(d[k:], k))
        if left > right:
            return False

    return True
