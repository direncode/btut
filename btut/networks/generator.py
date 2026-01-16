"""
Network topology generators for BTUT.

This module provides efficient generators for various network topologies,
with a focus on scale-free (Barabási-Albert) networks that model real-world
social and infrastructure networks.

Theory:
    BTUT works on scale-free networks because:
    1. Power-law degree distribution creates natural "hubs"
    2. Hubs can propagate cooperation efficiently (τ mechanism)
    3. Real traffic/robot networks exhibit scale-free properties

    The Barabási-Albert model generates networks via preferential attachment:
    - Start with m₀ connected nodes
    - Add new nodes one at a time
    - Each new node connects to m existing nodes
    - Connection probability proportional to existing degree

    This yields P(k) ~ k^(-3) power-law distribution.

Example:
    >>> network = NetworkGenerator.barabasi_albert(n_agents=10000, m=3)
    >>> print(f"Nodes: {network.n_nodes}, Max degree: {network.max_degree}")
    >>> print(f"Power-law exponent: {network.power_law_exponent:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class Network:
    """
    Container for network topology data.

    This class stores the degree sequence and provides utilities
    for analyzing network properties without storing the full
    adjacency matrix (O(N) storage instead of O(N²)).

    Attributes:
        degrees: Array of node degrees.
        n_nodes: Number of nodes in the network.
        m: Parameter used for generation (edges per new node).
        topology: Name of the topology type.
        seed: Random seed used for generation.

    Properties:
        max_degree: Maximum degree in the network.
        min_degree: Minimum degree in the network.
        mean_degree: Average degree.
        degree_variance: Variance of degree distribution.
        power_law_exponent: Estimated power-law exponent (for scale-free).
    """

    degrees: np.ndarray
    n_nodes: int
    m: int
    topology: str
    seed: Optional[int] = None
    _adjacency: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate network data."""
        assert len(self.degrees) == self.n_nodes
        assert np.all(self.degrees >= 0)

    @property
    def max_degree(self) -> int:
        """Maximum degree in the network."""
        return int(np.max(self.degrees))

    @property
    def min_degree(self) -> int:
        """Minimum degree in the network."""
        return int(np.min(self.degrees))

    @property
    def mean_degree(self) -> float:
        """Average degree."""
        return float(np.mean(self.degrees))

    @property
    def degree_variance(self) -> float:
        """Variance of degree distribution."""
        return float(np.var(self.degrees))

    @property
    def degree_std(self) -> float:
        """Standard deviation of degrees."""
        return float(np.std(self.degrees))

    @property
    def power_law_exponent(self) -> float:
        """
        Estimate power-law exponent using MLE.

        For true Barabási-Albert networks, this should be ~3.0.
        Uses the Hill estimator for power-law fitting.
        """
        k = self.degrees[self.degrees >= self.m]  # Only fit tail
        if len(k) < 10:
            return float('nan')

        k_min = np.min(k)
        n = len(k)

        # MLE estimator: α = 1 + n / Σlog(k_i / k_min)
        alpha = 1 + n / np.sum(np.log(k / k_min))
        return float(alpha)

    @property
    def hub_fraction(self) -> float:
        """Fraction of nodes that are hubs (top 10% by degree)."""
        threshold = np.percentile(self.degrees, 90)
        return float(np.mean(self.degrees >= threshold))

    def degree_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute empirical degree distribution.

        Returns:
            Tuple of (unique_degrees, counts).
        """
        unique, counts = np.unique(self.degrees, return_counts=True)
        return unique, counts

    def hub_indices(self, percentile: float = 90) -> np.ndarray:
        """
        Get indices of hub nodes.

        Args:
            percentile: Percentile threshold for hub classification.

        Returns:
            Array of hub node indices.
        """
        threshold = np.percentile(self.degrees, percentile)
        return np.where(self.degrees >= threshold)[0]

    def periphery_indices(self, percentile: float = 90) -> np.ndarray:
        """
        Get indices of periphery (non-hub) nodes.

        Args:
            percentile: Percentile threshold for hub classification.

        Returns:
            Array of periphery node indices.
        """
        threshold = np.percentile(self.degrees, percentile)
        return np.where(self.degrees < threshold)[0]

    def to_dict(self) -> dict:
        """Convert network to dictionary representation."""
        return {
            "n_nodes": self.n_nodes,
            "m": self.m,
            "topology": self.topology,
            "seed": self.seed,
            "max_degree": self.max_degree,
            "min_degree": self.min_degree,
            "mean_degree": self.mean_degree,
            "degree_variance": self.degree_variance,
            "power_law_exponent": self.power_law_exponent,
        }


class NetworkGenerator:
    """
    Factory class for generating network topologies.

    All methods are static and return Network instances with
    degree sequences (O(N) storage) rather than full adjacency
    matrices (O(N²) storage).

    Supported Topologies:
        - barabasi_albert: Scale-free with power-law degrees
        - erdos_renyi: Random with Poisson degrees
        - regular: Fixed degree for all nodes
        - watts_strogatz: Small-world with high clustering

    Example:
        >>> network = NetworkGenerator.barabasi_albert(1000, m=3)
        >>> print(network.max_degree)
    """

    @staticmethod
    def barabasi_albert(
        n_agents: int,
        m: int = 3,
        seed: Optional[int] = None,
    ) -> Network:
        """
        Generate Barabási-Albert scale-free network.

        Uses preferential attachment to create power-law degree distribution.
        This is the primary network type for BTUT simulations.

        Args:
            n_agents: Number of nodes in the network.
            m: Number of edges per new node (minimum degree).
            seed: Random seed for reproducibility.

        Returns:
            Network with scale-free degree distribution.

        Algorithm:
            1. Start with m fully connected nodes
            2. For each new node:
               a. Compute attachment probabilities p_i = k_i / Σk_j
               b. Select m existing nodes via weighted random choice
               c. Connect new node to selected nodes
            3. Extract degree sequence

        Complexity: O(N·m) time, O(N) storage.

        Example:
            >>> network = NetworkGenerator.barabasi_albert(10000, m=3, seed=42)
            >>> print(f"Power-law exponent: {network.power_law_exponent:.2f}")
        """
        rng = np.random.default_rng(seed)

        if n_agents < m:
            raise ValueError(f"n_agents ({n_agents}) must be >= m ({m})")

        # Initialize degree array
        degrees = np.zeros(n_agents, dtype=np.int32)

        # Start with m+1 fully connected nodes (clique)
        m0 = m + 1
        degrees[:m0] = m  # Each node in clique has m connections

        # Track total degree for normalization
        total_degree = m0 * m

        # Preferential attachment for remaining nodes
        for i in range(m0, n_agents):
            # Compute attachment probabilities
            probs = degrees[:i].astype(np.float64)
            probs /= probs.sum()

            # Select m nodes to connect to (without replacement)
            # Use a trick: sample with replacement but track unique selections
            selected = set()
            attempts = 0
            max_attempts = m * 10

            while len(selected) < m and attempts < max_attempts:
                # Weighted random selection
                idx = rng.choice(i, p=probs)
                selected.add(idx)
                attempts += 1

            # If we couldn't get m unique selections, fill randomly
            while len(selected) < m:
                idx = rng.integers(0, i)
                selected.add(idx)

            # Update degrees
            selected_list = list(selected)
            degrees[i] = len(selected_list)
            degrees[selected_list] += 1
            total_degree += 2 * len(selected_list)

        return Network(
            degrees=degrees,
            n_nodes=n_agents,
            m=m,
            topology="barabasi_albert",
            seed=seed,
        )

    @staticmethod
    def erdos_renyi(
        n_agents: int,
        p: float = 0.01,
        seed: Optional[int] = None,
    ) -> Network:
        """
        Generate Erdős-Rényi random network.

        Each edge exists independently with probability p.
        Degree distribution follows Poisson with mean (n-1)·p.

        Args:
            n_agents: Number of nodes.
            p: Edge probability.
            seed: Random seed.

        Returns:
            Network with Poisson degree distribution.

        Note:
            This generates only the degree sequence, not the full graph.
            The approximation uses Poisson sampling.
        """
        rng = np.random.default_rng(seed)

        # Expected degree
        expected_degree = (n_agents - 1) * p

        # Sample degrees from Poisson distribution
        degrees = rng.poisson(expected_degree, size=n_agents)

        # Ensure minimum degree of 1
        degrees = np.maximum(degrees, 1)

        return Network(
            degrees=degrees.astype(np.int32),
            n_nodes=n_agents,
            m=int(expected_degree),
            topology="erdos_renyi",
            seed=seed,
        )

    @staticmethod
    def regular(
        n_agents: int,
        degree: int = 4,
        seed: Optional[int] = None,
    ) -> Network:
        """
        Generate regular network where all nodes have same degree.

        Args:
            n_agents: Number of nodes.
            degree: Degree for all nodes.
            seed: Random seed (unused, included for API consistency).

        Returns:
            Network with uniform degree distribution.
        """
        degrees = np.full(n_agents, degree, dtype=np.int32)

        return Network(
            degrees=degrees,
            n_nodes=n_agents,
            m=degree,
            topology="regular",
            seed=seed,
        )

    @staticmethod
    def watts_strogatz(
        n_agents: int,
        k: int = 4,
        beta: float = 0.1,
        seed: Optional[int] = None,
    ) -> Network:
        """
        Generate Watts-Strogatz small-world network.

        Creates network with high clustering and short path lengths.
        Degree distribution is approximately regular with small variance.

        Args:
            n_agents: Number of nodes.
            k: Number of nearest neighbors in ring lattice.
            beta: Rewiring probability.
            seed: Random seed.

        Returns:
            Network with small-world properties.

        Note:
            Actual degrees vary slightly due to rewiring.
        """
        rng = np.random.default_rng(seed)

        # Start with ring lattice degrees (all nodes have k neighbors)
        degrees = np.full(n_agents, k, dtype=np.int32)

        # Simulate rewiring effects on degree distribution
        # Each edge has probability beta of being rewired
        # Rewiring changes degree of 2 nodes by ±1
        n_edges = n_agents * k // 2
        n_rewired = rng.binomial(n_edges, beta)

        # Adjust some degrees randomly to simulate rewiring effect
        for _ in range(n_rewired):
            # Random node loses an edge
            loser = rng.integers(n_agents)
            if degrees[loser] > 1:
                degrees[loser] -= 1
                # Random node gains an edge
                gainer = rng.integers(n_agents)
                degrees[gainer] += 1

        return Network(
            degrees=degrees,
            n_nodes=n_agents,
            m=k,
            topology="watts_strogatz",
            seed=seed,
        )

    @staticmethod
    def from_degree_sequence(
        degrees: np.ndarray,
        topology: str = "custom",
        seed: Optional[int] = None,
    ) -> Network:
        """
        Create network from existing degree sequence.

        Args:
            degrees: Array of node degrees.
            topology: Name for the topology.
            seed: Random seed (for reference).

        Returns:
            Network wrapping the provided degrees.
        """
        degrees = np.asarray(degrees, dtype=np.int32)
        return Network(
            degrees=degrees,
            n_nodes=len(degrees),
            m=int(np.min(degrees)),
            topology=topology,
            seed=seed,
        )
