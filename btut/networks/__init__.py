"""
btut.networks - Network topology generators
===========================================

Provides generators for scale-free and other network topologies
used in BTUT simulations.

Main Classes:
    - NetworkGenerator: Factory for creating network degree sequences
    - Network: Container for network topology data

Supported Topologies:
    - Barabási-Albert (scale-free): Power-law degree distribution
    - Erdős-Rényi (random): Poisson degree distribution
    - Regular: Fixed degree for all nodes
    - Watts-Strogatz (small-world): High clustering with short paths

Example:
    >>> from btut.networks import NetworkGenerator
    >>> network = NetworkGenerator.barabasi_albert(n_agents=1000, m=3)
    >>> print(f"Max degree: {network.max_degree}")
"""

from btut.networks.generator import NetworkGenerator, Network
from btut.networks.utils import (
    degree_distribution,
    clustering_coefficient,
    power_law_fit,
)

__all__ = [
    "NetworkGenerator",
    "Network",
    "degree_distribution",
    "clustering_coefficient",
    "power_law_fit",
]
