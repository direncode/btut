"""
BTUT - Bivariate Trajectory-Undercurrent Theory
================================================

O(N) multi-agent coordination via kernel-weighted mean-field dynamics.

Quick Start:
    >>> from btut import BTUTCoordinator, NetworkGenerator
    >>> network = NetworkGenerator.barabasi_albert(n_agents=1000, m=3)
    >>> coordinator = BTUTCoordinator(network, gamma=1.35, tau=0.7)
    >>> results = coordinator.run(iterations=50)
    >>> print(f"Cooperation: {results.cooperation_rate:.1%}")

Key Concepts:
    - gamma (γ): Cooperation bonus multiplier. Critical point γ_c ≈ 1.33
    - tau (τ): Hub influence weight. Higher = hubs drive cooperation
    - O(N) complexity via mean-field approximation (no adjacency matrix)

Modules:
    - btut.core: Agent class, payoff matrices, mean-field dynamics
    - btut.networks: Barabási-Albert and other network generators
    - btut.coordination: Main coordination loop with kernel-weighted updates
    - btut.calibration: Critical point finder (γ_c discovery)
    - btut.environments: Adapters for SUMO, ROS, PettingZoo
    - btut.analysis: Visualization and sensitivity analysis tools
"""

__version__ = "2.0.0"
__author__ = "Diren"

# High-level API imports
from btut.coordination.coordinator import BTUTCoordinator
from btut.networks.generator import NetworkGenerator
from btut.calibration.critical_point import CriticalPointFinder
from btut.core.agent import Agent
from btut.core.payoffs import PayoffMatrix, StagHuntPD

# Convenient presets
from btut.core.presets import PRESETS

__all__ = [
    # Main classes
    "BTUTCoordinator",
    "NetworkGenerator",
    "CriticalPointFinder",
    "Agent",
    "PayoffMatrix",
    "StagHuntPD",
    # Presets
    "PRESETS",
    # Version
    "__version__",
]
