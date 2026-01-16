"""
btut.coordination - Main coordination loop
==========================================

The coordination module contains the high-level BTUTCoordinator class
that orchestrates the simulation and provides a clean API for users.

Main Classes:
    - BTUTCoordinator: Primary interface for running BTUT simulations
    - SimulationResult: Container for simulation results

Example:
    >>> from btut import BTUTCoordinator, NetworkGenerator
    >>> network = NetworkGenerator.barabasi_albert(1000, m=3)
    >>> coordinator = BTUTCoordinator(network, gamma=1.5, tau=0.3)
    >>> result = coordinator.run(iterations=25)
    >>> print(f"Final cooperation: {result.cooperation_rate:.1%}")
"""

from btut.coordination.coordinator import BTUTCoordinator, SimulationResult

__all__ = ["BTUTCoordinator", "SimulationResult"]
