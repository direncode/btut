"""
btut.calibration - Critical point finder and domain calibration
===============================================================

Tools for discovering the phase transition critical point (γ_c)
and calibrating BTUT parameters for specific application domains.

Main Classes:
    - CriticalPointFinder: Binary search for γ_c
    - DomainCalibrator: Domain-specific parameter tuning

The critical point γ_c is where the system transitions from:
    - γ < γ_c: All-defect equilibrium (0% cooperation)
    - γ > γ_c: All-cooperate equilibrium (100% cooperation)
    - γ ≈ γ_c: Bistable region (sensitive to initial conditions)

Example:
    >>> from btut.calibration import CriticalPointFinder
    >>> finder = CriticalPointFinder(n_agents=1000, tau=0.3)
    >>> result = finder.find()
    >>> print(f"Critical gamma: {result.gamma_critical:.4f}")
    >>> print(f"Recommended: {result.recommended_gamma:.4f}")
"""

from btut.calibration.critical_point import CriticalPointFinder, CriticalPointResult

__all__ = ["CriticalPointFinder", "CriticalPointResult"]
