"""
btut.analysis - Visualization and analysis tools
================================================

Tools for analyzing BTUT simulation results, generating
visualizations, and performing sensitivity analysis.

Main Functions:
    - plot_convergence: Plot cooperation rate over iterations
    - plot_phase_diagram: Generate gamma-tau phase diagram
    - plot_degree_correlation: Degree vs cooperation scatter
    - sensitivity_analysis: Monte Carlo parameter sensitivity

Example:
    >>> from btut import BTUTCoordinator
    >>> from btut.analysis import plot_convergence, sensitivity_analysis
    >>>
    >>> coord = BTUTCoordinator.from_preset('standard')
    >>> result = coord.run()
    >>> plot_convergence(result, save_path='convergence.png')
    >>>
    >>> # Sensitivity analysis
    >>> sensitivity = sensitivity_analysis(
    ...     coord, param='gamma', values=[1.2, 1.3, 1.4, 1.5]
    ... )
"""

from btut.analysis.visualization import (
    plot_convergence,
    plot_phase_diagram,
    plot_degree_correlation,
    plot_parameter_sweep,
)
from btut.analysis.sensitivity import (
    sensitivity_analysis,
    monte_carlo_analysis,
    robustness_test,
)

__all__ = [
    "plot_convergence",
    "plot_phase_diagram",
    "plot_degree_correlation",
    "plot_parameter_sweep",
    "sensitivity_analysis",
    "monte_carlo_analysis",
    "robustness_test",
]
