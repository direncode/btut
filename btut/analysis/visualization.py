"""
Visualization utilities for BTUT analysis.

Provides plotting functions for analyzing BTUT simulation results.
All functions support both display and file output.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from btut.coordination.coordinator import SimulationResult


def plot_convergence(
    result: "SimulationResult",
    ax=None,
    save_path: Optional[str] = None,
    title: str = "BTUT Convergence Dynamics",
    show_threshold: bool = True,
) -> Any:
    """
    Plot cooperation rate convergence over iterations.

    Args:
        result: SimulationResult from coordinator.run().
        ax: Optional matplotlib axes.
        save_path: Optional path to save figure.
        title: Plot title.
        show_threshold: Whether to show convergence thresholds.

    Returns:
        Matplotlib figure or axes.

    Example:
        >>> result = coordinator.run()
        >>> plot_convergence(result, save_path='convergence.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    iterations = list(range(len(result.cooperation_history)))
    cooperation = result.cooperation_history

    # Main line
    ax.plot(iterations, cooperation, 'b-', linewidth=2, label='Cooperation Rate')

    # Thresholds
    if show_threshold:
        ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95% threshold')
        ax.axhline(y=0.50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
        ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')

    # Mark convergence
    if result.convergence_iteration:
        ax.axvline(
            x=result.convergence_iteration,
            color='purple',
            linestyle=':',
            label=f'Converged (iter {result.convergence_iteration})'
        )

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add annotation
    final_coop = cooperation[-1]
    ax.annotate(
        f'Final: {final_coop:.1%}',
        xy=(len(cooperation) - 1, final_coop),
        xytext=(len(cooperation) - 1 - 5, final_coop + 0.1),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray'),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig if ax is None else ax


def plot_phase_diagram(
    gamma_values: List[float],
    tau_values: List[float],
    cooperation_matrix: np.ndarray,
    ax=None,
    save_path: Optional[str] = None,
    title: str = "BTUT Phase Diagram",
) -> Any:
    """
    Plot gamma-tau phase diagram.

    Args:
        gamma_values: List of gamma values tested.
        tau_values: List of tau values tested.
        cooperation_matrix: 2D array of cooperation rates [gamma, tau].
        ax: Optional matplotlib axes.
        save_path: Optional path to save figure.
        title: Plot title.

    Returns:
        Matplotlib figure or axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Create meshgrid for pcolormesh
    G, T = np.meshgrid(gamma_values, tau_values)

    # Plot heatmap
    c = ax.pcolormesh(
        G, T, cooperation_matrix.T,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        shading='auto'
    )

    # Colorbar
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label('Cooperation Rate')

    # Mark critical line (approximate)
    critical_gamma = 1.33
    ax.axvline(x=critical_gamma, color='black', linestyle='--', linewidth=2,
               label=f'γ_c ≈ {critical_gamma}')

    ax.set_xlabel('γ (Cooperation Bonus)')
    ax.set_ylabel('τ (Hub Influence)')
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig if ax is None else ax


def plot_degree_correlation(
    degrees: np.ndarray,
    strategies: np.ndarray,
    ax=None,
    save_path: Optional[str] = None,
    title: str = "Degree-Cooperation Correlation",
) -> Any:
    """
    Plot correlation between node degree and cooperation.

    Args:
        degrees: Array of agent degrees.
        strategies: Boolean array of strategies (True=cooperate).
        ax: Optional matplotlib axes.
        save_path: Optional path to save figure.
        title: Plot title.

    Returns:
        Matplotlib figure or axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Bin degrees and compute cooperation rate per bin
    unique_degrees, indices = np.unique(degrees, return_inverse=True)
    coop_by_degree = np.zeros(len(unique_degrees))
    counts = np.zeros(len(unique_degrees))

    for i, deg_idx in enumerate(indices):
        coop_by_degree[deg_idx] += strategies[i]
        counts[deg_idx] += 1

    coop_by_degree /= np.maximum(counts, 1)

    # Scatter plot with size proportional to count
    sizes = np.sqrt(counts) * 10
    ax.scatter(unique_degrees, coop_by_degree, s=sizes, alpha=0.6, c='blue')

    # Trend line
    if len(unique_degrees) > 2:
        z = np.polyfit(np.log1p(unique_degrees), coop_by_degree, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(unique_degrees.min(), unique_degrees.max(), 100)
        ax.plot(x_trend, p(np.log1p(x_trend)), 'r--', linewidth=2, label='Trend')

    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig if ax is None else ax


def plot_parameter_sweep(
    param_values: List[float],
    cooperation_rates: List[float],
    param_name: str = "gamma",
    ax=None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    mark_critical: Optional[float] = None,
) -> Any:
    """
    Plot results of parameter sweep.

    Args:
        param_values: Parameter values tested.
        cooperation_rates: Cooperation rates for each value.
        param_name: Name of parameter.
        ax: Optional matplotlib axes.
        save_path: Optional path to save figure.
        title: Plot title.
        mark_critical: Optional critical value to mark.

    Returns:
        Matplotlib figure or axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if title is None:
        title = f"Parameter Sweep: {param_name}"

    ax.plot(param_values, cooperation_rates, 'bo-', linewidth=2, markersize=8)

    if mark_critical is not None:
        ax.axvline(x=mark_critical, color='red', linestyle='--',
                   label=f'{param_name}_c = {mark_critical:.3f}')

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel(param_name)
    ax.set_ylabel('Cooperation Rate')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig if ax is None else ax
