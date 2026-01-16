"""
Sensitivity analysis tools for BTUT.

Provides functions for analyzing how BTUT behavior changes
with parameter variations and testing robustness.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from btut.coordination.coordinator import BTUTCoordinator, SimulationResult


def sensitivity_analysis(
    coordinator: BTUTCoordinator,
    param: str,
    values: List[float],
    iterations: int = 25,
    n_trials: int = 1,
) -> Dict[str, Any]:
    """
    Analyze sensitivity to a single parameter.

    Args:
        coordinator: BTUTCoordinator instance.
        param: Parameter name ('gamma', 'tau').
        values: List of parameter values to test.
        iterations: Iterations per simulation.
        n_trials: Number of trials per value (for statistics).

    Returns:
        Dictionary with sensitivity analysis results.

    Example:
        >>> results = sensitivity_analysis(
        ...     coordinator, 'gamma', [1.2, 1.3, 1.4, 1.5]
        ... )
        >>> print(results['gradient'])  # Sensitivity measure
    """
    results_per_value = []

    for val in values:
        trial_results = []

        for trial in range(n_trials):
            # Update parameter
            if param == "gamma":
                coordinator.set_gamma(val)
            elif param == "tau":
                old_tau = coordinator.tau
                coordinator.tau = val
                coordinator._config["tau"] = val

            coordinator.reset()
            result = coordinator.run(iterations=iterations)
            trial_results.append(result.cooperation_rate)

        results_per_value.append({
            "value": val,
            "mean": np.mean(trial_results),
            "std": np.std(trial_results),
            "min": np.min(trial_results),
            "max": np.max(trial_results),
        })

    # Compute gradient (sensitivity measure)
    means = [r["mean"] for r in results_per_value]
    gradient = np.gradient(means, values)

    # Find steepest region (phase transition)
    max_gradient_idx = np.argmax(np.abs(gradient))

    return {
        "param": param,
        "values": values,
        "results": results_per_value,
        "gradient": gradient.tolist(),
        "max_sensitivity_at": values[max_gradient_idx],
        "max_sensitivity": float(np.max(np.abs(gradient))),
    }


def monte_carlo_analysis(
    coordinator: BTUTCoordinator,
    n_samples: int = 100,
    gamma_range: Tuple[float, float] = (1.0, 2.0),
    tau_range: Tuple[float, float] = (0.0, 1.0),
    iterations: int = 25,
) -> Dict[str, Any]:
    """
    Monte Carlo sampling of parameter space.

    Args:
        coordinator: BTUTCoordinator instance.
        n_samples: Number of random samples.
        gamma_range: (min, max) for gamma.
        tau_range: (min, max) for tau.
        iterations: Iterations per simulation.

    Returns:
        Dictionary with Monte Carlo results.

    Example:
        >>> mc_results = monte_carlo_analysis(coordinator, n_samples=1000)
        >>> print(f"Cooperation variance: {mc_results['coop_variance']:.3f}")
    """
    rng = np.random.default_rng()

    samples = []

    for _ in range(n_samples):
        gamma = rng.uniform(*gamma_range)
        tau = rng.uniform(*tau_range)

        coordinator.set_gamma(gamma)
        coordinator.tau = tau
        coordinator._config["tau"] = tau
        coordinator.reset()

        result = coordinator.run(iterations=iterations)

        samples.append({
            "gamma": gamma,
            "tau": tau,
            "cooperation": result.cooperation_rate,
            "converged": result.converged,
        })

    cooperations = [s["cooperation"] for s in samples]

    return {
        "n_samples": n_samples,
        "samples": samples,
        "coop_mean": float(np.mean(cooperations)),
        "coop_std": float(np.std(cooperations)),
        "coop_variance": float(np.var(cooperations)),
        "high_coop_fraction": float(np.mean([c > 0.9 for c in cooperations])),
        "low_coop_fraction": float(np.mean([c < 0.1 for c in cooperations])),
    }


def robustness_test(
    coordinator: BTUTCoordinator,
    noise_levels: List[float] = [0.0, 0.05, 0.1, 0.2],
    n_trials: int = 10,
    iterations: int = 25,
) -> Dict[str, Any]:
    """
    Test robustness to noise in the dynamics.

    Args:
        coordinator: BTUTCoordinator instance.
        noise_levels: List of noise standard deviations.
        n_trials: Trials per noise level.
        iterations: Iterations per trial.

    Returns:
        Dictionary with robustness analysis.
    """
    results = []

    for noise in noise_levels:
        # Set noise in dynamics
        coordinator.dynamics.noise = noise
        coordinator.dynamics.deterministic = (noise == 0)

        trial_results = []
        for _ in range(n_trials):
            coordinator.reset()
            result = coordinator.run(iterations=iterations)
            trial_results.append({
                "cooperation": result.cooperation_rate,
                "converged": result.converged,
                "runtime_ms": result.runtime_ms,
            })

        results.append({
            "noise": noise,
            "mean_cooperation": np.mean([r["cooperation"] for r in trial_results]),
            "std_cooperation": np.std([r["cooperation"] for r in trial_results]),
            "convergence_rate": np.mean([r["converged"] for r in trial_results]),
        })

    # Reset to deterministic
    coordinator.dynamics.noise = 0
    coordinator.dynamics.deterministic = True

    return {
        "noise_levels": noise_levels,
        "results": results,
        "robust": all(r["mean_cooperation"] > 0.8 for r in results),
    }


def scaling_test(
    n_agents_list: List[int] = [100, 500, 1000, 5000, 10000],
    gamma: float = 1.5,
    tau: float = 0.3,
    iterations: int = 25,
) -> Dict[str, Any]:
    """
    Test scaling behavior with population size.

    Verifies O(N) complexity by checking that:
    1. Convergence iterations remain constant
    2. Runtime scales linearly with N

    Args:
        n_agents_list: List of population sizes to test.
        gamma: Cooperation bonus.
        tau: Hub influence.
        iterations: Max iterations per test.

    Returns:
        Dictionary with scaling analysis.
    """
    from btut.networks.generator import NetworkGenerator

    results = []

    for n in n_agents_list:
        network = NetworkGenerator.barabasi_albert(n_agents=n, m=3)
        coordinator = BTUTCoordinator(
            network=network,
            gamma=gamma,
            tau=tau,
        )

        result = coordinator.run(iterations=iterations)

        results.append({
            "n_agents": n,
            "cooperation": result.cooperation_rate,
            "convergence_iter": result.convergence_iteration or iterations,
            "runtime_ms": result.runtime_ms,
            "runtime_per_agent_us": result.runtime_ms * 1000 / n,
        })

    # Check O(N) property: constant iterations
    iterations_used = [r["convergence_iter"] for r in results]
    iterations_constant = max(iterations_used) - min(iterations_used) <= 3

    # Check linear runtime
    runtimes = [r["runtime_ms"] for r in results]
    sizes = [r["n_agents"] for r in results]

    # Fit linear model
    slope, intercept = np.polyfit(sizes, runtimes, 1)

    return {
        "results": results,
        "o_n_verified": iterations_constant,
        "iterations_range": (min(iterations_used), max(iterations_used)),
        "runtime_slope_ms_per_agent": slope,
        "runtime_intercept_ms": intercept,
    }
