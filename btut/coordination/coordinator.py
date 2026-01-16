"""
Main BTUT Coordinator class.

This is the primary interface for running BTUT simulations.
It provides a clean, high-level API while internally managing
the network, payoffs, and mean-field dynamics.

Example:
    >>> from btut import BTUTCoordinator, NetworkGenerator
    >>>
    >>> # Create network and coordinator
    >>> network = NetworkGenerator.barabasi_albert(n_agents=10000, m=3)
    >>> coordinator = BTUTCoordinator(network, gamma=1.5, tau=0.3)
    >>>
    >>> # Run simulation
    >>> result = coordinator.run(iterations=25)
    >>> print(f"Cooperation: {result.cooperation_rate:.1%}")
    >>> print(f"Converged in {result.convergence_iteration} iterations")
    >>>
    >>> # Find critical point
    >>> gamma_c = coordinator.find_critical_point()
    >>> print(f"Critical gamma: {gamma_c:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import time
import numpy as np

from btut.networks.generator import Network, NetworkGenerator
from btut.core.agent import AgentPopulation
from btut.core.payoffs import PayoffMatrix, StagHuntPD
from btut.core.dynamics import MeanFieldDynamics, ConvergenceChecker
from btut.core.presets import PRESETS, get_preset


@dataclass
class SimulationResult:
    """
    Container for BTUT simulation results.

    Attributes:
        cooperation_rate: Final fraction of cooperators.
        cooperation_history: Cooperation rate at each iteration.
        hub_cooperation: Final cooperation rate among hubs.
        periphery_cooperation: Final cooperation rate among periphery.
        convergence_iteration: Iteration at which convergence detected.
        converged: Whether simulation converged.
        runtime_ms: Total runtime in milliseconds.
        config: Configuration parameters used.
        final_population: Final population state.
    """

    cooperation_rate: float
    cooperation_history: List[float]
    hub_cooperation: float
    periphery_cooperation: float
    convergence_iteration: Optional[int]
    converged: bool
    runtime_ms: float
    config: Dict[str, Any]
    final_population: Optional[AgentPopulation] = None

    @property
    def n_iterations(self) -> int:
        """Number of iterations run."""
        return len(self.cooperation_history)

    @property
    def cooperation_gain(self) -> float:
        """Change in cooperation from start to end."""
        if len(self.cooperation_history) < 2:
            return 0.0
        return self.cooperation_history[-1] - self.cooperation_history[0]

    @property
    def steady_state(self) -> str:
        """Classify final state: 'cooperation', 'defection', or 'mixed'."""
        if self.cooperation_rate > 0.95:
            return "cooperation"
        elif self.cooperation_rate < 0.05:
            return "defection"
        else:
            return "mixed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cooperation_rate": self.cooperation_rate,
            "cooperation_history": self.cooperation_history,
            "hub_cooperation": self.hub_cooperation,
            "periphery_cooperation": self.periphery_cooperation,
            "convergence_iteration": self.convergence_iteration,
            "converged": self.converged,
            "runtime_ms": self.runtime_ms,
            "n_iterations": self.n_iterations,
            "steady_state": self.steady_state,
            "config": self.config,
        }


class BTUTCoordinator:
    """
    Main interface for BTUT multi-agent coordination.

    The BTUTCoordinator orchestrates the simulation by:
    1. Managing the network topology
    2. Computing mean-field dynamics
    3. Tracking convergence
    4. Recording history for analysis

    Attributes:
        network: Network topology with degree sequence.
        gamma: Cooperation bonus multiplier.
        tau: Hub influence parameter.
        payoff: Payoff matrix (default: StagHuntPD).
        seed: Random seed for reproducibility.

    Key Methods:
        run: Execute simulation for specified iterations.
        step: Execute single iteration.
        find_critical_point: Discover phase transition gamma.
        reset: Reset to initial state.

    Example:
        >>> coordinator = BTUTCoordinator.from_preset('standard')
        >>> result = coordinator.run()
        >>> print(f"Cooperation: {result.cooperation_rate:.1%}")
    """

    def __init__(
        self,
        network: Network,
        gamma: float = 1.45,
        tau: float = 0.30,
        alpha: float = 0.60,
        cA_SH: float = 0.40,
        cB_SH: float = 0.10,
        cA_PD: float = 0.20,
        cB_PD: float = 0.08,
        seed: Optional[int] = None,
        initial_cooperation: float = 0.5,
        payoff: Optional[PayoffMatrix] = None,
    ):
        """
        Initialize BTUT coordinator.

        Args:
            network: Network topology.
            gamma: Cooperation bonus (critical ~1.33).
            tau: Hub influence (0=uniform, 1=proportional to degree).
            alpha: Stag Hunt vs PD mixing weight.
            cA_SH: Cost of cooperation in Stag Hunt.
            cB_SH: Cost of defection in Stag Hunt.
            cA_PD: Cost of cooperation in PD.
            cB_PD: Cost of defection in PD.
            seed: Random seed.
            initial_cooperation: Starting cooperation fraction.
            payoff: Optional custom payoff matrix.
        """
        self.network = network
        self.gamma = gamma
        self.tau = tau
        self.seed = seed
        self.initial_cooperation = initial_cooperation

        # Create payoff matrix
        if payoff is not None:
            self.payoff = payoff
        else:
            self.payoff = StagHuntPD(
                gamma=gamma,
                alpha=alpha,
                cA_SH=cA_SH,
                cB_SH=cB_SH,
                cA_PD=cA_PD,
                cB_PD=cB_PD,
            )

        # Initialize dynamics
        self.dynamics = MeanFieldDynamics(payoff=self.payoff)
        if seed is not None:
            self.dynamics.set_seed(seed)

        # Initialize population
        self._population = AgentPopulation.from_degrees(
            degrees=network.degrees,
            tau=tau,
            initial_cooperation=initial_cooperation,
            seed=seed,
        )

        # History tracking
        self._history: List[float] = [self._population.cooperation_rate]
        self._iteration = 0

        # Store config
        self._config = {
            "n_agents": network.n_nodes,
            "gamma": gamma,
            "tau": tau,
            "alpha": alpha,
            "cA_SH": cA_SH,
            "cB_SH": cB_SH,
            "cA_PD": cA_PD,
            "cB_PD": cB_PD,
            "seed": seed,
            "topology": network.topology,
            "m": network.m,
        }

    @classmethod
    def from_preset(cls, preset_name: str) -> BTUTCoordinator:
        """
        Create coordinator from a preset configuration.

        Args:
            preset_name: Name of preset ('standard', 'quick', etc.).

        Returns:
            Configured BTUTCoordinator instance.

        Example:
            >>> coord = BTUTCoordinator.from_preset('high_cooperation')
        """
        config = get_preset(preset_name)

        network = NetworkGenerator.barabasi_albert(
            n_agents=config["n_agents"],
            m=config["m"],
            seed=config.get("seed"),
        )

        return cls(
            network=network,
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            cA_SH=config["cA_SH"],
            cB_SH=config["cB_SH"],
            cA_PD=config["cA_PD"],
            cB_PD=config["cB_PD"],
            seed=config.get("seed"),
        )

    @property
    def cooperation_rate(self) -> float:
        """Current cooperation rate."""
        return self._population.cooperation_rate

    @property
    def hub_cooperation(self) -> float:
        """Current cooperation rate among hubs."""
        return self._population.hub_cooperation

    @property
    def periphery_cooperation(self) -> float:
        """Current cooperation rate among periphery."""
        return self._population.periphery_cooperation

    @property
    def iteration(self) -> int:
        """Current iteration number."""
        return self._iteration

    @property
    def history(self) -> List[float]:
        """Cooperation rate history."""
        return self._history.copy()

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration parameters."""
        return self._config.copy()

    def step(self) -> float:
        """
        Execute one iteration of mean-field dynamics.

        Returns:
            New cooperation rate after update.
        """
        self._population = self.dynamics.update(self._population)
        self._iteration += 1
        rate = self._population.cooperation_rate
        self._history.append(rate)
        return rate

    def run(
        self,
        iterations: int = 25,
        convergence_threshold: float = 1e-6,
        convergence_window: int = 5,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> SimulationResult:
        """
        Run simulation for specified iterations.

        Args:
            iterations: Maximum number of iterations.
            convergence_threshold: Threshold for early stopping.
            convergence_window: Window size for convergence check.
            callback: Optional function called after each iteration.

        Returns:
            SimulationResult with final state and history.

        Example:
            >>> result = coordinator.run(iterations=30)
            >>> print(f"Converged: {result.converged}")
        """
        start_time = time.perf_counter()

        checker = ConvergenceChecker(
            window=convergence_window,
            tolerance=convergence_threshold,
        )

        convergence_iter = None

        for i in range(iterations):
            rate = self.step()

            if callback:
                callback(i, rate)

            if checker.update(rate):
                convergence_iter = i + 1
                break

        end_time = time.perf_counter()
        runtime_ms = (end_time - start_time) * 1000

        return SimulationResult(
            cooperation_rate=self.cooperation_rate,
            cooperation_history=self.history,
            hub_cooperation=self.hub_cooperation,
            periphery_cooperation=self.periphery_cooperation,
            convergence_iteration=convergence_iter,
            converged=convergence_iter is not None,
            runtime_ms=runtime_ms,
            config=self.config,
            final_population=self._population,
        )

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._population = AgentPopulation.from_degrees(
            degrees=self.network.degrees,
            tau=self.tau,
            initial_cooperation=self.initial_cooperation,
            seed=self.seed,
        )
        self._history = [self._population.cooperation_rate]
        self._iteration = 0

    def set_gamma(self, gamma: float) -> None:
        """
        Update gamma parameter.

        Args:
            gamma: New cooperation bonus value.
        """
        self.gamma = gamma
        self.payoff.gamma = gamma
        self._config["gamma"] = gamma

    def find_critical_point(
        self,
        tolerance: float = 0.005,
        gamma_range: tuple = (1.0, 2.0),
        iterations_per_test: int = 20,
    ) -> float:
        """
        Find critical gamma using binary search.

        The critical point γ_c is where the system transitions
        from all-defect to all-cooperate equilibrium.

        Args:
            tolerance: Precision for γ_c estimate.
            gamma_range: (low, high) range to search.
            iterations_per_test: Iterations per gamma test.

        Returns:
            Estimated critical gamma value.

        Example:
            >>> gamma_c = coordinator.find_critical_point()
            >>> print(f"Critical gamma: {gamma_c:.3f}")
        """
        low, high = gamma_range
        original_gamma = self.gamma

        while high - low > tolerance:
            mid = (low + high) / 2
            self.set_gamma(mid)
            self.reset()
            result = self.run(iterations=iterations_per_test)

            if result.cooperation_rate > 0.5:
                high = mid
            else:
                low = mid

        # Restore original gamma
        self.set_gamma(original_gamma)
        self.reset()

        return (low + high) / 2

    def parameter_sweep(
        self,
        param_name: str,
        values: List[float],
        iterations: int = 25,
    ) -> List[SimulationResult]:
        """
        Run simulations across parameter values.

        Args:
            param_name: Parameter to sweep ('gamma', 'tau').
            values: List of parameter values.
            iterations: Iterations per simulation.

        Returns:
            List of SimulationResults.
        """
        results = []
        original_value = getattr(self, param_name)

        for val in values:
            if param_name == "gamma":
                self.set_gamma(val)
            elif param_name == "tau":
                self.tau = val
                self._population = AgentPopulation.from_degrees(
                    degrees=self.network.degrees,
                    tau=val,
                    initial_cooperation=self.initial_cooperation,
                    seed=self.seed,
                )
                self._config["tau"] = val

            self.reset()
            result = self.run(iterations=iterations)
            results.append(result)

        # Restore original
        if param_name == "gamma":
            self.set_gamma(original_value)
        elif param_name == "tau":
            self.tau = original_value

        self.reset()
        return results
