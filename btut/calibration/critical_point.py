"""
Critical point finder for BTUT phase transitions.

This module implements systematic discovery of the critical gamma (γ_c)
where BTUT systems undergo phase transition from defection to cooperation.

Theory:
    The BTUT dynamics exhibit a phase transition at γ_c:

    - For γ < γ_c: Defection is the unique stable equilibrium
      The payoff for defection exceeds cooperation regardless of population state.

    - For γ > γ_c: Cooperation is the unique stable equilibrium
      The γ bonus makes cooperation payoff-dominant.

    - At γ = γ_c: Bistable regime
      Both all-cooperate and all-defect are stable; outcome depends on
      initial conditions and perturbations.

    For the default StagHuntPD parameters, γ_c ≈ 1.33.

Algorithm:
    1. Binary search in [γ_low, γ_high]
    2. For each candidate γ, run simulation from p₀ = 0.5
    3. If final cooperation > threshold: γ is above critical
    4. Refine until |γ_high - γ_low| < tolerance

Example:
    >>> finder = CriticalPointFinder(n_agents=1000, tau=0.3)
    >>> result = finder.find(tolerance=0.001)
    >>> print(f"γ_c = {result.gamma_critical:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time
import numpy as np

from btut.networks.generator import NetworkGenerator, Network
from btut.coordination.coordinator import BTUTCoordinator


@dataclass
class CriticalPointResult:
    """
    Result of critical point discovery.

    Attributes:
        gamma_critical: Discovered critical gamma value.
        gamma_lower: Last gamma below critical (0% cooperation).
        gamma_upper: First gamma above critical (100% cooperation).
        transition_sharpness: Measure of how sharp the transition is.
        recommended_gamma: Suggested operating point (γ_c + margin).
        recommended_margin: Safety margin above γ_c.
        search_iterations: Number of binary search iterations.
        confidence: Confidence score based on transition sharpness.
        domain: Application domain (if specified).
        config_used: Configuration parameters.
        timestamp: When the search was performed.
    """

    gamma_critical: float
    gamma_lower: float
    gamma_upper: float
    transition_sharpness: float
    recommended_gamma: float
    recommended_margin: float
    search_iterations: int
    confidence: float
    domain: str
    config_used: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gamma_critical": self.gamma_critical,
            "gamma_lower": self.gamma_lower,
            "gamma_upper": self.gamma_upper,
            "transition_sharpness": self.transition_sharpness,
            "recommended_gamma": self.recommended_gamma,
            "recommended_margin": self.recommended_margin,
            "search_iterations": self.search_iterations,
            "confidence": self.confidence,
            "domain": self.domain,
            "config_used": self.config_used,
            "timestamp": self.timestamp,
        }


class CriticalPointFinder:
    """
    Find the critical gamma for BTUT phase transition.

    This class performs systematic binary search to discover γ_c
    with configurable precision and validation.

    Attributes:
        n_agents: Number of agents for test simulations.
        tau: Hub influence parameter.
        m: Barabási-Albert connectivity parameter.
        seed: Random seed for reproducibility.
        domain: Application domain name.

    Methods:
        find: Execute binary search for γ_c.
        validate: Verify transition sharpness.
        sweep: Generate full gamma sweep data.

    Example:
        >>> finder = CriticalPointFinder(n_agents=1000)
        >>> result = finder.find()
        >>> print(f"Use gamma >= {result.recommended_gamma:.3f}")
    """

    def __init__(
        self,
        n_agents: int = 1000,
        tau: float = 0.30,
        m: int = 3,
        seed: Optional[int] = None,
        domain: str = "abstract",
        alpha: float = 0.60,
        cA_SH: float = 0.40,
        cB_SH: float = 0.10,
        cA_PD: float = 0.20,
        cB_PD: float = 0.08,
    ):
        """
        Initialize critical point finder.

        Args:
            n_agents: Population size for tests.
            tau: Hub influence parameter.
            m: Network connectivity.
            seed: Random seed.
            domain: Domain name for results.
            alpha: Stag Hunt mixing weight.
            cA_SH, cB_SH, cA_PD, cB_PD: Payoff costs.
        """
        self.n_agents = n_agents
        self.tau = tau
        self.m = m
        self.seed = seed
        self.domain = domain

        # Payoff parameters
        self.alpha = alpha
        self.cA_SH = cA_SH
        self.cB_SH = cB_SH
        self.cA_PD = cA_PD
        self.cB_PD = cB_PD

        # Create reusable network
        self._network = NetworkGenerator.barabasi_albert(
            n_agents=n_agents,
            m=m,
            seed=seed,
        )

    def _create_coordinator(self, gamma: float) -> BTUTCoordinator:
        """Create coordinator with specified gamma."""
        return BTUTCoordinator(
            network=self._network,
            gamma=gamma,
            tau=self.tau,
            alpha=self.alpha,
            cA_SH=self.cA_SH,
            cB_SH=self.cB_SH,
            cA_PD=self.cA_PD,
            cB_PD=self.cB_PD,
            seed=self.seed,
            initial_cooperation=0.5,
        )

    def _test_gamma(
        self,
        gamma: float,
        iterations: int = 20,
    ) -> float:
        """
        Test a specific gamma value.

        Args:
            gamma: Gamma value to test.
            iterations: Simulation iterations.

        Returns:
            Final cooperation rate.
        """
        coordinator = self._create_coordinator(gamma)
        result = coordinator.run(iterations=iterations)
        return result.cooperation_rate

    def find(
        self,
        tolerance: float = 0.005,
        gamma_range: tuple = (1.0, 2.0),
        iterations_per_test: int = 20,
        cooperation_threshold: float = 0.5,
        margin: float = 0.05,
    ) -> CriticalPointResult:
        """
        Find critical gamma using binary search.

        Args:
            tolerance: Precision for γ_c (stop when range < tolerance).
            gamma_range: (low, high) initial search range.
            iterations_per_test: Iterations per gamma test.
            cooperation_threshold: Threshold for classifying as "cooperative".
            margin: Safety margin for recommended gamma.

        Returns:
            CriticalPointResult with discovered γ_c and metadata.

        Example:
            >>> result = finder.find(tolerance=0.001)
            >>> print(f"γ_c = {result.gamma_critical:.4f}")
        """
        from datetime import datetime

        low, high = gamma_range
        search_iterations = 0
        gamma_lower = low
        gamma_upper = high

        while high - low > tolerance:
            mid = (low + high) / 2
            coop = self._test_gamma(mid, iterations_per_test)
            search_iterations += 1

            if coop > cooperation_threshold:
                high = mid
                gamma_upper = mid
            else:
                low = mid
                gamma_lower = mid

        gamma_critical = (low + high) / 2

        # Validate transition sharpness
        sharpness = self._measure_sharpness(
            gamma_critical,
            delta=tolerance * 2,
            iterations=iterations_per_test,
        )

        # Compute confidence based on sharpness
        confidence = min(0.5 + sharpness * 0.5, 0.99)

        return CriticalPointResult(
            gamma_critical=gamma_critical,
            gamma_lower=gamma_lower,
            gamma_upper=gamma_upper,
            transition_sharpness=sharpness,
            recommended_gamma=gamma_critical + margin,
            recommended_margin=margin,
            search_iterations=search_iterations,
            confidence=confidence,
            domain=self.domain,
            config_used={
                "N": self.n_agents,
                "tau": self.tau,
                "tolerance": tolerance,
                "cA_SH": self.cA_SH,
                "cB_SH": self.cB_SH,
                "cA_PD": self.cA_PD,
                "cB_PD": self.cB_PD,
                "alpha": self.alpha,
            },
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _measure_sharpness(
        self,
        gamma_c: float,
        delta: float = 0.02,
        iterations: int = 20,
    ) -> float:
        """
        Measure how sharp the transition is at γ_c.

        Sharpness = |coop(γ_c + δ) - coop(γ_c - δ)|

        Args:
            gamma_c: Critical gamma estimate.
            delta: Offset for testing.
            iterations: Simulation iterations.

        Returns:
            Sharpness score in [0, 1].
        """
        coop_above = self._test_gamma(gamma_c + delta, iterations)
        coop_below = self._test_gamma(gamma_c - delta, iterations)
        return abs(coop_above - coop_below)

    def sweep(
        self,
        gamma_values: Optional[List[float]] = None,
        iterations: int = 20,
    ) -> Dict[float, float]:
        """
        Generate full gamma sweep data.

        Args:
            gamma_values: List of gamma values (default: 1.0 to 2.0).
            iterations: Iterations per test.

        Returns:
            Dictionary mapping gamma -> cooperation_rate.

        Example:
            >>> data = finder.sweep()
            >>> for gamma, coop in data.items():
            ...     print(f"γ={gamma:.2f}: {coop:.1%}")
        """
        if gamma_values is None:
            gamma_values = np.linspace(1.0, 2.0, 21).tolist()

        results = {}
        for gamma in gamma_values:
            results[gamma] = self._test_gamma(gamma, iterations)

        return results

    def validate(
        self,
        gamma_c: float,
        n_trials: int = 10,
        iterations: int = 20,
    ) -> Dict[str, Any]:
        """
        Validate critical point with multiple trials.

        Args:
            gamma_c: Critical gamma to validate.
            n_trials: Number of trials at each point.
            iterations: Iterations per trial.

        Returns:
            Validation statistics.
        """
        results_below = []
        results_above = []
        results_at = []

        delta = 0.02

        for trial in range(n_trials):
            # Vary seed for each trial
            self.seed = trial if self.seed is None else self.seed + trial
            self._network = NetworkGenerator.barabasi_albert(
                n_agents=self.n_agents,
                m=self.m,
                seed=self.seed,
            )

            results_below.append(self._test_gamma(gamma_c - delta, iterations))
            results_at.append(self._test_gamma(gamma_c, iterations))
            results_above.append(self._test_gamma(gamma_c + delta, iterations))

        return {
            "gamma_c": gamma_c,
            "below": {
                "mean": np.mean(results_below),
                "std": np.std(results_below),
            },
            "at": {
                "mean": np.mean(results_at),
                "std": np.std(results_at),
            },
            "above": {
                "mean": np.mean(results_above),
                "std": np.std(results_above),
            },
            "transition_detected": (
                np.mean(results_below) < 0.3 and np.mean(results_above) > 0.7
            ),
        }
