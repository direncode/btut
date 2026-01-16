"""
Mean-field dynamics for O(N) strategy updates.

This module implements the core BTUT update rule using kernel-weighted
mean-field approximation. This is the key innovation that enables O(N)
complexity instead of O(N²) pairwise interactions.

Theory:
    Instead of computing pairwise interactions between all agents,
    we approximate the local environment using mean-field statistics:

    1. Compute population cooperation rate: p = Σs_i / N
    2. Compute hub-weighted cooperation: p_hub = Σ(w_i · s_i) / Σw_i
    3. For each agent, compute expected utility using p (and p_hub for τ > 0)
    4. Update strategy based on utility comparison

    The τ parameter controls hub influence:
    - τ = 0: All agents have equal influence (pure mean-field)
    - τ > 0: High-degree hubs have more influence on perceived cooperation

    This gives O(N) per iteration with constant number of iterations
    (typically ~12) regardless of population size.

Example:
    >>> from btut.core import MeanFieldDynamics, AgentPopulation, StagHuntPD
    >>> pop = AgentPopulation.from_degrees(degrees, tau=0.3)
    >>> dynamics = MeanFieldDynamics(StagHuntPD(gamma=1.5))
    >>> new_pop = dynamics.update(pop)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np

from btut.core.agent import AgentPopulation
from btut.core.payoffs import PayoffMatrix, StagHuntPD


@dataclass
class MeanFieldDynamics:
    """
    O(N) mean-field dynamics for BTUT coordination.

    This class implements the kernel-weighted mean-field update rule
    that enables linear-time strategy updates.

    Attributes:
        payoff: Payoff matrix for utility computation.
        noise: Optional noise level for stochastic updates (default: 0).
        deterministic: If True, use deterministic best-response (default: True).

    Key Methods:
        update: Perform one iteration of strategy updates.
        compute_field: Compute mean-field statistics.
        compute_utilities: Compute per-agent utilities.

    Example:
        >>> dynamics = MeanFieldDynamics(StagHuntPD(gamma=1.5))
        >>> new_strategies = dynamics.update_strategies(pop)
    """

    payoff: PayoffMatrix
    noise: float = 0.0
    deterministic: bool = True
    _rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        """Initialize random generator if needed."""
        if self._rng is None:
            self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def compute_field(
        self,
        population: AgentPopulation,
    ) -> Tuple[float, float]:
        """
        Compute mean-field statistics for the population.

        This is the core O(N) operation that summarizes the population
        state into two statistics:
        1. Overall cooperation rate p
        2. Hub-weighted cooperation rate p_hub

        Args:
            population: Current population state.

        Returns:
            Tuple of (p, p_hub) cooperation rates.

        Complexity: O(N) - single pass over agents.
        """
        strategies = population.strategies.astype(np.float64)
        weights = population.weights

        # Overall cooperation rate
        p = np.mean(strategies)

        # Hub-weighted cooperation rate
        total_weight = np.sum(weights)
        p_hub = np.sum(weights * strategies) / total_weight if total_weight > 0 else p

        return float(p), float(p_hub)

    def compute_utilities(
        self,
        population: AgentPopulation,
        p: float,
        p_hub: float,
    ) -> np.ndarray:
        """
        Compute expected utilities for all agents.

        Each agent's utility depends on:
        - The mean-field cooperation rate (p)
        - Their degree/weight (for τ-modulated perception)
        - The payoff matrix

        Args:
            population: Current population state.
            p: Overall cooperation rate.
            p_hub: Hub-weighted cooperation rate.

        Returns:
            Array of shape (N, 2) with [U_cooperate, U_defect] per agent.

        Complexity: O(N) - vectorized over all agents.
        """
        n = population.n_agents
        utilities = np.zeros((n, 2), dtype=np.float64)

        # Get base utilities from payoff matrix
        u_coop, u_defect = self.payoff.expected_utilities(p, p_hub)

        # For τ > 0, agents perceive cooperation rate based on their position
        # Hub nodes "see" more cooperation due to their connections
        if population.tau > 0:
            weights = population.weights
            max_weight = np.max(weights)

            # Effective cooperation rate perceived by each agent
            # Hubs perceive rate closer to p_hub, periphery perceives p
            weight_factor = weights / max_weight  # in [0, 1]
            p_effective = (1 - weight_factor) * p + weight_factor * p_hub

            # Compute utilities for each effective rate
            for i in range(n):
                u_c, u_d = self.payoff.expected_utilities(p_effective[i], p_hub)
                utilities[i, 0] = u_c
                utilities[i, 1] = u_d
        else:
            # All agents perceive same mean field
            utilities[:, 0] = u_coop
            utilities[:, 1] = u_defect

        # Apply degree-based scaling
        # Higher-degree agents get slightly more reliable utility estimates
        degrees = population.degrees.astype(np.float64)
        reliability = np.sqrt(degrees / np.max(degrees))
        utilities *= reliability[:, np.newaxis]

        return utilities

    def update_strategies(
        self,
        population: AgentPopulation,
        p: float,
        p_hub: float,
    ) -> np.ndarray:
        """
        Compute new strategies based on utilities.

        Args:
            population: Current population state.
            p: Overall cooperation rate.
            p_hub: Hub-weighted cooperation rate.

        Returns:
            Boolean array of new strategies (True = cooperate).

        Complexity: O(N).
        """
        utilities = self.compute_utilities(population, p, p_hub)

        if self.deterministic:
            # Best response: choose strategy with higher utility
            new_strategies = utilities[:, 0] > utilities[:, 1]
        else:
            # Logit response with noise
            diff = utilities[:, 0] - utilities[:, 1]
            if self.noise > 0:
                diff += self._rng.normal(0, self.noise, size=len(diff))
            prob_coop = 1 / (1 + np.exp(-diff))
            new_strategies = self._rng.random(len(diff)) < prob_coop

        return new_strategies

    def update(
        self,
        population: AgentPopulation,
    ) -> AgentPopulation:
        """
        Perform one iteration of mean-field dynamics.

        This is the main update method that:
        1. Computes mean-field statistics
        2. Calculates expected utilities
        3. Updates all strategies

        Args:
            population: Current population state.

        Returns:
            New AgentPopulation with updated strategies.

        Complexity: O(N) total.
        """
        # Step 1: Compute mean-field statistics (O(N))
        p, p_hub = self.compute_field(population)

        # Step 2: Update strategies (O(N))
        new_strategies = self.update_strategies(population, p, p_hub)

        # Step 3: Compute new utilities for tracking
        utilities = self.compute_utilities(population, p, p_hub)
        new_utilities = np.where(
            new_strategies,
            utilities[:, 0],
            utilities[:, 1]
        )

        # Return new population state
        return AgentPopulation(
            n_agents=population.n_agents,
            degrees=population.degrees.copy(),
            tau=population.tau,
            strategies=new_strategies,
            utilities=new_utilities,
        )


@dataclass
class ConvergenceChecker:
    """
    Check for convergence of mean-field dynamics.

    Convergence is detected when cooperation rate stabilizes
    within a tolerance over a window of iterations.

    Attributes:
        window: Number of iterations to check.
        tolerance: Maximum change in cooperation rate.
        history: List of recent cooperation rates.
    """

    window: int = 5
    tolerance: float = 1e-6
    history: list = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []

    def update(self, cooperation_rate: float) -> bool:
        """
        Record cooperation rate and check convergence.

        Args:
            cooperation_rate: Current cooperation rate.

        Returns:
            True if converged, False otherwise.
        """
        self.history.append(cooperation_rate)

        if len(self.history) < self.window:
            return False

        # Check if all recent values are within tolerance
        recent = self.history[-self.window:]
        return max(recent) - min(recent) < self.tolerance

    def reset(self) -> None:
        """Reset convergence history."""
        self.history = []
