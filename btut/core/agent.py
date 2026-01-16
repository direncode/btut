"""
Agent primitives for BTUT coordination.

This module defines the core Agent class and Strategy enum used throughout
the BTUT framework. Agents maintain their strategy (cooperate/defect),
network degree, and computed utilities.

Theory:
    Each agent i has:
    - Strategy s_i ∈ {A (cooperate), B (defect)}
    - Degree k_i from the scale-free network
    - Weight w_i = k_i^τ (hub influence)
    - Utility U_i computed from mean-field payoffs

Example:
    >>> agent = Agent(id=0, degree=5, tau=0.3)
    >>> agent.strategy = Strategy.COOPERATE
    >>> print(agent.weight)  # 5^0.3 ≈ 1.62
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import numpy as np


class Strategy(Enum):
    """
    Agent strategy in the coordination game.

    Attributes:
        COOPERATE: Strategy A - cooperative behavior (yielding, sharing)
        DEFECT: Strategy B - selfish behavior (prioritize own gain)
    """
    COOPERATE = "A"
    DEFECT = "B"

    def __str__(self) -> str:
        return self.value

    @property
    def is_cooperative(self) -> bool:
        """Return True if this is the cooperative strategy."""
        return self == Strategy.COOPERATE


@dataclass
class Agent:
    """
    An agent in the BTUT coordination system.

    Agents exist on a scale-free network and update their strategies
    based on expected utilities computed via mean-field dynamics.

    Attributes:
        id: Unique identifier for this agent.
        degree: Number of connections in the network (k_i).
        tau: Hub influence parameter (0 = uniform, 1 = degree-proportional).
        strategy: Current strategy (cooperate or defect).
        utility: Last computed expected utility.
        neighbors: Optional list of neighbor agent IDs (not used in O(N) mode).

    Properties:
        weight: Hub influence weight w_i = k_i^τ.
        is_hub: True if degree is in top 10% of network.

    Example:
        >>> agent = Agent(id=0, degree=10, tau=0.5)
        >>> agent.strategy = Strategy.COOPERATE
        >>> print(f"Weight: {agent.weight:.2f}")  # 10^0.5 ≈ 3.16
    """

    id: int
    degree: int
    tau: float = 0.3
    strategy: Strategy = field(default=Strategy.DEFECT)
    utility: float = 0.0
    neighbors: Optional[List[int]] = None

    # Internal state
    _max_degree: int = field(default=1, repr=False)

    def __post_init__(self) -> None:
        """Validate agent parameters after initialization."""
        if self.degree < 1:
            raise ValueError(f"Agent degree must be >= 1, got {self.degree}")
        if not 0 <= self.tau <= 1:
            raise ValueError(f"Tau must be in [0, 1], got {self.tau}")

    @property
    def weight(self) -> float:
        """
        Compute hub influence weight.

        The weight determines how much this agent's strategy influences
        the mean-field computation. Higher-degree nodes (hubs) have
        more influence when τ > 0.

        Formula: w_i = k_i^τ

        Returns:
            Hub influence weight (float >= 1).
        """
        return float(self.degree ** self.tau)

    @property
    def normalized_weight(self) -> float:
        """
        Compute normalized weight relative to max degree.

        Formula: w̃_i = (k_i / k_max)^τ

        Returns:
            Normalized weight in [0, 1].
        """
        if self._max_degree <= 0:
            return 1.0
        return (self.degree / self._max_degree) ** self.tau

    @property
    def is_hub(self) -> bool:
        """
        Check if this agent is a hub (high-degree node).

        An agent is considered a hub if its degree is >= 50% of max degree.
        This is a heuristic threshold; actual hub classification may vary.

        Returns:
            True if this agent is a hub node.
        """
        return self.degree >= 0.5 * self._max_degree

    def set_max_degree(self, max_degree: int) -> None:
        """
        Set the maximum degree in the network for normalization.

        Args:
            max_degree: Maximum degree across all agents.
        """
        self._max_degree = max(1, max_degree)

    def switch_strategy(self) -> None:
        """Toggle between cooperate and defect strategies."""
        if self.strategy == Strategy.COOPERATE:
            self.strategy = Strategy.DEFECT
        else:
            self.strategy = Strategy.COOPERATE

    def copy(self) -> Agent:
        """Create a deep copy of this agent."""
        return Agent(
            id=self.id,
            degree=self.degree,
            tau=self.tau,
            strategy=self.strategy,
            utility=self.utility,
            neighbors=self.neighbors.copy() if self.neighbors else None,
            _max_degree=self._max_degree,
        )

    def to_dict(self) -> dict:
        """
        Convert agent state to dictionary.

        Returns:
            Dictionary with agent properties.
        """
        return {
            "id": self.id,
            "degree": self.degree,
            "tau": self.tau,
            "strategy": self.strategy.value,
            "utility": self.utility,
            "weight": self.weight,
            "is_hub": self.is_hub,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Agent:
        """
        Create agent from dictionary.

        Args:
            data: Dictionary with agent properties.

        Returns:
            New Agent instance.
        """
        agent = cls(
            id=data["id"],
            degree=data["degree"],
            tau=data.get("tau", 0.3),
        )
        agent.strategy = Strategy(data.get("strategy", "B"))
        agent.utility = data.get("utility", 0.0)
        return agent


@dataclass
class AgentPopulation:
    """
    Collection of agents with vectorized operations.

    This class provides efficient NumPy-based operations over
    a population of agents, enabling O(N) mean-field updates.

    Attributes:
        n_agents: Number of agents in population.
        degrees: Array of agent degrees.
        tau: Hub influence parameter.
        strategies: Boolean array (True = cooperate).
        utilities: Array of agent utilities.

    Example:
        >>> pop = AgentPopulation.from_degrees([3, 5, 10, 2], tau=0.5)
        >>> print(f"Cooperation rate: {pop.cooperation_rate:.1%}")
    """

    n_agents: int
    degrees: np.ndarray
    tau: float
    strategies: np.ndarray  # bool array: True = cooperate
    utilities: np.ndarray

    def __post_init__(self) -> None:
        """Validate population arrays."""
        assert len(self.degrees) == self.n_agents
        assert len(self.strategies) == self.n_agents
        assert len(self.utilities) == self.n_agents

    @classmethod
    def from_degrees(
        cls,
        degrees: np.ndarray,
        tau: float = 0.3,
        initial_cooperation: float = 0.5,
        seed: Optional[int] = None,
    ) -> AgentPopulation:
        """
        Create population from degree sequence.

        Args:
            degrees: Array of agent degrees.
            tau: Hub influence parameter.
            initial_cooperation: Fraction starting as cooperators.
            seed: Random seed for reproducibility.

        Returns:
            New AgentPopulation instance.
        """
        rng = np.random.default_rng(seed)
        n = len(degrees)

        return cls(
            n_agents=n,
            degrees=np.asarray(degrees, dtype=np.int32),
            tau=tau,
            strategies=rng.random(n) < initial_cooperation,
            utilities=np.zeros(n, dtype=np.float64),
        )

    @property
    def weights(self) -> np.ndarray:
        """Compute hub influence weights: w_i = k_i^τ."""
        return np.power(self.degrees.astype(np.float64), self.tau)

    @property
    def cooperation_rate(self) -> float:
        """Fraction of agents cooperating."""
        return float(np.mean(self.strategies))

    @property
    def weighted_cooperation(self) -> float:
        """
        Weight-averaged cooperation rate.

        Hub nodes contribute more to this metric.
        Formula: Σ(w_i · s_i) / Σ(w_i)
        """
        w = self.weights
        return float(np.sum(w * self.strategies) / np.sum(w))

    @property
    def hub_cooperation(self) -> float:
        """Cooperation rate among hub nodes (top 10% by degree)."""
        threshold = np.percentile(self.degrees, 90)
        hub_mask = self.degrees >= threshold
        if not np.any(hub_mask):
            return 0.0
        return float(np.mean(self.strategies[hub_mask]))

    @property
    def periphery_cooperation(self) -> float:
        """Cooperation rate among non-hub nodes."""
        threshold = np.percentile(self.degrees, 90)
        periphery_mask = self.degrees < threshold
        if not np.any(periphery_mask):
            return 0.0
        return float(np.mean(self.strategies[periphery_mask]))

    def to_agents(self) -> List[Agent]:
        """Convert to list of Agent objects."""
        max_degree = int(np.max(self.degrees))
        agents = []
        for i in range(self.n_agents):
            agent = Agent(
                id=i,
                degree=int(self.degrees[i]),
                tau=self.tau,
                strategy=Strategy.COOPERATE if self.strategies[i] else Strategy.DEFECT,
                utility=float(self.utilities[i]),
            )
            agent.set_max_degree(max_degree)
            agents.append(agent)
        return agents
