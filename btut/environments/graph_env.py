"""
Simple graph-based environment for BTUT testing.

This environment provides a lightweight testbed for BTUT coordination
without requiring external simulators like SUMO.

Agents exist on a graph and interact via coordination games.
Rewards are based on payoffs from the StagHunt+PD game.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from btut.environments.base import BTUTEnv
from btut.networks.generator import NetworkGenerator, Network
from btut.coordination.coordinator import BTUTCoordinator


class GraphEnv(BTUTEnv):
    """
    Graph-based coordination environment.

    Agents exist on a scale-free network and play coordination games.
    This environment is useful for:
    - Testing BTUT algorithms
    - Benchmarking performance
    - Educational demonstrations

    Attributes:
        network: Underlying network topology.
        coordinator: BTUT coordinator instance.
        gamma: Cooperation bonus parameter.
        tau: Hub influence parameter.
        max_steps: Maximum timesteps per episode.

    Example:
        >>> env = GraphEnv(n_agents=100, gamma=1.5)
        >>> obs = env.reset()
        >>> for _ in range(25):
        ...     actions = env.get_btut_actions()
        ...     obs, rewards, done, info = env.step(actions)
        >>> print(f"Final cooperation: {info['cooperation_rate']:.1%}")
    """

    def __init__(
        self,
        n_agents: int = 100,
        gamma: float = 1.5,
        tau: float = 0.3,
        m: int = 3,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize graph environment.

        Args:
            n_agents: Number of agents.
            gamma: Cooperation bonus.
            tau: Hub influence.
            m: Network connectivity.
            max_steps: Episode length.
            seed: Random seed.
        """
        self._n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.m = m
        self.max_steps = max_steps
        self.seed = seed

        # Create network
        self.network = NetworkGenerator.barabasi_albert(
            n_agents=n_agents,
            m=m,
            seed=seed,
        )

        # Create coordinator
        self.coordinator = BTUTCoordinator(
            network=self.network,
            gamma=gamma,
            tau=tau,
            seed=seed,
        )

        # State tracking
        self._timestep = 0
        self._strategies = np.random.random(n_agents) < 0.5
        self._rewards = np.zeros(n_agents)

    @property
    def n_agents(self) -> int:
        """Number of agents."""
        return self._n_agents

    @property
    def cooperation_rate(self) -> float:
        """Current cooperation rate."""
        return float(np.mean(self._strategies))

    def reset(self) -> np.ndarray:
        """
        Reset environment.

        Returns:
            Initial observation (strategies and degrees).
        """
        self._timestep = 0
        self._strategies = np.random.random(self._n_agents) < 0.5
        self._rewards = np.zeros(self._n_agents)
        self.coordinator.reset()

        return self._get_observation()

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            actions: Boolean array of strategies (True=cooperate).

        Returns:
            Tuple of (observation, rewards, done, info).
        """
        self._timestep += 1
        self._strategies = actions.astype(bool)

        # Compute rewards based on coordination outcomes
        self._rewards = self._compute_rewards()

        # Check termination
        done = self._timestep >= self.max_steps

        obs = self._get_observation()
        info = self.get_state()

        return obs, self._rewards, done, info

    def get_btut_actions(self) -> np.ndarray:
        """
        Get BTUT-recommended actions.

        Returns:
            Boolean array of recommended strategies.
        """
        # Run one BTUT iteration
        self.coordinator.step()

        # Get strategies from coordinator's population
        return self.coordinator._population.strategies

    def get_state(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            "cooperation_rate": self.cooperation_rate,
            "n_agents": self._n_agents,
            "timestep": self._timestep,
            "mean_reward": float(np.mean(self._rewards)),
            "hub_cooperation": self._compute_hub_cooperation(),
        }

    def _get_observation(self) -> np.ndarray:
        """
        Create observation array.

        Returns:
            Array of shape (n_agents, 2) with [strategy, normalized_degree].
        """
        obs = np.zeros((self._n_agents, 2), dtype=np.float32)
        obs[:, 0] = self._strategies.astype(np.float32)
        obs[:, 1] = self.network.degrees / self.network.max_degree
        return obs

    def _compute_rewards(self) -> np.ndarray:
        """
        Compute rewards based on coordination game.

        Cooperators get reward if enough others cooperate.
        Defectors get temptation payoff against cooperators.
        """
        coop_rate = self.cooperation_rate
        rewards = np.zeros(self._n_agents)

        # Cooperators: reward if coop rate above threshold
        coop_mask = self._strategies
        rewards[coop_mask] = coop_rate * self.gamma - 0.2  # Cost of cooperation

        # Defectors: temptation against cooperators
        defect_mask = ~self._strategies
        rewards[defect_mask] = coop_rate * 1.5  # Exploit cooperators

        # Scale by degree (hubs have more interactions)
        degree_factor = np.sqrt(self.network.degrees / self.network.mean_degree)
        rewards *= degree_factor

        return rewards

    def _compute_hub_cooperation(self) -> float:
        """Cooperation rate among hubs."""
        hub_indices = self.network.hub_indices(percentile=90)
        if len(hub_indices) == 0:
            return 0.0
        return float(np.mean(self._strategies[hub_indices]))
