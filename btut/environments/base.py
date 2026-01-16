"""
Base environment interface for BTUT coordination.

Defines the abstract interface that all BTUT environments must implement,
enabling consistent integration with different simulation backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BTUTEnv(ABC):
    """
    Abstract base class for BTUT environments.

    All environments must implement:
    - reset(): Initialize/reset the environment
    - step(): Execute one timestep
    - get_btut_actions(): Get BTUT-recommended actions
    - get_state(): Get current state for BTUT computation

    Environments optionally implement:
    - render(): Visualize current state
    - close(): Clean up resources
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation array.
        """
        pass

    @abstractmethod
    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            actions: Array of agent actions (0=cooperate, 1=defect).

        Returns:
            Tuple of (observations, rewards, done, info).
        """
        pass

    @abstractmethod
    def get_btut_actions(self) -> np.ndarray:
        """
        Get BTUT-recommended actions for all agents.

        Uses the internal BTUT coordinator to compute optimal
        strategies based on current state.

        Returns:
            Boolean array where True=cooperate.
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current environment state.

        Returns:
            Dictionary with state information including:
            - cooperation_rate: Current cooperation fraction
            - n_agents: Number of active agents
            - timestep: Current timestep
        """
        pass

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """Number of agents in the environment."""
        pass

    @property
    @abstractmethod
    def cooperation_rate(self) -> float:
        """Current cooperation rate."""
        pass

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.

        Args:
            mode: Rendering mode ('human', 'rgb_array').

        Returns:
            RGB array if mode='rgb_array', else None.
        """
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        pass
