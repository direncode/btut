"""
btut.environments - External simulation adapters
================================================

Provides adapters for integrating BTUT coordination with
external simulation environments.

Supported Environments:
    - BTUTEnv: Abstract base environment
    - SUMOEnv: Eclipse SUMO traffic simulator (optional)
    - GraphEnv: Simple graph-based toy environment
    - PettingZooWrapper: Multi-agent RL compatibility

Installation:
    SUMO support requires additional dependencies:
        pip install btut[sumo]

    PettingZoo support:
        pip install btut[pettingzoo]

Example (GraphEnv):
    >>> from btut.environments import GraphEnv
    >>> env = GraphEnv(n_agents=100, gamma=1.5)
    >>> obs = env.reset()
    >>> for _ in range(25):
    ...     actions = env.get_btut_actions()
    ...     obs, rewards, done, info = env.step(actions)
    >>> print(f"Cooperation: {info['cooperation_rate']:.1%}")
"""

from btut.environments.base import BTUTEnv
from btut.environments.graph_env import GraphEnv

__all__ = ["BTUTEnv", "GraphEnv"]

# Optional imports
try:
    from btut.environments.sumo_env import SUMOEnv
    __all__.append("SUMOEnv")
except ImportError:
    pass

try:
    from btut.environments.pettingzoo_wrapper import PettingZooWrapper
    __all__.append("PettingZooWrapper")
except ImportError:
    pass
