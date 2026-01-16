"""
Pre-configured parameter presets for common use cases.

These presets provide starting points for different scenarios,
from quick testing to production-scale simulations.

Example:
    >>> from btut import BTUTCoordinator, PRESETS
    >>> coordinator = BTUTCoordinator.from_preset('standard')
    >>> results = coordinator.run()
"""

from typing import Dict, Any

PRESETS: Dict[str, Dict[str, Any]] = {
    # Quick testing preset - small scale, fast iterations
    "quick": {
        "n_agents": 1_000,
        "gamma": 1.45,
        "tau": 0.30,
        "cA_SH": 0.40,
        "cB_SH": 0.10,
        "cA_PD": 0.20,
        "cB_PD": 0.08,
        "alpha": 0.60,
        "iterations": 20,
        "m": 3,
        "seed": 42,
        "description": "Quick test with 1K agents",
    },

    # Standard preset - balanced for most use cases
    "standard": {
        "n_agents": 10_000,
        "gamma": 1.45,
        "tau": 0.30,
        "cA_SH": 0.40,
        "cB_SH": 0.10,
        "cA_PD": 0.20,
        "cB_PD": 0.08,
        "alpha": 0.60,
        "iterations": 25,
        "m": 3,
        "seed": 42,
        "description": "Standard simulation with 10K agents",
    },

    # Large scale - production workloads
    "large": {
        "n_agents": 100_000,
        "gamma": 1.45,
        "tau": 0.30,
        "cA_SH": 0.40,
        "cB_SH": 0.10,
        "cA_PD": 0.20,
        "cB_PD": 0.08,
        "alpha": 0.60,
        "iterations": 30,
        "m": 3,
        "seed": 42,
        "description": "Large-scale simulation with 100K agents",
    },

    # Massive scale - stress testing
    "massive": {
        "n_agents": 1_000_000,
        "gamma": 1.45,
        "tau": 0.30,
        "cA_SH": 0.40,
        "cB_SH": 0.10,
        "cA_PD": 0.20,
        "cB_PD": 0.08,
        "alpha": 0.60,
        "iterations": 30,
        "m": 3,
        "seed": 42,
        "description": "Massive-scale stress test with 1M agents",
    },

    # High cooperation - parameters tuned for >95% cooperation
    "high_cooperation": {
        "n_agents": 10_000,
        "gamma": 2.0,
        "tau": 0.50,
        "cA_SH": 0.20,
        "cB_SH": 0.05,
        "cA_PD": 0.15,
        "cB_PD": 0.05,
        "alpha": 0.60,
        "iterations": 25,
        "m": 3,
        "seed": 42,
        "description": "High gamma for guaranteed cooperation",
    },

    # Low cooperation - below critical point
    "low_cooperation": {
        "n_agents": 10_000,
        "gamma": 1.1,
        "tau": 0.30,
        "cA_SH": 0.60,
        "cB_SH": 0.30,
        "cA_PD": 0.40,
        "cB_PD": 0.20,
        "alpha": 0.60,
        "iterations": 25,
        "m": 3,
        "seed": 42,
        "description": "Below critical gamma - expect defection",
    },

    # Hub dominated - high tau for hub influence
    "hub_dominated": {
        "n_agents": 10_000,
        "gamma": 1.45,
        "tau": 0.80,
        "cA_SH": 0.40,
        "cB_SH": 0.10,
        "cA_PD": 0.20,
        "cB_PD": 0.08,
        "alpha": 0.60,
        "iterations": 25,
        "m": 5,
        "seed": 42,
        "description": "High hub influence - hubs drive cooperation",
    },

    # Critical point - at phase transition
    "critical": {
        "n_agents": 10_000,
        "gamma": 1.33,
        "tau": 0.30,
        "cA_SH": 0.40,
        "cB_SH": 0.10,
        "cA_PD": 0.20,
        "cB_PD": 0.08,
        "alpha": 0.60,
        "iterations": 50,
        "m": 3,
        "seed": 42,
        "description": "At critical gamma - bistable behavior",
    },

    # Traffic domain - calibrated for SUMO integration
    "traffic": {
        "n_agents": 800,
        "gamma": 1.50,
        "tau": 0.30,
        "cA_SH": 0.40,
        "cB_SH": 0.10,
        "cA_PD": 0.20,
        "cB_PD": 0.08,
        "alpha": 0.60,
        "iterations": 25,
        "m": 3,
        "seed": 42,
        "description": "Traffic coordination - 800 vehicles",
    },

    # Drone swarm - calibrated for ROS integration
    "drone_swarm": {
        "n_agents": 100,
        "gamma": 1.50,
        "tau": 0.50,
        "cA_SH": 0.35,
        "cB_SH": 0.12,
        "cA_PD": 0.18,
        "cB_PD": 0.10,
        "alpha": 0.50,
        "iterations": 20,
        "m": 4,
        "seed": 42,
        "description": "Drone swarm coordination - 100 drones",
    },

    # Benchmark - for performance testing
    "benchmark": {
        "n_agents": 50_000,
        "gamma": 1.45,
        "tau": 0.30,
        "cA_SH": 0.40,
        "cB_SH": 0.10,
        "cA_PD": 0.20,
        "cB_PD": 0.08,
        "alpha": 0.60,
        "iterations": 20,
        "m": 3,
        "seed": 42,
        "description": "Performance benchmark configuration",
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """
    Get preset configuration by name.

    Args:
        name: Preset name (e.g., 'standard', 'high_cooperation').

    Returns:
        Dictionary of configuration parameters.

    Raises:
        KeyError: If preset name not found.
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name].copy()


def list_presets() -> Dict[str, str]:
    """
    List all available presets with descriptions.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return {name: cfg["description"] for name, cfg in PRESETS.items()}
