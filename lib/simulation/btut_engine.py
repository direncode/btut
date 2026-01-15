"""
BTUT Python Simulation Engine
==============================
Pure Python implementation for API backend (native Rust version also available)
O(N) kernel-weighted mean-field dynamics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


# Preset configurations
PRESETS = {
    'quick': {
        'N': 10_000,
        'gamma': 1.45,
        'tau': 0.30,
        'cA_SH': 0.40,
        'cB_SH': 0.10,
        'cA_PD': 0.20,
        'cB_PD': 0.08,
        'alpha': 0.60,
        'iterations': 20,
        'm': 3,
        'seed': 42
    },
    'standard': {
        'N': 100_000,
        'gamma': 1.45,
        'tau': 0.30,
        'cA_SH': 0.40,
        'cB_SH': 0.10,
        'cA_PD': 0.20,
        'cB_PD': 0.08,
        'alpha': 0.60,
        'iterations': 25,
        'm': 3,
        'seed': 42
    },
    'massive': {
        'N': 1_000_000,
        'gamma': 1.45,
        'tau': 0.30,
        'cA_SH': 0.40,
        'cB_SH': 0.10,
        'cA_PD': 0.20,
        'cB_PD': 0.08,
        'alpha': 0.60,
        'iterations': 30,
        'm': 3,
        'seed': 42
    },
    'high_cooperation': {
        'N': 50_000,
        'gamma': 2.0,
        'tau': 0.30,
        'cA_SH': 0.20,
        'cB_SH': 0.05,
        'cA_PD': 0.15,
        'cB_PD': 0.05,
        'alpha': 0.60,
        'iterations': 25,
        'm': 3,
        'seed': 42
    },
    'low_cooperation': {
        'N': 50_000,
        'gamma': 1.1,
        'tau': 0.30,
        'cA_SH': 0.60,
        'cB_SH': 0.30,
        'cA_PD': 0.40,
        'cB_PD': 0.20,
        'alpha': 0.60,
        'iterations': 25,
        'm': 3,
        'seed': 42
    },
    'hub_dominated': {
        'N': 50_000,
        'gamma': 1.45,
        'tau': 0.80,
        'cA_SH': 0.40,
        'cB_SH': 0.10,
        'cA_PD': 0.20,
        'cB_PD': 0.08,
        'alpha': 0.60,
        'iterations': 25,
        'm': 5,
        'seed': 42
    }
}


class BTUTSimulator:
    """
    BTUT Simulation Engine - O(N) Scalable Multi-Agent Coordination

    Uses kernel-weighted mean-field dynamics to avoid explicit graph storage
    and achieve linear complexity in agent count.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.N = config['N']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.cA_SH = config['cA_SH']
        self.cB_SH = config['cB_SH']
        self.cA_PD = config['cA_PD']
        self.cB_PD = config['cB_PD']
        self.alpha = config['alpha']
        self.iterations = config['iterations']
        self.m = config['m']
        self.seed = config.get('seed', 42)

        np.random.seed(self.seed)

        # Initialize agents
        self.degrees = self._sample_degrees()
        self.max_degree = np.max(self.degrees)
        self.weights = self._compute_weights()
        self.strategies = np.random.randint(0, 2, size=self.N)  # 0=B, 1=A

        # Simulation state
        self.p = 0.5  # Fraction of cooperators
        self.history = []

    def _sample_degrees(self) -> np.ndarray:
        """Sample degrees from Barabási-Albert power-law distribution"""
        u = np.random.uniform(0, 1, size=self.N)
        degrees = self.m / np.sqrt(1 - u)
        return np.maximum(degrees, self.m)

    def _compute_weights(self) -> np.ndarray:
        """Compute kernel weights: w_i = (k_i / k_max)^tau"""
        if self.tau == 0 or self.max_degree == 0:
            return np.ones(self.N)
        return (self.degrees / self.max_degree) ** self.tau

    def _compute_expected_utilities(self, p: float) -> Tuple[float, float]:
        """Compute expected utilities for both strategies"""

        # Base payoffs (log-scaled)
        u_A_PD = np.log(2.0)
        u_B_PD = np.log(1.2)
        u_A_HD = np.log(2.2)
        u_B_HD = np.log(1.5)
        u_A_SH = np.log(2.5)
        u_B_SH = np.log(1.2)

        # Prisoner's Dilemma
        EU_PD_A = 0.5 * (u_A_PD + (p * u_A_PD + (1 - p) * u_B_PD)) - self.cA_PD * u_A_PD
        EU_PD_B = 0.5 * (u_B_PD + (p * u_A_PD + (1 - p) * u_B_PD)) - self.cB_PD * u_B_PD

        # Hawk-Dove
        EU_HD_A = 0.5 * (u_A_HD + p * u_A_HD + (1 - p) * u_B_HD) * (p * self.alpha + (1 - p)) - 0.22 * u_A_HD
        EU_HD_B = 0.5 * (u_B_HD + p * u_A_HD + (1 - p) * u_B_HD) * (p + (1 - p) * 1.08) - 0.12 * u_B_HD

        # Stag Hunt
        EU_SH_A = 0.5 * (u_A_SH + p * u_A_SH + (1 - p) * u_B_SH) * (p * self.gamma + (1 - p) * 0.03) - self.cA_SH * u_A_SH
        EU_SH_B = 0.5 * (u_B_SH + p * u_A_SH + (1 - p) * u_B_SH) * (p * 0.03 + (1 - p)) - self.cB_SH * u_B_SH

        # Mixed game (equal weighting)
        mix = 1.0 / 3.0
        U_A = mix * (EU_PD_A + EU_HD_A + EU_SH_A)
        U_B = mix * (EU_PD_B + EU_HD_B + EU_SH_B)

        return U_A, U_B

    def _update_strategies(self):
        """O(N) strategy update using mean-field approximation"""
        U_A, U_B = self._compute_expected_utilities(self.p)

        # Weighted voting for each agent
        weighted_A = 0.0
        total_weight = 0.0

        for i in range(self.N):
            agent_U_A = U_A * self.weights[i] * self.degrees[i]
            agent_U_B = U_B * self.weights[i] * self.degrees[i]

            if agent_U_A > agent_U_B:
                self.strategies[i] = 1
                weighted_A += self.weights[i]
            else:
                self.strategies[i] = 0

            total_weight += self.weights[i]

        p_new = weighted_A / total_weight if total_weight > 0 else self.p

        # Momentum update
        self.p = 0.5 * self.p + 0.5 * p_new
        self.history.append(self.p)

    def run(self) -> List[Dict]:
        """Run complete simulation and return state history"""
        states = []

        for iteration in range(self.iterations):
            self._update_strategies()

            state = {
                'iteration': iteration + 1,
                'fractionA': float(self.p),
                'strategyA_count': int(np.sum(self.strategies)),
                'strategyB_count': int(self.N - np.sum(self.strategies)),
                'converged': self._check_convergence()
            }
            states.append(state)

            if state['converged'] and iteration >= 10:
                break

        return states

    def _check_convergence(self, window: int = 5, threshold: float = 1e-6) -> bool:
        """Check if simulation has converged"""
        if len(self.history) < window:
            return False

        recent = self.history[-window:]
        variance = np.var(recent)
        return bool(variance < threshold)

    def get_final_state(self) -> Dict:
        """Get final simulation state"""
        return {
            'fractionA': float(self.p),
            'strategyA_count': int(np.sum(self.strategies)),
            'strategyB_count': int(self.N - np.sum(self.strategies)),
            'convergence_history': [float(x) for x in self.history],
            'iterations_completed': len(self.history),
            'converged': self._check_convergence()
        }


def benchmark_performance(agent_counts: List[int], iterations: int = 20) -> List[Dict]:
    """Run performance benchmarks"""
    results = []

    for N in agent_counts:
        config = {
            **PRESETS['standard'],
            'N': N,
            'iterations': iterations
        }

        start = time.time()
        sim = BTUTSimulator(config)
        sim.run()
        elapsed = time.time() - start

        results.append({
            'agents': N,
            'iterations': iterations,
            'runtime_seconds': elapsed,
            'throughput': (N * iterations) / elapsed
        })

    return results
