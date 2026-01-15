#!/usr/bin/env python3
"""
BTUT Universal Critical Point Finder

Automatically discovers the phase transition critical point (gamma_c) for any
BTUT application domain. This is THE key systematic step that must run before
any simulation to ensure optimal parameter selection.

Theory:
- Below gamma_c: System tips to all-defect (0% cooperation)
- Above gamma_c: System tips to all-cooperate (100% cooperation)
- At gamma_c: Bistable regime, sensitive to initial conditions

Algorithm:
1. Binary search to find gamma_c within tolerance
2. Validate transition sharpness
3. Return optimal operating point (gamma_c + margin)

Usage:
    from critical_point_finder import CriticalPointFinder

    finder = CriticalPointFinder()
    result = finder.find_critical_point()
    print(f"Critical gamma: {result.gamma_critical}")
    print(f"Recommended operating point: {result.recommended_gamma}")
"""

import sys
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
import json
import time
import logging

# Add path for btut_engine
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Domain(Enum):
    """Supported application domains"""
    ABSTRACT = "abstract"      # Pure BTUT simulation
    TRAFFIC = "traffic"        # SUMO traffic
    DRONE_SWARM = "drone"      # Drone coordination
    ROBOT_FLEET = "robot"      # Multi-robot
    CUSTOM = "custom"          # User-defined


@dataclass
class CriticalPointResult:
    """Result of critical point discovery"""
    gamma_critical: float           # Critical gamma value
    gamma_lower: float              # Last gamma with 0% cooperation
    gamma_upper: float              # First gamma with 100% cooperation
    transition_sharpness: float     # How sharp is the transition (0-1)
    recommended_gamma: float        # Recommended operating point
    recommended_margin: float       # Safety margin above critical
    search_iterations: int          # Binary search iterations
    confidence: float               # Confidence in result (0-1)
    domain: str                     # Application domain
    config_used: Dict               # Configuration parameters
    timestamp: str                  # When discovered


@dataclass
class DomainConfig:
    """Domain-specific configuration"""
    name: str
    default_N: int = 1000
    default_tau: float = 0.3
    gamma_search_min: float = 1.0
    gamma_search_max: float = 2.0
    cost_params: Dict = None

    def __post_init__(self):
        if self.cost_params is None:
            self.cost_params = {
                'cA_SH': 0.40, 'cB_SH': 0.10,
                'cA_PD': 0.20, 'cB_PD': 0.08,
                'alpha': 0.6
            }


# Pre-defined domain configurations
DOMAIN_CONFIGS = {
    Domain.ABSTRACT: DomainConfig(
        name="abstract",
        default_N=1000,
        gamma_search_min=1.0,
        gamma_search_max=2.0
    ),
    Domain.TRAFFIC: DomainConfig(
        name="traffic",
        default_N=500,
        gamma_search_min=1.2,
        gamma_search_max=1.8,
        cost_params={
            'cA_SH': 0.40, 'cB_SH': 0.10,
            'cA_PD': 0.20, 'cB_PD': 0.08,
            'alpha': 0.6
        }
    ),
    Domain.DRONE_SWARM: DomainConfig(
        name="drone",
        default_N=100,
        gamma_search_min=1.2,
        gamma_search_max=2.0,
        cost_params={
            'cA_SH': 0.35, 'cB_SH': 0.12,
            'cA_PD': 0.18, 'cB_PD': 0.10,
            'alpha': 0.5
        }
    ),
    Domain.ROBOT_FLEET: DomainConfig(
        name="robot",
        default_N=50,
        gamma_search_min=1.1,
        gamma_search_max=1.8,
        cost_params={
            'cA_SH': 0.45, 'cB_SH': 0.08,
            'cA_PD': 0.22, 'cB_PD': 0.06,
            'alpha': 0.7
        }
    ),
}


class CriticalPointFinder:
    """
    Universal critical point finder for BTUT systems.

    Automatically discovers gamma_c using binary search with
    domain-specific configurations.
    """

    def __init__(
        self,
        domain: Domain = Domain.ABSTRACT,
        custom_config: DomainConfig = None
    ):
        self.domain = domain

        if custom_config:
            self.config = custom_config
        elif domain in DOMAIN_CONFIGS:
            self.config = DOMAIN_CONFIGS[domain]
        else:
            self.config = DOMAIN_CONFIGS[Domain.ABSTRACT]

        # Import BTUT engine
        try:
            from btut_engine import BTUTSimulator
            self.BTUTSimulator = BTUTSimulator
        except ImportError:
            raise ImportError("btut_engine not found. Ensure it's in the path.")

    def _run_simulation(
        self,
        gamma: float,
        tau: float = None,
        N: int = None,
        iterations: int = 25
    ) -> float:
        """Run single BTUT simulation, return cooperation fraction"""
        if tau is None:
            tau = self.config.default_tau
        if N is None:
            N = self.config.default_N

        config = {
            'N': N,
            'gamma': gamma,
            'tau': tau,
            'iterations': iterations,
            'm': 3,
            **self.config.cost_params
        }

        sim = self.BTUTSimulator(config)
        states = sim.run()
        return states[-1]['fractionA']

    def find_critical_point(
        self,
        tau: float = None,
        N: int = None,
        tolerance: float = 0.005,
        max_iterations: int = 20,
        margin: float = 0.05,
        verbose: bool = True
    ) -> CriticalPointResult:
        """
        Find the critical gamma using binary search.

        Args:
            tau: Hub influence parameter (uses domain default if None)
            N: Number of agents (uses domain default if None)
            tolerance: Search tolerance for gamma
            max_iterations: Maximum binary search iterations
            margin: Safety margin above gamma_c for recommended value
            verbose: Print progress

        Returns:
            CriticalPointResult with all findings
        """
        if tau is None:
            tau = self.config.default_tau
        if N is None:
            N = self.config.default_N

        if verbose:
            logger.info(f"Finding critical point for domain: {self.config.name}")
            logger.info(f"  N={N}, tau={tau}")
            logger.info(f"  Search range: [{self.config.gamma_search_min}, {self.config.gamma_search_max}]")

        low = self.config.gamma_search_min
        high = self.config.gamma_search_max

        # Verify bounds
        coop_low = self._run_simulation(low, tau, N)
        coop_high = self._run_simulation(high, tau, N)

        if verbose:
            logger.info(f"  Bounds: gamma={low:.2f} -> {coop_low:.0%}, gamma={high:.2f} -> {coop_high:.0%}")

        if coop_low > 0.5:
            logger.warning(f"Lower bound gamma={low} already shows cooperation. Extending search.")
            low = max(0.5, low - 0.5)
            coop_low = self._run_simulation(low, tau, N)

        if coop_high < 0.5:
            logger.warning(f"Upper bound gamma={high} still shows defection. Extending search.")
            high = high + 0.5
            coop_high = self._run_simulation(high, tau, N)

        # Binary search
        iterations = 0
        gamma_history = []

        while high - low > tolerance and iterations < max_iterations:
            mid = (low + high) / 2
            coop_mid = self._run_simulation(mid, tau, N)
            gamma_history.append((mid, coop_mid))

            if verbose:
                logger.info(f"  Iter {iterations}: gamma={mid:.4f} -> {coop_mid:.0%}")

            if coop_mid < 0.5:
                low = mid
            else:
                high = mid

            iterations += 1

        gamma_critical = (low + high) / 2

        # Validate transition sharpness
        test_gammas = np.linspace(gamma_critical - 0.05, gamma_critical + 0.05, 11)
        test_coops = [self._run_simulation(g, tau, N) for g in test_gammas]

        # Sharpness = how quickly cooperation jumps from 0 to 1
        transition_width = 0.0
        for i in range(len(test_coops) - 1):
            if test_coops[i] < 0.1 and test_coops[i+1] > 0.9:
                transition_width = test_gammas[i+1] - test_gammas[i]
                break

        sharpness = 1.0 - min(1.0, transition_width / 0.1)  # Normalize

        # Confidence based on iterations and sharpness
        confidence = min(1.0, (iterations / max_iterations) * sharpness + 0.5)

        # Recommended operating point
        recommended_gamma = gamma_critical + margin

        result = CriticalPointResult(
            gamma_critical=gamma_critical,
            gamma_lower=low,
            gamma_upper=high,
            transition_sharpness=sharpness,
            recommended_gamma=recommended_gamma,
            recommended_margin=margin,
            search_iterations=iterations,
            confidence=confidence,
            domain=self.config.name,
            config_used={
                'N': N,
                'tau': tau,
                'tolerance': tolerance,
                **self.config.cost_params
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        if verbose:
            logger.info(f"\n{'='*50}")
            logger.info(f"CRITICAL POINT FOUND")
            logger.info(f"{'='*50}")
            logger.info(f"  Gamma critical: {gamma_critical:.4f}")
            logger.info(f"  Transition sharpness: {sharpness:.2f}")
            logger.info(f"  Recommended gamma: {recommended_gamma:.4f}")
            logger.info(f"  Confidence: {confidence:.2f}")

        return result

    def validate_critical_point(
        self,
        gamma_critical: float,
        tau: float = None,
        N: int = None,
        num_tests: int = 10
    ) -> Dict:
        """
        Validate a critical point by running multiple tests around it.
        """
        if tau is None:
            tau = self.config.default_tau
        if N is None:
            N = self.config.default_N

        results = {
            'below': [],
            'at': [],
            'above': []
        }

        for _ in range(num_tests):
            results['below'].append(self._run_simulation(gamma_critical - 0.02, tau, N))
            results['at'].append(self._run_simulation(gamma_critical, tau, N))
            results['above'].append(self._run_simulation(gamma_critical + 0.02, tau, N))

        return {
            'gamma_critical': gamma_critical,
            'below_mean': np.mean(results['below']),
            'below_std': np.std(results['below']),
            'at_mean': np.mean(results['at']),
            'at_std': np.std(results['at']),
            'above_mean': np.mean(results['above']),
            'above_std': np.std(results['above']),
            'valid': np.mean(results['below']) < 0.5 < np.mean(results['above'])
        }

    def sweep_tau(
        self,
        tau_values: List[float] = None,
        N: int = None,
        verbose: bool = True
    ) -> Dict[float, CriticalPointResult]:
        """
        Find critical points across different tau values.

        Useful for understanding how hub influence affects the phase transition.
        """
        if tau_values is None:
            tau_values = [0.0, 0.2, 0.3, 0.5, 0.7]

        results = {}

        for tau in tau_values:
            if verbose:
                logger.info(f"\n--- Tau = {tau} ---")
            result = self.find_critical_point(tau=tau, N=N, verbose=verbose)
            results[tau] = result

        return results


class AutoCalibrator:
    """
    Automatic calibration for BTUT deployments.

    Runs critical point discovery and stores results for use by
    all simulation components.
    """

    def __init__(self, cache_file: str = "btut_calibration.json"):
        self.cache_file = cache_file
        self.calibrations: Dict[str, CriticalPointResult] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached calibrations"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for key, val in data.items():
                        self.calibrations[key] = CriticalPointResult(**val)
                logger.info(f"Loaded {len(self.calibrations)} cached calibrations")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")

    def _save_cache(self):
        """Save calibrations to cache"""
        data = {k: asdict(v) for k, v in self.calibrations.items()}
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _make_key(self, domain: str, tau: float, N: int) -> str:
        """Create cache key"""
        return f"{domain}_tau{tau:.2f}_N{N}"

    def calibrate(
        self,
        domain: Domain = Domain.ABSTRACT,
        tau: float = 0.3,
        N: int = 1000,
        force: bool = False
    ) -> CriticalPointResult:
        """
        Get or compute calibration for given parameters.

        Uses cache if available, otherwise runs discovery.
        """
        key = self._make_key(domain.value, tau, N)

        if not force and key in self.calibrations:
            logger.info(f"Using cached calibration for {key}")
            return self.calibrations[key]

        logger.info(f"Running calibration for {key}")
        finder = CriticalPointFinder(domain)
        result = finder.find_critical_point(tau=tau, N=N)

        self.calibrations[key] = result
        self._save_cache()

        return result

    def calibrate_all_domains(self, tau: float = 0.3) -> Dict[str, CriticalPointResult]:
        """Calibrate all supported domains"""
        results = {}

        for domain in [Domain.ABSTRACT, Domain.TRAFFIC, Domain.DRONE_SWARM]:
            config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS[Domain.ABSTRACT])
            result = self.calibrate(domain, tau, config.default_N)
            results[domain.value] = result

        return results

    def get_recommended_gamma(
        self,
        domain: Domain = Domain.ABSTRACT,
        tau: float = 0.3,
        N: int = None
    ) -> float:
        """
        Get recommended gamma for a domain.

        This is the primary interface for other modules.
        """
        if N is None:
            config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS[Domain.ABSTRACT])
            N = config.default_N

        result = self.calibrate(domain, tau, N)
        return result.recommended_gamma


# Global calibrator instance
_calibrator = None

def get_calibrator() -> AutoCalibrator:
    """Get global calibrator instance"""
    global _calibrator
    if _calibrator is None:
        _calibrator = AutoCalibrator()
    return _calibrator


def get_critical_gamma(
    domain: str = "abstract",
    tau: float = 0.3,
    N: int = 1000
) -> float:
    """
    Convenience function to get recommended gamma.

    Usage:
        from critical_point_finder import get_critical_gamma
        gamma = get_critical_gamma("traffic", tau=0.3)
    """
    domain_enum = Domain(domain) if domain in [d.value for d in Domain] else Domain.ABSTRACT
    return get_calibrator().get_recommended_gamma(domain_enum, tau, N)


def main():
    """CLI interface for critical point finder"""
    import argparse

    parser = argparse.ArgumentParser(description='BTUT Critical Point Finder')
    parser.add_argument('--domain', type=str, default='abstract',
                       choices=['abstract', 'traffic', 'drone', 'robot'],
                       help='Application domain')
    parser.add_argument('--tau', type=float, default=0.3, help='Hub influence')
    parser.add_argument('--N', type=int, default=1000, help='Number of agents')
    parser.add_argument('--sweep-tau', action='store_true', help='Sweep across tau values')
    parser.add_argument('--calibrate-all', action='store_true', help='Calibrate all domains')
    parser.add_argument('--output', type=str, default='critical_point_result.json')

    args = parser.parse_args()

    if args.calibrate_all:
        calibrator = AutoCalibrator()
        results = calibrator.calibrate_all_domains(args.tau)

        print("\n" + "="*60)
        print("ALL DOMAINS CALIBRATED")
        print("="*60)
        for domain, result in results.items():
            print(f"  {domain}: gamma_c = {result.gamma_critical:.4f}, "
                  f"recommended = {result.recommended_gamma:.4f}")

        with open(args.output, 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)

    elif args.sweep_tau:
        domain = Domain(args.domain)
        finder = CriticalPointFinder(domain)
        results = finder.sweep_tau(N=args.N)

        print("\n" + "="*60)
        print("TAU SWEEP RESULTS")
        print("="*60)
        print(f"{'Tau':<10} {'Gamma_c':<12} {'Recommended':<12} {'Sharpness':<10}")
        print("-"*60)
        for tau, result in sorted(results.items()):
            print(f"{tau:<10.2f} {result.gamma_critical:<12.4f} "
                  f"{result.recommended_gamma:<12.4f} {result.transition_sharpness:<10.2f}")

        with open(args.output, 'w') as f:
            json.dump({str(k): asdict(v) for k, v in results.items()}, f, indent=2)

    else:
        domain = Domain(args.domain)
        finder = CriticalPointFinder(domain)
        result = finder.find_critical_point(tau=args.tau, N=args.N)

        with open(args.output, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\nResult saved to {args.output}")


if __name__ == '__main__':
    main()
