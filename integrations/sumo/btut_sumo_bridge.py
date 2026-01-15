"""
BTUT-SUMO Bridge: Enhanced Traffic Coordinator

This module bridges BTUT multi-agent coordination with SUMO traffic simulation,
enabling validation of BTUT's phase transition behavior in realistic traffic scenarios.

Key Features:
- Spatial neighbor detection (bridging BTUT's mean-field to SUMO's spatial model)
- Degree-based hub identification (vehicles at intersections = high degree)
- Baseline comparison mode (tau=0 vs optimal tau)
- Comprehensive metrics collection for validation

Usage:
    python btut_sumo_bridge.py --config example_scenario.sumocfg --experiment comparison
"""

import sys
import os
import math
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

# Add python-sdk to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python-sdk'))

import numpy as np

try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("Warning: traci not available. Install SUMO to use traffic simulation.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VehicleState:
    """State of a single vehicle"""
    id: str
    position: Tuple[float, float]
    speed: float
    lane: str
    route: str
    strategy: str = "A"  # A=cooperative, B=competitive
    degree: int = 0  # Number of neighbors
    weight: float = 1.0  # Kernel weight for BTUT


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    step: int = 0
    num_vehicles: int = 0
    cooperation_rate: float = 0.5
    avg_speed: float = 0.0
    avg_waiting_time: float = 0.0
    throughput: int = 0  # Vehicles completed
    total_travel_time: float = 0.0
    collisions: int = 0
    avg_degree: float = 0.0
    max_degree: int = 0
    hub_cooperation: float = 0.0  # Cooperation rate among hubs


@dataclass
class ExperimentResults:
    """Results from a complete experiment"""
    experiment_name: str
    tau: float
    gamma: float
    total_steps: int
    final_cooperation: float
    avg_speed: float
    avg_waiting_time: float
    total_throughput: int
    metrics_history: List[Dict] = field(default_factory=list)


class SpatialNeighborDetector:
    """
    Detects neighbors based on spatial proximity in SUMO.

    This bridges BTUT's abstract network topology to real spatial positions,
    computing "degree" as the number of vehicles within communication range.
    """

    def __init__(self, comm_range: float = 50.0):
        """
        Args:
            comm_range: Communication/influence range in meters
        """
        self.comm_range = comm_range
        self._position_cache: Dict[str, Tuple[float, float]] = {}

    def update_positions(self, vehicles: Dict[str, VehicleState]):
        """Update position cache"""
        self._position_cache = {
            vid: v.position for vid, v in vehicles.items()
        }

    def get_neighbors(self, vehicle_id: str) -> List[str]:
        """Get IDs of vehicles within communication range"""
        if vehicle_id not in self._position_cache:
            return []

        my_pos = self._position_cache[vehicle_id]
        neighbors = []

        for vid, pos in self._position_cache.items():
            if vid == vehicle_id:
                continue

            dist = math.sqrt(
                (pos[0] - my_pos[0])**2 +
                (pos[1] - my_pos[1])**2
            )

            if dist <= self.comm_range:
                neighbors.append(vid)

        return neighbors

    def compute_degrees(self, vehicles: Dict[str, VehicleState]) -> Dict[str, int]:
        """Compute degree (neighbor count) for all vehicles"""
        self.update_positions(vehicles)
        degrees = {}

        for vid in vehicles:
            neighbors = self.get_neighbors(vid)
            degrees[vid] = len(neighbors)

        return degrees


class BTUTSpatialSimulator:
    """
    BTUT simulator adapted for spatial traffic networks.

    Unlike the pure mean-field approach, this version uses actual neighbor
    relationships from the traffic simulation to compute local cooperation
    pressure, while still leveraging the O(N) kernel-weighted update.
    """

    def __init__(self, gamma: float = 1.5, tau: float = 0.3):
        self.gamma = gamma
        self.tau = tau

        # Game payoffs (from btut_engine.py)
        self.u_A_SH = np.log(2.5)  # Stag Hunt payoff for A
        self.u_B_SH = np.log(1.2)  # Stag Hunt payoff for B
        self.c_A_SH = 0.05
        self.c_B_SH = 0.03

    def compute_kernel_weights(
        self,
        degrees: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Compute BTUT kernel weights: w_i = (k_i / k_max)^tau
        """
        if not degrees:
            return {}

        k_max = max(degrees.values()) if max(degrees.values()) > 0 else 1

        weights = {}
        for vid, k in degrees.items():
            if k_max > 0 and k > 0:
                weights[vid] = (k / k_max) ** self.tau
            else:
                weights[vid] = 0.01  # Minimum weight for isolated vehicles

        return weights

    def compute_expected_utility(self, p: float) -> Tuple[float, float]:
        """
        Compute expected utilities for strategies A and B.

        Uses Stag Hunt game with cooperation bonus gamma.

        Args:
            p: Current cooperation fraction

        Returns:
            (EU_A, EU_B) expected utilities
        """
        # Stag Hunt expected utilities (simplified from btut_engine.py)
        EU_A = 0.5 * (
            self.u_A_SH + p * self.u_A_SH + (1 - p) * self.u_B_SH
        ) * (p * self.gamma + (1 - p) * 0.03) - self.c_A_SH * self.u_A_SH

        EU_B = 0.5 * (
            self.u_B_SH + p * self.u_A_SH + (1 - p) * self.u_B_SH
        ) * (p * 0.03 + (1 - p)) - self.c_B_SH * self.u_B_SH

        return EU_A, EU_B

    def update_strategies(
        self,
        vehicles: Dict[str, VehicleState],
        weights: Dict[str, float],
        p_current: float
    ) -> Tuple[Dict[str, str], float]:
        """
        Update vehicle strategies using BTUT kernel-weighted mean-field.

        Returns:
            (strategies, new_p) - Strategy assignments and new cooperation fraction
        """
        if not vehicles:
            return {}, 0.5

        EU_A, EU_B = self.compute_expected_utility(p_current)

        strategies = {}
        weighted_A = 0.0
        total_weight = 0.0

        for vid, vehicle in vehicles.items():
            w = weights.get(vid, 0.01)
            k = vehicle.degree

            # Weighted utility comparison
            agent_U_A = EU_A * w * (k + 1)  # +1 to avoid zero
            agent_U_B = EU_B * w * (k + 1)

            if agent_U_A > agent_U_B:
                strategies[vid] = "A"
                weighted_A += w
            else:
                strategies[vid] = "B"

            total_weight += w

        # New cooperation fraction (weighted)
        new_p = weighted_A / total_weight if total_weight > 0 else 0.5

        # Momentum smoothing
        smoothed_p = 0.5 * p_current + 0.5 * new_p

        return strategies, smoothed_p

    def run_iterations(
        self,
        vehicles: Dict[str, VehicleState],
        weights: Dict[str, float],
        iterations: int = 10
    ) -> Tuple[Dict[str, str], float]:
        """
        Run BTUT iterations until convergence.

        Returns:
            (final_strategies, final_p)
        """
        p = 0.5
        strategies = {vid: "A" for vid in vehicles}

        for _ in range(iterations):
            strategies, p = self.update_strategies(vehicles, weights, p)

        return strategies, p


class BTUTTrafficCoordinator:
    """
    Main coordinator class bridging BTUT with SUMO.

    Supports:
    - Real-time coordination updates
    - Baseline comparison (tau=0 vs optimal tau)
    - Comprehensive metrics collection
    - Spatial neighbor detection
    """

    def __init__(
        self,
        sumo_config: str,
        gamma: float = 1.5,
        tau: float = 0.3,
        comm_range: float = 50.0,
        use_gui: bool = True,
        headless: bool = False
    ):
        """
        Initialize traffic coordinator.

        Args:
            sumo_config: Path to SUMO configuration file
            gamma: Cooperation bonus parameter
            tau: Hub influence exponent
            comm_range: Communication range for neighbor detection (meters)
            use_gui: Whether to use sumo-gui (visualization)
            headless: If True, run without GUI even if use_gui is True
        """
        if not TRACI_AVAILABLE:
            raise RuntimeError("traci not available. Install SUMO first.")

        self.sumo_config = sumo_config
        self.gamma = gamma
        self.tau = tau
        self.use_gui = use_gui and not headless

        # Components
        self.neighbor_detector = SpatialNeighborDetector(comm_range)
        self.btut_sim = BTUTSpatialSimulator(gamma, tau)

        # State
        self.vehicles: Dict[str, VehicleState] = {}
        self.current_p = 0.5
        self.step_count = 0
        self.completed_vehicles = 0

        # Metrics
        self.metrics_history: List[SimulationMetrics] = []

        # Strategy behavior parameters
        self.strategy_params = {
            "A": {  # Cooperative
                "speed": 13.89,  # 50 km/h
                "speed_mode": 0b011111,  # All safety checks
                "lane_change_mode": 0b1001010101,  # Conservative
                "min_gap": 3.5,
                "tau": 2.0,  # Larger time headway
            },
            "B": {  # Competitive
                "speed": 19.44,  # 70 km/h
                "speed_mode": 0b010111,  # Less conservative
                "lane_change_mode": 0b1001011001,  # More aggressive
                "min_gap": 2.0,
                "tau": 1.0,  # Smaller time headway
            }
        }

        self._start_sumo()

    def _start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        cmd = [
            sumo_binary,
            "-c", self.sumo_config,
            "--start",
            "--quit-on-end"
        ]

        if not self.use_gui:
            cmd.extend(["--no-step-log", "true"])

        try:
            traci.start(cmd)
            logger.info(f"SUMO started with {sumo_binary}")
        except Exception as e:
            logger.error(f"Failed to start SUMO: {e}")
            raise

    def _update_vehicle_states(self):
        """Update internal vehicle state from SUMO"""
        vehicle_ids = traci.vehicle.getIDList()

        new_vehicles = {}
        for vid in vehicle_ids:
            try:
                pos = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                lane = traci.vehicle.getLaneID(vid)
                route = traci.vehicle.getRouteID(vid)

                # Preserve existing strategy if vehicle existed
                old_strategy = self.vehicles.get(vid, VehicleState(
                    id=vid, position=(0,0), speed=0, lane="", route=""
                )).strategy

                new_vehicles[vid] = VehicleState(
                    id=vid,
                    position=pos,
                    speed=speed,
                    lane=lane,
                    route=route,
                    strategy=old_strategy
                )
            except traci.exceptions.TraCIException:
                pass

        # Track completed vehicles
        departed = set(self.vehicles.keys()) - set(new_vehicles.keys())
        self.completed_vehicles += len(departed)

        self.vehicles = new_vehicles

    def _compute_vehicle_degrees(self):
        """Compute degrees (neighbor counts) for all vehicles"""
        degrees = self.neighbor_detector.compute_degrees(self.vehicles)

        for vid, degree in degrees.items():
            if vid in self.vehicles:
                self.vehicles[vid].degree = degree

    def _run_btut_coordination(self) -> float:
        """
        Run BTUT coordination simulation.

        Returns:
            New cooperation fraction
        """
        if not self.vehicles:
            return 0.5

        # Compute kernel weights from degrees
        degrees = {vid: v.degree for vid, v in self.vehicles.items()}
        weights = self.btut_sim.compute_kernel_weights(degrees)

        # Update weights in vehicle states
        for vid, w in weights.items():
            if vid in self.vehicles:
                self.vehicles[vid].weight = w

        # Run BTUT iterations
        strategies, new_p = self.btut_sim.run_iterations(
            self.vehicles, weights, iterations=10
        )

        # Update vehicle strategies
        for vid, strategy in strategies.items():
            if vid in self.vehicles:
                self.vehicles[vid].strategy = strategy

        return new_p

    def _apply_strategies(self):
        """Apply strategies to SUMO vehicle behaviors"""
        for vid, vehicle in self.vehicles.items():
            params = self.strategy_params[vehicle.strategy]

            try:
                traci.vehicle.setSpeed(vid, params["speed"])
                traci.vehicle.setSpeedMode(vid, params["speed_mode"])
                traci.vehicle.setLaneChangeMode(vid, params["lane_change_mode"])
                traci.vehicle.setMinGap(vid, params["min_gap"])
                traci.vehicle.setTau(vid, params["tau"])
            except traci.exceptions.TraCIException as e:
                logger.debug(f"Failed to set params for {vid}: {e}")

    def _collect_metrics(self) -> SimulationMetrics:
        """Collect current simulation metrics"""
        if not self.vehicles:
            return SimulationMetrics(step=self.step_count)

        speeds = []
        waiting_times = []
        degrees = []
        hub_strategies = []

        avg_degree = np.mean([v.degree for v in self.vehicles.values()])
        hub_threshold = avg_degree * 1.5  # Hubs have above-average degree

        for vid, vehicle in self.vehicles.items():
            try:
                speeds.append(traci.vehicle.getSpeed(vid))
                waiting_times.append(traci.vehicle.getWaitingTime(vid))
            except traci.exceptions.TraCIException:
                pass

            degrees.append(vehicle.degree)

            if vehicle.degree > hub_threshold:
                hub_strategies.append(1 if vehicle.strategy == "A" else 0)

        coop_count = sum(1 for v in self.vehicles.values() if v.strategy == "A")
        coop_rate = coop_count / len(self.vehicles) if self.vehicles else 0.5

        hub_coop = np.mean(hub_strategies) if hub_strategies else coop_rate

        return SimulationMetrics(
            step=self.step_count,
            num_vehicles=len(self.vehicles),
            cooperation_rate=coop_rate,
            avg_speed=np.mean(speeds) if speeds else 0,
            avg_waiting_time=np.mean(waiting_times) if waiting_times else 0,
            throughput=self.completed_vehicles,
            avg_degree=np.mean(degrees) if degrees else 0,
            max_degree=max(degrees) if degrees else 0,
            hub_cooperation=hub_coop
        )

    def step(self) -> bool:
        """Execute one simulation step"""
        try:
            traci.simulationStep()
            self.step_count += 1
            return True
        except traci.exceptions.TraCIException:
            return False

    def run_coordination_cycle(self) -> SimulationMetrics:
        """
        Run one complete coordination cycle.

        1. Update vehicle states from SUMO
        2. Compute neighbor degrees
        3. Run BTUT coordination
        4. Apply strategies
        5. Collect metrics
        """
        self._update_vehicle_states()
        self._compute_vehicle_degrees()
        self.current_p = self._run_btut_coordination()
        self._apply_strategies()
        return self._collect_metrics()

    def run(
        self,
        steps: int = 3600,
        coordination_interval: int = 100,
        verbose: bool = True
    ) -> ExperimentResults:
        """
        Run complete simulation.

        Args:
            steps: Total simulation steps
            coordination_interval: Steps between coordination updates
            verbose: Print progress

        Returns:
            ExperimentResults with all metrics
        """
        logger.info(f"Running simulation: {steps} steps, gamma={self.gamma}, tau={self.tau}")

        for step in range(steps):
            if not self.step():
                logger.info(f"Simulation ended at step {step}")
                break

            if step % coordination_interval == 0:
                metrics = self.run_coordination_cycle()
                self.metrics_history.append(metrics)

                if verbose and step % (coordination_interval * 5) == 0:
                    logger.info(
                        f"Step {step}: {metrics.num_vehicles} vehicles, "
                        f"coop={metrics.cooperation_rate:.1%}, "
                        f"speed={metrics.avg_speed:.1f}m/s, "
                        f"wait={metrics.avg_waiting_time:.1f}s"
                    )

        # Final metrics
        if self.metrics_history:
            final_coop = np.mean([m.cooperation_rate for m in self.metrics_history[-10:]])
            avg_speed = np.mean([m.avg_speed for m in self.metrics_history])
            avg_wait = np.mean([m.avg_waiting_time for m in self.metrics_history])
        else:
            final_coop = 0.5
            avg_speed = 0
            avg_wait = 0

        return ExperimentResults(
            experiment_name=f"tau_{self.tau}_gamma_{self.gamma}",
            tau=self.tau,
            gamma=self.gamma,
            total_steps=self.step_count,
            final_cooperation=final_coop,
            avg_speed=avg_speed,
            avg_waiting_time=avg_wait,
            total_throughput=self.completed_vehicles,
            metrics_history=[asdict(m) for m in self.metrics_history]
        )

    def close(self):
        """Close SUMO connection"""
        try:
            traci.close()
            logger.info("SUMO closed")
        except:
            pass


def run_baseline_comparison(
    sumo_config: str,
    steps: int = 3600,
    gamma: float = 1.5,
    tau_values: List[float] = None,
    use_gui: bool = False
) -> Dict[str, ExperimentResults]:
    """
    Run comparison experiments: baseline (tau=0) vs optimal tau.

    Args:
        sumo_config: Path to SUMO config
        steps: Simulation steps per experiment
        gamma: Cooperation bonus
        tau_values: List of tau values to test (default: [0, 0.3, 0.5])
        use_gui: Show GUI for last experiment

    Returns:
        Dict mapping tau value to results
    """
    if tau_values is None:
        tau_values = [0.0, 0.3, 0.5]

    results = {}

    for i, tau in enumerate(tau_values):
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: tau={tau}")
        logger.info(f"{'='*60}")

        # Only show GUI for last experiment
        show_gui = use_gui and (i == len(tau_values) - 1)

        coordinator = BTUTTrafficCoordinator(
            sumo_config=sumo_config,
            gamma=gamma,
            tau=tau,
            use_gui=show_gui
        )

        try:
            result = coordinator.run(steps=steps)
            results[f"tau_{tau}"] = result
        finally:
            coordinator.close()

        # Brief pause between experiments
        time.sleep(1)

    # Print comparison
    print_comparison(results)

    return results


def print_comparison(results: Dict[str, ExperimentResults]):
    """Print comparison table"""
    print("\n" + "="*70)
    print("EXPERIMENT COMPARISON")
    print("="*70)
    print(f"{'Tau':<10} {'Coop Rate':<12} {'Avg Speed':<12} {'Avg Wait':<12} {'Throughput':<12}")
    print("-"*70)

    baseline_wait = None
    baseline_speed = None

    for key, result in sorted(results.items()):
        if baseline_wait is None:
            baseline_wait = result.avg_waiting_time
            baseline_speed = result.avg_speed

        wait_change = ((result.avg_waiting_time - baseline_wait) / baseline_wait * 100) if baseline_wait > 0 else 0
        speed_change = ((result.avg_speed - baseline_speed) / baseline_speed * 100) if baseline_speed > 0 else 0

        print(
            f"{result.tau:<10.2f} "
            f"{result.final_cooperation:<12.1%} "
            f"{result.avg_speed:<12.1f} "
            f"{result.avg_waiting_time:<12.1f} "
            f"{result.total_throughput:<12}"
        )

    print("="*70)

    # Improvement summary
    if len(results) > 1:
        baseline = list(results.values())[0]
        best = max(results.values(), key=lambda r: r.avg_speed)

        if best.tau != baseline.tau:
            speed_imp = (best.avg_speed - baseline.avg_speed) / baseline.avg_speed * 100 if baseline.avg_speed > 0 else 0
            wait_imp = (baseline.avg_waiting_time - best.avg_waiting_time) / baseline.avg_waiting_time * 100 if baseline.avg_waiting_time > 0 else 0

            print(f"\nBest tau={best.tau}:")
            print(f"  Speed improvement: {speed_imp:+.1f}%")
            print(f"  Wait time reduction: {wait_imp:+.1f}%")


def save_results(results: Dict[str, ExperimentResults], output_path: str):
    """Save results to JSON"""
    data = {
        key: asdict(result) if hasattr(result, '__dataclass_fields__') else result
        for key, result in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='BTUT-SUMO Traffic Coordination Bridge'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to SUMO config file'
    )
    parser.add_argument(
        '--experiment', type=str, default='single',
        choices=['single', 'comparison', 'sweep'],
        help='Experiment type'
    )
    parser.add_argument(
        '--steps', type=int, default=3600,
        help='Simulation steps'
    )
    parser.add_argument(
        '--gamma', type=float, default=1.5,
        help='Cooperation bonus'
    )
    parser.add_argument(
        '--tau', type=float, default=0.3,
        help='Hub influence (for single experiment)'
    )
    parser.add_argument(
        '--interval', type=int, default=100,
        help='Coordination update interval'
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='Show SUMO GUI'
    )
    parser.add_argument(
        '--output', type=str, default='btut_sumo_results.json',
        help='Output file path'
    )

    args = parser.parse_args()

    if args.experiment == 'single':
        # Single experiment
        coordinator = BTUTTrafficCoordinator(
            sumo_config=args.config,
            gamma=args.gamma,
            tau=args.tau,
            use_gui=args.gui
        )

        try:
            result = coordinator.run(
                steps=args.steps,
                coordination_interval=args.interval
            )
            save_results({f"tau_{args.tau}": result}, args.output)
        finally:
            coordinator.close()

    elif args.experiment == 'comparison':
        # Baseline comparison
        results = run_baseline_comparison(
            sumo_config=args.config,
            steps=args.steps,
            gamma=args.gamma,
            tau_values=[0.0, 0.3, 0.5],
            use_gui=args.gui
        )
        save_results(results, args.output)

    elif args.experiment == 'sweep':
        # Parameter sweep
        tau_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        results = run_baseline_comparison(
            sumo_config=args.config,
            steps=args.steps,
            gamma=args.gamma,
            tau_values=tau_values,
            use_gui=False
        )
        save_results(results, args.output)


if __name__ == '__main__':
    main()
