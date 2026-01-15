#!/usr/bin/env python3
"""
BTUT Drone Swarm Core

Core coordination logic for drone swarms using BTUT game-theoretic framework.
Works standalone (pure Python) or integrates with ROS 2 / Gazebo.

Key Concepts:
- Agents = Drones with 3D positions
- Strategy A = Cooperative (formation-keeping, yield to neighbors)
- Strategy B = Competitive (goal-seeking, ignore formation)
- Hub = Drones with high neighbor count (center of local cluster)
- Tau = How much hubs influence collective behavior

Metrics:
- Formation stability (variance from target positions)
- Collision avoidance rate
- Task completion time
- Energy efficiency
"""

import sys
import os
import math
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

# Add BTUT engine
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'lib', 'simulation'))

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Strategy(Enum):
    COOPERATIVE = "A"  # Formation-keeping, yield
    COMPETITIVE = "B"  # Goal-seeking, aggressive


@dataclass
class DroneState:
    """State of a single drone"""
    id: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    target: np.ndarray    # Target position
    strategy: Strategy = Strategy.COOPERATIVE
    degree: int = 0       # Number of neighbors
    weight: float = 1.0   # BTUT kernel weight
    energy: float = 100.0 # Battery percentage

    def distance_to(self, other: 'DroneState') -> float:
        return np.linalg.norm(self.position - other.position)

    def distance_to_target(self) -> float:
        return np.linalg.norm(self.position - self.target)


@dataclass
class SwarmMetrics:
    """Metrics for swarm performance"""
    step: int
    num_drones: int
    cooperation_rate: float
    formation_error: float      # RMS distance from formation
    min_separation: float       # Minimum drone-drone distance
    collisions: int             # Count of separation < safety threshold
    avg_speed: float
    avg_energy: float
    task_progress: float        # Fraction of drones at target


@dataclass
class SwarmConfig:
    """Configuration for swarm simulation"""
    num_drones: int = 100
    comm_range: float = 50.0      # Communication range (meters)
    safety_distance: float = 2.0  # Minimum safe separation
    max_speed: float = 10.0       # Maximum drone speed (m/s)
    gamma: float = 1.5            # BTUT cooperation bonus
    tau: float = 0.3              # BTUT hub influence
    formation: str = "grid"       # Formation type: grid, circle, v, random
    arena_size: float = 200.0     # Arena size (meters)


class SpatialGraph:
    """
    Builds communication graph based on spatial proximity.

    Drones within comm_range are neighbors.
    """

    def __init__(self, comm_range: float = 50.0):
        self.comm_range = comm_range

    def compute_neighbors(
        self,
        drones: Dict[int, DroneState]
    ) -> Dict[int, List[int]]:
        """Compute neighbor lists for all drones"""
        neighbors = {drone_id: [] for drone_id in drones}

        drone_list = list(drones.values())
        n = len(drone_list)

        for i in range(n):
            for j in range(i + 1, n):
                dist = drone_list[i].distance_to(drone_list[j])
                if dist <= self.comm_range:
                    neighbors[drone_list[i].id].append(drone_list[j].id)
                    neighbors[drone_list[j].id].append(drone_list[i].id)

        return neighbors

    def compute_degrees(
        self,
        drones: Dict[int, DroneState]
    ) -> Dict[int, int]:
        """Compute degree (neighbor count) for all drones"""
        neighbors = self.compute_neighbors(drones)
        return {drone_id: len(nbrs) for drone_id, nbrs in neighbors.items()}


class BTUTSwarmCoordinator:
    """
    BTUT-based coordination for drone swarms.

    Maps BTUT game theory to swarm behaviors:
    - Cooperation = maintain formation, yield to neighbors
    - Defection = pursue individual goal, ignore others
    """

    def __init__(self, gamma: float = 1.5, tau: float = 0.3):
        self.gamma = gamma
        self.tau = tau
        self._p = 0.5  # Current cooperation fraction

        # Payoffs (Stag Hunt style)
        self.u_A = np.log(2.5)  # Cooperation payoff
        self.u_B = np.log(1.2)  # Defection payoff

    def compute_kernel_weights(
        self,
        degrees: Dict[int, int]
    ) -> Dict[int, float]:
        """Compute BTUT kernel weights from degrees"""
        if not degrees:
            return {}

        k_max = max(degrees.values()) if max(degrees.values()) > 0 else 1

        weights = {}
        for drone_id, k in degrees.items():
            if k_max > 0 and k > 0:
                weights[drone_id] = (k / k_max) ** self.tau
            else:
                weights[drone_id] = 0.01

        return weights

    def update_strategies(
        self,
        drones: Dict[int, DroneState],
        degrees: Dict[int, int]
    ) -> Tuple[Dict[int, Strategy], float]:
        """
        Update drone strategies using BTUT coordination.

        Returns:
            (strategies, new_cooperation_rate)
        """
        if not drones:
            return {}, 0.5

        weights = self.compute_kernel_weights(degrees)

        # Update weights in drone states
        for drone_id, w in weights.items():
            if drone_id in drones:
                drones[drone_id].weight = w
                drones[drone_id].degree = degrees.get(drone_id, 0)

        # Compute expected utilities
        p = self._p
        EU_A = 0.5 * (self.u_A + p * self.u_A + (1-p) * self.u_B) * \
               (p * self.gamma + (1-p) * 0.03)
        EU_B = 0.5 * (self.u_B + p * self.u_A + (1-p) * self.u_B) * \
               (p * 0.03 + (1-p))

        # Assign strategies
        strategies = {}
        weighted_A = 0.0
        total_weight = 0.0

        for drone_id, drone in drones.items():
            w = weights.get(drone_id, 0.01)
            k = degrees.get(drone_id, 0)

            agent_U_A = EU_A * w * (k + 1)
            agent_U_B = EU_B * w * (k + 1)

            if agent_U_A > agent_U_B:
                strategies[drone_id] = Strategy.COOPERATIVE
                weighted_A += w
            else:
                strategies[drone_id] = Strategy.COMPETITIVE

            total_weight += w

        # Update cooperation fraction
        new_p = weighted_A / total_weight if total_weight > 0 else 0.5
        self._p = 0.5 * self._p + 0.5 * new_p

        return strategies, self._p


class FormationController:
    """
    Controls drone movements based on strategy and formation goals.
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.formation_targets: Dict[int, np.ndarray] = {}

    def generate_formation(
        self,
        num_drones: int,
        center: np.ndarray = None
    ) -> Dict[int, np.ndarray]:
        """Generate target positions for formation"""
        if center is None:
            center = np.array([
                self.config.arena_size / 2,
                self.config.arena_size / 2,
                20.0  # Default altitude
            ])

        targets = {}

        if self.config.formation == "grid":
            # Square grid formation
            side = int(np.ceil(np.sqrt(num_drones)))
            spacing = 10.0

            for i in range(num_drones):
                row = i // side
                col = i % side
                targets[i] = center + np.array([
                    (col - side/2) * spacing,
                    (row - side/2) * spacing,
                    0.0
                ])

        elif self.config.formation == "circle":
            # Circular formation
            radius = 5.0 * np.sqrt(num_drones)
            for i in range(num_drones):
                angle = 2 * np.pi * i / num_drones
                targets[i] = center + np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0.0
                ])

        elif self.config.formation == "v":
            # V formation (like birds)
            spacing = 8.0
            for i in range(num_drones):
                side = i % 2  # Alternate sides
                pos_in_line = (i + 1) // 2
                targets[i] = center + np.array([
                    -pos_in_line * spacing,
                    (1 if side == 0 else -1) * pos_in_line * spacing * 0.5,
                    0.0
                ])

        else:  # random
            for i in range(num_drones):
                targets[i] = center + np.random.uniform(
                    -50, 50, size=3
                )
                targets[i][2] = max(5.0, targets[i][2])  # Min altitude

        self.formation_targets = targets
        return targets

    def compute_velocity(
        self,
        drone: DroneState,
        neighbors: List[DroneState],
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Compute desired velocity based on strategy.

        Cooperative: Balance formation + separation + alignment
        Competitive: Direct path to target, minimal avoidance
        """
        velocity = np.zeros(3)

        # Target attraction
        to_target = drone.target - drone.position
        dist_to_target = np.linalg.norm(to_target)

        if dist_to_target > 0.1:
            target_direction = to_target / dist_to_target
        else:
            target_direction = np.zeros(3)

        if drone.strategy == Strategy.COOPERATIVE:
            # Cooperative behavior: Reynolds flocking rules

            # 1. Separation (avoid neighbors)
            separation = np.zeros(3)
            for neighbor in neighbors:
                diff = drone.position - neighbor.position
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < self.config.safety_distance * 3:
                    separation += diff / (dist * dist)

            # 2. Alignment (match neighbor velocities)
            alignment = np.zeros(3)
            if neighbors:
                avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
                alignment = avg_velocity - drone.velocity

            # 3. Cohesion (move toward formation target)
            cohesion = target_direction

            # Combine with weights
            velocity = (
                0.3 * separation +      # Avoid collisions
                0.2 * alignment +       # Match neighbors
                0.5 * cohesion          # Go to formation
            )

        else:  # Competitive
            # Direct to target, minimal separation only for safety
            velocity = target_direction * 1.5  # Faster

            # Only avoid imminent collisions
            for neighbor in neighbors:
                diff = drone.position - neighbor.position
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < self.config.safety_distance * 1.5:
                    velocity += diff / (dist * dist) * 0.5

        # Limit speed
        speed = np.linalg.norm(velocity)
        if speed > self.config.max_speed:
            velocity = velocity / speed * self.config.max_speed

        return velocity


class SwarmSimulator:
    """
    Complete swarm simulation with BTUT coordination.
    """

    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()

        self.drones: Dict[int, DroneState] = {}
        self.spatial_graph = SpatialGraph(self.config.comm_range)
        self.coordinator = BTUTSwarmCoordinator(
            self.config.gamma, self.config.tau
        )
        self.formation_controller = FormationController(self.config)

        self.step_count = 0
        self.metrics_history: List[SwarmMetrics] = []
        self.collision_count = 0

    def initialize(self, initial_positions: Dict[int, np.ndarray] = None):
        """Initialize swarm with random or specified positions"""

        # Generate formation targets
        center = np.array([
            self.config.arena_size / 2,
            self.config.arena_size / 2,
            20.0
        ])
        targets = self.formation_controller.generate_formation(
            self.config.num_drones, center
        )

        # Initialize drones
        for i in range(self.config.num_drones):
            if initial_positions and i in initial_positions:
                pos = initial_positions[i]
            else:
                # Random start positions around arena edges
                angle = 2 * np.pi * i / self.config.num_drones
                radius = self.config.arena_size * 0.4
                pos = np.array([
                    self.config.arena_size/2 + radius * np.cos(angle),
                    self.config.arena_size/2 + radius * np.sin(angle),
                    10.0 + np.random.uniform(-2, 2)
                ])

            self.drones[i] = DroneState(
                id=i,
                position=pos.copy(),
                velocity=np.zeros(3),
                target=targets[i].copy(),
                energy=100.0
            )

        logger.info(f"Initialized {self.config.num_drones} drones")

    def step(self, dt: float = 0.1) -> SwarmMetrics:
        """Execute one simulation step"""

        # 1. Compute spatial graph (neighbors)
        degrees = self.spatial_graph.compute_degrees(self.drones)
        neighbors_map = self.spatial_graph.compute_neighbors(self.drones)

        # 2. Update strategies using BTUT
        strategies, coop_rate = self.coordinator.update_strategies(
            self.drones, degrees
        )

        # Apply strategies
        for drone_id, strategy in strategies.items():
            self.drones[drone_id].strategy = strategy

        # 3. Compute velocities and update positions
        step_collisions = 0
        min_separation = float('inf')

        for drone_id, drone in self.drones.items():
            # Get neighbor states
            neighbor_ids = neighbors_map.get(drone_id, [])
            neighbors = [self.drones[nid] for nid in neighbor_ids]

            # Compute velocity
            velocity = self.formation_controller.compute_velocity(
                drone, neighbors, dt
            )

            # Update state
            drone.velocity = velocity
            drone.position = drone.position + velocity * dt

            # Clamp to arena
            drone.position = np.clip(
                drone.position,
                [0, 0, 1],
                [self.config.arena_size, self.config.arena_size, 100]
            )

            # Update energy (simple model)
            speed = np.linalg.norm(velocity)
            drone.energy -= 0.01 * (1 + speed / self.config.max_speed)
            drone.energy = max(0, drone.energy)

            # Check collisions
            for neighbor in neighbors:
                dist = drone.distance_to(neighbor)
                min_separation = min(min_separation, dist)
                if dist < self.config.safety_distance:
                    step_collisions += 1

        self.collision_count += step_collisions // 2  # Each collision counted twice
        self.step_count += 1

        # 4. Compute metrics
        metrics = self._compute_metrics(coop_rate, min_separation, step_collisions // 2)
        self.metrics_history.append(metrics)

        return metrics

    def _compute_metrics(
        self,
        coop_rate: float,
        min_sep: float,
        collisions: int
    ) -> SwarmMetrics:
        """Compute current swarm metrics"""

        # Formation error (RMS distance to targets)
        formation_errors = []
        at_target_count = 0
        speeds = []
        energies = []

        for drone in self.drones.values():
            dist = drone.distance_to_target()
            formation_errors.append(dist * dist)

            if dist < 5.0:  # Within 5m of target
                at_target_count += 1

            speeds.append(np.linalg.norm(drone.velocity))
            energies.append(drone.energy)

        rms_error = np.sqrt(np.mean(formation_errors)) if formation_errors else 0
        task_progress = at_target_count / len(self.drones) if self.drones else 0

        return SwarmMetrics(
            step=self.step_count,
            num_drones=len(self.drones),
            cooperation_rate=coop_rate,
            formation_error=rms_error,
            min_separation=min_sep if min_sep != float('inf') else 0,
            collisions=collisions,
            avg_speed=np.mean(speeds) if speeds else 0,
            avg_energy=np.mean(energies) if energies else 0,
            task_progress=task_progress
        )

    def run(
        self,
        duration: float = 60.0,
        dt: float = 0.1,
        verbose: bool = True
    ) -> List[SwarmMetrics]:
        """Run simulation for specified duration"""

        steps = int(duration / dt)

        logger.info(f"Running swarm simulation: {duration}s, {steps} steps")
        logger.info(f"  Drones: {self.config.num_drones}")
        logger.info(f"  Gamma: {self.config.gamma}, Tau: {self.config.tau}")
        logger.info(f"  Formation: {self.config.formation}")

        for step in range(steps):
            metrics = self.step(dt)

            if verbose and step % 100 == 0:
                logger.info(
                    f"  Step {step}: coop={metrics.cooperation_rate:.0%}, "
                    f"error={metrics.formation_error:.1f}m, "
                    f"progress={metrics.task_progress:.0%}"
                )

        return self.metrics_history

    def get_final_metrics(self) -> Dict:
        """Get summary metrics from simulation"""
        if not self.metrics_history:
            return {}

        final = self.metrics_history[-1]

        # Compute averages over last 10% of simulation
        last_n = max(1, len(self.metrics_history) // 10)
        recent = self.metrics_history[-last_n:]

        return {
            "final_cooperation": final.cooperation_rate,
            "final_formation_error": final.formation_error,
            "final_task_progress": final.task_progress,
            "total_collisions": self.collision_count,
            "avg_formation_error": np.mean([m.formation_error for m in recent]),
            "avg_speed": np.mean([m.avg_speed for m in recent]),
            "final_energy": final.avg_energy,
            "convergence_step": self._find_convergence_step()
        }

    def _find_convergence_step(self, threshold: float = 0.9) -> int:
        """Find step where task progress first exceeds threshold"""
        for metrics in self.metrics_history:
            if metrics.task_progress >= threshold:
                return metrics.step
        return -1  # Never converged


def run_comparison(
    num_drones: int = 100,
    duration: float = 60.0,
    tau_values: List[float] = None,
    gamma: float = 1.5
) -> Dict[str, Dict]:
    """
    Run comparison experiments across tau values.
    """
    if tau_values is None:
        tau_values = [0.0, 0.3, 0.5]

    results = {}

    for tau in tau_values:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running with tau={tau}")
        logger.info(f"{'='*50}")

        config = SwarmConfig(
            num_drones=num_drones,
            gamma=gamma,
            tau=tau,
            formation="grid"
        )

        sim = SwarmSimulator(config)
        sim.initialize()
        sim.run(duration=duration, verbose=False)

        metrics = sim.get_final_metrics()
        results[f"tau_{tau}"] = metrics

        logger.info(f"  Cooperation: {metrics['final_cooperation']:.0%}")
        logger.info(f"  Formation error: {metrics['avg_formation_error']:.1f}m")
        logger.info(f"  Task progress: {metrics['final_task_progress']:.0%}")
        logger.info(f"  Collisions: {metrics['total_collisions']}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='BTUT Drone Swarm Simulation')
    parser.add_argument('--drones', type=int, default=100, help='Number of drones')
    parser.add_argument('--duration', type=float, default=60.0, help='Simulation duration (s)')
    parser.add_argument('--gamma', type=float, default=1.5, help='Cooperation bonus')
    parser.add_argument('--tau', type=float, default=0.3, help='Hub influence')
    parser.add_argument('--formation', type=str, default='grid',
                       choices=['grid', 'circle', 'v', 'random'])
    parser.add_argument('--compare', action='store_true', help='Run tau comparison')
    parser.add_argument('--output', type=str, default='swarm_results.json')

    args = parser.parse_args()

    if args.compare:
        results = run_comparison(
            num_drones=args.drones,
            duration=args.duration,
            tau_values=[0.0, 0.3, 0.5],
            gamma=args.gamma
        )
    else:
        config = SwarmConfig(
            num_drones=args.drones,
            gamma=args.gamma,
            tau=args.tau,
            formation=args.formation
        )

        sim = SwarmSimulator(config)
        sim.initialize()
        sim.run(duration=args.duration)

        results = {"single_run": sim.get_final_metrics()}

    # Print summary
    print("\n" + "="*60)
    print("SWARM SIMULATION RESULTS")
    print("="*60)

    for key, metrics in results.items():
        print(f"\n{key}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
