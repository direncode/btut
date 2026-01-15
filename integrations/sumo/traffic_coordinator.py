"""
SUMO Traffic Coordinator

Integrates BTUT with SUMO traffic simulator for coordinated vehicle behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python-sdk'))

import traci
import numpy as np
from btut import Simulator
import time
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficCoordinator:
    """Coordinates traffic using BTUT and SUMO"""

    def __init__(self, sumo_config: str, gamma: float = 1.5, tau: float = 0.3):
        """
        Initialize traffic coordinator

        Args:
            sumo_config: Path to SUMO configuration file
            gamma: Cooperation bonus parameter
            tau: Hub influence parameter
        """
        self.sumo_config = sumo_config
        self.gamma = gamma
        self.tau = tau

        # State
        self.vehicle_strategies = {}
        self.coordination_history = []
        self.current_cooperation = 0.5

        # Statistics
        self.total_waiting_time = 0.0
        self.total_travel_time = 0.0
        self.num_collisions = 0

        # Start SUMO
        self._start_sumo()

    def _start_sumo(self):
        """Start SUMO simulation"""
        try:
            traci.start(["sumo-gui", "-c", self.sumo_config,
                        "--start", "--quit-on-end"])
            logger.info("SUMO started successfully")
        except Exception as e:
            logger.error(f"Failed to start SUMO: {e}")
            raise

    def get_vehicle_count(self) -> int:
        """Get current number of vehicles"""
        return traci.vehicle.getIDCount()

    def get_vehicles(self) -> List[str]:
        """Get list of all vehicle IDs"""
        return traci.vehicle.getIDList()

    def run_coordination(self) -> Dict:
        """
        Run BTUT coordination simulation

        Returns:
            Dictionary with coordination results
        """
        num_vehicles = self.get_vehicle_count()

        if num_vehicles == 0:
            return {
                'cooperation': 0.5,
                'converged': False,
                'num_vehicles': 0
            }

        # Run BTUT simulation
        sim = Simulator(
            agents=max(num_vehicles, 10),  # Minimum 10 for stability
            gamma=self.gamma,
            tau=self.tau,
            alpha=0.1
        )

        result = sim.run()

        self.current_cooperation = result.final_cooperation
        self.coordination_history.append(result.final_cooperation)

        logger.info(f"Coordination: {result.final_cooperation:.2%} cooperation, "
                   f"{num_vehicles} vehicles")

        return {
            'cooperation': result.final_cooperation,
            'converged': result.converged,
            'num_vehicles': num_vehicles,
            'iterations': result.iterations_completed
        }

    def assign_strategies(self, cooperation_rate: float):
        """
        Assign strategies to vehicles based on cooperation rate

        Args:
            cooperation_rate: Target cooperation rate from BTUT
        """
        vehicles = self.get_vehicles()

        if not vehicles:
            return

        # Sort vehicles by some criterion (e.g., speed, position)
        vehicle_speeds = [(vid, traci.vehicle.getSpeed(vid)) for vid in vehicles]
        vehicle_speeds.sort(key=lambda x: x[1], reverse=True)

        # Assign strategies: faster vehicles are more competitive
        num_cooperative = int(cooperation_rate * len(vehicles))

        for i, (vid, _) in enumerate(vehicle_speeds):
            if i >= num_cooperative:
                self.vehicle_strategies[vid] = "A"  # Cooperative
            else:
                self.vehicle_strategies[vid] = "B"  # Competitive

    def apply_strategies(self):
        """Apply assigned strategies to vehicle behaviors"""
        for vid in self.get_vehicles():
            if vid not in self.vehicle_strategies:
                continue

            strategy = self.vehicle_strategies[vid]

            try:
                if strategy == "A":
                    # Cooperative: maintain safe speed, respect others
                    target_speed = 13.89  # 50 km/h in m/s
                    traci.vehicle.setSpeed(vid, target_speed)
                    traci.vehicle.setSpeedMode(vid, 0b011111)  # All safety checks
                else:
                    # Competitive: faster, more aggressive
                    target_speed = 19.44  # 70 km/h in m/s
                    traci.vehicle.setSpeed(vid, target_speed)
                    traci.vehicle.setSpeedMode(vid, 0b010111)  # Less conservative

            except traci.exceptions.TraCIException as e:
                logger.warning(f"Failed to set speed for vehicle {vid}: {e}")

    def collect_statistics(self):
        """Collect traffic statistics"""
        vehicles = self.get_vehicles()

        total_waiting = 0.0
        total_speed = 0.0

        for vid in vehicles:
            try:
                waiting_time = traci.vehicle.getWaitingTime(vid)
                speed = traci.vehicle.getSpeed(vid)

                total_waiting += waiting_time
                total_speed += speed

            except traci.exceptions.TraCIException:
                pass

        self.total_waiting_time += total_waiting

        return {
            'num_vehicles': len(vehicles),
            'avg_waiting_time': total_waiting / len(vehicles) if vehicles else 0,
            'avg_speed': total_speed / len(vehicles) if vehicles else 0,
            'total_waiting': total_waiting
        }

    def step(self):
        """Execute one simulation step"""
        try:
            traci.simulationStep()
            return True
        except traci.exceptions.TraCIException:
            return False

    def run(self, steps: int = 3600, coordination_interval: int = 100):
        """
        Run coordinated traffic simulation

        Args:
            steps: Number of simulation steps
            coordination_interval: Steps between coordination updates
        """
        logger.info(f"Running coordinated traffic simulation for {steps} steps")

        stats_history = []

        for step in range(steps):
            # Simulation step
            if not self.step():
                logger.info("Simulation ended")
                break

            # Periodic coordination
            if step % coordination_interval == 0:
                # Run coordination
                coord_result = self.run_coordination()

                # Assign and apply strategies
                self.assign_strategies(coord_result['cooperation'])
                self.apply_strategies()

                # Collect statistics
                stats = self.collect_statistics()
                stats['step'] = step
                stats['cooperation'] = coord_result['cooperation']
                stats_history.append(stats)

                logger.info(f"Step {step}: {stats['num_vehicles']} vehicles, "
                           f"avg wait: {stats['avg_waiting_time']:.2f}s, "
                           f"avg speed: {stats['avg_speed']:.2f} m/s, "
                           f"cooperation: {coord_result['cooperation']:.2%}")

        # Final statistics
        self.print_summary(stats_history)

        return stats_history

    def print_summary(self, stats_history: List[Dict]):
        """Print simulation summary"""
        if not stats_history:
            return

        avg_cooperation = np.mean([s['cooperation'] for s in stats_history])
        avg_vehicles = np.mean([s['num_vehicles'] for s in stats_history])
        avg_waiting = np.mean([s['avg_waiting_time'] for s in stats_history])
        avg_speed = np.mean([s['avg_speed'] for s in stats_history])

        logger.info("\n" + "="*60)
        logger.info("SIMULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Average cooperation rate: {avg_cooperation:.2%}")
        logger.info(f"Average vehicles: {avg_vehicles:.1f}")
        logger.info(f"Average waiting time: {avg_waiting:.2f}s")
        logger.info(f"Average speed: {avg_speed:.2f} m/s ({avg_speed*3.6:.1f} km/h)")
        logger.info("="*60)

    def close(self):
        """Close SUMO connection"""
        try:
            traci.close()
            logger.info("SUMO connection closed")
        except:
            pass


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='BTUT-SUMO Traffic Coordinator')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to SUMO config file')
    parser.add_argument('--steps', type=int, default=3600,
                       help='Number of simulation steps')
    parser.add_argument('--gamma', type=float, default=1.5,
                       help='Cooperation bonus')
    parser.add_argument('--tau', type=float, default=0.3,
                       help='Hub influence')
    parser.add_argument('--interval', type=int, default=100,
                       help='Coordination update interval')

    args = parser.parse_args()

    # Create coordinator
    coordinator = TrafficCoordinator(
        sumo_config=args.config,
        gamma=args.gamma,
        tau=args.tau
    )

    try:
        # Run simulation
        stats = coordinator.run(
            steps=args.steps,
            coordination_interval=args.interval
        )

        # Save results
        import json
        with open('traffic_results.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("Results saved to traffic_results.json")

    finally:
        coordinator.close()


if __name__ == '__main__':
    main()
