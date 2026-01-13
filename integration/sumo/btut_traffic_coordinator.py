"""
BTUT + SUMO Traffic Simulation
===============================
Integration with SUMO (Simulation of Urban MObility) for traffic coordination.

Use Case: Coordinate autonomous vehicles at intersections to optimize flow
without traditional traffic signals.

Requirements:
    - SUMO installed (sumo.dlr.de)
    - traci (SUMO Python API)

Example:
    python btut_sumo_integration.py --network city.net.xml --routes traffic.rou.xml
"""

import traci
import sumolib
import numpy as np
from typing import Dict, List, Tuple
import time

# Import BTUT
import sys
sys.path.append('../../lib/simulation')
from btut_engine import BTUTSimulator


class BTUTTrafficCoordinator:
    """
    Coordinate autonomous vehicles using BTUT
    
    Strategy A: Cooperative driving (yielding, platooning)
    Strategy B: Aggressive driving (prioritize own progress)
    """
    
    def __init__(self, 
                 sumo_config: str,
                 coordination_radius: float = 50.0,
                 update_interval: int = 10):
        """
        Initialize traffic coordinator
        
        Args:
            sumo_config: Path to SUMO config file
            coordination_radius: Distance for neighbor detection (meters)
            update_interval: Steps between BTUT updates
        """
        
        self.config_file = sumo_config
        self.coord_radius = coordination_radius
        self.update_interval = update_interval
        
        # Vehicle states
        self.vehicle_strategies: Dict[str, str] = {}
        self.vehicle_positions: Dict[str, Tuple[float, float]] = {}
        self.vehicle_speeds: Dict[str, float] = {}
        
        # BTUT simulator
        self.btut = None
        self.step_count = 0
        
        # Metrics
        self.metrics = {
            'avg_speed': [],
            'cooperation_level': [],
            'collision_count': 0,
            'throughput': []
        }
    
    def start_simulation(self, gui: bool = False):
        """Start SUMO simulation"""
        
        if gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')
        
        traci.start([sumo_binary, "-c", self.config_file])
        print(f"SUMO simulation started")
    
    def run(self, max_steps: int = 3600):
        """
        Run coordinated traffic simulation
        
        Args:
            max_steps: Maximum simulation steps (1 step = 1 second)
        """
        
        print("="*60)
        print("BTUT Traffic Coordination")
        print("="*60)
        
        for step in range(max_steps):
            # Advance SUMO
            traci.simulationStep()
            self.step_count = step
            
            # Update vehicle information
            self.update_vehicle_data()
            
            # Coordinate using BTUT every N steps
            if step % self.update_interval == 0:
                self.coordinate_vehicles()
            
            # Apply strategies to vehicle behavior
            self.apply_strategies()
            
            # Collect metrics
            self.collect_metrics()
            
            # Print progress
            if step % 100 == 0:
                self.print_status()
        
        # Finalize
        traci.close()
        self.print_summary()
    
    def update_vehicle_data(self):
        """Update positions and speeds of all vehicles"""
        
        vehicle_ids = traci.vehicle.getIDList()
        
        for veh_id in vehicle_ids:
            # Position
            x, y = traci.vehicle.getPosition(veh_id)
            self.vehicle_positions[veh_id] = (x, y)
            
            # Speed
            speed = traci.vehicle.getSpeed(veh_id)
            self.vehicle_speeds[veh_id] = speed
            
            # Initialize strategy if new
            if veh_id not in self.vehicle_strategies:
                self.vehicle_strategies[veh_id] = 'B'  # Start aggressive
    
    def coordinate_vehicles(self):
        """Run BTUT to determine coordination strategies"""
        
        vehicle_ids = list(self.vehicle_positions.keys())
        N = len(vehicle_ids)
        
        if N < 2:
            return
        
        # Build neighbor graph
        neighbor_counts = self.compute_neighbor_counts(vehicle_ids)
        
        # BTUT configuration
        config = {
            'N': N,
            'gamma': 1.6,  # High cooperation reward for traffic
            'tau': 0.40,   # Moderate hub influence
            'cA_SH': 0.30,
            'cB_SH': 0.10,
            'cA_PD': 0.20,
            'cB_PD': 0.08,
            'alpha': 0.70,  # High cost for aggression
            'iterations': 1,  # Single update
            'm': int(np.mean(neighbor_counts)) if neighbor_counts else 3,
            'seed': self.step_count
        }
        
        # Initialize or update BTUT
        if self.btut is None:
            self.btut = BTUTSimulator(config)
        
        # Run BTUT step
        state = self.btut.step()
        
        # Update strategies
        agents = state['agents']
        for i, veh_id in enumerate(vehicle_ids):
            if i < len(agents):
                self.vehicle_strategies[veh_id] = agents[i]['strategy']
    
    def compute_neighbor_counts(self, vehicle_ids: List[str]) -> List[int]:
        """Count neighbors within coordination radius"""
        
        counts = []
        
        for veh_id in vehicle_ids:
            pos = self.vehicle_positions[veh_id]
            neighbors = 0
            
            for other_id in vehicle_ids:
                if other_id == veh_id:
                    continue
                
                other_pos = self.vehicle_positions[other_id]
                dist = np.linalg.norm(
                    np.array(pos) - np.array(other_pos)
                )
                
                if dist < self.coord_radius:
                    neighbors += 1
            
            counts.append(neighbors)
        
        return counts
    
    def apply_strategies(self):
        """Apply coordination strategies to vehicle behavior"""
        
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in self.vehicle_strategies:
                continue
            
            strategy = self.vehicle_strategies[veh_id]
            
            if strategy == 'A':  # Cooperative
                # Reduce speed variation
                traci.vehicle.setSpeedMode(veh_id, 31)
                
                # Increase safety gap
                traci.vehicle.setTau(veh_id, 2.0)
                
                # Enable cooperative lane change
                traci.vehicle.setLaneChangeMode(veh_id, 512)
            
            else:  # Aggressive (B)
                # Normal speed variation
                traci.vehicle.setSpeedMode(veh_id, 30)
                
                # Reduce safety gap
                traci.vehicle.setTau(veh_id, 1.0)
                
                # Normal lane change
                traci.vehicle.setLaneChangeMode(veh_id, 597)
    
    def collect_metrics(self):
        """Collect performance metrics"""
        
        if not self.vehicle_speeds:
            return
        
        # Average speed
        avg_speed = np.mean(list(self.vehicle_speeds.values()))
        self.metrics['avg_speed'].append(avg_speed)
        
        # Cooperation level
        coop_count = sum(1 for s in self.vehicle_strategies.values() if s == 'A')
        total = len(self.vehicle_strategies)
        coop_level = coop_count / total if total > 0 else 0
        self.metrics['cooperation_level'].append(coop_level)
        
        # Throughput (vehicles passed)
        arrived = traci.simulation.getArrivedNumber()
        self.metrics['throughput'].append(arrived)
    
    def print_status(self):
        """Print current status"""
        
        if not self.metrics['avg_speed']:
            return
        
        avg_speed = self.metrics['avg_speed'][-1]
        coop = self.metrics['cooperation_level'][-1]
        total_veh = len(self.vehicle_positions)
        
        print(f"Step {self.step_count:5d} | "
              f"Vehicles: {total_veh:3d} | "
              f"Avg Speed: {avg_speed:5.2f} m/s | "
              f"Cooperation: {coop:5.2%}")
    
    def print_summary(self):
        """Print final summary"""
        
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        
        if self.metrics['avg_speed']:
            print(f"Average Speed: {np.mean(self.metrics['avg_speed']):.2f} m/s")
            print(f"Average Cooperation: {np.mean(self.metrics['cooperation_level']):.2%}")
            print(f"Total Throughput: {self.metrics['throughput'][-1] if self.metrics['throughput'] else 0}")
        
        print("="*60)
    
    def export_results(self, filename: str = 'traffic_results.json'):
        """Export metrics to file"""
        import json
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Results exported to {filename}")


def create_sample_network():
    """Create a simple test network"""
    
    network_xml = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <edge id="e1" from="n1" to="n2" priority="-1">
        <lane id="e1_0" index="0" speed="13.89" length="500.00" shape="0.00,0.00 500.00,0.00"/>
    </edge>
    <edge id="e2" from="n2" to="n3" priority="-1">
        <lane id="e2_0" index="0" speed="13.89" length="500.00" shape="500.00,0.00 1000.00,0.00"/>
    </edge>
</net>
"""
    
    with open('test_network.net.xml', 'w') as f:
        f.write(network_xml)
    
    print("Sample network created: test_network.net.xml")


def main():
    """Run BTUT traffic coordination"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='BTUT Traffic Coordination')
    parser.add_argument('--config', default='city.sumocfg', 
                       help='SUMO config file')
    parser.add_argument('--gui', action='store_true', 
                       help='Use SUMO GUI')
    parser.add_argument('--steps', type=int, default=3600,
                       help='Simulation duration (seconds)')
    parser.add_argument('--radius', type=float, default=50.0,
                       help='Coordination radius (meters)')
    
    args = parser.parse_args()
    
    # Create coordinator
    coordinator = BTUTTrafficCoordinator(
        sumo_config=args.config,
        coordination_radius=args.radius
    )
    
    # Run simulation
    coordinator.start_simulation(gui=args.gui)
    coordinator.run(max_steps=args.steps)
    coordinator.export_results()


if __name__ == '__main__':
    main()
