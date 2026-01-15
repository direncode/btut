#!/usr/bin/env python3
"""
BTUT ROS 2 Bridge for Drone Swarms

Integrates BTUT coordination with ROS 2 / Gazebo multi-UAV simulation.

Requirements:
- ROS 2 Humble or Iron
- Gazebo Garden/Harmonic
- PX4 or ArduPilot SITL (optional, for realistic flight dynamics)

Usage (with ROS 2):
    ros2 run btut_swarm btut_coordinator --ros-args -p num_drones:=10 -p tau:=0.3

Usage (standalone mock):
    python ros2_bridge.py --mock --drones 50

Architecture:
    +------------------+     +------------------+
    |  BTUT Coordinator|     |  Gazebo / SITL   |
    |  (This Node)     |<--->|  (Physics)       |
    +------------------+     +------------------+
           |                        |
           v                        v
    +------------------+     +------------------+
    | /btut/strategies |     | /drone_N/pose    |
    | /btut/metrics    |     | /drone_N/cmd_vel |
    +------------------+     +------------------+
"""

import sys
import os
import time
import json
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

# Add core module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np

from btut_swarm_core import (
    DroneState, Strategy, SwarmConfig, SwarmMetrics,
    SpatialGraph, BTUTSwarmCoordinator, FormationController
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Try to import ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped, Twist, Point
    from std_msgs.msg import String, Float32
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("ROS 2 not available. Running in mock mode only.")


@dataclass
class ROS2Config:
    """ROS 2 specific configuration"""
    node_name: str = "btut_coordinator"
    pose_topic_prefix: str = "/drone_"
    cmd_topic_prefix: str = "/drone_"
    coordination_rate: float = 10.0  # Hz
    publish_metrics: bool = True


class BTUTCoordinatorNode:
    """
    ROS 2 Node for BTUT swarm coordination.

    Subscribes to drone poses, runs BTUT coordination,
    publishes velocity commands and metrics.
    """

    def __init__(
        self,
        swarm_config: SwarmConfig,
        ros_config: ROS2Config = None
    ):
        self.swarm_config = swarm_config
        self.ros_config = ros_config or ROS2Config()

        # State
        self.drones: Dict[int, DroneState] = {}
        self.spatial_graph = SpatialGraph(swarm_config.comm_range)
        self.coordinator = BTUTSwarmCoordinator(
            swarm_config.gamma, swarm_config.tau
        )
        self.formation_controller = FormationController(swarm_config)

        # Initialize formation targets
        self.formation_controller.generate_formation(
            swarm_config.num_drones,
            np.array([0, 0, 10])  # Default center
        )

        self._running = False

        if ROS2_AVAILABLE:
            self._init_ros2()
        else:
            logger.info("Running without ROS 2")

    def _init_ros2(self):
        """Initialize ROS 2 node"""
        rclpy.init()
        self.node = rclpy.create_node(self.ros_config.node_name)

        # QoS for reliable communication
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE

        # Create subscribers for each drone
        self.pose_subs = {}
        self.cmd_pubs = {}

        for i in range(self.swarm_config.num_drones):
            # Subscribe to pose
            topic = f"{self.ros_config.pose_topic_prefix}{i}/pose"
            self.pose_subs[i] = self.node.create_subscription(
                PoseStamped,
                topic,
                lambda msg, drone_id=i: self._pose_callback(drone_id, msg),
                qos
            )

            # Publisher for velocity commands
            cmd_topic = f"{self.ros_config.cmd_topic_prefix}{i}/cmd_vel"
            self.cmd_pubs[i] = self.node.create_publisher(Twist, cmd_topic, qos)

            # Initialize drone state
            self.drones[i] = DroneState(
                id=i,
                position=np.zeros(3),
                velocity=np.zeros(3),
                target=self.formation_controller.formation_targets.get(
                    i, np.zeros(3)
                )
            )

        # Metrics publisher
        if self.ros_config.publish_metrics:
            self.metrics_pub = self.node.create_publisher(
                String, "/btut/metrics", qos
            )
            self.strategies_pub = self.node.create_publisher(
                String, "/btut/strategies", qos
            )

        # Coordination timer
        period = 1.0 / self.ros_config.coordination_rate
        self.coord_timer = self.node.create_timer(
            period, self._coordination_callback
        )

        logger.info(f"ROS 2 node initialized: {self.ros_config.node_name}")

    def _pose_callback(self, drone_id: int, msg):
        """Handle incoming pose message"""
        if drone_id in self.drones:
            self.drones[drone_id].position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])

    def _coordination_callback(self):
        """Run BTUT coordination (called periodically)"""
        # Compute neighbors and degrees
        degrees = self.spatial_graph.compute_degrees(self.drones)
        neighbors_map = self.spatial_graph.compute_neighbors(self.drones)

        # Run BTUT strategy update
        strategies, coop_rate = self.coordinator.update_strategies(
            self.drones, degrees
        )

        # Apply strategies
        for drone_id, strategy in strategies.items():
            self.drones[drone_id].strategy = strategy

        # Compute and publish velocity commands
        for drone_id, drone in self.drones.items():
            neighbor_ids = neighbors_map.get(drone_id, [])
            neighbors = [self.drones[nid] for nid in neighbor_ids]

            velocity = self.formation_controller.compute_velocity(
                drone, neighbors
            )

            # Publish command
            if drone_id in self.cmd_pubs:
                cmd = Twist()
                cmd.linear.x = float(velocity[0])
                cmd.linear.y = float(velocity[1])
                cmd.linear.z = float(velocity[2])
                self.cmd_pubs[drone_id].publish(cmd)

        # Publish metrics
        if self.ros_config.publish_metrics:
            self._publish_metrics(coop_rate, strategies)

    def _publish_metrics(self, coop_rate: float, strategies: Dict):
        """Publish coordination metrics"""
        metrics = {
            "timestamp": time.time(),
            "cooperation_rate": coop_rate,
            "num_cooperative": sum(
                1 for s in strategies.values() if s == Strategy.COOPERATIVE
            ),
            "num_competitive": sum(
                1 for s in strategies.values() if s == Strategy.COMPETITIVE
            )
        }

        msg = String()
        msg.data = json.dumps(metrics)
        self.metrics_pub.publish(msg)

        # Strategies
        strat_msg = String()
        strat_msg.data = json.dumps({
            str(k): v.value for k, v in strategies.items()
        })
        self.strategies_pub.publish(strat_msg)

    def spin(self):
        """Run ROS 2 event loop"""
        if ROS2_AVAILABLE:
            self._running = True
            rclpy.spin(self.node)

    def shutdown(self):
        """Clean shutdown"""
        self._running = False
        if ROS2_AVAILABLE:
            self.node.destroy_node()
            rclpy.shutdown()


class MockROS2Bridge:
    """
    Mock ROS 2 bridge for testing without ROS 2 installed.

    Simulates the ROS 2 communication and Gazebo physics.
    """

    def __init__(self, swarm_config: SwarmConfig):
        self.swarm_config = swarm_config

        # Import full simulator
        from btut_swarm_core import SwarmSimulator

        self.simulator = SwarmSimulator(swarm_config)
        self.simulator.initialize()

    def run(self, duration: float = 60.0, dt: float = 0.1):
        """Run mock simulation"""
        logger.info("Running mock ROS 2 simulation...")
        return self.simulator.run(duration=duration, dt=dt)

    def get_results(self) -> Dict:
        """Get simulation results"""
        return self.simulator.get_final_metrics()


def create_gazebo_world(num_drones: int, output_file: str):
    """
    Generate Gazebo world file for multi-UAV simulation.

    This is a template - actual UAV models depend on your Gazebo setup.
    """
    world_template = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="btut_swarm_world">

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Spawn drones in circle -->
'''

    radius = 20 + num_drones * 0.5

    for i in range(num_drones):
        angle = 2 * 3.14159 * i / num_drones
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.5

        world_template += f'''
    <!-- Drone {i} -->
    <model name="drone_{i}">
      <include>
        <uri>model://iris_with_standoffs</uri>
        <pose>{x} {y} {z} 0 0 0</pose>
      </include>
    </model>
'''

    world_template += '''
  </world>
</sdf>
'''

    with open(output_file, 'w') as f:
        f.write(world_template)

    logger.info(f"Gazebo world file written to {output_file}")


def create_launch_file(num_drones: int, output_file: str):
    """
    Generate ROS 2 launch file for BTUT swarm.
    """
    launch_template = f'''"""
BTUT Drone Swarm Launch File

Launches:
- Gazebo simulation with {num_drones} drones
- BTUT coordinator node
- Visualization (optional)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'num_drones',
            default_value='{num_drones}',
            description='Number of drones in swarm'
        ),
        DeclareLaunchArgument(
            'tau',
            default_value='0.3',
            description='BTUT hub influence parameter'
        ),
        DeclareLaunchArgument(
            'gamma',
            default_value='1.5',
            description='BTUT cooperation bonus'
        ),

        # Gazebo simulation
        ExecuteProcess(
            cmd=['gazebo', '--verbose', 'btut_swarm.world'],
            output='screen'
        ),

        # BTUT Coordinator
        Node(
            package='btut_swarm',
            executable='btut_coordinator',
            name='btut_coordinator',
            parameters=[{{
                'num_drones': LaunchConfiguration('num_drones'),
                'tau': LaunchConfiguration('tau'),
                'gamma': LaunchConfiguration('gamma'),
            }}],
            output='screen'
        ),

        # Metrics visualization (optional)
        # Node(
        #     package='btut_swarm',
        #     executable='metrics_visualizer',
        #     name='metrics_vis',
        #     output='screen'
        # ),
    ])
'''

    with open(output_file, 'w') as f:
        f.write(launch_template)

    logger.info(f"Launch file written to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='BTUT ROS 2 Drone Swarm Bridge')
    parser.add_argument('--mock', action='store_true', help='Run mock simulation')
    parser.add_argument('--drones', type=int, default=50, help='Number of drones')
    parser.add_argument('--duration', type=float, default=60.0, help='Duration (s)')
    parser.add_argument('--gamma', type=float, default=1.5, help='Cooperation bonus')
    parser.add_argument('--tau', type=float, default=0.3, help='Hub influence')
    parser.add_argument('--generate-world', action='store_true',
                       help='Generate Gazebo world file')
    parser.add_argument('--generate-launch', action='store_true',
                       help='Generate ROS 2 launch file')
    parser.add_argument('--output', type=str, default='ros2_swarm_results.json')

    args = parser.parse_args()

    config = SwarmConfig(
        num_drones=args.drones,
        gamma=args.gamma,
        tau=args.tau
    )

    if args.generate_world:
        create_gazebo_world(args.drones, "btut_swarm.world")
        return

    if args.generate_launch:
        create_launch_file(args.drones, "btut_swarm_launch.py")
        return

    if args.mock or not ROS2_AVAILABLE:
        # Run mock simulation
        bridge = MockROS2Bridge(config)
        bridge.run(duration=args.duration)
        results = bridge.get_results()

        print("\n" + "="*50)
        print("MOCK SWARM RESULTS")
        print("="*50)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    else:
        # Run real ROS 2 node
        ros_config = ROS2Config()
        coordinator = BTUTCoordinatorNode(config, ros_config)

        try:
            coordinator.spin()
        except KeyboardInterrupt:
            pass
        finally:
            coordinator.shutdown()


if __name__ == '__main__':
    main()
