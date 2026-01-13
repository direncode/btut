#!/usr/bin/env python3
"""
BTUT ROS Integration
====================
Integrate BTUT with Robot Operating System (ROS) for real-time
coordination of robot swarms, autonomous vehicles, and drones.

Requirements:
    - ROS Noetic or ROS2 Humble
    - rospy or rclpy

Usage:
    roslaunch btut_ros coordinator.launch
"""

import rospy
from std_msgs.msg import Float32, Int32, String
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from btut_msgs.msg import AgentState, CoordinationCommand
import numpy as np
from typing import Dict, List
import json

# Import BTUT
import sys
sys.path.append('../../lib/simulation')
from btut_engine import BTUTSimulator


class BTUTCoordinator:
    """
    ROS node that uses BTUT for multi-robot coordination
    
    Subscribes to:
        /robot_*/odom - Robot positions
        /robot_*/state - Robot internal states
    
    Publishes to:
        /coordination/strategy - Coordination strategy per robot
        /coordination/global_state - Global cooperation level
    """
    
    def __init__(self):
        rospy.init_node('btut_coordinator', anonymous=False)
        
        # Parameters
        self.num_robots = rospy.get_param('~num_robots', 10)
        self.update_rate = rospy.get_param('~update_rate', 1.0)  # Hz
        self.gamma = rospy.get_param('~gamma', 1.45)
        self.tau = rospy.get_param('~tau', 0.30)
        
        # Robot states
        self.robot_positions: Dict[str, np.ndarray] = {}
        self.robot_states: Dict[str, str] = {}  # A or B
        self.robot_neighbors: Dict[str, List[str]] = {}
        
        # BTUT simulator
        self.simulator = None
        self.current_cooperation = 0.5
        
        # Publishers
        self.global_state_pub = rospy.Publisher(
            '/coordination/global_state',
            Float32,
            queue_size=10
        )
        
        self.strategy_pubs: Dict[str, rospy.Publisher] = {}
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            self.strategy_pubs[robot_name] = rospy.Publisher(
                f'/{robot_name}/coordination/strategy',
                String,
                queue_size=10
            )
        
        # Subscribers
        for i in range(self.num_robots):
            robot_name = f'robot_{i}'
            rospy.Subscriber(
                f'/{robot_name}/odom',
                Odometry,
                self.odom_callback,
                callback_args=robot_name
            )
        
        # Timer for coordination updates
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.update_rate),
            self.coordination_callback
        )
        
        rospy.loginfo("BTUT Coordinator initialized")
        rospy.loginfo(f"  Robots: {self.num_robots}")
        rospy.loginfo(f"  Update rate: {self.update_rate} Hz")
        rospy.loginfo(f"  Gamma: {self.gamma}")
        rospy.loginfo(f"  Tau: {self.tau}")
    
    def odom_callback(self, msg: Odometry, robot_name: str):
        """Update robot position from odometry"""
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.robot_positions[robot_name] = position
        
        # Update neighbors based on proximity
        self.update_neighbors(robot_name)
    
    def update_neighbors(self, robot_name: str):
        """
        Update neighbor list based on proximity
        (within communication range)
        """
        comm_range = rospy.get_param('~comm_range', 10.0)
        
        if robot_name not in self.robot_positions:
            return
        
        pos = self.robot_positions[robot_name]
        neighbors = []
        
        for other_name, other_pos in self.robot_positions.items():
            if other_name == robot_name:
                continue
            
            distance = np.linalg.norm(pos - other_pos)
            if distance < comm_range:
                neighbors.append(other_name)
        
        self.robot_neighbors[robot_name] = neighbors
    
    def coordination_callback(self, event):
        """
        Main coordination loop using BTUT
        Called at update_rate frequency
        """
        
        if len(self.robot_positions) < self.num_robots:
            rospy.logwarn_throttle(
                10.0,
                f"Waiting for all robots: {len(self.robot_positions)}/{self.num_robots}"
            )
            return
        
        # Build network topology from neighbors
        degree_distribution = [
            len(neighbors) for neighbors in self.robot_neighbors.values()
        ]
        
        if not degree_distribution:
            return
        
        # Run BTUT simulation step
        config = {
            'N': self.num_robots,
            'gamma': self.gamma,
            'tau': self.tau,
            'cA_SH': 0.40,
            'cB_SH': 0.10,
            'cA_PD': 0.20,
            'cB_PD': 0.08,
            'alpha': 0.60,
            'iterations': 1,  # Single step update
            'm': 3,
            'seed': int(rospy.Time.now().to_sec())
        }
        
        if self.simulator is None:
            self.simulator = BTUTSimulator(config)
        
        # Update and get new state
        state = self.simulator.step()
        self.current_cooperation = state['fractionA']
        
        # Publish global cooperation level
        self.global_state_pub.publish(Float32(self.current_cooperation))
        
        # Assign strategies to robots based on BTUT output
        self.assign_strategies(state['agents'])
        
        rospy.loginfo_throttle(
            5.0,
            f"Cooperation level: {self.current_cooperation:.3f}"
        )
    
    def assign_strategies(self, agents: List[Dict]):
        """
        Assign coordination strategies to robots
        Strategy A = Cooperative behavior
        Strategy B = Individual/greedy behavior
        """
        
        for i, (robot_name, publisher) in enumerate(self.strategy_pubs.items()):
            if i < len(agents):
                strategy = agents[i]['strategy']
                publisher.publish(String(strategy))
                self.robot_states[robot_name] = strategy


class BTUTDroneSwarm:
    """
    Specialized coordinator for drone swarms
    Handles formation control and collision avoidance
    """
    
    def __init__(self):
        rospy.init_node('btut_drone_coordinator', anonymous=False)
        
        self.num_drones = rospy.get_param('~num_drones', 20)
        self.formation_type = rospy.get_param('~formation', 'grid')
        
        # Formation targets
        self.target_positions = self.generate_formation()
        
        # BTUT for coordination decision
        self.coordinator = BTUTCoordinator()
        
        rospy.loginfo(f"Drone swarm coordinator initialized")
        rospy.loginfo(f"  Drones: {self.num_drones}")
        rospy.loginfo(f"  Formation: {self.formation_type}")
    
    def generate_formation(self) -> Dict[str, np.ndarray]:
        """Generate target positions for formation"""
        
        targets = {}
        
        if self.formation_type == 'grid':
            side = int(np.ceil(np.sqrt(self.num_drones)))
            spacing = 5.0
            
            for i in range(self.num_drones):
                x = (i % side) * spacing
                y = (i // side) * spacing
                z = 10.0  # Fixed altitude
                targets[f'drone_{i}'] = np.array([x, y, z])
        
        elif self.formation_type == 'circle':
            radius = 20.0
            for i in range(self.num_drones):
                angle = 2 * np.pi * i / self.num_drones
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = 10.0
                targets[f'drone_{i}'] = np.array([x, y, z])
        
        return targets


def main():
    """Launch BTUT ROS coordinator"""
    
    try:
        coordinator_type = rospy.get_param('~type', 'ground')
        
        if coordinator_type == 'drone':
            coordinator = BTUTDroneSwarm()
        else:
            coordinator = BTUTCoordinator()
        
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
