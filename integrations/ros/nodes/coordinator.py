#!/usr/bin/env python3
"""
Robot Swarm Coordinator

Coordinates robot behaviors based on BTUT simulations.
Uses robot positions and states to determine optimal coordination strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python-sdk'))

import rospy
import numpy as np
from btut import Simulator

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from btut_ros.msg import CoordinationResult
from btut_ros.srv import RunSimulation, RunSimulationRequest


class SwarmCoordinator:
    """Coordinates robot swarm using BTUT"""

    def __init__(self):
        rospy.init_node('swarm_coordinator', anonymous=False)

        # Parameters
        self.robot_namespace = rospy.get_param('~robot_namespace', '/robot')
        self.num_robots = rospy.get_param('~num_robots', 10)
        self.coordination_rate = rospy.get_param('~coordination_rate', 0.5)  # Hz
        self.gamma = rospy.get_param('~gamma', 1.5)
        self.tau = rospy.get_param('~tau', 0.3)

        # State
        self.robot_poses = {}
        self.robot_strategies = {}
        self.current_cooperation = 0.5

        # Subscribers
        self.odom_subscribers = []
        for i in range(self.num_robots):
            topic = f'{self.robot_namespace}_{i}/odom'
            sub = rospy.Subscriber(
                topic,
                Odometry,
                lambda msg, robot_id=i: self.odom_callback(msg, robot_id)
            )
            self.odom_subscribers.append(sub)

        # Publishers
        self.cmd_vel_publishers = []
        for i in range(self.num_robots):
            topic = f'{self.robot_namespace}_{i}/cmd_vel'
            pub = rospy.Publisher(topic, Twist, queue_size=10)
            self.cmd_vel_publishers.append(pub)

        # Service client
        rospy.wait_for_service('/btut/run_simulation')
        self.sim_client = rospy.ServiceProxy('/btut/run_simulation', RunSimulation)

        # Timer
        self.coordination_timer = rospy.Timer(
            rospy.Duration(1.0 / self.coordination_rate),
            self.coordination_callback
        )

        rospy.loginfo(f"Swarm Coordinator initialized for {self.num_robots} robots")

    def odom_callback(self, msg, robot_id):
        """Store robot odometry"""
        self.robot_poses[robot_id] = msg.pose.pose

    def coordination_callback(self, event):
        """Run coordination simulation and update robot behaviors"""
        try:
            # Only coordinate if we have pose info for all robots
            if len(self.robot_poses) < self.num_robots:
                rospy.logwarn_throttle(5.0, f"Waiting for all robot poses ({len(self.robot_poses)}/{self.num_robots})")
                return

            # Run BTUT simulation
            req = RunSimulationRequest()
            req.config.num_agents = self.num_robots
            req.config.gamma = self.gamma
            req.config.tau = self.tau
            req.config.alpha = 0.1
            req.config.max_iterations = 100
            req.config.threshold = 1e-6

            response = self.sim_client(req)

            if not response.success:
                rospy.logerr(f"Simulation failed: {response.message}")
                return

            # Update cooperation rate
            result = response.result
            self.current_cooperation = result.final_cooperation

            rospy.loginfo(f"Coordination update: {self.current_cooperation:.2%} cooperation")

            # Assign strategies to robots
            self.assign_strategies(self.current_cooperation)

            # Update robot behaviors
            self.update_robot_behaviors()

        except Exception as e:
            rospy.logerr(f"Coordination failed: {str(e)}")

    def assign_strategies(self, cooperation_rate):
        """Assign strategies to robots based on cooperation rate"""
        # Spatial assignment: robots closer to center are more cooperative
        robot_positions = []
        for i in range(self.num_robots):
            if i in self.robot_poses:
                pose = self.robot_poses[i]
                robot_positions.append((i, pose.position.x, pose.position.y))

        if not robot_positions:
            return

        # Calculate distances from centroid
        positions = np.array([[x, y] for _, x, y in robot_positions])
        centroid = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - centroid) for pos in positions]

        # Sort by distance
        sorted_indices = np.argsort(distances)

        # Assign strategies: closest robots are cooperative
        num_cooperative = int(cooperation_rate * len(robot_positions))

        for idx, (robot_id, _, _) in enumerate(robot_positions):
            if idx in sorted_indices[:num_cooperative]:
                self.robot_strategies[robot_id] = "A"  # Cooperative
            else:
                self.robot_strategies[robot_id] = "B"  # Competitive

    def update_robot_behaviors(self):
        """Update robot velocities based on assigned strategies"""
        for robot_id in range(self.num_robots):
            if robot_id not in self.robot_strategies:
                continue

            strategy = self.robot_strategies[robot_id]

            # Create velocity command
            cmd = Twist()

            if strategy == "A":
                # Cooperative: move slowly, maintain formation
                cmd.linear.x = 0.2  # m/s
                cmd.angular.z = 0.1  # rad/s
            else:
                # Competitive: move faster, explore more
                cmd.linear.x = 0.5  # m/s
                cmd.angular.z = 0.3  # rad/s

            # Publish command
            if robot_id < len(self.cmd_vel_publishers):
                self.cmd_vel_publishers[robot_id].publish(cmd)

    def spin(self):
        """Main loop"""
        rospy.spin()


def main():
    """Main entry point"""
    try:
        coordinator = SwarmCoordinator()
        coordinator.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Swarm Coordinator shutting down")


if __name__ == '__main__':
    main()
