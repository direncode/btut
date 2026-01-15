#!/usr/bin/env python3
"""
BTUT ROS Node

Provides coordination services for robot swarms using BTUT simulations.
"""

import sys
import os

# Add BTUT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python-sdk'))

import rospy
import numpy as np
from btut import Simulator

from btut_ros.msg import AgentState, SimulationConfig, CoordinationResult
from btut_ros.srv import RunSimulation, RunSimulationResponse
from btut_ros.srv import GetStrategy, GetStrategyResponse
from std_msgs.msg import Header


class BTUTNode:
    """ROS node for BTUT multi-agent coordination"""

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('btut_coordination', anonymous=False)

        # Parameters
        self.update_rate = rospy.get_param('~update_rate', 1.0)  # Hz
        self.default_gamma = rospy.get_param('~default_gamma', 1.5)
        self.default_tau = rospy.get_param('~default_tau', 0.3)
        self.default_alpha = rospy.get_param('~default_alpha', 0.1)

        # State
        self.last_result = None
        self.agent_states = {}

        # Publishers
        self.result_pub = rospy.Publisher(
            '/btut/coordination_result',
            CoordinationResult,
            queue_size=10
        )

        self.agent_state_pub = rospy.Publisher(
            '/btut/agent_states',
            AgentState,
            queue_size=100
        )

        # Services
        self.sim_service = rospy.Service(
            '/btut/run_simulation',
            RunSimulation,
            self.handle_simulation_request
        )

        self.strategy_service = rospy.Service(
            '/btut/get_strategy',
            GetStrategy,
            self.handle_strategy_request
        )

        # Timer for periodic updates
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.update_rate),
            self.periodic_update
        )

        rospy.loginfo("BTUT Node initialized and ready")

    def handle_simulation_request(self, req):
        """Handle simulation service request"""
        try:
            rospy.loginfo(f"Running simulation with {req.config.num_agents} agents")

            # Extract configuration
            config = req.config

            # Create simulator
            sim = Simulator(
                agents=config.num_agents,
                gamma=config.gamma if config.gamma > 0 else self.default_gamma,
                tau=config.tau if config.tau >= 0 else self.default_tau,
                alpha=config.alpha if config.alpha > 0 else self.default_alpha,
                iterations=config.max_iterations if config.max_iterations > 0 else 100
            )

            # Run simulation
            result = sim.run()

            # Store result
            self.last_result = result

            # Create response message
            coord_result = CoordinationResult()
            coord_result.header = Header()
            coord_result.header.stamp = rospy.Time.now()
            coord_result.simulation_id = f"sim_{rospy.Time.now().to_nsec()}"
            coord_result.final_cooperation = result.final_cooperation
            coord_result.converged = result.converged
            coord_result.iterations_completed = result.iterations_completed
            coord_result.runtime_seconds = result.runtime_seconds
            coord_result.cooperation_history = result.convergence_history

            # Publish result
            self.result_pub.publish(coord_result)

            # Create service response
            response = RunSimulationResponse()
            response.success = True
            response.message = f"Simulation completed in {result.iterations_completed} iterations"
            response.result = coord_result

            rospy.loginfo(f"Simulation complete: cooperation={result.final_cooperation:.3f}, converged={result.converged}")

            return response

        except Exception as e:
            rospy.logerr(f"Simulation failed: {str(e)}")
            response = RunSimulationResponse()
            response.success = False
            response.message = f"Error: {str(e)}"
            return response

    def handle_strategy_request(self, req):
        """Handle strategy recommendation request"""
        try:
            # If we have a recent simulation result, use it
            if self.last_result is not None:
                cooperation_rate = self.last_result.final_cooperation

                # Simple strategy: recommend A if we want more cooperation
                if cooperation_rate > 0.55:
                    strategy = "A"
                    confidence = cooperation_rate
                    reasoning = f"High cooperation rate ({cooperation_rate:.2%}), recommend strategy A"
                else:
                    strategy = "B"
                    confidence = 1.0 - cooperation_rate
                    reasoning = f"Low cooperation rate ({cooperation_rate:.2%}), recommend strategy B"

                response = GetStrategyResponse()
                response.success = True
                response.recommended_strategy = strategy
                response.confidence = confidence
                response.reasoning = reasoning

                return response
            else:
                # No simulation run yet, default to balanced
                response = GetStrategyResponse()
                response.success = True
                response.recommended_strategy = "A" if np.random.rand() > 0.5 else "B"
                response.confidence = 0.5
                response.reasoning = "No simulation data available, random strategy"
                return response

        except Exception as e:
            rospy.logerr(f"Strategy request failed: {str(e)}")
            response = GetStrategyResponse()
            response.success = False
            response.recommended_strategy = "A"
            response.confidence = 0.0
            response.reasoning = f"Error: {str(e)}"
            return response

    def periodic_update(self, event):
        """Periodic update callback"""
        # Publish current coordination state if available
        if self.last_result is not None:
            # Could publish periodic updates here
            pass

    def spin(self):
        """Main loop"""
        rospy.spin()


def main():
    """Main entry point"""
    try:
        node = BTUTNode()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("BTUT Node shutting down")


if __name__ == '__main__':
    main()
