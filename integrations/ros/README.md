# BTUT ROS Integration

ROS package for integrating BTUT multi-agent coordination with robot swarms.

## Overview

This package provides ROS nodes and services for running BTUT simulations and coordinating robot behaviors based on game-theoretic principles.

## Features

- **BTUT Node**: Core coordination service
- **Swarm Coordinator**: Applies coordination strategies to robot swarm
- **Custom Messages**: Agent states, simulation configs, results
- **Services**: Run simulations, get strategy recommendations
- **Launch Files**: Easy deployment

## Installation

### Prerequisites

- ROS Melodic, Noetic, or later
- Python 3.8+
- BTUT Python SDK

### Install BTUT SDK

```bash
pip3 install btut-sdk
```

### Build Package

```bash
# Navigate to your catkin workspace
cd ~/catkin_ws/src

# Clone or copy btut_ros package
cp -r /path/to/btut/integrations/ros btut_ros

# Build
cd ~/catkin_ws
catkin_make

# Source workspace
source devel/setup.bash
```

## Usage

### Launch BTUT Coordination

```bash
# Basic launch
roslaunch btut_ros btut_coordination.launch

# With custom parameters
roslaunch btut_ros btut_coordination.launch \
  num_robots:=20 \
  gamma:=2.0 \
  tau:=0.4
```

### Call Simulation Service

```bash
# Run a simulation
rosservice call /btut/run_simulation \
  "{config: {num_agents: 100, gamma: 1.5, tau: 0.3, alpha: 0.1, max_iterations: 100}}"
```

### Get Strategy Recommendation

```bash
# Get recommended strategy for a robot
rosservice call /btut/get_strategy \
  "{agent_id: 0, current_pose: {position: {x: 1.0, y: 2.0, z: 0.0}}}"
```

### Monitor Coordination Results

```bash
# Subscribe to coordination results
rostopic echo /btut/coordination_result

# Subscribe to agent states
rostopic echo /btut/agent_states
```

## Nodes

### btut_node

Core BTUT coordination service.

**Published Topics:**
- `/btut/coordination_result` (btut_ros/CoordinationResult): Simulation results
- `/btut/agent_states` (btut_ros/AgentState): Individual agent states

**Services:**
- `/btut/run_simulation` (btut_ros/RunSimulation): Run a coordination simulation
- `/btut/get_strategy` (btut_ros/GetStrategy): Get recommended strategy for an agent

**Parameters:**
- `~update_rate` (double, default: 1.0): Update frequency in Hz
- `~default_gamma` (double, default: 1.5): Default cooperation bonus
- `~default_tau` (double, default: 0.3): Default hub influence
- `~default_alpha` (double, default: 0.1): Default adaptation rate

### coordinator

Swarm coordinator that applies BTUT strategies to robots.

**Subscribed Topics:**
- `/{robot_namespace}_{i}/odom` (nav_msgs/Odometry): Robot odometry

**Published Topics:**
- `/{robot_namespace}_{i}/cmd_vel` (geometry_msgs/Twist): Velocity commands

**Parameters:**
- `~num_robots` (int, default: 10): Number of robots in swarm
- `~gamma` (double, default: 1.5): Cooperation bonus
- `~tau` (double, default: 0.3): Hub influence
- `~coordination_rate` (double, default: 0.5): Coordination update frequency
- `~robot_namespace` (string, default: "/robot"): Robot topic namespace

## Messages

### AgentState.msg

```
Header header
int32 agent_id
string strategy
geometry_msgs/Pose pose
float64 cooperation_rate
bool converged
int32 iteration
int32 degree
float64 centrality
bool is_hub
```

### SimulationConfig.msg

```
int32 num_agents
float64 gamma
float64 tau
float64 alpha
string network_type
int32 network_param
int32 max_iterations
float64 threshold
float64 initial_cooperation
```

### CoordinationResult.msg

```
Header header
string simulation_id
float64 final_cooperation
bool converged
int32 iterations_completed
float64 runtime_seconds
float64[] cooperation_history
string[] strategies
int32 num_hubs
float64 avg_degree
float64 clustering_coefficient
```

## Services

### RunSimulation.srv

**Request:**
```
SimulationConfig config
```

**Response:**
```
bool success
string message
CoordinationResult result
```

### GetStrategy.srv

**Request:**
```
int32 agent_id
geometry_msgs/Pose current_pose
float64[] neighbor_strategies
```

**Response:**
```
bool success
string recommended_strategy
float64 confidence
string reasoning
```

## Example: Coordinating Turtlebot3 Swarm

```python
#!/usr/bin/env python3
import rospy
from btut_ros.srv import RunSimulation, RunSimulationRequest

def coordinate_turtlebots():
    rospy.init_node('turtlebot_coordinator')

    # Wait for service
    rospy.wait_for_service('/btut/run_simulation')
    sim_client = rospy.ServiceProxy('/btut/run_simulation', RunSimulation)

    # Configure simulation
    req = RunSimulationRequest()
    req.config.num_agents = 5  # 5 turtlebots
    req.config.gamma = 1.8
    req.config.tau = 0.4
    req.config.alpha = 0.1
    req.config.max_iterations = 100

    # Run coordination
    response = sim_client(req)

    if response.success:
        rospy.loginfo(f"Coordination result: {response.result.final_cooperation:.2%} cooperation")
        rospy.loginfo(f"Converged in {response.result.iterations_completed} iterations")
    else:
        rospy.logerr(f"Coordination failed: {response.message}")

if __name__ == '__main__':
    coordinate_turtlebots()
```

## Integration with Gazebo

See `examples/gazebo_swarm.launch` for a complete example with simulated robots.

## Performance

- Coordination latency: < 100ms for 100 agents
- Scales linearly: O(N) complexity
- Real-time capable for swarms up to 10,000 agents

## Troubleshooting

### Service not available

```bash
# Check if btut_node is running
rosnode list | grep btut

# Check service list
rosservice list | grep btut
```

### Import errors

```bash
# Ensure BTUT SDK is installed
python3 -c "import btut; print(btut.__version__)"

# Check Python path
echo $PYTHONPATH
```

### Message generation errors

```bash
# Clean and rebuild
cd ~/catkin_ws
catkin_make clean
catkin_make
source devel/setup.bash
```

## Citation

If you use BTUT in your research, please cite:

```bibtex
@software{btut2025,
  title={BTUT: Scalable Multi-Agent Coordination Framework},
  author={BTUT Team},
  year={2025},
  url={https://btut.ai}
}
```

## Support

- Documentation: https://btut.ai/docs/integrations/ros
- Issues: https://github.com/direncode/btut/issues
- Discussions: https://github.com/direncode/btut/discussions

## License

Apache License 2.0 - see LICENSE file for details.
