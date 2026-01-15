# BTUT Drone Swarm Integration

Coordinate drone swarms using BTUT game-theoretic framework for formation control, collision avoidance, and cooperative task execution.

## Quick Start

```bash
# Check setup
python quickstart.py --check

# Run basic demo (50 drones, 30s)
python quickstart.py

# Compare tau values
python quickstart.py --compare --drones 100

# Scale test
python quickstart.py --scale 500

# Generate visualization
python quickstart.py --visualize
```

## Architecture

```
                    +------------------+
                    |  BTUT Coordinator|
                    |  (Game Theory)   |
                    +------------------+
                           |
          +----------------+----------------+
          |                                 |
   +------+------+                   +------+------+
   | Pure Python |                   |   ROS 2     |
   | Simulator   |                   |   Bridge    |
   +-------------+                   +-------------+
          |                                 |
          v                                 v
   [Standalone]                     [Gazebo/SITL]
```

## Strategy Mapping

| BTUT Strategy | Drone Behavior |
|--------------|----------------|
| **Cooperative (A)** | Formation-keeping, yield to neighbors, smooth trajectories |
| **Competitive (B)** | Goal-seeking, aggressive pathing, minimal collision avoidance |

## Key Concepts

### Hub Influence (Tau)

Drones with more neighbors (hubs) have greater influence on collective behavior:
- **tau=0**: All drones weighted equally
- **tau=0.3**: Moderate hub influence (recommended)
- **tau=0.5+**: Strong hub influence

### Spatial Communication Graph

Unlike SUMO (road network), drone neighbors are determined by **spatial proximity**:
- Drones within `comm_range` (default 50m) are neighbors
- Degree = number of neighbors
- Hubs = drones in dense clusters

## Metrics

| Metric | Description |
|--------|-------------|
| **Formation Error** | RMS distance from target positions |
| **Collision Rate** | Violations of safety distance |
| **Task Progress** | Fraction of drones at targets |
| **Energy** | Remaining battery (simple model) |

## Files

```
integrations/drone_swarm/
├── README.md              # This file
├── quickstart.py          # Quick start and validation
├── btut_swarm_core.py     # Core simulation (pure Python)
├── ros2_bridge.py         # ROS 2 / Gazebo integration
└── swarm_results.json     # Output (generated)
```

## ROS 2 Integration

### Requirements

- ROS 2 Humble or Iron
- Gazebo Garden or Harmonic
- PX4 or ArduPilot SITL (optional)

### Generate Launch Files

```bash
# Generate Gazebo world
python ros2_bridge.py --generate-world --drones 20

# Generate ROS 2 launch file
python ros2_bridge.py --generate-launch --drones 20
```

### Run with ROS 2

```bash
# Terminal 1: Launch Gazebo
ros2 launch btut_swarm btut_swarm_launch.py

# Terminal 2: Monitor metrics
ros2 topic echo /btut/metrics
```

## Expected Results

Based on BTUT theory, cooperative swarms should show:

| Configuration | Formation Error | Collisions | Progress |
|--------------|-----------------|------------|----------|
| No BTUT (random) | ~30-50m | High | ~50% |
| BTUT tau=0 | ~15-25m | Medium | ~70% |
| BTUT tau=0.3 | ~8-15m | Low | ~85% |
| BTUT tau=0.5 | ~5-12m | Very Low | ~90% |

## Alternatives to Gazebo

For larger swarms (1000+):

1. **Webots** - Good multi-robot support, Python API
2. **CoppeliaSim (V-REP)** - Less resource-heavy
3. **AirSim** - High-fidelity UAV, Unreal Engine

The `btut_swarm_core.py` works standalone without any simulator.

## Scaling

| Drones | Platform | Notes |
|--------|----------|-------|
| 50-100 | Laptop (pure Python) | Real-time capable |
| 100-500 | Desktop | May need headless |
| 500-1000 | Cloud | Distributed recommended |
| 1000+ | Multi-machine | Use ROS 2 distributed |

## Citation

```bibtex
@article{btut_swarm2025,
  title={BTUT: Hub-Mediated Coordination for Drone Swarms},
  author={BTUT Team},
  year={2025}
}
```
