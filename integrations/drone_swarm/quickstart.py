#!/usr/bin/env python3
"""
BTUT Drone Swarm Quick Start

Run demos and validate BTUT coordination in drone swarms.

Usage:
    python quickstart.py              # Check setup, run demo
    python quickstart.py --compare    # Compare tau values
    python quickstart.py --scale 500  # Test at larger scale
"""

import sys
import os
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np


def check_dependencies():
    """Check required dependencies"""
    print("="*60)
    print("BTUT Drone Swarm Setup Checker")
    print("="*60)

    status = {
        "python": False,
        "numpy": False,
        "matplotlib": False,
        "btut_core": False,
        "ros2": False,
    }

    # Python
    py_version = sys.version_info
    status["python"] = py_version >= (3, 8)
    print(f"\n[{'OK' if status['python'] else 'FAIL'}] Python {py_version.major}.{py_version.minor}")

    # NumPy
    try:
        import numpy
        status["numpy"] = True
        print(f"[OK] NumPy {numpy.__version__}")
    except ImportError:
        print("[FAIL] NumPy not installed")

    # Matplotlib
    try:
        import matplotlib
        status["matplotlib"] = True
        print(f"[OK] Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("[WARN] Matplotlib not installed (optional)")

    # BTUT Core
    try:
        from btut_swarm_core import SwarmSimulator
        status["btut_core"] = True
        print("[OK] BTUT Swarm Core")
    except ImportError as e:
        print(f"[FAIL] BTUT Swarm Core: {e}")

    # ROS 2
    try:
        import rclpy
        status["ros2"] = True
        print("[OK] ROS 2")
    except ImportError:
        print("[WARN] ROS 2 not available (mock mode only)")

    print("\n" + "-"*60)

    if status["python"] and status["numpy"] and status["btut_core"]:
        print("Core dependencies OK! Ready for simulation.")
        return True
    else:
        print("Missing core dependencies.")
        return False


def run_demo(num_drones: int = 50, duration: float = 30.0):
    """Run basic demonstration"""
    print("\n" + "="*60)
    print("BTUT Drone Swarm Demo")
    print("="*60)

    from btut_swarm_core import SwarmSimulator, SwarmConfig

    config = SwarmConfig(
        num_drones=num_drones,
        gamma=1.5,
        tau=0.3,
        formation="grid"
    )

    print(f"\nConfiguration:")
    print(f"  Drones: {num_drones}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Tau: {config.tau}")
    print(f"  Formation: {config.formation}")
    print(f"  Duration: {duration}s")

    sim = SwarmSimulator(config)
    sim.initialize()

    print("\nRunning simulation...")
    start = time.time()
    sim.run(duration=duration, verbose=True)
    elapsed = time.time() - start

    metrics = sim.get_final_metrics()

    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"  Cooperation rate: {metrics['final_cooperation']:.0%}")
    print(f"  Formation error: {metrics['avg_formation_error']:.1f}m")
    print(f"  Task progress: {metrics['final_task_progress']:.0%}")
    print(f"  Total collisions: {metrics['total_collisions']}")
    print(f"  Final energy: {metrics['final_energy']:.1f}%")
    print(f"  Runtime: {elapsed:.1f}s")

    return metrics


def run_comparison(num_drones: int = 100, duration: float = 60.0):
    """Compare different tau values"""
    print("\n" + "="*60)
    print("BTUT Swarm Tau Comparison")
    print("="*60)

    from btut_swarm_core import run_comparison

    results = run_comparison(
        num_drones=num_drones,
        duration=duration,
        tau_values=[0.0, 0.3, 0.5],
        gamma=1.5
    )

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Tau':<10} {'Coop':<12} {'Formation':<15} {'Progress':<12} {'Collisions':<12}")
    print("-"*70)

    for key, metrics in sorted(results.items()):
        tau = float(key.split("_")[1])
        print(f"{tau:<10.1f} {metrics['final_cooperation']:<12.0%} "
              f"{metrics['avg_formation_error']:<15.1f} "
              f"{metrics['final_task_progress']:<12.0%} "
              f"{metrics['total_collisions']:<12}")

    print("="*70)

    # Analysis
    baseline = results.get("tau_0.0", {})
    best_key = min(results.keys(), key=lambda k: results[k].get('avg_formation_error', float('inf')))
    best = results[best_key]

    if baseline and best_key != "tau_0.0":
        formation_imp = (baseline['avg_formation_error'] - best['avg_formation_error']) / baseline['avg_formation_error'] * 100 if baseline['avg_formation_error'] > 0 else 0
        collision_red = (baseline['total_collisions'] - best['total_collisions']) / max(1, baseline['total_collisions']) * 100

        print(f"\nBest: {best_key}")
        print(f"  Formation error reduction: {formation_imp:+.1f}%")
        print(f"  Collision reduction: {collision_red:+.1f}%")

    return results


def run_scale_test(max_drones: int = 500):
    """Test performance at different scales"""
    print("\n" + "="*60)
    print("BTUT Swarm Scale Test")
    print("="*60)

    from btut_swarm_core import SwarmSimulator, SwarmConfig

    drone_counts = [50, 100, 200, 500]
    drone_counts = [n for n in drone_counts if n <= max_drones]

    results = {}

    for n in drone_counts:
        print(f"\n--- Testing {n} drones ---")

        config = SwarmConfig(
            num_drones=n,
            gamma=1.5,
            tau=0.3,
            formation="grid"
        )

        sim = SwarmSimulator(config)
        sim.initialize()

        start = time.time()
        sim.run(duration=30.0, verbose=False)
        elapsed = time.time() - start

        metrics = sim.get_final_metrics()
        metrics['runtime'] = elapsed
        metrics['drones_per_second'] = n / elapsed

        results[f"n_{n}"] = metrics

        print(f"  Cooperation: {metrics['final_cooperation']:.0%}")
        print(f"  Formation error: {metrics['avg_formation_error']:.1f}m")
        print(f"  Runtime: {elapsed:.2f}s ({metrics['drones_per_second']:.0f} drones/s)")

    # Print scale summary
    print("\n" + "="*70)
    print("SCALE TEST SUMMARY")
    print("="*70)
    print(f"{'Drones':<12} {'Coop':<10} {'Error':<12} {'Runtime':<12} {'Throughput':<15}")
    print("-"*70)

    for key, m in sorted(results.items(), key=lambda x: int(x[0].split("_")[1])):
        n = int(key.split("_")[1])
        print(f"{n:<12} {m['final_cooperation']:<10.0%} {m['avg_formation_error']:<12.1f} "
              f"{m['runtime']:<12.2f} {m['drones_per_second']:<15.0f}")

    print("="*70)

    return results


def visualize_results(sim, output_file: str = "swarm_trajectory.png"):
    """Generate visualization of swarm trajectory"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    fig = plt.figure(figsize=(12, 5))

    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')

    for drone in sim.drones.values():
        ax1.scatter(
            drone.position[0], drone.position[1], drone.position[2],
            c='green' if drone.strategy.value == 'A' else 'red',
            s=50
        )
        ax1.scatter(
            drone.target[0], drone.target[1], drone.target[2],
            c='blue', marker='x', s=30, alpha=0.5
        )

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Swarm Positions (Green=Coop, Red=Comp)')

    # Metrics over time
    ax2 = fig.add_subplot(122)

    steps = [m.step for m in sim.metrics_history]
    coop = [m.cooperation_rate for m in sim.metrics_history]
    progress = [m.task_progress for m in sim.metrics_history]

    ax2.plot(steps, coop, 'g-', label='Cooperation', linewidth=2)
    ax2.plot(steps, progress, 'b--', label='Task Progress', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Rate')
    ax2.set_title('Swarm Metrics Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Visualization saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='BTUT Drone Swarm Quick Start')
    parser.add_argument('--check', action='store_true', help='Check dependencies only')
    parser.add_argument('--compare', action='store_true', help='Run tau comparison')
    parser.add_argument('--scale', type=int, help='Run scale test up to N drones')
    parser.add_argument('--drones', type=int, default=50, help='Number of drones')
    parser.add_argument('--duration', type=float, default=30.0, help='Duration (s)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--output', type=str, default='swarm_results.json')

    args = parser.parse_args()

    # Always check dependencies
    ok = check_dependencies()

    if args.check:
        return

    if not ok:
        print("\nCannot run simulation - missing dependencies")
        return

    if args.compare:
        results = run_comparison(args.drones, args.duration)
    elif args.scale:
        results = run_scale_test(args.scale)
    else:
        results = {"demo": run_demo(args.drones, args.duration)}

        if args.visualize:
            from btut_swarm_core import SwarmSimulator, SwarmConfig
            config = SwarmConfig(num_drones=args.drones, gamma=1.5, tau=0.3)
            sim = SwarmSimulator(config)
            sim.initialize()
            sim.run(duration=args.duration, verbose=False)
            visualize_results(sim)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
