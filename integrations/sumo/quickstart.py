#!/usr/bin/env python3
"""
BTUT-SUMO Quick Start

Immediate testing script - validates your setup and runs a demo.

Usage:
    python quickstart.py              # Check setup, run mock demo
    python quickstart.py --sumo       # Run with SUMO (requires SUMO installed)
    python quickstart.py --install    # Show installation instructions
"""

import sys
import os
import subprocess
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / '..' / '..' / 'python-sdk'))


def check_dependencies():
    """Check all required dependencies"""
    print("="*60)
    print("BTUT-SUMO Setup Checker")
    print("="*60)

    status = {
        "python": False,
        "numpy": False,
        "matplotlib": False,
        "btut_sdk": False,
        "sumo": False,
        "traci": False
    }

    # Python version
    py_version = sys.version_info
    status["python"] = py_version >= (3, 8)
    print(f"\n[{'OK' if status['python'] else 'FAIL'}] Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # NumPy
    try:
        import numpy as np
        status["numpy"] = True
        print(f"[OK] NumPy {np.__version__}")
    except ImportError:
        print("[FAIL] NumPy not installed")

    # Matplotlib
    try:
        import matplotlib
        status["matplotlib"] = True
        print(f"[OK] Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("[WARN] Matplotlib not installed (optional, for visualization)")

    # BTUT SDK
    try:
        from btut import Simulator
        status["btut_sdk"] = True
        print("[OK] BTUT SDK")
    except ImportError:
        print("[FAIL] BTUT SDK not found")

    # SUMO
    try:
        result = subprocess.run(
            ["sumo", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            status["sumo"] = True
            print(f"[OK] SUMO: {version_line}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[WARN] SUMO not installed or not in PATH")

    # TraCI
    try:
        import traci
        status["traci"] = True
        print("[OK] TraCI")
    except ImportError:
        print("[WARN] TraCI not installed")

    # SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        print(f"[OK] SUMO_HOME: {sumo_home}")
    else:
        print("[WARN] SUMO_HOME not set")

    print("\n" + "-"*60)

    # Summary
    core_ok = status["python"] and status["numpy"] and status["btut_sdk"]
    sumo_ok = status["sumo"] and status["traci"]

    if core_ok and sumo_ok:
        print("All dependencies OK! Ready for SUMO simulation.")
        return "full"
    elif core_ok:
        print("Core dependencies OK. SUMO not available (mock mode only).")
        return "mock"
    else:
        print("Missing core dependencies. See installation instructions.")
        return "fail"


def show_install_instructions():
    """Show installation instructions"""
    print("""
================================================================================
INSTALLATION INSTRUCTIONS
================================================================================

1. PYTHON DEPENDENCIES
   pip install numpy matplotlib

2. BTUT SDK (from project root)
   pip install -e python-sdk/

3. SUMO INSTALLATION

   Windows:
   - Download from https://sumo.dlr.de/docs/Downloads.php
   - Install to default location
   - Add to PATH: C:\\Program Files (x86)\\Eclipse\\Sumo\\bin
   - Set SUMO_HOME: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo

   Linux (Ubuntu/Debian):
   sudo add-apt-repository ppa:sumo/stable
   sudo apt-get update
   sudo apt-get install sumo sumo-tools
   export SUMO_HOME=/usr/share/sumo

   macOS:
   brew install sumo
   export SUMO_HOME=/usr/local/opt/sumo/share/sumo

4. VERIFY INSTALLATION
   sumo --version
   python -c "import traci; print('TraCI OK')"

================================================================================
""")


def run_mock_demo():
    """Run mock demonstration without SUMO"""
    print("\n" + "="*60)
    print("BTUT-SUMO Mock Demo")
    print("="*60)

    try:
        # Import the engine directly
        sys.path.insert(0, str(SCRIPT_DIR / '..' / '..' / 'lib' / 'simulation'))
        from btut_engine import BTUTSimulator

        print("\nRunning BTUT coordination simulation (no SUMO)...")
        print("-"*60)

        # Simulate different tau values
        results = []

        for tau in [0.0, 0.3, 0.5]:
            config = {
                'N': 1000,
                'gamma': 1.5,
                'tau': tau,
                'iterations': 20,
                'cA_SH': 0.05,
                'cB_SH': 0.03,
                'cA_PD': 0.02,
                'cB_PD': 0.01,
                'alpha': 0.1,
                'm': 3
            }

            sim = BTUTSimulator(config)
            states = sim.run()
            final_state = states[-1]

            results.append({
                "tau": tau,
                "cooperation": final_state['fractionA'],
                "converged": final_state['fractionA'] > 0.99,
                "iterations": len(states)
            })

            print(f"  tau={tau:.1f}: cooperation={final_state['fractionA']:.1%}, "
                  f"iterations={len(states)}")

        # Show improvement
        baseline = results[0]["cooperation"]
        best = max(results, key=lambda r: r["cooperation"])

        print("-"*60)
        print(f"\nBaseline (tau=0): {baseline:.1%} cooperation")
        print(f"Best (tau={best['tau']}): {best['cooperation']:.1%} cooperation")

        if baseline > 0:
            improvement = (best['cooperation'] - baseline) / baseline * 100
            print(f"Improvement: {improvement:+.1f}%")

        print("\n[SUCCESS] Mock demo complete!")
        print("To run with SUMO traffic simulation: python quickstart.py --sumo")

    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def run_sumo_demo():
    """Run demonstration with SUMO"""
    print("\n" + "="*60)
    print("BTUT-SUMO Traffic Demo")
    print("="*60)

    try:
        from btut_sumo_bridge import BTUTTrafficCoordinator, run_baseline_comparison

        config_path = SCRIPT_DIR / "example_scenario.sumocfg"

        if not config_path.exists():
            print(f"[ERROR] Config not found: {config_path}")
            return False

        print(f"\nConfig: {config_path}")
        print("Running quick comparison (tau=0 vs tau=0.3 vs tau=0.5)...")
        print("-"*60)

        results = run_baseline_comparison(
            sumo_config=str(config_path),
            steps=500,  # Quick demo
            gamma=1.5,
            tau_values=[0.0, 0.3, 0.5],
            use_gui=False
        )

        print("\n[SUCCESS] SUMO demo complete!")
        print("Results saved to btut_sumo_results.json")

        # Show quick summary
        print("\n" + "-"*60)
        print("Quick Results:")
        for key, result in results.items():
            print(f"  {key}: coop={result.final_cooperation:.1%}, "
                  f"speed={result.avg_speed:.1f}m/s, "
                  f"wait={result.avg_waiting_time:.1f}s")

    except Exception as e:
        print(f"[ERROR] SUMO demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def run_full_validation():
    """Run full validation experiment"""
    print("\n" + "="*60)
    print("BTUT-SUMO Full Validation")
    print("="*60)

    try:
        from validation_experiment import ValidationExperiment

        experiment = ValidationExperiment(
            output_dir="validation_results",
            sumo_config=str(SCRIPT_DIR / "example_scenario.sumocfg")
        )

        print("\nRunning full validation sweep...")
        analysis = experiment.run_full_validation()

        print("\nGenerating visualizations...")
        experiment.generate_plots()

        print("\nGenerating report...")
        experiment.generate_report()
        experiment.save_results()

        print(f"\n[SUCCESS] Validation complete!")
        print(f"Results saved to: {experiment.output_dir}/")

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='BTUT-SUMO Quick Start')
    parser.add_argument('--install', action='store_true', help='Show installation instructions')
    parser.add_argument('--check', action='store_true', help='Check dependencies only')
    parser.add_argument('--sumo', action='store_true', help='Run with SUMO')
    parser.add_argument('--full', action='store_true', help='Run full validation')

    args = parser.parse_args()

    if args.install:
        show_install_instructions()
        return

    # Always check dependencies first
    setup_status = check_dependencies()

    if args.check:
        return

    if setup_status == "fail":
        print("\nRun 'python quickstart.py --install' for installation instructions.")
        return

    print("\n")

    if args.full:
        if setup_status != "full":
            print("[ERROR] Full validation requires SUMO. Install SUMO first.")
            return
        run_full_validation()
    elif args.sumo:
        if setup_status != "full":
            print("[ERROR] SUMO mode requires SUMO. Install SUMO first.")
            return
        run_sumo_demo()
    else:
        # Default: run mock demo
        run_mock_demo()


if __name__ == '__main__':
    main()
