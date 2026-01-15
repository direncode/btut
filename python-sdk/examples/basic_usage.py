"""
BTUT SDK - Basic Usage Examples
================================
Simple examples demonstrating core functionality
"""

from btut import Simulator, Presets, quick_run

print("="*70)
print("BTUT Python SDK - Basic Examples")
print("="*70)

# Example 1: Simplest possible usage
print("\n1. Quick Run (One-Liner):")
print("-" * 40)
result = quick_run(agents=5000, gamma=1.5)
print(f"Final cooperation: {result.final_cooperation:.2%}")
print(f"Runtime: {result.runtime_seconds:.3f}s")

# Example 2: Basic simulation with configuration
print("\n2. Basic Simulation:")
print("-" * 40)
sim = Simulator(
    agents=10000,
    gamma=1.45,
    tau=0.30,
    iterations=20
)
result = sim.run()

print(f"Agents: {result.agent_count:,}")
print(f"Iterations: {result.iterations_completed}")
print(f"Final cooperation: {result.final_cooperation:.2%}")
print(f"Converged: {'Yes' if result.converged else 'No'}")

# Example 3: Using presets
print("\n3. Using Presets:")
print("-" * 40)

# Quick test
sim = Presets.quick()
result = sim.run()
print(f"Quick (10K agents): {result.final_cooperation:.2%} in {result.runtime_seconds:.2f}s")

# Standard simulation
sim = Presets.standard()
result = sim.run()
print(f"Standard (100K): {result.final_cooperation:.2%} in {result.runtime_seconds:.2f}s")

# Example 4: Custom configuration
print("\n4. Custom Configuration:")
print("-" * 40)
sim = Simulator(
    agents=25000,
    gamma=1.8,      # High cooperation bonus
    tau=0.5,        # Strong hub influence
    alpha=0.7,      # More aggressive
    iterations=30,
    seed=42         # Reproducible
)
result = sim.run()
print(f"Custom config: {result.final_cooperation:.2%}")
print(f"Convergence: {result.convergence_history[-5:]}")

# Example 5: Accessing detailed results
print("\n5. Result Details:")
print("-" * 40)
print(f"Agent count: {result.agent_count:,}")
print(f"Iterations completed: {result.iterations_completed}")
print(f"Final strategy A: {int(result.final_cooperation * result.agent_count):,} agents")
print(f"Final strategy B: {int((1 - result.final_cooperation) * result.agent_count):,} agents")

print("\n" + "="*70)
print("Examples complete! Try modifying parameters to see different dynamics.")
print("="*70)
