# Quick Start Guide

Get up and running with BTUT in 5 minutes.

## Installation

```bash
pip install btut-sdk
```

## Your First Simulation

```python
from btut import Simulator

# Create simulator with 10,000 agents
sim = Simulator(agents=10000, gamma=1.45)

# Run simulation
results = sim.run()

# View results
print(f"Final cooperation: {results.final_cooperation:.2%}")
print(f"Converged: {results.converged}")
print(f"Iterations: {results.iterations_completed}")

# Plot convergence
results.plot()
```

## What Just Happened?

You simulated 10,000 agents playing coordination games on a scale-free network:

1. **Agents** started with random strategies (A or B)
2. **Network** connected agents in a scale-free topology
3. **Dynamics** evolved through mean-field updates
4. **Convergence** reached equilibrium in ~20 iterations

## Next Steps

### Use Presets

```python
from btut import Presets

# Quick test (10K agents)
sim = Presets.quick()
results = sim.run()

# Standard simulation (100K agents)
sim = Presets.standard()
results = sim.run()

# Massive scale (500K+ agents)
sim = Presets.massive()
results = sim.run()
```

### Explore Parameters

```python
sim = Simulator(
    agents=50000,
    gamma=1.8,      # Higher cooperation bonus
    tau=0.5,        # Stronger hub influence
    alpha=0.7,      # More aggressive dynamics
    iterations=30   # Longer convergence window
)

results = sim.run()
```

### Run Parameter Sweeps

```python
sim = Simulator(agents=10000)

# Test different cooperation bonuses
results = sim.sweep('gamma', [1.2, 1.4, 1.6, 1.8, 2.0])

for gamma, result in zip([1.2, 1.4, 1.6, 1.8, 2.0], results):
    print(f"γ={gamma}: {result.final_cooperation:.2%}")
```

### Use the REST API

```python
# Connect to hosted API
sim = Simulator(
    agents=100000,
    api_url="https://btut-api.fly.dev"
)

results = sim.run()
```

## Understanding Results

```python
results = sim.run()

# Access detailed data
print(f"Agent count: {results.agent_count:,}")
print(f"Final cooperation: {results.final_cooperation:.2%}")
print(f"Convergence history: {results.convergence_history}")
print(f"Runtime: {results.runtime_seconds:.2f}s")
print(f"Converged: {results.converged}")

# Export to JSON
import json
with open('results.json', 'w') as f:
    json.dump(results.to_dict(), f, indent=2)

# Visualize
results.plot(save_path='convergence.png')
```

## Performance Expectations

| Agents | Typical Runtime |
|--------|----------------|
| 1,000 | ~10ms |
| 10,000 | ~100ms |
| 100,000 | ~1s |
| 1,000,000 | ~10s |

Linear O(N) scaling enables unprecedented simulation sizes!

## What's Next?

- [Basic Concepts](concepts.md) - Understand the theory
- [Parameter Reference](../reference/parameters.md) - Detailed parameter docs
- [API Reference](../sdk/api-reference.md) - Complete SDK documentation
- [Tutorials](../tutorials/parameter-tuning.md) - In-depth guides

## Need Help?

- **Examples**: https://github.com/direncode/btut/tree/main/examples
- **Issues**: https://github.com/direncode/btut/issues
- **Discussions**: https://github.com/direncode/btut/discussions
