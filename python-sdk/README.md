# BTUT Python SDK

Official Python SDK for the BTUT multi-agent simulation platform.

## Installation

```bash
pip install btut-sdk
```

## Quick Start

```python
from btut import Simulator

# Run a quick simulation
sim = Simulator(agents=10000, gamma=1.45)
results = sim.run()

print(f"Final cooperation: {results.final_cooperation:.2%}")
print(f"Converged: {results.converged}")
results.plot()  # Visualize convergence
```

## Features

- **O(N) Scalability**: Handle millions of agents efficiently
- **Game Theory**: Stag Hunt, Prisoner's Dilemma, Hawk-Dove dynamics
- **Network Effects**: Scale-free (Barabási-Albert) network topology
- **Mean-Field Dynamics**: Hub-weighted convergence
- **Parameter Sweeps**: Test ranges of parameter values
- **Visualization**: Built-in plotting with matplotlib
- **API Integration**: Use local engine or remote API

## Usage Examples

### Using Presets

```python
from btut import Presets

# Quick test (10K agents)
sim = Presets.quick()
results = sim.run()

# Standard simulation (100K agents)
sim = Presets.standard(gamma=1.6)
results = sim.run()

# Massive scale (500K+ agents)
sim = Presets.massive()
results = sim.run()
```

### Parameter Sweeps

```python
sim = Simulator(agents=10000)
results = sim.sweep('gamma', [1.2, 1.4, 1.6, 1.8, 2.0])

for gamma, result in zip([1.2, 1.4, 1.6, 1.8, 2.0], results):
    print(f"γ={gamma}: {result.final_cooperation:.2%} cooperation")
```

### Using Remote API

```python
# Connect to hosted API
sim = Simulator(agents=100000, api_url="https://btut-api.fly.dev")
results = sim.run()

# Async execution for long-running simulations
results = sim.run(async_mode=True)
```

### Performance Benchmarking

```python
sim = Simulator()
bench = sim.benchmark(agent_counts=[1000, 10000, 100000, 1000000])

for result in bench['benchmark_results']:
    print(f"{result['agents']:>8} agents: {result['runtime_seconds']:.3f}s")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agents` | 10000 | Number of agents in simulation |
| `gamma` | 1.45 | Cooperation bonus (Stag Hunt) |
| `tau` | 0.30 | Hub influence exponent (0-1) |
| `cA_SH` | 0.40 | Cost for strategy A in Stag Hunt |
| `cB_SH` | 0.10 | Cost for strategy B in Stag Hunt |
| `cA_PD` | 0.20 | Cost for strategy A in Prisoner's Dilemma |
| `cB_PD` | 0.08 | Cost for strategy B in Prisoner's Dilemma |
| `alpha` | 0.60 | Aggression parameter (Hawk-Dove) |
| `iterations` | 20 | Maximum number of update iterations |
| `m` | 3 | Minimum degree (Barabási-Albert network) |
| `seed` | None | Random seed for reproducibility |

## Advanced Usage

### Custom Configurations

```python
sim = Simulator(
    agents=50000,
    gamma=1.8,          # Higher cooperation bonus
    tau=0.5,            # Stronger hub influence
    alpha=0.7,          # More aggressive
    iterations=50,      # Longer convergence
    seed=42             # Reproducible results
)

results = sim.run()
```

### Visualization

```python
results = sim.run()

# Interactive plot
results.plot()

# Save to file
results.plot(save_path='convergence.png')
```

### Export Results

```python
results = sim.run()

# As dictionary
data = results.to_dict()

# Save to JSON
import json
with open('results.json', 'w') as f:
    json.dump(results.to_dict(), f, indent=2)
```

## Performance

Typical performance on modern hardware:

- **1,000 agents**: ~10ms
- **10,000 agents**: ~100ms
- **100,000 agents**: ~1s
- **1,000,000 agents**: ~10s

Linear O(N) scaling enables unprecedented simulation sizes.

## Requirements

- Python ≥3.8
- NumPy ≥1.20
- Requests ≥2.28

Optional dependencies:
- `matplotlib` for visualization
- `seaborn` for advanced plotting

## Documentation

- **Full Documentation**: https://btut.ai/docs
- **API Reference**: https://btut-api.fly.dev/docs
- **Examples**: https://github.com/direncode/btut/tree/main/examples
- **Paper**: [Link to research paper]

## License

Apache License 2.0 - see LICENSE file for details

## Citation

If you use BTUT in your research, please cite:

```bibtex
@software{btut2026,
  title={BTUT: Scalable O(N) Multi-Agent Coordination Engine},
  author={Akkocdemir, Diren},
  year={2026},
  url={https://btut.ai}
}
```

## Support

- **Issues**: https://github.com/direncode/btut/issues
- **Discussions**: https://github.com/direncode/btut/discussions
- **Email**: support@btut.ai
