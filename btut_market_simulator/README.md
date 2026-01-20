# BTUT Market Simulator

A **planetary-scale multi-agent financial market simulator** using Bivariate Trajectory-Undercurrent Theory (BTUT) as the core coordination engine.

## Overview

BTUT Market Simulator models emergent behaviors in financial markets through multi-agent simulation with up to **1 million+ agents** on a single machine. The simulator uses BTUT's O(N) kernel-weighted mean-field dynamics to enable efficient coordination without explicit pairwise agent interactions.

### Key Features

- **O(N) Complexity**: No explicit graph storage; kernel-weighted mean-field dynamics
- **Scalable**: 1M agents on single machine, path to 1B+ on cluster
- **Fast Convergence**: 20-30 BTUT iterations even at scale
- **Heterogeneous Agents**: Retail, HFT, Institutions, Market Makers
- **Emergent Behaviors**: Flash crashes, liquidity spirals, market regeneration
- **Production Ready**: Full logging, configurable parameters, comprehensive metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/btut/btut_market_simulator.git
cd btut_market_simulator

# Build release version
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Basic Usage

```bash
# Run with default configuration (100k agents, 1000 timesteps)
cargo run --release -- run

# Run 1M agent simulation
cargo run --release -- run --preset million

# Run liquidity shock experiment
cargo run --release -- run --preset shock --shock-at 250

# Run baseline comparison (BTUT vs Zero-Intelligence)
cargo run --release -- run --preset shock --compare

# Generate configuration file
cargo run --release -- generate-config my_config.toml

# Run with custom config
cargo run --release -- run --config my_config.toml
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Simulation Loop                          │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │  Agents  │──▶│   BTUT   │──▶│  Market  │──▶│   Metrics   │ │
│  │  (1M+)   │   │  Kernel  │   │ OrderBook│   │  Collector  │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘ │
│       │              │              │               │          │
│       ▼              ▼              ▼               ▼          │
│   Strategies    Convergence     Matching        Analysis       │
│   (cooperate/     Loop         Engine         (crashes,        │
│    defect)      (20-30 iter)   (FIFO)         regeneration)   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
btut_market_simulator/
├── Cargo.toml           # Dependencies and project config
├── README.md            # This file
├── src/
│   ├── main.rs          # CLI entry point
│   ├── lib.rs           # Library root, re-exports
│   ├── config.rs        # Configuration structures
│   ├── utils.rs         # RNG, logging, plotting helpers
│   ├── simulation.rs    # Simulation orchestration
│   ├── metrics.rs       # Metrics collection, crash detection
│   ├── btut/
│   │   ├── mod.rs       # BTUT module root
│   │   └── kernel.rs    # Core BTUT algorithm
│   ├── market/
│   │   ├── mod.rs       # Market module root
│   │   └── order_book.rs# Limit order book, matching engine
│   └── agent/
│       └── mod.rs       # Agent types and behaviors
└── benches/
    └── btut_benchmarks.rs # Performance benchmarks
```

## BTUT Algorithm

### Core Concept

BTUT (Bivariate Trajectory-Undercurrent Theory) achieves O(N) complexity by computing agent strategy updates through kernel-weighted mean-field dynamics rather than explicit pairwise interactions.

### Algorithm Steps

For each timestep:

1. **Compute Global Mean Field**
   ```
   C = Σᵢ wᵢ · sᵢ / Σᵢ wᵢ
   ```
   Where `wᵢ` is agent weight and `sᵢ` is strategy (cooperation level).

2. **Update Each Agent's Strategy**
   ```
   Kᵢ = exp(-λ · dᵢ)          # Kernel weight (distance decay)
   tᵢ = C + bias               # Target strategy
   sᵢ' = α·sᵢ + (1-α)·Kᵢ·tᵢ   # Momentum update
   ```

3. **Check Convergence**
   ```
   converged = max|sᵢ' - sᵢ| < threshold
   ```

### Strategy Interpretation

| Strategy | Behavior | Market Effect |
|----------|----------|---------------|
| **Cooperate** (s→1) | Provide liquidity, tight spreads | Stabilizing, regeneration |
| **Defect** (s→0) | Front-run, wide spreads, spoof | Destabilizing, drag |

## Agent Types

### Retail (70% default)
- **Behavior**: Uninformed, random timing, small orders
- **Latency**: High (10x)
- **Cooperation Influence**: Higher cooperation → limit orders; Lower → market orders

### HFT (5% default)
- **Behavior**: Informed, continuous trading, small-medium orders
- **Latency**: Very low (0.1x)
- **Cooperation Influence**: Higher → market making; Lower → directional aggression

### Institution (10% default)
- **Behavior**: Informed, patient execution, large orders
- **Latency**: Medium (2x)
- **Cooperation Influence**: Higher → small slices, passive; Lower → large slices, aggressive

### Market Maker (15% default)
- **Behavior**: Liquidity provider, two-sided quotes
- **Latency**: Low (0.5x)
- **Cooperation Influence**: Higher → tight spreads, large size; Lower → wide spreads, potential spoofing

## Configuration

### Configuration File (TOML)

```toml
[simulation]
num_agents = 100000
num_timesteps = 1000
seed = 42
parallel = true

[btut]
kernel_decay = 0.1        # λ: Higher = faster decay = more localized
momentum_strength = 0.7   # α: Higher = more persistent strategies
convergence_iterations = 25
convergence_threshold = 1e-6
temperature = 1.0         # Lower = sharper strategy selection
cooperation_bias = 0.0    # Shift equilibrium toward cooperation (+) or defection (-)
adaptive_kernel = false   # Adapt decay based on volatility

[market]
initial_price = 100.0
tick_size = 0.01
max_book_depth = 1000

[agents]
retail_fraction = 0.70
hft_fraction = 0.05
institution_fraction = 0.10
market_maker_fraction = 0.15
order_size_exponent = 1.5  # Pareto α for order sizes
initial_cooperation = 0.5
cooperation_noise = 0.1

[output]
output_dir = "output"
format = "csv"             # csv, json, or both
log_level = "info"
metrics_sample_rate = 1
show_progress = true
```

### Preset Configurations

| Preset | Agents | Timesteps | Use Case |
|--------|--------|-----------|----------|
| `default` | 100,000 | 1,000 | General purpose |
| `million` | 1,000,000 | 1,000 | Large-scale testing |
| `test` | 1,000 | 100 | Development/debugging |
| `shock` | 100,000 | 500 | Liquidity shock experiments |

### Environment Variables

Override config values with environment variables:

```bash
export BTUT_SIMULATION_NUM_AGENTS=500000
export BTUT_SIMULATION_SEED=12345
export BTUT_BTUT_KERNEL_DECAY=0.15
export BTUT_OUTPUT_LOG_LEVEL=debug
```

## Metrics

### Output Metrics

| Metric | Description |
|--------|-------------|
| `mid_price` | Order book mid price |
| `spread_bps` | Bid-ask spread in basis points |
| `bid_volume` / `ask_volume` | Total volume at each side |
| `imbalance` | Volume imbalance (-1 to +1) |
| `volatility` | Rolling return volatility |
| `mean_cooperation` | Average agent cooperation level |
| `regeneration_score` | Market stability (0-1, higher = more stable) |
| `drag_score` | Market friction (0-1, higher = more unstable) |
| `is_crash` | Flash crash detected |

### Crash Detection

Flash crashes are detected when:
- Price drops > 2% from baseline
- Spread widens > 3x normal
- Liquidity drops < 50% of normal

## Scaling

### Single Machine Performance

| Agents | Expected Runtime | Memory |
|--------|------------------|--------|
| 10,000 | ~1s | ~10MB |
| 100,000 | ~10s | ~100MB |
| 1,000,000 | ~60s | ~1GB |

### Optimization Tips

1. **Use Release Build**: `cargo build --release`
2. **Enable Parallel**: Set `parallel = true` in config
3. **Tune BTUT Iterations**: Reduce `convergence_iterations` if acceptable
4. **Sample Metrics**: Increase `metrics_sample_rate` to reduce I/O
5. **Disable Progress**: Use `--no-progress` for batch runs

### Future: Distributed Scaling

The architecture supports future extension to distributed computing:

```
Cluster Mode (planned):
├── Coordinator Node
│   └── Global mean field aggregation
├── Worker Nodes (N)
│   └── Local agent updates
└── Shared State
    └── Order book synchronization
```

## Examples

### Basic Simulation

```bash
cargo run --release -- run --agents 50000 --timesteps 500 --seed 42
```

### Liquidity Shock Experiment

```bash
# Run shock at timestep 200 with 60% severity
cargo run --release -- run \
    --preset shock \
    --shock-at 200 \
    --shock-severity 0.6 \
    --output shock_results
```

### Baseline Comparison

```bash
# Compare BTUT agents vs zero-intelligence during crisis
cargo run --release -- run \
    --agents 100000 \
    --timesteps 500 \
    --compare \
    --shock-at 250
```

### Library Usage

```rust
use btut_market_simulator::{Simulation, SimulationConfig};

fn main() {
    // Create configuration
    let config = SimulationConfig::million_agent_preset();

    // Initialize simulation
    let mut sim = Simulation::new(config);

    // Schedule a liquidity crisis at timestep 500
    sim.schedule_liquidity_crisis(500, 0.5);

    // Run simulation
    let results = sim.run();

    // Analyze results
    println!("Crashes detected: {}", results.crashes.len());
    println!("Mean regeneration: {:.4}", results.summary.mean_regeneration);

    // Save results
    results.save_json("results.json").unwrap();
    results.save_metrics_csv("metrics.csv").unwrap();
}
```

## Output Files

After running a simulation, the output directory contains:

```
output/
├── results.json      # Full simulation results (config, metrics, crashes)
├── metrics.csv       # Time series metrics for analysis
└── [comparison files if --compare used]
    ├── btut_results.json
    ├── btut_metrics.csv
    ├── zi_results.json
    └── zi_metrics.csv
```

## Development

### Running Tests

```bash
# All tests
cargo test

# Specific module
cargo test btut::

# With output
cargo test -- --nocapture
```

### Running Benchmarks

```bash
cargo bench
```

### Generating Documentation

```bash
cargo doc --open
```

## References

- **BTUT Theory**: Bivariate Trajectory-Undercurrent Theory for multi-agent coordination
- **Order Book**: Price-time priority matching (FIFO within price levels)
- **Volatility**: Heston-style stochastic volatility model
- **Order Sizes**: Power-law (Pareto) distribution

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

**Built with 🦀 Rust for maximum performance and safety.**
