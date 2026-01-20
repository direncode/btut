//! Benchmarks for BTUT Market Simulator
//!
//! Run with: cargo bench

use btut_market_simulator::{
    agent::AgentFactory,
    btut::{BTUTKernel, KernelConfig},
    config::SimulationConfig,
    market::OrderBook,
    simulation::Simulation,
    utils::SimRng,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Benchmark BTUT kernel update cycle at various scales
fn bench_btut_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("btut_kernel");

    for num_agents in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(num_agents as u64));

        group.bench_with_input(
            BenchmarkId::new("update_cycle", num_agents),
            &num_agents,
            |b, &n| {
                let config = SimulationConfig::default();
                let rng = SimRng::from_seed(42);
                let agent_states =
                    AgentFactory::create_agent_states(&config.agents, n, rng.clone());
                let mut kernel = BTUTKernel::new(KernelConfig::default(), &agent_states, rng);

                b.iter(|| {
                    black_box(kernel.update_cycle(0.01));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark order book operations
fn bench_order_book(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_book");

    // Benchmark order submission
    group.bench_function("submit_limit_order", |b| {
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 50, 100.0);
        let mut id = 1000u64;

        b.iter(|| {
            let order = btut_market_simulator::market::Order::limit(
                btut_market_simulator::market::OrderId(id),
                btut_market_simulator::agent::AgentId(1),
                btut_market_simulator::market::OrderSide::Bid,
                99.5,
                10.0,
                0,
            );
            id += 1;
            black_box(book.submit_order(order));
        });
    });

    // Benchmark market order execution
    group.bench_function("execute_market_order", |b| {
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 100, 1000.0);
        let mut id = 1000u64;

        b.iter(|| {
            // Reinitialize liquidity periodically
            if book.total_ask_volume() < 100.0 {
                book.initialize_liquidity(100.0, 100, 1000.0);
            }

            let order = btut_market_simulator::market::Order::market(
                btut_market_simulator::market::OrderId(id),
                btut_market_simulator::agent::AgentId(1),
                btut_market_simulator::market::OrderSide::Bid,
                10.0,
                0,
            );
            id += 1;
            black_box(book.submit_order(order));
        });
    });

    // Benchmark snapshot generation
    group.bench_function("snapshot_100_levels", |b| {
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 100, 100.0);

        b.iter(|| {
            black_box(book.snapshot(100));
        });
    });

    group.finish();
}

/// Benchmark full simulation step
fn bench_simulation_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation");
    group.sample_size(50); // Reduce sample size for slower benchmarks

    for num_agents in [1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(num_agents as u64));

        group.bench_with_input(
            BenchmarkId::new("full_step", num_agents),
            &num_agents,
            |b, &n| {
                let mut config = SimulationConfig::test_preset();
                config.simulation.num_agents = n;
                config.simulation.num_timesteps = 10;
                config.output.show_progress = false;

                b.iter(|| {
                    let mut sim = Simulation::new(config.clone());
                    black_box(sim.run());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark agent state creation
fn bench_agent_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("agent_creation");

    for num_agents in [1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(num_agents as u64));

        group.bench_with_input(
            BenchmarkId::new("create_states", num_agents),
            &num_agents,
            |b, &n| {
                let config = SimulationConfig::default();

                b.iter(|| {
                    let rng = SimRng::from_seed(42);
                    black_box(AgentFactory::create_agent_states(&config.agents, n, rng));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RNG operations
fn bench_rng(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng");

    group.bench_function("uniform", |b| {
        let mut rng = SimRng::from_seed(42);
        b.iter(|| black_box(rng.uniform(0.0, 1.0)));
    });

    group.bench_function("normal", |b| {
        let mut rng = SimRng::from_seed(42);
        b.iter(|| black_box(rng.normal(0.0, 1.0)));
    });

    group.bench_function("pareto", |b| {
        let mut rng = SimRng::from_seed(42);
        b.iter(|| black_box(rng.pareto(1.0, 1.5)));
    });

    group.bench_function("fork", |b| {
        let mut rng = SimRng::from_seed(42);
        b.iter(|| black_box(rng.fork()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_btut_kernel,
    bench_order_book,
    bench_simulation_step,
    bench_agent_creation,
    bench_rng,
);
criterion_main!(benches);
