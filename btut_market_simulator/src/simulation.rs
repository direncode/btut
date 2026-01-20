//! Simulation Module
//!
//! This module provides the main simulation orchestration, coordinating:
//! - Agent order generation and execution
//! - BTUT kernel updates for strategy coordination
//! - Order book management
//! - Metrics collection
//!
//! ## Simulation Loop
//!
//! Each timestep:
//! 1. BTUT kernel updates agent strategies (convergence loop)
//! 2. Agents generate orders based on market state and strategy
//! 3. Orders are submitted to the order book
//! 4. Matching engine executes trades
//! 5. Metrics are collected
//! 6. External shocks may be applied

use crate::agent::{AgentFactory, AgentState};
use crate::btut::{BTUTKernel, KernelConfig, ShockType};
use crate::config::SimulationConfig;
use crate::market::OrderBook;
use crate::metrics::{CrashEvent, MarketMetrics, MetricsCollector, MetricsSummary};
use crate::utils::{SimRng, SyntheticGenerator};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};

/// Current state of the simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulationState {
    /// Not yet started
    NotStarted,
    /// Currently running
    Running,
    /// Completed successfully
    Completed,
    /// Stopped due to error
    Error,
}

/// Results from a completed simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResults {
    /// Configuration used
    pub config: SimulationConfig,
    /// All collected metrics
    pub metrics: Vec<MarketMetrics>,
    /// Summary statistics
    pub summary: MetricsSummary,
    /// Detected crash events
    pub crashes: Vec<CrashEvent>,
    /// Total runtime
    pub runtime_ms: u64,
    /// Final state
    pub state: SimulationState,
}

impl SimulationResults {
    /// Save results to JSON file
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Save metrics to CSV file
    pub fn save_metrics_csv(&self, path: &str) -> std::io::Result<()> {
        let mut csv = String::new();

        // Header
        csv.push_str("timestep,mid_price,spread_bps,bid_volume,ask_volume,");
        csv.push_str("volatility,mean_cooperation,regeneration_score,is_crash\n");

        for m in &self.metrics {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{}\n",
                m.timestep,
                m.mid_price.map_or("".to_string(), |v| format!("{:.4}", v)),
                m.spread_bps.map_or("".to_string(), |v| format!("{:.2}", v)),
                m.bid_volume,
                m.ask_volume,
                m.volatility,
                m.mean_cooperation,
                m.regeneration_score,
                m.is_crash
            ));
        }

        std::fs::write(path, csv)
    }
}

/// Scheduled shock event
#[derive(Debug, Clone)]
pub struct ScheduledShock {
    /// Timestep to apply shock
    pub timestep: u64,
    /// Type of shock
    pub shock_type: ShockType,
}

/// Main simulation orchestrator
pub struct Simulation {
    /// Configuration
    config: SimulationConfig,
    /// Random number generator
    rng: SimRng,
    /// Synthetic data generator
    generator: SyntheticGenerator,
    /// Order book
    book: OrderBook,
    /// BTUT kernel
    kernel: BTUTKernel,
    /// Agent states (for BTUT)
    agent_states: Vec<AgentState>,
    /// Metrics collector
    metrics: MetricsCollector,
    /// Current timestep
    timestep: u64,
    /// Current state
    state: SimulationState,
    /// Scheduled shocks
    scheduled_shocks: Vec<ScheduledShock>,
    /// Start time
    start_time: Option<Instant>,
    /// Progress bar
    progress: Option<ProgressBar>,
}

impl Simulation {
    /// Create a new simulation from configuration
    #[instrument(skip(config))]
    pub fn new(config: SimulationConfig) -> Self {
        info!(
            num_agents = config.simulation.num_agents,
            num_timesteps = config.simulation.num_timesteps,
            "Initializing simulation"
        );

        // Initialize RNG
        let mut rng = SimRng::new(config.simulation.seed);

        // Initialize synthetic generator
        let generator = SyntheticGenerator::new(&config, config.simulation.seed);

        // Initialize order book
        let mut book = OrderBook::new(config.market.tick_size);
        book.initialize_liquidity(
            config.market.initial_price,
            config.market.max_book_depth / 2,
            1000.0,
        );

        // Create agent states
        let agent_states = AgentFactory::create_agent_states(
            &config.agents,
            config.simulation.num_agents,
            rng.fork(),
        );

        // Initialize BTUT kernel
        let kernel_config = KernelConfig::from(&config.btut);
        let kernel = BTUTKernel::new(kernel_config, &agent_states, rng.fork());

        // Initialize metrics collector
        let metrics = MetricsCollector::new(config.output.metrics_sample_rate);

        // Set up progress bar
        let progress = if config.output.show_progress {
            let pb = ProgressBar::new(config.simulation.num_timesteps as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        Self {
            config,
            rng,
            generator,
            book,
            kernel,
            agent_states,
            metrics,
            timestep: 0,
            state: SimulationState::NotStarted,
            scheduled_shocks: Vec::new(),
            start_time: None,
            progress,
        }
    }

    /// Schedule a shock event
    pub fn schedule_shock(&mut self, timestep: u64, shock_type: ShockType) {
        self.scheduled_shocks.push(ScheduledShock {
            timestep,
            shock_type,
        });
        // Sort by timestep
        self.scheduled_shocks.sort_by_key(|s| s.timestep);
    }

    /// Schedule a liquidity crisis at a specific timestep
    pub fn schedule_liquidity_crisis(&mut self, timestep: u64, severity: f64) {
        self.schedule_shock(timestep, ShockType::LiquidityCrisis { severity });
    }

    /// Run the full simulation
    #[instrument(skip(self))]
    pub fn run(&mut self) -> SimulationResults {
        self.start_time = Some(Instant::now());
        self.state = SimulationState::Running;

        info!("Starting simulation");

        // Warmup period
        for _ in 0..self.config.simulation.warmup_steps {
            self.step_warmup();
        }

        debug!(
            warmup_steps = self.config.simulation.warmup_steps,
            "Warmup complete"
        );

        // Main simulation loop
        for t in 0..self.config.simulation.num_timesteps {
            self.timestep = t as u64;
            self.step();

            if let Some(ref pb) = self.progress {
                pb.set_position(t as u64 + 1);
            }
        }

        if let Some(ref pb) = self.progress {
            pb.finish_with_message("Simulation complete");
        }

        // Finalize
        self.metrics.finalize(self.timestep);
        self.state = SimulationState::Completed;

        let runtime = self.start_time.map_or(0, |t| t.elapsed().as_millis() as u64);

        info!(
            runtime_ms = runtime,
            num_crashes = self.metrics.crashes().len(),
            "Simulation complete"
        );

        SimulationResults {
            config: self.config.clone(),
            metrics: self.metrics.metrics().to_vec(),
            summary: self.metrics.summary(),
            crashes: self.metrics.crashes().to_vec(),
            runtime_ms: runtime,
            state: self.state,
        }
    }

    /// Run a single warmup step (no metrics collection)
    fn step_warmup(&mut self) {
        // Simple BTUT update without full agent interaction
        let volatility = self.generator.volatility();
        self.kernel.update_cycle(volatility);

        // Update synthetic price
        self.generator.update_price();

        // Basic order flow
        self.generate_simple_flow();
    }

    /// Run a single simulation step
    fn step(&mut self) {
        self.book.set_timestamp(self.timestep);

        // Check for scheduled shocks
        self.apply_scheduled_shocks();

        // Update synthetic volatility
        let volatility = self.generator.update_volatility();

        // BTUT kernel update
        let btut_stats = self.kernel.update_cycle(volatility);

        // Sync strategies back to agent states
        self.kernel.sync_to_agents(&mut self.agent_states);

        // Generate and execute orders
        self.generate_and_execute_orders();

        // Collect metrics
        let cooperation_by_type = self.kernel.cooperation_by_type(&self.agent_states);
        self.metrics.collect(
            self.timestep,
            &self.book,
            btut_stats.mean_cooperation,
            Some(cooperation_by_type),
        );

        // Periodic logging
        if self.timestep % 100 == 0 {
            debug!(
                timestep = self.timestep,
                mid_price = ?self.book.mid_price(),
                spread_bps = ?self.book.spread_bps(),
                cooperation = btut_stats.mean_cooperation,
                "Step complete"
            );
        }
    }

    /// Apply any scheduled shocks for current timestep
    fn apply_scheduled_shocks(&mut self) {
        let shocks: Vec<_> = self
            .scheduled_shocks
            .iter()
            .filter(|s| s.timestep == self.timestep)
            .map(|s| s.shock_type)
            .collect();

        for shock in shocks {
            warn!(timestep = self.timestep, shock = ?shock, "Applying shock");
            self.kernel.apply_shock(shock);
        }
    }

    /// Generate orders from all agents and execute them
    fn generate_and_execute_orders(&mut self) {
        // In a real implementation, we would have agent objects generate orders.
        // For scalability with 1M+ agents, we use a simplified flow model.
        self.generate_scaled_flow();
    }

    /// Generate simple order flow for warmup
    fn generate_simple_flow(&mut self) {
        let mid = self.book.mid_price().unwrap_or(100.0);
        let spread = self.book.spread().unwrap_or(0.02);

        // Add some liquidity
        for _ in 0..10 {
            let offset = self.rng.uniform(0.0, spread * 2.0);
            let qty = self.rng.log_normal(3.0, 0.5);

            if self.rng.bernoulli(0.5) {
                // Bid
                let order = crate::market::Order::limit(
                    self.book.next_order_id(),
                    crate::agent::AgentId(0),
                    crate::market::OrderSide::Bid,
                    mid - offset,
                    qty,
                    self.timestep,
                );
                self.book.submit_order(order);
            } else {
                // Ask
                let order = crate::market::Order::limit(
                    self.book.next_order_id(),
                    crate::agent::AgentId(0),
                    crate::market::OrderSide::Ask,
                    mid + offset,
                    qty,
                    self.timestep,
                );
                self.book.submit_order(order);
            }
        }
    }

    /// Generate scaled order flow based on agent population statistics
    ///
    /// This is an O(1) approximation of agent behavior that scales to millions of agents.
    fn generate_scaled_flow(&mut self) {
        let num_agents = self.config.simulation.num_agents;
        let mid = self.book.mid_price().unwrap_or(100.0);
        let spread = self.book.spread().unwrap_or(0.02);
        let mean_coop = self.kernel.mean_cooperation();

        // Calculate expected order flow based on agent population
        let retail_activity = self.config.agents.retail_fraction * 0.01;
        let hft_activity = self.config.agents.hft_fraction * 0.50;
        let inst_activity = self.config.agents.institution_fraction * 0.05;
        let mm_activity = self.config.agents.market_maker_fraction * 0.80;

        let total_activity = retail_activity + hft_activity + inst_activity + mm_activity;
        let expected_orders = (num_agents as f64 * total_activity) as usize;

        // Scale down to reasonable number while preserving statistics
        let actual_orders = expected_orders.min(1000);
        let order_scale = expected_orders as f64 / actual_orders as f64;

        for _ in 0..actual_orders {
            // Sample agent type based on activity-weighted distribution
            let roll = self.rng.uniform(0.0, total_activity);
            let (is_cooperative, size_mult, is_liquidity_provider) = if roll < retail_activity {
                // Retail
                let coop_idx = (self.rng.uniform(0.0, self.config.agents.retail_fraction)
                    * num_agents as f64) as usize;
                let coop = self
                    .agent_states
                    .get(coop_idx)
                    .map_or(mean_coop, |s| s.cooperation);
                (coop > 0.5, 1.0, false)
            } else if roll < retail_activity + hft_activity {
                // HFT
                let base_idx = (self.config.agents.retail_fraction * num_agents as f64) as usize;
                let coop_idx = base_idx
                    + (self.rng.uniform(0.0, self.config.agents.hft_fraction) * num_agents as f64)
                        as usize;
                let coop = self
                    .agent_states
                    .get(coop_idx)
                    .map_or(mean_coop, |s| s.cooperation);
                (coop > 0.4, 5.0, coop > 0.5)
            } else if roll < retail_activity + hft_activity + inst_activity {
                // Institution
                (mean_coop > 0.4, 50.0, false)
            } else {
                // Market Maker
                (true, 20.0, true)
            };

            // Generate order
            let base_size = self.generator.generate_order_size() * size_mult * order_scale;

            if is_liquidity_provider && is_cooperative {
                // Two-sided quotes
                let half_spread = spread * (0.3 + 0.7 * (1.0 - mean_coop));

                // Bid
                let bid = crate::market::Order::limit(
                    self.book.next_order_id(),
                    crate::agent::AgentId(0),
                    crate::market::OrderSide::Bid,
                    mid - half_spread,
                    base_size,
                    self.timestep,
                );
                self.book.submit_order(bid);

                // Ask
                let ask = crate::market::Order::limit(
                    self.book.next_order_id(),
                    crate::agent::AgentId(0),
                    crate::market::OrderSide::Ask,
                    mid + half_spread,
                    base_size,
                    self.timestep,
                );
                self.book.submit_order(ask);
            } else if is_cooperative {
                // Passive limit order
                let side = if self.rng.bernoulli(0.5) {
                    crate::market::OrderSide::Bid
                } else {
                    crate::market::OrderSide::Ask
                };
                let offset = spread * self.rng.uniform(0.5, 2.0);
                let price = match side {
                    crate::market::OrderSide::Bid => mid - offset,
                    crate::market::OrderSide::Ask => mid + offset,
                };

                let order = crate::market::Order::limit(
                    self.book.next_order_id(),
                    crate::agent::AgentId(0),
                    side,
                    price,
                    base_size,
                    self.timestep,
                );
                self.book.submit_order(order);
            } else {
                // Aggressive market order (defecting)
                let side = if self.rng.bernoulli(0.5) {
                    crate::market::OrderSide::Bid
                } else {
                    crate::market::OrderSide::Ask
                };

                let order = crate::market::Order::market(
                    self.book.next_order_id(),
                    crate::agent::AgentId(0),
                    side,
                    base_size * 0.5, // Smaller for aggression
                    self.timestep,
                );
                self.book.submit_order(order);
            }
        }

        // Cancel some old orders to maintain book health
        self.prune_order_book();
    }

    /// Remove stale orders to prevent unbounded book growth
    fn prune_order_book(&mut self) {
        // Simple approach: cancel orders older than 50 timesteps
        // In a real implementation, this would be more sophisticated
        if self.timestep > 50 && self.timestep % 10 == 0 {
            // Reinitialize some liquidity to maintain depth
            let mid = self.book.mid_price().unwrap_or(100.0);
            let depth_to_add = (self.book.num_bid_levels() < 50) as usize * 20
                + (self.book.num_ask_levels() < 50) as usize * 20;

            for i in 1..=depth_to_add {
                let offset = i as f64 * self.config.market.tick_size;
                let qty = 100.0 * (-0.05 * i as f64).exp();

                if self.book.num_bid_levels() < 50 {
                    let bid = crate::market::Order::limit(
                        self.book.next_order_id(),
                        crate::agent::AgentId(0),
                        crate::market::OrderSide::Bid,
                        mid - offset,
                        qty,
                        self.timestep,
                    );
                    self.book.submit_order(bid);
                }

                if self.book.num_ask_levels() < 50 {
                    let ask = crate::market::Order::limit(
                        self.book.next_order_id(),
                        crate::agent::AgentId(0),
                        crate::market::OrderSide::Ask,
                        mid + offset,
                        qty,
                        self.timestep,
                    );
                    self.book.submit_order(ask);
                }
            }
        }
    }

    /// Get current simulation state
    pub fn state(&self) -> SimulationState {
        self.state
    }

    /// Get current timestep
    pub fn timestep(&self) -> u64 {
        self.timestep
    }

    /// Get reference to order book
    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    /// Get reference to BTUT kernel
    pub fn kernel(&self) -> &BTUTKernel {
        &self.kernel
    }

    /// Get reference to metrics
    pub fn metrics(&self) -> &MetricsCollector {
        &self.metrics
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.map_or(Duration::ZERO, |t| t.elapsed())
    }
}

/// Run a baseline comparison between BTUT and zero-intelligence agents
#[instrument]
pub fn run_baseline_comparison(
    config: &SimulationConfig,
    shock_timestep: u64,
) -> (SimulationResults, SimulationResults) {
    info!("Running baseline comparison");

    // Run BTUT simulation
    let mut btut_config = config.clone();
    btut_config.output.show_progress = false;

    let mut btut_sim = Simulation::new(btut_config.clone());
    btut_sim.schedule_liquidity_crisis(shock_timestep, 0.5);
    let btut_results = btut_sim.run();

    // Run zero-intelligence simulation (cooperation fixed at 0.5)
    let mut zi_config = config.clone();
    zi_config.btut.momentum_strength = 1.0; // No BTUT updates (strategies don't change)
    zi_config.agents.initial_cooperation = 0.5;
    zi_config.agents.cooperation_noise = 0.0;
    zi_config.output.show_progress = false;

    let mut zi_sim = Simulation::new(zi_config);
    zi_sim.schedule_liquidity_crisis(shock_timestep, 0.5);
    let zi_results = zi_sim.run();

    info!(
        btut_crashes = btut_results.crashes.len(),
        zi_crashes = zi_results.crashes.len(),
        "Baseline comparison complete"
    );

    (btut_results, zi_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_creation() {
        let config = SimulationConfig::test_preset();
        let sim = Simulation::new(config);
        assert_eq!(sim.state(), SimulationState::NotStarted);
    }

    #[test]
    fn test_simulation_run() {
        let mut config = SimulationConfig::test_preset();
        config.simulation.num_agents = 100;
        config.simulation.num_timesteps = 50;
        config.output.show_progress = false;

        let mut sim = Simulation::new(config);
        let results = sim.run();

        assert_eq!(results.state, SimulationState::Completed);
        assert!(!results.metrics.is_empty());
        assert!(results.runtime_ms > 0);
    }

    #[test]
    fn test_scheduled_shock() {
        let mut config = SimulationConfig::test_preset();
        config.simulation.num_agents = 100;
        config.simulation.num_timesteps = 100;
        config.output.show_progress = false;

        let mut sim = Simulation::new(config);
        sim.schedule_liquidity_crisis(50, 0.5);

        let results = sim.run();
        assert_eq!(results.state, SimulationState::Completed);
    }

    #[test]
    fn test_baseline_comparison() {
        let mut config = SimulationConfig::test_preset();
        config.simulation.num_agents = 100;
        config.simulation.num_timesteps = 50;
        config.output.show_progress = false;

        let (btut_results, zi_results) = run_baseline_comparison(&config, 25);

        assert_eq!(btut_results.state, SimulationState::Completed);
        assert_eq!(zi_results.state, SimulationState::Completed);
    }
}
