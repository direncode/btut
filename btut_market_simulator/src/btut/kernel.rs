//! Core BTUT Kernel Implementation
//!
//! This module contains the main BTUT algorithm for coordinating agent strategies
//! using kernel-weighted mean-field dynamics with O(N) complexity.

use crate::agent::{AgentId, AgentState, AgentType};
use crate::config::BTUTConfig;
use crate::utils::SimRng;
use ndarray::{Array1, Zip};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, instrument, trace};

/// Agent strategy value (0 = full defect, 1 = full cooperate)
pub type Strategy = f64;

/// Kernel configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Kernel decay parameter (λ) - controls influence falloff
    pub decay: f64,
    /// Momentum strength (α) for strategy updates
    pub momentum: f64,
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Convergence threshold
    pub threshold: f64,
    /// Temperature for softmax-style updates
    pub temperature: f64,
    /// Cooperation bias (shifts equilibrium)
    pub bias: f64,
    /// Enable adaptive kernel decay
    pub adaptive: bool,
}

impl From<&BTUTConfig> for KernelConfig {
    fn from(config: &BTUTConfig) -> Self {
        Self {
            decay: config.kernel_decay,
            momentum: config.momentum_strength,
            max_iterations: config.convergence_iterations,
            threshold: config.convergence_threshold,
            temperature: config.temperature,
            bias: config.cooperation_bias,
            adaptive: config.adaptive_kernel,
        }
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            decay: 0.1,
            momentum: 0.7,
            max_iterations: 25,
            threshold: 1e-6,
            temperature: 1.0,
            bias: 0.0,
            adaptive: false,
        }
    }
}

/// Strategy update result for a single agent
#[derive(Debug, Clone, Copy)]
pub struct StrategyUpdate {
    /// Agent ID
    pub agent_id: AgentId,
    /// Previous strategy
    pub old_strategy: Strategy,
    /// New strategy
    pub new_strategy: Strategy,
    /// Change magnitude
    pub delta: f64,
}

/// Statistics from a BTUT update cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTUTStats {
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Final maximum strategy change
    pub max_delta: f64,
    /// Mean cooperation level after update
    pub mean_cooperation: f64,
    /// Standard deviation of cooperation
    pub std_cooperation: f64,
    /// Fraction of agents cooperating (s > 0.5)
    pub cooperation_fraction: f64,
    /// Did the update converge?
    pub converged: bool,
}

/// The core BTUT kernel for multi-agent coordination
///
/// This struct maintains the state needed to compute kernel-weighted mean-field
/// updates for agent strategies. It achieves O(N) complexity by avoiding explicit
/// pairwise interactions.
pub struct BTUTKernel {
    /// Kernel configuration
    config: KernelConfig,
    /// Agent strategies (indexed by agent order, not ID)
    strategies: Array1<f64>,
    /// Agent weights (based on size/type)
    weights: Array1<f64>,
    /// Agent distances (for kernel computation)
    distances: Array1<f64>,
    /// Precomputed kernel values
    kernel_values: Array1<f64>,
    /// Scratch space for new strategies
    new_strategies: Array1<f64>,
    /// Random noise generator
    rng: SimRng,
    /// Current adaptive decay value
    current_decay: f64,
    /// Iteration counter for stats
    total_iterations: AtomicU64,
}

impl BTUTKernel {
    /// Create a new BTUT kernel with the given configuration
    #[instrument(skip(agents, rng))]
    pub fn new(config: KernelConfig, agents: &[AgentState], rng: SimRng) -> Self {
        let n = agents.len();
        debug!(num_agents = n, "Initializing BTUT kernel");

        // Initialize arrays
        let mut strategies = Array1::zeros(n);
        let mut weights = Array1::zeros(n);
        let mut distances = Array1::zeros(n);

        // Populate from agent states
        for (i, agent) in agents.iter().enumerate() {
            strategies[i] = agent.cooperation;
            weights[i] = Self::compute_weight(agent);
            distances[i] = Self::compute_distance(agent);
        }

        // Precompute kernel values
        let kernel_values = distances.mapv(|d| (-config.decay * d).exp());

        Self {
            current_decay: config.decay,
            config,
            strategies,
            weights,
            distances,
            kernel_values,
            new_strategies: Array1::zeros(n),
            rng,
            total_iterations: AtomicU64::new(0),
        }
    }

    /// Compute agent weight based on type and size
    fn compute_weight(agent: &AgentState) -> f64 {
        let type_weight = match agent.agent_type {
            AgentType::Retail => 1.0,
            AgentType::HFT => 5.0,
            AgentType::Institution => 20.0,
            AgentType::MarketMaker => 10.0,
        };
        // Weight scales with sqrt of capital (diminishing influence)
        type_weight * agent.capital.sqrt().max(1.0)
    }

    /// Compute agent "distance" in the market space
    /// Distance represents how far from the core market the agent operates
    fn compute_distance(agent: &AgentState) -> f64 {
        // Combine factors: order book depth position, typical spread, trade frequency
        let type_distance = match agent.agent_type {
            AgentType::MarketMaker => 0.1, // Closest to the market
            AgentType::HFT => 0.3,
            AgentType::Institution => 0.6,
            AgentType::Retail => 1.0, // Farthest from core
        };

        // Add some variation based on capital
        let capital_factor = 1.0 / (1.0 + (agent.capital / 10000.0).ln().max(0.0));
        type_distance * capital_factor
    }

    /// Perform one complete BTUT update cycle (multiple iterations until convergence)
    #[instrument(skip(self))]
    pub fn update_cycle(&mut self, market_volatility: f64) -> BTUTStats {
        // Adapt kernel decay based on volatility if enabled
        if self.config.adaptive {
            // Higher volatility = slower decay = longer-range correlations
            self.current_decay = self.config.decay / (1.0 + market_volatility * 10.0);
            self.kernel_values = self.distances.mapv(|d| (-self.current_decay * d).exp());
            trace!(decay = self.current_decay, "Adapted kernel decay");
        }

        let mut iterations = 0;
        let mut max_delta = f64::MAX;
        let mut converged = false;

        while iterations < self.config.max_iterations && !converged {
            max_delta = self.update_iteration();
            iterations += 1;
            converged = max_delta < self.config.threshold;

            trace!(
                iteration = iterations,
                max_delta = max_delta,
                "BTUT iteration"
            );
        }

        self.total_iterations
            .fetch_add(iterations as u64, Ordering::Relaxed);

        // Compute statistics
        let mean_cooperation = self.strategies.mean().unwrap_or(0.5);
        let variance = self
            .strategies
            .mapv(|s| (s - mean_cooperation).powi(2))
            .mean()
            .unwrap_or(0.0);
        let std_cooperation = variance.sqrt();
        let cooperation_fraction =
            self.strategies.iter().filter(|&&s| s > 0.5).count() as f64 / self.strategies.len() as f64;

        debug!(
            iterations = iterations,
            converged = converged,
            mean_cooperation = mean_cooperation,
            "BTUT cycle complete"
        );

        BTUTStats {
            iterations,
            max_delta,
            mean_cooperation,
            std_cooperation,
            cooperation_fraction,
            converged,
        }
    }

    /// Perform a single BTUT iteration, returning the maximum strategy change
    fn update_iteration(&mut self) -> f64 {
        let n = self.strategies.len();
        if n == 0 {
            return 0.0;
        }

        // Step 1: Compute weighted mean field (global cooperation level)
        let total_weight: f64 = self.weights.sum();
        let weighted_sum: f64 = Zip::from(&self.strategies)
            .and(&self.weights)
            .fold(0.0, |acc, &s, &w| acc + s * w);

        let global_cooperation = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.5
        };

        // Step 2: Update each agent's strategy
        // Using parallel iteration if enabled
        #[cfg(feature = "parallel")]
        {
            use ndarray::parallel::prelude::*;

            let config = &self.config;
            let strategies = &self.strategies;
            let kernel_values = &self.kernel_values;

            self.new_strategies
                .as_slice_mut()
                .unwrap()
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, new_s)| {
                    let old_s = strategies[i];
                    let k = kernel_values[i];

                    // Target strategy based on mean field + bias
                    let target = (global_cooperation + config.bias).clamp(0.0, 1.0);

                    // Softmax-style update with temperature
                    let influence = k * (target - old_s) / config.temperature;

                    // Momentum-based update
                    *new_s = config.momentum * old_s + (1.0 - config.momentum) * (old_s + influence);
                    *new_s = new_s.clamp(0.0, 1.0);
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n {
                let old_s = self.strategies[i];
                let k = self.kernel_values[i];

                let target = (global_cooperation + self.config.bias).clamp(0.0, 1.0);
                let influence = k * (target - old_s) / self.config.temperature;

                self.new_strategies[i] =
                    self.config.momentum * old_s + (1.0 - self.config.momentum) * (old_s + influence);
                self.new_strategies[i] = self.new_strategies[i].clamp(0.0, 1.0);
            }
        }

        // Step 3: Compute max delta and swap strategies
        let max_delta = Zip::from(&self.strategies)
            .and(&self.new_strategies)
            .fold(0.0f64, |acc, &old, &new| acc.max((new - old).abs()));

        std::mem::swap(&mut self.strategies, &mut self.new_strategies);

        max_delta
    }

    /// Apply an external shock to agent strategies (e.g., liquidity crisis)
    pub fn apply_shock(&mut self, shock_type: ShockType) {
        match shock_type {
            ShockType::LiquidityCrisis { severity } => {
                // Reduce cooperation proportionally to severity
                self.strategies
                    .mapv_inplace(|s| (s - severity * 0.5).clamp(0.0, 1.0));
            }
            ShockType::FlashCrash { duration: _ } => {
                // HFT and MM agents defect rapidly
                for i in 0..self.strategies.len() {
                    if self.distances[i] < 0.5 {
                        // Close to market = more affected
                        self.strategies[i] = (self.strategies[i] * 0.3).clamp(0.0, 1.0);
                    }
                }
            }
            ShockType::Regeneration { strength } => {
                // Boost cooperation
                self.strategies
                    .mapv_inplace(|s| (s + strength * 0.3).clamp(0.0, 1.0));
            }
            ShockType::Noise { magnitude } => {
                // Add random perturbation
                for i in 0..self.strategies.len() {
                    let noise = self.rng.normal(0.0, magnitude);
                    self.strategies[i] = (self.strategies[i] + noise).clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Get current strategy for an agent by index
    pub fn get_strategy(&self, index: usize) -> Option<Strategy> {
        self.strategies.get(index).copied()
    }

    /// Get all current strategies
    pub fn strategies(&self) -> &Array1<f64> {
        &self.strategies
    }

    /// Get strategy as a slice for efficient iteration
    pub fn strategies_slice(&self) -> &[f64] {
        self.strategies.as_slice().unwrap()
    }

    /// Update a single agent's parameters (e.g., after trade)
    pub fn update_agent(&mut self, index: usize, new_state: &AgentState) {
        if index < self.strategies.len() {
            self.strategies[index] = new_state.cooperation;
            self.weights[index] = Self::compute_weight(new_state);
            self.distances[index] = Self::compute_distance(new_state);
            self.kernel_values[index] = (-self.current_decay * self.distances[index]).exp();
        }
    }

    /// Sync all agent states from external source
    pub fn sync_from_agents(&mut self, agents: &[AgentState]) {
        for (i, agent) in agents.iter().enumerate() {
            if i < self.strategies.len() {
                self.strategies[i] = agent.cooperation;
            }
        }
    }

    /// Write updated strategies back to agent states
    pub fn sync_to_agents(&self, agents: &mut [AgentState]) {
        for (i, agent) in agents.iter_mut().enumerate() {
            if let Some(&s) = self.strategies.get(i) {
                agent.cooperation = s;
            }
        }
    }

    /// Get the current mean cooperation level
    pub fn mean_cooperation(&self) -> f64 {
        self.strategies.mean().unwrap_or(0.5)
    }

    /// Get cooperation level by agent type
    pub fn cooperation_by_type(&self, agents: &[AgentState]) -> CooperationByType {
        let mut retail_sum = 0.0;
        let mut retail_count = 0;
        let mut hft_sum = 0.0;
        let mut hft_count = 0;
        let mut inst_sum = 0.0;
        let mut inst_count = 0;
        let mut mm_sum = 0.0;
        let mut mm_count = 0;

        for (i, agent) in agents.iter().enumerate() {
            if let Some(&s) = self.strategies.get(i) {
                match agent.agent_type {
                    AgentType::Retail => {
                        retail_sum += s;
                        retail_count += 1;
                    }
                    AgentType::HFT => {
                        hft_sum += s;
                        hft_count += 1;
                    }
                    AgentType::Institution => {
                        inst_sum += s;
                        inst_count += 1;
                    }
                    AgentType::MarketMaker => {
                        mm_sum += s;
                        mm_count += 1;
                    }
                }
            }
        }

        CooperationByType {
            retail: if retail_count > 0 {
                retail_sum / retail_count as f64
            } else {
                0.5
            },
            hft: if hft_count > 0 {
                hft_sum / hft_count as f64
            } else {
                0.5
            },
            institution: if inst_count > 0 {
                inst_sum / inst_count as f64
            } else {
                0.5
            },
            market_maker: if mm_count > 0 {
                mm_sum / mm_count as f64
            } else {
                0.5
            },
        }
    }

    /// Get total iterations performed
    pub fn total_iterations(&self) -> u64 {
        self.total_iterations.load(Ordering::Relaxed)
    }

    /// Get current kernel configuration
    pub fn config(&self) -> &KernelConfig {
        &self.config
    }

    /// Update kernel configuration
    pub fn set_config(&mut self, config: KernelConfig) {
        self.config = config;
        self.current_decay = self.config.decay;
        self.kernel_values = self.distances.mapv(|d| (-self.current_decay * d).exp());
    }

    /// Get number of agents
    pub fn num_agents(&self) -> usize {
        self.strategies.len()
    }
}

/// Types of external shocks that can be applied
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ShockType {
    /// Liquidity crisis: agents reduce cooperation
    LiquidityCrisis {
        /// Severity from 0.0 (mild) to 1.0 (severe)
        severity: f64,
    },
    /// Flash crash: rapid defection by fast agents
    FlashCrash {
        /// Duration in timesteps
        duration: usize,
    },
    /// Market regeneration: cooperation boost
    Regeneration {
        /// Strength of regeneration
        strength: f64,
    },
    /// Random noise injection
    Noise {
        /// Noise magnitude (std dev)
        magnitude: f64,
    },
}

/// Cooperation levels broken down by agent type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CooperationByType {
    /// Retail agent cooperation
    pub retail: f64,
    /// HFT agent cooperation
    pub hft: f64,
    /// Institutional agent cooperation
    pub institution: f64,
    /// Market maker cooperation
    pub market_maker: f64,
}

impl std::fmt::Debug for BTUTKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BTUTKernel")
            .field("num_agents", &self.strategies.len())
            .field("config", &self.config)
            .field("mean_cooperation", &self.mean_cooperation())
            .field(
                "total_iterations",
                &self.total_iterations.load(Ordering::Relaxed),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_agents(n: usize) -> Vec<AgentState> {
        (0..n)
            .map(|i| AgentState {
                id: AgentId(i as u64),
                agent_type: match i % 4 {
                    0 => AgentType::Retail,
                    1 => AgentType::HFT,
                    2 => AgentType::Institution,
                    _ => AgentType::MarketMaker,
                },
                capital: 10000.0 + (i as f64 * 100.0),
                cooperation: 0.5,
                position: 0.0,
                pnl: 0.0,
            })
            .collect()
    }

    #[test]
    fn test_kernel_initialization() {
        let agents = create_test_agents(100);
        let rng = SimRng::from_seed(42);
        let kernel = BTUTKernel::new(KernelConfig::default(), &agents, rng);

        assert_eq!(kernel.num_agents(), 100);
        assert!((kernel.mean_cooperation() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_convergence() {
        let agents = create_test_agents(1000);
        let rng = SimRng::from_seed(42);
        let mut kernel = BTUTKernel::new(KernelConfig::default(), &agents, rng);

        let stats = kernel.update_cycle(0.01);

        assert!(stats.converged || stats.iterations == kernel.config().max_iterations);
        assert!(stats.max_delta < 0.1);
    }

    #[test]
    fn test_shock_application() {
        let agents = create_test_agents(100);
        let rng = SimRng::from_seed(42);
        let mut kernel = BTUTKernel::new(KernelConfig::default(), &agents, rng);

        let initial_coop = kernel.mean_cooperation();

        kernel.apply_shock(ShockType::LiquidityCrisis { severity: 0.5 });

        let post_shock_coop = kernel.mean_cooperation();
        assert!(post_shock_coop < initial_coop);
    }

    #[test]
    fn test_cooperation_by_type() {
        let agents = create_test_agents(100);
        let rng = SimRng::from_seed(42);
        let kernel = BTUTKernel::new(KernelConfig::default(), &agents, rng);

        let coop = kernel.cooperation_by_type(&agents);

        assert!(coop.retail >= 0.0 && coop.retail <= 1.0);
        assert!(coop.hft >= 0.0 && coop.hft <= 1.0);
        assert!(coop.institution >= 0.0 && coop.institution <= 1.0);
        assert!(coop.market_maker >= 0.0 && coop.market_maker <= 1.0);
    }

    #[test]
    fn test_large_scale() {
        // Test with 100k agents
        let agents = create_test_agents(100_000);
        let rng = SimRng::from_seed(42);
        let mut kernel = BTUTKernel::new(KernelConfig::default(), &agents, rng);

        let start = std::time::Instant::now();
        let stats = kernel.update_cycle(0.01);
        let elapsed = start.elapsed();

        println!("100k agents: {:?}, stats: {:?}", elapsed, stats);
        assert!(stats.iterations <= 25);
        // Should complete in under 1 second
        assert!(elapsed.as_secs() < 1);
    }
}
