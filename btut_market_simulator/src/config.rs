//! Configuration structures for the BTUT Market Simulator.
//!
//! This module provides strongly-typed configuration for all simulation parameters,
//! supporting loading from TOML files, environment variables, and programmatic construction.
//!
//! # Example Configuration File (config.toml)
//!
//! ```toml
//! [simulation]
//! num_agents = 1000000
//! num_timesteps = 1000
//! seed = 42
//!
//! [btut]
//! kernel_decay = 0.1
//! momentum_strength = 0.7
//! convergence_iterations = 25
//!
//! [market]
//! initial_price = 100.0
//! tick_size = 0.01
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    /// IO error when reading config file
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),

    /// TOML parsing error
    #[error("Failed to parse config: {0}")]
    ParseError(#[from] toml::de::Error),

    /// Validation error
    #[error("Invalid configuration: {0}")]
    ValidationError(String),
}

/// Master configuration for the entire simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Core simulation parameters
    pub simulation: CoreConfig,

    /// BTUT kernel parameters
    pub btut: BTUTConfig,

    /// Market/order book parameters
    pub market: MarketConfig,

    /// Agent population parameters
    pub agents: AgentConfig,

    /// Output and logging parameters
    pub output: OutputConfig,
}

/// Core simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    /// Total number of agents in the simulation
    #[serde(default = "default_num_agents")]
    pub num_agents: usize,

    /// Number of simulation timesteps to run
    #[serde(default = "default_num_timesteps")]
    pub num_timesteps: usize,

    /// Random seed for reproducibility (None = random)
    #[serde(default)]
    pub seed: Option<u64>,

    /// Enable parallel execution with Rayon
    #[serde(default = "default_true")]
    pub parallel: bool,

    /// Number of threads (0 = auto-detect)
    #[serde(default)]
    pub num_threads: usize,

    /// Warmup period (timesteps before metrics collection)
    #[serde(default = "default_warmup")]
    pub warmup_steps: usize,
}

/// BTUT kernel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTUTConfig {
    /// Kernel decay parameter (λ) - controls influence falloff with distance
    /// Higher values = faster decay = more localized interactions
    #[serde(default = "default_kernel_decay")]
    pub kernel_decay: f64,

    /// Momentum strength (α) for strategy updates
    /// new_strategy = α * old_strategy + (1-α) * btut_update
    #[serde(default = "default_momentum")]
    pub momentum_strength: f64,

    /// Number of BTUT convergence iterations per timestep
    #[serde(default = "default_iterations")]
    pub convergence_iterations: usize,

    /// Convergence threshold (stop early if change < threshold)
    #[serde(default = "default_convergence_threshold")]
    pub convergence_threshold: f64,

    /// Temperature parameter for softmax strategy selection
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Cooperation bias (shifts equilibrium toward cooperation)
    #[serde(default)]
    pub cooperation_bias: f64,

    /// Enable adaptive kernel decay based on market volatility
    #[serde(default)]
    pub adaptive_kernel: bool,
}

/// Market configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConfig {
    /// Initial mid-price for the order book
    #[serde(default = "default_initial_price")]
    pub initial_price: f64,

    /// Minimum price increment (tick size)
    #[serde(default = "default_tick_size")]
    pub tick_size: f64,

    /// Maximum order book depth (price levels per side)
    #[serde(default = "default_book_depth")]
    pub max_book_depth: usize,

    /// Lot size (minimum order quantity)
    #[serde(default = "default_lot_size")]
    pub lot_size: f64,

    /// Maximum order size
    #[serde(default = "default_max_order_size")]
    pub max_order_size: f64,

    /// Enable order book snapshots
    #[serde(default = "default_true")]
    pub enable_snapshots: bool,

    /// Snapshot frequency (every N timesteps)
    #[serde(default = "default_snapshot_freq")]
    pub snapshot_frequency: usize,
}

/// Agent population configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Fraction of retail agents (uninformed, random)
    #[serde(default = "default_retail_fraction")]
    pub retail_fraction: f64,

    /// Fraction of HFT agents (fast, informed, small size)
    #[serde(default = "default_hft_fraction")]
    pub hft_fraction: f64,

    /// Fraction of institutional agents (slow, large size)
    #[serde(default = "default_institution_fraction")]
    pub institution_fraction: f64,

    /// Fraction of market maker agents (liquidity providers)
    #[serde(default = "default_market_maker_fraction")]
    pub market_maker_fraction: f64,

    /// Power-law exponent for order size distribution
    #[serde(default = "default_order_size_exponent")]
    pub order_size_exponent: f64,

    /// Base capital for retail agents
    #[serde(default = "default_retail_capital")]
    pub retail_capital: f64,

    /// Capital multiplier for institutions
    #[serde(default = "default_institution_multiplier")]
    pub institution_capital_multiplier: f64,

    /// Initial cooperation level (0 = full defect, 1 = full cooperate)
    #[serde(default = "default_initial_cooperation")]
    pub initial_cooperation: f64,

    /// Cooperation noise (std dev of initial cooperation)
    #[serde(default = "default_cooperation_noise")]
    pub cooperation_noise: f64,
}

/// Output and logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory for results
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Output format (csv, json, both)
    #[serde(default)]
    pub format: OutputFormat,

    /// Log level (trace, debug, info, warn, error)
    #[serde(default)]
    pub log_level: LogLevelConfig,

    /// Enable JSON-formatted logs
    #[serde(default)]
    pub json_logs: bool,

    /// Metrics sampling rate (collect every N timesteps)
    #[serde(default = "default_one")]
    pub metrics_sample_rate: usize,

    /// Enable progress bar
    #[serde(default = "default_true")]
    pub show_progress: bool,

    /// Compression for output files
    #[serde(default)]
    pub compress_output: bool,
}

/// Output format options
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// CSV format
    #[default]
    Csv,
    /// JSON format
    Json,
    /// Both CSV and JSON
    Both,
}

/// Log level configuration
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevelConfig {
    /// Trace level (most verbose)
    Trace,
    /// Debug level
    Debug,
    /// Info level (default)
    #[default]
    Info,
    /// Warning level
    Warn,
    /// Error level (least verbose)
    Error,
}

// Default value functions
fn default_num_agents() -> usize {
    100_000
}
fn default_num_timesteps() -> usize {
    1000
}
fn default_warmup() -> usize {
    100
}
fn default_kernel_decay() -> f64 {
    0.1
}
fn default_momentum() -> f64 {
    0.7
}
fn default_iterations() -> usize {
    25
}
fn default_convergence_threshold() -> f64 {
    1e-6
}
fn default_temperature() -> f64 {
    1.0
}
fn default_initial_price() -> f64 {
    100.0
}
fn default_tick_size() -> f64 {
    0.01
}
fn default_book_depth() -> usize {
    1000
}
fn default_lot_size() -> f64 {
    1.0
}
fn default_max_order_size() -> f64 {
    10000.0
}
fn default_snapshot_freq() -> usize {
    10
}
fn default_retail_fraction() -> f64 {
    0.70
}
fn default_hft_fraction() -> f64 {
    0.05
}
fn default_institution_fraction() -> f64 {
    0.10
}
fn default_market_maker_fraction() -> f64 {
    0.15
}
fn default_order_size_exponent() -> f64 {
    1.5
}
fn default_retail_capital() -> f64 {
    10_000.0
}
fn default_institution_multiplier() -> f64 {
    1000.0
}
fn default_initial_cooperation() -> f64 {
    0.5
}
fn default_cooperation_noise() -> f64 {
    0.1
}
fn default_output_dir() -> String {
    "output".to_string()
}
fn default_true() -> bool {
    true
}
fn default_one() -> usize {
    1
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            simulation: CoreConfig::default(),
            btut: BTUTConfig::default(),
            market: MarketConfig::default(),
            agents: AgentConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            num_agents: default_num_agents(),
            num_timesteps: default_num_timesteps(),
            seed: None,
            parallel: true,
            num_threads: 0,
            warmup_steps: default_warmup(),
        }
    }
}

impl Default for BTUTConfig {
    fn default() -> Self {
        Self {
            kernel_decay: default_kernel_decay(),
            momentum_strength: default_momentum(),
            convergence_iterations: default_iterations(),
            convergence_threshold: default_convergence_threshold(),
            temperature: default_temperature(),
            cooperation_bias: 0.0,
            adaptive_kernel: false,
        }
    }
}

impl Default for MarketConfig {
    fn default() -> Self {
        Self {
            initial_price: default_initial_price(),
            tick_size: default_tick_size(),
            max_book_depth: default_book_depth(),
            lot_size: default_lot_size(),
            max_order_size: default_max_order_size(),
            enable_snapshots: true,
            snapshot_frequency: default_snapshot_freq(),
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            retail_fraction: default_retail_fraction(),
            hft_fraction: default_hft_fraction(),
            institution_fraction: default_institution_fraction(),
            market_maker_fraction: default_market_maker_fraction(),
            order_size_exponent: default_order_size_exponent(),
            retail_capital: default_retail_capital(),
            institution_capital_multiplier: default_institution_multiplier(),
            initial_cooperation: default_initial_cooperation(),
            cooperation_noise: default_cooperation_noise(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            output_dir: default_output_dir(),
            format: OutputFormat::default(),
            log_level: LogLevelConfig::default(),
            json_logs: false,
            metrics_sample_rate: 1,
            show_progress: true,
            compress_output: false,
        }
    }
}

impl SimulationConfig {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration with overrides from environment variables
    pub fn from_file_with_env<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let mut config = Self::from_file(path)?;
        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }

    /// Apply environment variable overrides
    /// Format: BTUT_SECTION_KEY (e.g., BTUT_SIMULATION_NUM_AGENTS)
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("BTUT_SIMULATION_NUM_AGENTS") {
            if let Ok(n) = val.parse() {
                self.simulation.num_agents = n;
            }
        }
        if let Ok(val) = std::env::var("BTUT_SIMULATION_NUM_TIMESTEPS") {
            if let Ok(n) = val.parse() {
                self.simulation.num_timesteps = n;
            }
        }
        if let Ok(val) = std::env::var("BTUT_SIMULATION_SEED") {
            if let Ok(n) = val.parse() {
                self.simulation.seed = Some(n);
            }
        }
        if let Ok(val) = std::env::var("BTUT_BTUT_KERNEL_DECAY") {
            if let Ok(n) = val.parse() {
                self.btut.kernel_decay = n;
            }
        }
        if let Ok(val) = std::env::var("BTUT_BTUT_MOMENTUM_STRENGTH") {
            if let Ok(n) = val.parse() {
                self.btut.momentum_strength = n;
            }
        }
        if let Ok(val) = std::env::var("BTUT_OUTPUT_LOG_LEVEL") {
            match val.to_lowercase().as_str() {
                "trace" => self.output.log_level = LogLevelConfig::Trace,
                "debug" => self.output.log_level = LogLevelConfig::Debug,
                "info" => self.output.log_level = LogLevelConfig::Info,
                "warn" => self.output.log_level = LogLevelConfig::Warn,
                "error" => self.output.log_level = LogLevelConfig::Error,
                _ => {}
            }
        }
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Agent fractions must sum to 1.0
        let total_fraction = self.agents.retail_fraction
            + self.agents.hft_fraction
            + self.agents.institution_fraction
            + self.agents.market_maker_fraction;

        if (total_fraction - 1.0).abs() > 1e-6 {
            return Err(ConfigError::ValidationError(format!(
                "Agent fractions must sum to 1.0, got {}",
                total_fraction
            )));
        }

        // Kernel decay must be positive
        if self.btut.kernel_decay <= 0.0 {
            return Err(ConfigError::ValidationError(
                "Kernel decay must be positive".to_string(),
            ));
        }

        // Momentum must be in [0, 1]
        if !(0.0..=1.0).contains(&self.btut.momentum_strength) {
            return Err(ConfigError::ValidationError(
                "Momentum strength must be in [0, 1]".to_string(),
            ));
        }

        // Temperature must be positive
        if self.btut.temperature <= 0.0 {
            return Err(ConfigError::ValidationError(
                "Temperature must be positive".to_string(),
            ));
        }

        // Must have at least some agents
        if self.simulation.num_agents == 0 {
            return Err(ConfigError::ValidationError(
                "Must have at least one agent".to_string(),
            ));
        }

        // Tick size must be positive
        if self.market.tick_size <= 0.0 {
            return Err(ConfigError::ValidationError(
                "Tick size must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Create a config optimized for 1M agent simulations
    pub fn million_agent_preset() -> Self {
        Self {
            simulation: CoreConfig {
                num_agents: 1_000_000,
                num_timesteps: 1000,
                seed: Some(42),
                parallel: true,
                num_threads: 0,
                warmup_steps: 50,
            },
            btut: BTUTConfig {
                kernel_decay: 0.15,
                momentum_strength: 0.8,
                convergence_iterations: 20,
                convergence_threshold: 1e-5,
                temperature: 0.5,
                cooperation_bias: 0.0,
                adaptive_kernel: true,
            },
            market: MarketConfig {
                initial_price: 100.0,
                tick_size: 0.01,
                max_book_depth: 500,
                lot_size: 1.0,
                max_order_size: 10000.0,
                enable_snapshots: true,
                snapshot_frequency: 50,
            },
            agents: AgentConfig::default(),
            output: OutputConfig {
                output_dir: "output_1m".to_string(),
                format: OutputFormat::Csv,
                log_level: LogLevelConfig::Info,
                json_logs: false,
                metrics_sample_rate: 10,
                show_progress: true,
                compress_output: true,
            },
        }
    }

    /// Create a config for quick testing
    pub fn test_preset() -> Self {
        Self {
            simulation: CoreConfig {
                num_agents: 1000,
                num_timesteps: 100,
                seed: Some(12345),
                parallel: true,
                num_threads: 0,
                warmup_steps: 10,
            },
            btut: BTUTConfig::default(),
            market: MarketConfig::default(),
            agents: AgentConfig::default(),
            output: OutputConfig {
                output_dir: "output_test".to_string(),
                format: OutputFormat::Both,
                log_level: LogLevelConfig::Debug,
                json_logs: false,
                metrics_sample_rate: 1,
                show_progress: false,
                compress_output: false,
            },
        }
    }

    /// Create a config for liquidity shock experiments
    pub fn liquidity_shock_preset() -> Self {
        Self {
            simulation: CoreConfig {
                num_agents: 100_000,
                num_timesteps: 500,
                seed: Some(999),
                parallel: true,
                num_threads: 0,
                warmup_steps: 100,
            },
            btut: BTUTConfig {
                kernel_decay: 0.05, // Slower decay = longer-range correlations
                momentum_strength: 0.9, // High momentum = more persistent strategies
                convergence_iterations: 30,
                convergence_threshold: 1e-7,
                temperature: 0.3, // Lower temperature = sharper strategy selection
                cooperation_bias: -0.1, // Slight defection bias during shock
                adaptive_kernel: true,
            },
            market: MarketConfig {
                initial_price: 100.0,
                tick_size: 0.01,
                max_book_depth: 2000,
                lot_size: 1.0,
                max_order_size: 50000.0,
                enable_snapshots: true,
                snapshot_frequency: 5,
            },
            agents: AgentConfig {
                retail_fraction: 0.60,
                hft_fraction: 0.10,
                institution_fraction: 0.15,
                market_maker_fraction: 0.15,
                order_size_exponent: 2.0, // Heavier tails
                retail_capital: 10_000.0,
                institution_capital_multiplier: 2000.0,
                initial_cooperation: 0.6,
                cooperation_noise: 0.2,
            },
            output: OutputConfig {
                output_dir: "output_shock".to_string(),
                format: OutputFormat::Both,
                log_level: LogLevelConfig::Info,
                json_logs: false,
                metrics_sample_rate: 1,
                show_progress: true,
                compress_output: false,
            },
        }
    }

    /// Serialize config to TOML string
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Save config to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = self.to_toml().map_err(|e| {
            ConfigError::ValidationError(format!("Failed to serialize config: {}", e))
        })?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = SimulationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_preset_configs_are_valid() {
        assert!(SimulationConfig::million_agent_preset().validate().is_ok());
        assert!(SimulationConfig::test_preset().validate().is_ok());
        assert!(SimulationConfig::liquidity_shock_preset()
            .validate()
            .is_ok());
    }

    #[test]
    fn test_invalid_agent_fractions() {
        let mut config = SimulationConfig::default();
        config.agents.retail_fraction = 0.5;
        // Now fractions don't sum to 1.0
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_kernel_decay() {
        let mut config = SimulationConfig::default();
        config.btut.kernel_decay = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = SimulationConfig::default();
        let toml_str = config.to_toml().unwrap();
        let parsed: SimulationConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.simulation.num_agents, parsed.simulation.num_agents);
    }
}
