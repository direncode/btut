//! Utility functions for the BTUT Market Simulator.
//!
//! This module provides:
//! - Random number generation with reproducible seeding
//! - Synthetic market data generation (power-law distributions, volatility)
//! - Logging setup with tracing
//! - Statistical helper functions
//! - Simple text-based plotting for terminal output

use crate::config::{LogLevelConfig, SimulationConfig};
use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Exp, LogNormal, Normal, Pareto};
use serde::{Deserialize, Serialize};
use std::fmt;
use tracing::Level;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

/// Log level wrapper for external use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Trace level
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warn level
    Warn,
    /// Error level
    Error,
}

impl From<LogLevelConfig> for LogLevel {
    fn from(config: LogLevelConfig) -> Self {
        match config {
            LogLevelConfig::Trace => LogLevel::Trace,
            LogLevelConfig::Debug => LogLevel::Debug,
            LogLevelConfig::Info => LogLevel::Info,
            LogLevelConfig::Warn => LogLevel::Warn,
            LogLevelConfig::Error => LogLevel::Error,
        }
    }
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

/// Initialize the logging system with the given configuration
pub fn init_logging(config: &SimulationConfig) {
    let level = match config.output.log_level {
        LogLevelConfig::Trace => "trace",
        LogLevelConfig::Debug => "debug",
        LogLevelConfig::Info => "info",
        LogLevelConfig::Warn => "warn",
        LogLevelConfig::Error => "error",
    };

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("btut_market_simulator={}", level)));

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(false)
        .with_line_number(false);

    if config.output.json_logs {
        subscriber.json().init();
    } else {
        subscriber.init();
    }
}

/// Thread-safe random number generator wrapper
#[derive(Clone)]
pub struct SimRng {
    rng: ChaCha8Rng,
}

impl SimRng {
    /// Create a new RNG with optional seed
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self { rng }
    }

    /// Create a new RNG from a seed
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Fork the RNG for use in a parallel context
    /// Each fork gets a unique seed derived from the current state
    pub fn fork(&mut self) -> Self {
        let new_seed: u64 = self.uniform(0, u64::MAX);
        Self::from_seed(new_seed)
    }

    /// Generate multiple forked RNGs for parallel execution
    pub fn fork_n(&mut self, n: usize) -> Vec<Self> {
        (0..n).map(|_| self.fork()).collect()
    }

    /// Sample from uniform distribution [low, high)
    pub fn uniform<T>(&mut self, low: T, high: T) -> T
    where
        T: rand::distributions::uniform::SampleUniform + Copy,
    {
        Uniform::new(low, high).sample(&mut self.rng)
    }

    /// Sample from uniform distribution [low, high]
    pub fn uniform_inclusive<T>(&mut self, low: T, high: T) -> T
    where
        T: rand::distributions::uniform::SampleUniform + Copy,
    {
        Uniform::new_inclusive(low, high).sample(&mut self.rng)
    }

    /// Sample from normal distribution
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        Normal::new(mean, std_dev)
            .expect("Invalid normal parameters")
            .sample(&mut self.rng)
    }

    /// Sample from log-normal distribution
    pub fn log_normal(&mut self, mu: f64, sigma: f64) -> f64 {
        LogNormal::new(mu, sigma)
            .expect("Invalid log-normal parameters")
            .sample(&mut self.rng)
    }

    /// Sample from exponential distribution
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        Exp::new(lambda)
            .expect("Invalid exponential parameter")
            .sample(&mut self.rng)
    }

    /// Sample from Pareto (power-law) distribution
    /// Returns values >= scale with P(X > x) ~ (scale/x)^alpha
    pub fn pareto(&mut self, scale: f64, alpha: f64) -> f64 {
        Pareto::new(scale, alpha)
            .expect("Invalid Pareto parameters")
            .sample(&mut self.rng)
    }

    /// Sample from truncated Pareto distribution
    pub fn pareto_truncated(&mut self, scale: f64, alpha: f64, max: f64) -> f64 {
        loop {
            let x = self.pareto(scale, alpha);
            if x <= max {
                return x;
            }
        }
    }

    /// Generate a random boolean with given probability
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform(0.0, 1.0) < p
    }

    /// Choose a random element from a slice
    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        if slice.is_empty() {
            None
        } else {
            let idx = self.uniform(0, slice.len());
            Some(&slice[idx])
        }
    }

    /// Shuffle a slice in place
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        use rand::seq::SliceRandom;
        slice.shuffle(&mut self.rng);
    }

    /// Get the underlying RNG (for advanced use)
    pub fn inner(&mut self) -> &mut ChaCha8Rng {
        &mut self.rng
    }
}

impl fmt::Debug for SimRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimRng").finish()
    }
}

/// Synthetic market data generator
#[derive(Debug, Clone)]
pub struct SyntheticGenerator {
    rng: SimRng,
    /// Pareto alpha for order sizes
    pub order_size_alpha: f64,
    /// Minimum order size
    pub min_order_size: f64,
    /// Maximum order size
    pub max_order_size: f64,
    /// Base volatility
    pub base_volatility: f64,
    /// Volatility mean reversion speed
    pub vol_mean_reversion: f64,
    /// Volatility of volatility
    pub vol_of_vol: f64,
    /// Current volatility state
    current_volatility: f64,
    /// Current price
    current_price: f64,
}

impl SyntheticGenerator {
    /// Create a new synthetic generator with given parameters
    pub fn new(config: &SimulationConfig, seed: Option<u64>) -> Self {
        Self {
            rng: SimRng::new(seed),
            order_size_alpha: config.agents.order_size_exponent,
            min_order_size: config.market.lot_size,
            max_order_size: config.market.max_order_size,
            base_volatility: 0.02,
            vol_mean_reversion: 0.1,
            vol_of_vol: 0.3,
            current_volatility: 0.02,
            current_price: config.market.initial_price,
        }
    }

    /// Generate a power-law distributed order size
    pub fn generate_order_size(&mut self) -> f64 {
        self.rng
            .pareto_truncated(self.min_order_size, self.order_size_alpha, self.max_order_size)
    }

    /// Generate multiple order sizes
    pub fn generate_order_sizes(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.generate_order_size()).collect()
    }

    /// Update stochastic volatility (Heston-like model)
    pub fn update_volatility(&mut self) -> f64 {
        // Mean-reverting volatility with vol-of-vol
        let dw = self.rng.normal(0.0, 1.0);
        let variance = self.current_volatility * self.current_volatility;

        // Heston-style update
        let new_variance = variance
            + self.vol_mean_reversion * (self.base_volatility * self.base_volatility - variance)
            + self.vol_of_vol * variance.sqrt() * dw;

        self.current_volatility = new_variance.max(0.0001).sqrt();
        self.current_volatility
    }

    /// Generate a price return with current volatility
    pub fn generate_return(&mut self) -> f64 {
        let vol = self.current_volatility;
        self.rng.normal(0.0, vol)
    }

    /// Update price with generated return
    pub fn update_price(&mut self) -> f64 {
        self.update_volatility();
        let ret = self.generate_return();
        self.current_price *= (1.0 + ret).max(0.01);
        self.current_price
    }

    /// Generate a momentum shock (large directional move)
    pub fn generate_momentum_shock(&mut self, magnitude: f64) -> f64 {
        let direction = if self.rng.bernoulli(0.5) { 1.0 } else { -1.0 };
        direction * magnitude * self.rng.exponential(1.0)
    }

    /// Generate initial order book depth profile (exponentially decaying from mid)
    pub fn generate_book_profile(&mut self, mid_price: f64, num_levels: usize) -> BookProfile {
        let tick_size = 0.01;
        let mut bids = Vec::with_capacity(num_levels);
        let mut asks = Vec::with_capacity(num_levels);

        for i in 0..num_levels {
            let distance = (i + 1) as f64;
            // Exponentially decaying liquidity with noise
            let base_qty = 1000.0 * (-0.1 * distance).exp();
            let bid_qty = base_qty * self.rng.uniform(0.8, 1.2);
            let ask_qty = base_qty * self.rng.uniform(0.8, 1.2);

            let bid_price = mid_price - distance * tick_size;
            let ask_price = mid_price + distance * tick_size;

            bids.push((bid_price, bid_qty.max(1.0)));
            asks.push((ask_price, ask_qty.max(1.0)));
        }

        BookProfile { bids, asks }
    }

    /// Get current volatility
    pub fn volatility(&self) -> f64 {
        self.current_volatility
    }

    /// Get current price
    pub fn price(&self) -> f64 {
        self.current_price
    }

    /// Set current price
    pub fn set_price(&mut self, price: f64) {
        self.current_price = price;
    }

    /// Fork for parallel use
    pub fn fork(&mut self) -> Self {
        Self {
            rng: self.rng.fork(),
            order_size_alpha: self.order_size_alpha,
            min_order_size: self.min_order_size,
            max_order_size: self.max_order_size,
            base_volatility: self.base_volatility,
            vol_mean_reversion: self.vol_mean_reversion,
            vol_of_vol: self.vol_of_vol,
            current_volatility: self.current_volatility,
            current_price: self.current_price,
        }
    }
}

/// Order book depth profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookProfile {
    /// Bid prices and quantities (sorted by price descending)
    pub bids: Vec<(f64, f64)>,
    /// Ask prices and quantities (sorted by price ascending)
    pub asks: Vec<(f64, f64)>,
}

/// Simple ASCII plotting for terminal output
pub struct PlotHelper;

impl PlotHelper {
    /// Plot a time series as ASCII art
    pub fn plot_ascii(data: &[f64], title: &str, width: usize, height: usize) -> String {
        if data.is_empty() {
            return format!("{}: No data\n", title);
        }

        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range < 1e-10 {
            return format!("{}: Constant value {:.4}\n", title, min_val);
        }

        let mut output = String::new();
        output.push_str(&format!("{}:\n", title));
        output.push_str(&format!("  Max: {:.4}\n", max_val));

        // Create the plot grid
        let mut grid = vec![vec![' '; width]; height];

        // Sample data to fit width
        let step = data.len().max(1) as f64 / width as f64;
        for x in 0..width {
            let idx = (x as f64 * step).min((data.len() - 1) as f64) as usize;
            let val = data[idx];
            let y = ((val - min_val) / range * (height - 1) as f64) as usize;
            let y = y.min(height - 1);
            grid[height - 1 - y][x] = '█';
        }

        // Render grid
        for row in &grid {
            output.push_str("  │");
            for &c in row {
                output.push(c);
            }
            output.push('\n');
        }

        // Bottom border
        output.push_str("  └");
        output.push_str(&"─".repeat(width));
        output.push('\n');
        output.push_str(&format!("  Min: {:.4}\n", min_val));

        output
    }

    /// Plot two series overlaid (e.g., for comparison)
    pub fn plot_comparison(
        data1: &[f64],
        data2: &[f64],
        title: &str,
        label1: &str,
        label2: &str,
        width: usize,
        height: usize,
    ) -> String {
        if data1.is_empty() && data2.is_empty() {
            return format!("{}: No data\n", title);
        }

        let all_data: Vec<f64> = data1.iter().chain(data2.iter()).cloned().collect();
        let min_val = all_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = all_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range < 1e-10 {
            return format!("{}: Constant value {:.4}\n", title, min_val);
        }

        let mut output = String::new();
        output.push_str(&format!("{} ({}: █, {}: ░):\n", title, label1, label2));
        output.push_str(&format!("  Max: {:.4}\n", max_val));

        let mut grid = vec![vec![' '; width]; height];

        // Plot first series
        let step1 = data1.len().max(1) as f64 / width as f64;
        for x in 0..width {
            if !data1.is_empty() {
                let idx = (x as f64 * step1).min((data1.len() - 1) as f64) as usize;
                let val = data1[idx];
                let y = ((val - min_val) / range * (height - 1) as f64) as usize;
                let y = y.min(height - 1);
                grid[height - 1 - y][x] = '█';
            }
        }

        // Plot second series
        let step2 = data2.len().max(1) as f64 / width as f64;
        for x in 0..width {
            if !data2.is_empty() {
                let idx = (x as f64 * step2).min((data2.len() - 1) as f64) as usize;
                let val = data2[idx];
                let y = ((val - min_val) / range * (height - 1) as f64) as usize;
                let y = y.min(height - 1);
                if grid[height - 1 - y][x] == '█' {
                    grid[height - 1 - y][x] = '▓'; // Overlap
                } else {
                    grid[height - 1 - y][x] = '░';
                }
            }
        }

        for row in &grid {
            output.push_str("  │");
            for &c in row {
                output.push(c);
            }
            output.push('\n');
        }

        output.push_str("  └");
        output.push_str(&"─".repeat(width));
        output.push('\n');
        output.push_str(&format!("  Min: {:.4}\n", min_val));

        output
    }

    /// Create a histogram
    pub fn histogram(data: &[f64], title: &str, num_bins: usize, width: usize) -> String {
        if data.is_empty() {
            return format!("{}: No data\n", title);
        }

        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range < 1e-10 {
            return format!("{}: All values = {:.4}\n", title, min_val);
        }

        let bin_width = range / num_bins as f64;
        let mut bins = vec![0usize; num_bins];

        for &val in data {
            let bin = ((val - min_val) / bin_width) as usize;
            let bin = bin.min(num_bins - 1);
            bins[bin] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&1);

        let mut output = String::new();
        output.push_str(&format!("{} (n={}):\n", title, data.len()));

        for (i, &count) in bins.iter().enumerate() {
            let bar_len = (count as f64 / max_count as f64 * width as f64) as usize;
            let bin_start = min_val + i as f64 * bin_width;
            let bin_end = bin_start + bin_width;
            output.push_str(&format!(
                "  [{:>8.3}, {:>8.3}): {:>6} │{}│\n",
                bin_start,
                bin_end,
                count,
                "█".repeat(bar_len)
            ));
        }

        output
    }
}

/// Statistical helper functions
pub mod stats {
    /// Calculate mean of a slice
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate variance of a slice
    pub fn variance(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let m = mean(data);
        let sum_sq: f64 = data.iter().map(|x| (x - m).powi(2)).sum();
        sum_sq / (data.len() - 1) as f64
    }

    /// Calculate standard deviation
    pub fn std_dev(data: &[f64]) -> f64 {
        variance(data).sqrt()
    }

    /// Calculate percentile (0-100)
    pub fn percentile(data: &mut [f64], p: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (data.len() - 1) as f64) as usize;
        data[idx.min(data.len() - 1)]
    }

    /// Calculate median
    pub fn median(data: &mut [f64]) -> f64 {
        percentile(data, 50.0)
    }

    /// Calculate exponential moving average
    pub fn ema(data: &[f64], alpha: f64) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }
        let mut result = Vec::with_capacity(data.len());
        result.push(data[0]);
        for i in 1..data.len() {
            let prev = result[i - 1];
            result.push(alpha * data[i] + (1.0 - alpha) * prev);
        }
        result
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![0.0; data.len()];
        }
        let mut result = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            if i < window - 1 {
                result.push(0.0);
            } else {
                let window_data = &data[i + 1 - window..=i];
                result.push(std_dev(window_data));
            }
        }
        result
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown(prices: &[f64]) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }
        let mut peak = prices[0];
        let mut max_dd = 0.0;
        for &price in prices {
            if price > peak {
                peak = price;
            }
            let dd = (peak - price) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        max_dd
    }

    /// Calculate Sharpe-like ratio (mean/std of returns)
    pub fn returns_ratio(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
        let m = mean(&returns);
        let s = std_dev(&returns);
        if s < 1e-10 {
            0.0
        } else {
            m / s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_reproducibility() {
        let mut rng1 = SimRng::from_seed(42);
        let mut rng2 = SimRng::from_seed(42);

        for _ in 0..100 {
            assert_eq!(rng1.uniform(0.0, 1.0), rng2.uniform(0.0, 1.0));
        }
    }

    #[test]
    fn test_rng_fork() {
        let mut rng = SimRng::from_seed(42);
        let mut fork1 = rng.fork();
        let mut fork2 = rng.fork();

        // Forked RNGs should produce different sequences
        let v1: f64 = fork1.uniform(0.0, 1.0);
        let v2: f64 = fork2.uniform(0.0, 1.0);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_pareto_distribution() {
        let mut rng = SimRng::from_seed(42);
        let mut samples: Vec<f64> = (0..10000).map(|_| rng.pareto(1.0, 1.5)).collect();

        // All values should be >= scale
        assert!(samples.iter().all(|&x| x >= 1.0));

        // Distribution should be heavy-tailed
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[5000];
        let p99 = samples[9900];
        assert!(p99 > 10.0 * median); // Heavy tail
    }

    #[test]
    fn test_synthetic_generator() {
        let config = SimulationConfig::default();
        let mut gen = SyntheticGenerator::new(&config, Some(42));

        let sizes: Vec<f64> = (0..1000).map(|_| gen.generate_order_size()).collect();
        assert!(sizes.iter().all(|&x| x >= config.market.lot_size));
        assert!(sizes.iter().all(|&x| x <= config.market.max_order_size));
    }

    #[test]
    fn test_stats_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((stats::mean(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = stats::std_dev(&data);
        assert!((std - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_ascii_plot() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let plot = PlotHelper::plot_ascii(&data, "Sine Wave", 40, 10);
        assert!(plot.contains("Sine Wave"));
        assert!(plot.contains("Max:"));
        assert!(plot.contains("Min:"));
    }
}
