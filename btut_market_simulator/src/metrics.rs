//! Metrics Module
//!
//! This module provides market metrics collection, crash detection, and regeneration scoring.
//!
//! ## Key Metrics
//!
//! - **Liquidity Depth**: Total volume at various levels from mid-price
//! - **Spread**: Bid-ask spread (absolute and relative)
//! - **Volatility**: Rolling volatility of returns
//! - **Regeneration Score**: Measure of market stability/recovery
//! - **Crash Detection**: Automated detection of flash crashes

use crate::btut::CooperationByType;
use crate::market::{BookSnapshot, OrderBook};
use crate::utils::stats;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, warn};

/// Market metrics at a single point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMetrics {
    /// Simulation timestep
    pub timestep: u64,

    // Price metrics
    /// Mid price
    pub mid_price: Option<f64>,
    /// Last trade price
    pub last_price: Option<f64>,
    /// VWAP
    pub vwap: Option<f64>,

    // Spread metrics
    /// Bid-ask spread (absolute)
    pub spread: Option<f64>,
    /// Spread in basis points
    pub spread_bps: Option<f64>,

    // Volume metrics
    /// Total bid volume
    pub bid_volume: f64,
    /// Total ask volume
    pub ask_volume: f64,
    /// Volume imbalance (-1 to +1)
    pub imbalance: f64,
    /// Total traded volume this timestep
    pub traded_volume: f64,
    /// Number of trades this timestep
    pub num_trades: u64,

    // Depth metrics
    /// Liquidity at 1% from mid
    pub depth_1pct: f64,
    /// Liquidity at 5% from mid
    pub depth_5pct: f64,
    /// Liquidity at 10% from mid
    pub depth_10pct: f64,

    // Volatility metrics
    /// Rolling volatility (std of returns)
    pub volatility: f64,
    /// Realized volatility (annualized)
    pub realized_vol: f64,

    // BTUT metrics
    /// Mean cooperation level
    pub mean_cooperation: f64,
    /// Cooperation by agent type
    pub cooperation_by_type: Option<CooperationByType>,

    // Stability metrics
    /// Regeneration score (higher = more stable)
    pub regeneration_score: f64,
    /// Drag score (higher = more friction/instability)
    pub drag_score: f64,

    // Event flags
    /// Is this a crash timestep?
    pub is_crash: bool,
    /// Is market recovering?
    pub is_recovering: bool,
}

impl Default for MarketMetrics {
    fn default() -> Self {
        Self {
            timestep: 0,
            mid_price: None,
            last_price: None,
            vwap: None,
            spread: None,
            spread_bps: None,
            bid_volume: 0.0,
            ask_volume: 0.0,
            imbalance: 0.0,
            traded_volume: 0.0,
            num_trades: 0,
            depth_1pct: 0.0,
            depth_5pct: 0.0,
            depth_10pct: 0.0,
            volatility: 0.0,
            realized_vol: 0.0,
            mean_cooperation: 0.5,
            cooperation_by_type: None,
            regeneration_score: 1.0,
            drag_score: 0.0,
            is_crash: false,
            is_recovering: false,
        }
    }
}

/// Crash event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashEvent {
    /// Start timestep
    pub start_timestep: u64,
    /// End timestep (when recovered or simulation ended)
    pub end_timestep: Option<u64>,
    /// Price at crash start
    pub start_price: f64,
    /// Minimum price during crash
    pub min_price: f64,
    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,
    /// Spread at peak of crash
    pub peak_spread_bps: f64,
    /// Minimum liquidity during crash
    pub min_liquidity: f64,
    /// Mean cooperation during crash
    pub mean_cooperation: f64,
    /// Did market fully recover?
    pub recovered: bool,
    /// Recovery time in timesteps
    pub recovery_time: Option<u64>,
}

/// Crash detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashDetectorConfig {
    /// Minimum price drop percentage to trigger crash detection
    pub price_drop_threshold: f64,
    /// Spread widening threshold (multiple of normal spread)
    pub spread_threshold: f64,
    /// Liquidity drop threshold (percentage of normal)
    pub liquidity_threshold: f64,
    /// Window size for detecting price drops
    pub lookback_window: usize,
    /// Window for calculating "normal" conditions
    pub baseline_window: usize,
    /// Recovery threshold (percentage of drop recovered)
    pub recovery_threshold: f64,
}

impl Default for CrashDetectorConfig {
    fn default() -> Self {
        Self {
            price_drop_threshold: 0.02,  // 2% drop
            spread_threshold: 3.0,       // 3x normal spread
            liquidity_threshold: 0.5,    // 50% of normal liquidity
            lookback_window: 10,
            baseline_window: 50,
            recovery_threshold: 0.8, // 80% recovery
        }
    }
}

/// Crash detector for identifying flash crashes
pub struct CrashDetector {
    config: CrashDetectorConfig,
    /// Price history
    price_history: VecDeque<f64>,
    /// Spread history
    spread_history: VecDeque<f64>,
    /// Liquidity history
    liquidity_history: VecDeque<f64>,
    /// Baseline metrics
    baseline_price: f64,
    baseline_spread: f64,
    baseline_liquidity: f64,
    /// Current crash state
    in_crash: bool,
    current_crash: Option<CrashEvent>,
    /// All detected crashes
    crashes: Vec<CrashEvent>,
}

impl CrashDetector {
    /// Create a new crash detector
    pub fn new(config: CrashDetectorConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::with_capacity(100),
            spread_history: VecDeque::with_capacity(100),
            liquidity_history: VecDeque::with_capacity(100),
            baseline_price: 0.0,
            baseline_spread: 0.0,
            baseline_liquidity: 0.0,
            in_crash: false,
            current_crash: None,
            crashes: Vec::new(),
        }
    }

    /// Update detector with new metrics
    pub fn update(&mut self, metrics: &MarketMetrics) -> bool {
        let price = metrics.mid_price.unwrap_or(self.baseline_price);
        let spread = metrics.spread_bps.unwrap_or(0.0);
        let liquidity = metrics.bid_volume + metrics.ask_volume;

        // Update histories
        self.price_history.push_back(price);
        self.spread_history.push_back(spread);
        self.liquidity_history.push_back(liquidity);

        // Trim histories
        while self.price_history.len() > self.config.baseline_window {
            self.price_history.pop_front();
        }
        while self.spread_history.len() > self.config.baseline_window {
            self.spread_history.pop_front();
        }
        while self.liquidity_history.len() > self.config.baseline_window {
            self.liquidity_history.pop_front();
        }

        // Update baselines (use median for robustness)
        if self.price_history.len() >= self.config.baseline_window / 2 {
            let mut prices: Vec<f64> = self.price_history.iter().cloned().collect();
            self.baseline_price = stats::median(&mut prices);

            let mut spreads: Vec<f64> = self.spread_history.iter().cloned().collect();
            self.baseline_spread = stats::median(&mut spreads).max(1.0);

            let mut liquidity: Vec<f64> = self.liquidity_history.iter().cloned().collect();
            self.baseline_liquidity = stats::median(&mut liquidity).max(1.0);
        }

        // Check for crash conditions
        let is_crash = self.detect_crash(price, spread, liquidity, metrics.timestep);

        // Check for recovery
        if self.in_crash {
            self.check_recovery(price, metrics.timestep);
        }

        is_crash
    }

    /// Detect if current conditions constitute a crash
    fn detect_crash(&mut self, price: f64, spread: f64, liquidity: f64, timestep: u64) -> bool {
        if self.baseline_price == 0.0 {
            return false;
        }

        let price_drop = (self.baseline_price - price) / self.baseline_price;
        let spread_multiple = spread / self.baseline_spread;
        let liquidity_ratio = liquidity / self.baseline_liquidity;

        let conditions_met = price_drop > self.config.price_drop_threshold
            || spread_multiple > self.config.spread_threshold
            || liquidity_ratio < self.config.liquidity_threshold;

        if conditions_met && !self.in_crash {
            // Start new crash
            self.in_crash = true;
            self.current_crash = Some(CrashEvent {
                start_timestep: timestep,
                end_timestep: None,
                start_price: self.baseline_price,
                min_price: price,
                max_drawdown_pct: price_drop * 100.0,
                peak_spread_bps: spread,
                min_liquidity: liquidity,
                mean_cooperation: 0.0,
                recovered: false,
                recovery_time: None,
            });

            warn!(
                timestep = timestep,
                price_drop_pct = price_drop * 100.0,
                spread_multiple = spread_multiple,
                "Flash crash detected"
            );

            true
        } else if self.in_crash {
            // Update crash metrics
            if let Some(ref mut crash) = self.current_crash {
                crash.min_price = crash.min_price.min(price);
                crash.max_drawdown_pct = crash
                    .max_drawdown_pct
                    .max((crash.start_price - price) / crash.start_price * 100.0);
                crash.peak_spread_bps = crash.peak_spread_bps.max(spread);
                crash.min_liquidity = crash.min_liquidity.min(liquidity);
            }
            true
        } else {
            false
        }
    }

    /// Check if market has recovered from crash
    fn check_recovery(&mut self, price: f64, timestep: u64) {
        if let Some(ref mut crash) = self.current_crash {
            let recovery_ratio = (price - crash.min_price) / (crash.start_price - crash.min_price);

            if recovery_ratio >= self.config.recovery_threshold {
                crash.end_timestep = Some(timestep);
                crash.recovered = true;
                crash.recovery_time = Some(timestep - crash.start_timestep);

                debug!(
                    start = crash.start_timestep,
                    end = timestep,
                    recovery_time = crash.recovery_time,
                    "Market recovered from crash"
                );

                self.crashes.push(crash.clone());
                self.current_crash = None;
                self.in_crash = false;
            }
        }
    }

    /// Finalize any ongoing crash at simulation end
    pub fn finalize(&mut self, timestep: u64) {
        if let Some(ref mut crash) = self.current_crash {
            crash.end_timestep = Some(timestep);
            crash.recovered = false;
            self.crashes.push(crash.clone());
            self.current_crash = None;
            self.in_crash = false;
        }
    }

    /// Get all detected crashes
    pub fn crashes(&self) -> &[CrashEvent] {
        &self.crashes
    }

    /// Is market currently in crash state?
    pub fn is_in_crash(&self) -> bool {
        self.in_crash
    }

    /// Get current crash if any
    pub fn current_crash(&self) -> Option<&CrashEvent> {
        self.current_crash.as_ref()
    }
}

/// Metrics collector for the simulation
pub struct MetricsCollector {
    /// All collected metrics
    metrics: Vec<MarketMetrics>,
    /// Crash detector
    crash_detector: CrashDetector,
    /// Price history for volatility calculation
    price_history: VecDeque<f64>,
    /// Return history
    return_history: VecDeque<f64>,
    /// Volatility window size
    volatility_window: usize,
    /// Previous traded volume (for delta calculation)
    prev_traded_volume: f64,
    /// Previous trade count
    prev_trade_count: u64,
    /// Sample rate
    sample_rate: usize,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(sample_rate: usize) -> Self {
        Self {
            metrics: Vec::with_capacity(10000),
            crash_detector: CrashDetector::new(CrashDetectorConfig::default()),
            price_history: VecDeque::with_capacity(100),
            return_history: VecDeque::with_capacity(100),
            volatility_window: 20,
            prev_traded_volume: 0.0,
            prev_trade_count: 0,
            sample_rate,
        }
    }

    /// Create with custom crash detector config
    pub fn with_crash_config(sample_rate: usize, crash_config: CrashDetectorConfig) -> Self {
        Self {
            metrics: Vec::with_capacity(10000),
            crash_detector: CrashDetector::new(crash_config),
            price_history: VecDeque::with_capacity(100),
            return_history: VecDeque::with_capacity(100),
            volatility_window: 20,
            prev_traded_volume: 0.0,
            prev_trade_count: 0,
            sample_rate,
        }
    }

    /// Collect metrics for current timestep
    pub fn collect(
        &mut self,
        timestep: u64,
        book: &OrderBook,
        mean_cooperation: f64,
        cooperation_by_type: Option<CooperationByType>,
    ) -> MarketMetrics {
        // Check sample rate
        if timestep % self.sample_rate as u64 != 0 {
            return MarketMetrics {
                timestep,
                mean_cooperation,
                ..Default::default()
            };
        }

        let snapshot = book.snapshot(100);

        // Calculate depth at various levels
        let mid = snapshot.mid_price.unwrap_or(100.0);
        let depth_1pct = self.calculate_depth(&snapshot, mid, 0.01);
        let depth_5pct = self.calculate_depth(&snapshot, mid, 0.05);
        let depth_10pct = self.calculate_depth(&snapshot, mid, 0.10);

        // Update price history
        if let Some(price) = snapshot.mid_price {
            if let Some(&prev_price) = self.price_history.back() {
                let ret = (price - prev_price) / prev_price;
                self.return_history.push_back(ret);
                if self.return_history.len() > self.volatility_window {
                    self.return_history.pop_front();
                }
            }
            self.price_history.push_back(price);
            if self.price_history.len() > self.volatility_window {
                self.price_history.pop_front();
            }
        }

        // Calculate volatility
        let returns: Vec<f64> = self.return_history.iter().cloned().collect();
        let volatility = stats::std_dev(&returns);
        let realized_vol = volatility * (252.0_f64).sqrt(); // Annualized

        // Calculate volume delta
        let current_volume = book.total_volume();
        let current_trades = book.num_trades();
        let traded_volume = current_volume - self.prev_traded_volume;
        let num_trades = current_trades - self.prev_trade_count;
        self.prev_traded_volume = current_volume;
        self.prev_trade_count = current_trades;

        // Calculate regeneration and drag scores
        let (regeneration_score, drag_score) = self.calculate_scores(
            &snapshot,
            volatility,
            mean_cooperation,
        );

        let mut metrics = MarketMetrics {
            timestep,
            mid_price: snapshot.mid_price,
            last_price: book.last_price(),
            vwap: book.vwap(100),
            spread: snapshot.spread,
            spread_bps: book.spread_bps(),
            bid_volume: snapshot.total_bid_volume,
            ask_volume: snapshot.total_ask_volume,
            imbalance: book.imbalance(),
            traded_volume,
            num_trades,
            depth_1pct,
            depth_5pct,
            depth_10pct,
            volatility,
            realized_vol,
            mean_cooperation,
            cooperation_by_type,
            regeneration_score,
            drag_score,
            is_crash: false,
            is_recovering: false,
        };

        // Update crash detector
        metrics.is_crash = self.crash_detector.update(&metrics);
        metrics.is_recovering = self.crash_detector.is_in_crash()
            && metrics.mid_price.map_or(false, |p| {
                self.crash_detector
                    .current_crash()
                    .map_or(false, |c| p > c.min_price * 1.01)
            });

        self.metrics.push(metrics.clone());
        metrics
    }

    /// Calculate depth within a percentage range of mid price
    fn calculate_depth(&self, snapshot: &BookSnapshot, mid: f64, pct: f64) -> f64 {
        let lower = mid * (1.0 - pct);
        let upper = mid * (1.0 + pct);

        let bid_depth: f64 = snapshot
            .bids
            .iter()
            .filter(|(p, _)| *p >= lower)
            .map(|(_, q)| q)
            .sum();

        let ask_depth: f64 = snapshot
            .asks
            .iter()
            .filter(|(p, _)| *p <= upper)
            .map(|(_, q)| q)
            .sum();

        bid_depth + ask_depth
    }

    /// Calculate regeneration and drag scores
    ///
    /// Regeneration: Higher when market is stable and recovering
    /// - High cooperation + low volatility + tight spreads = high regeneration
    ///
    /// Drag: Higher when market is stressed
    /// - Low cooperation + high volatility + wide spreads = high drag
    fn calculate_scores(
        &self,
        snapshot: &BookSnapshot,
        volatility: f64,
        cooperation: f64,
    ) -> (f64, f64) {
        // Normalize spread (assume 10 bps is normal)
        let spread_score = snapshot.spread.map_or(0.5, |s| {
            let mid = snapshot.mid_price.unwrap_or(100.0);
            let spread_bps = s / mid * 10000.0;
            (10.0 / spread_bps.max(1.0)).min(1.0)
        });

        // Normalize volatility (assume 2% is high)
        let vol_score = 1.0 - (volatility / 0.02).min(1.0);

        // Liquidity score
        let total_liq = snapshot.total_bid_volume + snapshot.total_ask_volume;
        let liq_score = (total_liq / 10000.0).min(1.0); // Normalize to 10k

        // Imbalance penalty
        let imbalance = ((snapshot.total_bid_volume - snapshot.total_ask_volume).abs()
            / (total_liq.max(1.0)))
        .min(1.0);
        let balance_score = 1.0 - imbalance;

        // Regeneration = cooperation * stability_factors
        let regeneration = cooperation * 0.4
            + spread_score * 0.2
            + vol_score * 0.2
            + liq_score * 0.1
            + balance_score * 0.1;

        // Drag = opposite of regeneration factors
        let drag = (1.0 - cooperation) * 0.4
            + (1.0 - spread_score) * 0.2
            + (1.0 - vol_score) * 0.2
            + (1.0 - liq_score) * 0.1
            + imbalance * 0.1;

        (regeneration.clamp(0.0, 1.0), drag.clamp(0.0, 1.0))
    }

    /// Finalize collection at simulation end
    pub fn finalize(&mut self, timestep: u64) {
        self.crash_detector.finalize(timestep);
    }

    /// Get all collected metrics
    pub fn metrics(&self) -> &[MarketMetrics] {
        &self.metrics
    }

    /// Get detected crashes
    pub fn crashes(&self) -> &[CrashEvent] {
        self.crash_detector.crashes()
    }

    /// Get summary statistics
    pub fn summary(&self) -> MetricsSummary {
        if self.metrics.is_empty() {
            return MetricsSummary::default();
        }

        let prices: Vec<f64> = self
            .metrics
            .iter()
            .filter_map(|m| m.mid_price)
            .collect();
        let spreads: Vec<f64> = self
            .metrics
            .iter()
            .filter_map(|m| m.spread_bps)
            .collect();
        let volatilities: Vec<f64> = self.metrics.iter().map(|m| m.volatility).collect();
        let cooperations: Vec<f64> = self.metrics.iter().map(|m| m.mean_cooperation).collect();
        let regenerations: Vec<f64> = self.metrics.iter().map(|m| m.regeneration_score).collect();

        MetricsSummary {
            num_timesteps: self.metrics.len(),
            mean_price: stats::mean(&prices),
            price_std: stats::std_dev(&prices),
            max_drawdown: stats::max_drawdown(&prices),
            mean_spread_bps: stats::mean(&spreads),
            max_spread_bps: spreads.iter().cloned().fold(0.0, f64::max),
            mean_volatility: stats::mean(&volatilities),
            max_volatility: volatilities.iter().cloned().fold(0.0, f64::max),
            mean_cooperation: stats::mean(&cooperations),
            min_cooperation: cooperations.iter().cloned().fold(1.0, f64::min),
            mean_regeneration: stats::mean(&regenerations),
            num_crashes: self.crash_detector.crashes().len(),
            total_crash_duration: self
                .crash_detector
                .crashes()
                .iter()
                .filter_map(|c| c.recovery_time)
                .sum(),
        }
    }

    /// Export metrics to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("timestep,mid_price,last_price,spread_bps,bid_volume,ask_volume,");
        csv.push_str("imbalance,traded_volume,num_trades,depth_1pct,depth_5pct,depth_10pct,");
        csv.push_str("volatility,realized_vol,mean_cooperation,regeneration_score,drag_score,");
        csv.push_str("is_crash,is_recovering\n");

        for m in &self.metrics {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                m.timestep,
                m.mid_price.map_or("".to_string(), |v| v.to_string()),
                m.last_price.map_or("".to_string(), |v| v.to_string()),
                m.spread_bps.map_or("".to_string(), |v| v.to_string()),
                m.bid_volume,
                m.ask_volume,
                m.imbalance,
                m.traded_volume,
                m.num_trades,
                m.depth_1pct,
                m.depth_5pct,
                m.depth_10pct,
                m.volatility,
                m.realized_vol,
                m.mean_cooperation,
                m.regeneration_score,
                m.drag_score,
                m.is_crash,
                m.is_recovering
            ));
        }

        csv
    }
}

/// Summary statistics from a simulation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// Number of timesteps collected
    pub num_timesteps: usize,
    /// Mean price
    pub mean_price: f64,
    /// Price standard deviation
    pub price_std: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Mean spread in basis points
    pub mean_spread_bps: f64,
    /// Maximum spread
    pub max_spread_bps: f64,
    /// Mean volatility
    pub mean_volatility: f64,
    /// Maximum volatility
    pub max_volatility: f64,
    /// Mean cooperation level
    pub mean_cooperation: f64,
    /// Minimum cooperation level
    pub min_cooperation: f64,
    /// Mean regeneration score
    pub mean_regeneration: f64,
    /// Number of detected crashes
    pub num_crashes: usize,
    /// Total duration of all crashes
    pub total_crash_duration: u64,
}

impl std::fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Simulation Summary ===")?;
        writeln!(f, "Timesteps: {}", self.num_timesteps)?;
        writeln!(f, "Mean Price: {:.4}", self.mean_price)?;
        writeln!(f, "Price Std Dev: {:.4}", self.price_std)?;
        writeln!(f, "Max Drawdown: {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Mean Spread: {:.2} bps", self.mean_spread_bps)?;
        writeln!(f, "Max Spread: {:.2} bps", self.max_spread_bps)?;
        writeln!(f, "Mean Volatility: {:.4}", self.mean_volatility)?;
        writeln!(f, "Max Volatility: {:.4}", self.max_volatility)?;
        writeln!(f, "Mean Cooperation: {:.4}", self.mean_cooperation)?;
        writeln!(f, "Min Cooperation: {:.4}", self.min_cooperation)?;
        writeln!(f, "Mean Regeneration: {:.4}", self.mean_regeneration)?;
        writeln!(f, "Flash Crashes: {}", self.num_crashes)?;
        writeln!(f, "Total Crash Duration: {} ticks", self.total_crash_duration)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::OrderBook;

    #[test]
    fn test_metrics_collection() {
        let mut collector = MetricsCollector::new(1);
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 10, 100.0);

        for t in 0..100 {
            collector.collect(t, &book, 0.5, None);
        }

        assert_eq!(collector.metrics().len(), 100);
    }

    #[test]
    fn test_crash_detection() {
        let config = CrashDetectorConfig {
            price_drop_threshold: 0.05,
            spread_threshold: 2.0,
            liquidity_threshold: 0.5,
            lookback_window: 5,
            baseline_window: 10,
            recovery_threshold: 0.8,
        };

        let mut detector = CrashDetector::new(config);

        // Normal conditions
        for t in 0..20 {
            let metrics = MarketMetrics {
                timestep: t,
                mid_price: Some(100.0),
                spread_bps: Some(10.0),
                bid_volume: 1000.0,
                ask_volume: 1000.0,
                ..Default::default()
            };
            detector.update(&metrics);
        }

        assert!(!detector.is_in_crash());

        // Crash conditions
        let crash_metrics = MarketMetrics {
            timestep: 20,
            mid_price: Some(90.0), // 10% drop
            spread_bps: Some(50.0), // 5x spread
            bid_volume: 200.0, // Low liquidity
            ask_volume: 200.0,
            ..Default::default()
        };
        let is_crash = detector.update(&crash_metrics);

        assert!(is_crash);
        assert!(detector.is_in_crash());
    }

    #[test]
    fn test_regeneration_score() {
        let collector = MetricsCollector::new(1);

        // Good conditions
        let good_snapshot = BookSnapshot {
            bids: vec![(99.0, 1000.0)],
            asks: vec![(101.0, 1000.0)],
            best_bid: Some(99.0),
            best_ask: Some(101.0),
            mid_price: Some(100.0),
            spread: Some(2.0),
            total_bid_volume: 5000.0,
            total_ask_volume: 5000.0,
            timestamp: 0,
        };

        let (regen, drag) = collector.calculate_scores(&good_snapshot, 0.01, 0.8);
        assert!(regen > 0.5);
        assert!(drag < 0.5);

        // Bad conditions
        let bad_snapshot = BookSnapshot {
            bids: vec![(90.0, 100.0)],
            asks: vec![(110.0, 100.0)],
            best_bid: Some(90.0),
            best_ask: Some(110.0),
            mid_price: Some(100.0),
            spread: Some(20.0),
            total_bid_volume: 100.0,
            total_ask_volume: 500.0,
            timestamp: 0,
        };

        let (regen, drag) = collector.calculate_scores(&bad_snapshot, 0.05, 0.2);
        assert!(regen < 0.5);
        assert!(drag > 0.5);
    }

    #[test]
    fn test_summary() {
        let mut collector = MetricsCollector::new(1);
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 10, 100.0);

        for t in 0..50 {
            collector.collect(t, &book, 0.5 + (t as f64 * 0.01), None);
        }

        let summary = collector.summary();
        assert!(summary.num_timesteps > 0);
        assert!(summary.mean_price > 0.0);
    }

    #[test]
    fn test_csv_export() {
        let mut collector = MetricsCollector::new(1);
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 10, 100.0);

        for t in 0..10 {
            collector.collect(t, &book, 0.5, None);
        }

        let csv = collector.to_csv();
        assert!(csv.contains("timestep"));
        assert!(csv.contains("mid_price"));
        assert!(csv.lines().count() > 1);
    }
}
