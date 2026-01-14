// Convergence detection and analysis for BTUT simulations

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Convergence detector using multiple criteria
pub struct ConvergenceDetector {
    history: VecDeque<f64>,
    window_size: usize,
    threshold: f64,
}

impl ConvergenceDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
            threshold,
        }
    }

    /// Add new observation
    pub fn observe(&mut self, value: f64) {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(value);
    }

    /// Check if converged using variance criterion
    pub fn is_converged(&self) -> bool {
        if self.history.len() < self.window_size {
            return false;
        }

        let variance = self.compute_variance();
        variance < self.threshold
    }

    /// Compute variance of recent history
    pub fn compute_variance(&self) -> f64 {
        if self.history.is_empty() {
            return f64::INFINITY;
        }

        let mean = self.history.iter().sum::<f64>() / (self.history.len() as f64);
        let variance = self.history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (self.history.len() as f64);

        variance
    }

    /// Compute standard deviation
    pub fn compute_std(&self) -> f64 {
        self.compute_variance().sqrt()
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().sum::<f64>() / (self.history.len() as f64)
    }

    /// Check for oscillation pattern
    pub fn is_oscillating(&self) -> bool {
        if self.history.len() < 4 {
            return false;
        }

        let values: Vec<f64> = self.history.iter().copied().collect();
        let n = values.len();

        // Count direction changes
        let mut direction_changes = 0;
        for i in 1..n-1 {
            let prev_slope = values[i] - values[i-1];
            let next_slope = values[i+1] - values[i];
            if prev_slope * next_slope < 0.0 {
                direction_changes += 1;
            }
        }

        // Oscillating if more than 50% changes in direction
        direction_changes > (n - 2) / 2
    }

    /// Get convergence quality score (0-1, higher is better)
    pub fn convergence_quality(&self) -> f64 {
        if self.history.len() < self.window_size {
            return 0.0;
        }

        let variance = self.compute_variance();
        let quality = (-variance / self.threshold).exp();
        quality.min(1.0)
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Get history as vector
    pub fn get_history(&self) -> Vec<f64> {
        self.history.iter().copied().collect()
    }
}

/// Convergence metrics for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    pub converged: bool,
    pub iterations_to_converge: Option<usize>,
    pub final_value: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub quality_score: f64,
    pub oscillating: bool,
}

impl ConvergenceMetrics {
    pub fn from_detector(detector: &ConvergenceDetector, iterations: usize) -> Self {
        Self {
            converged: detector.is_converged(),
            iterations_to_converge: if detector.is_converged() {
                Some(iterations)
            } else {
                None
            },
            final_value: detector.mean(),
            variance: detector.compute_variance(),
            std_dev: detector.compute_std(),
            quality_score: detector.convergence_quality(),
            oscillating: detector.is_oscillating(),
        }
    }
}

/// Adaptive convergence detector with dynamic threshold
pub struct AdaptiveConvergenceDetector {
    base_detector: ConvergenceDetector,
    initial_threshold: f64,
    min_threshold: f64,
    decay_rate: f64,
    iteration: usize,
}

impl AdaptiveConvergenceDetector {
    pub fn new(
        window_size: usize,
        initial_threshold: f64,
        min_threshold: f64,
        decay_rate: f64,
    ) -> Self {
        Self {
            base_detector: ConvergenceDetector::new(window_size, initial_threshold),
            initial_threshold,
            min_threshold,
            decay_rate,
            iteration: 0,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.base_detector.observe(value);
        self.iteration += 1;

        // Adaptive threshold decay
        let new_threshold = (self.initial_threshold * (-self.decay_rate * self.iteration as f64).exp())
            .max(self.min_threshold);
        self.base_detector.threshold = new_threshold;
    }

    pub fn is_converged(&self) -> bool {
        self.base_detector.is_converged()
    }

    pub fn current_threshold(&self) -> f64 {
        self.base_detector.threshold
    }

    pub fn metrics(&self) -> ConvergenceMetrics {
        ConvergenceMetrics::from_detector(&self.base_detector, self.iteration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_detection() {
        let mut detector = ConvergenceDetector::new(5, 1e-4);

        // Converging sequence
        for val in [0.5, 0.51, 0.499, 0.501, 0.5001] {
            detector.observe(val);
        }

        assert!(detector.is_converged());
    }

    #[test]
    fn test_oscillation_detection() {
        let mut detector = ConvergenceDetector::new(10, 1e-4);

        // Oscillating sequence
        for i in 0..10 {
            detector.observe(if i % 2 == 0 { 0.3 } else { 0.7 });
        }

        assert!(detector.is_oscillating());
    }

    #[test]
    fn test_convergence_quality() {
        let mut detector = ConvergenceDetector::new(5, 1e-3);

        // Perfect convergence
        for _ in 0..5 {
            detector.observe(0.99);
        }

        let quality = detector.convergence_quality();
        assert!(quality > 0.99);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut detector = AdaptiveConvergenceDetector::new(5, 1e-2, 1e-6, 0.1);

        let initial = detector.current_threshold();

        for val in [0.5, 0.51, 0.52, 0.53, 0.54] {
            detector.observe(val);
        }

        let final_threshold = detector.current_threshold();
        assert!(final_threshold < initial);
        assert!(final_threshold >= 1e-6);
    }
}
