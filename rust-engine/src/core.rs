// BTUT Core Algorithm - O(N) Kernel-Weighted Mean-Field Dynamics
// Mathematical foundation for PDE-free, scalable game theory

use serde::{Deserialize, Serialize};

/// Core configuration for BTUT simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTUTConfig {
    pub n: usize,              // Number of agents
    pub gamma: f64,            // Cooperation bonus (Stag Hunt)
    pub tau: f64,              // Hub influence exponent
    pub c_a_sh: f64,           // Cost for strategy A in Stag Hunt
    pub c_b_sh: f64,           // Cost for strategy B in Stag Hunt
    pub c_a_pd: f64,           // Cost for strategy A in Prisoner's Dilemma
    pub c_b_pd: f64,           // Cost for strategy B in Prisoner's Dilemma
    pub alpha: f64,            // Hawk-Dove aggression parameter
    pub iterations: usize,     // Maximum iterations
    pub m: usize,              // Minimum degree (BA model)
    pub seed: u64,             // Random seed
    pub convergence_threshold: f64, // Variance threshold for convergence
}

impl Default for BTUTConfig {
    fn default() -> Self {
        Self {
            n: 10_000,
            gamma: 1.45,
            tau: 0.30,
            c_a_sh: 0.40,
            c_b_sh: 0.10,
            c_a_pd: 0.20,
            c_b_pd: 0.08,
            alpha: 0.60,
            iterations: 30,
            m: 3,
            seed: 42,
            convergence_threshold: 1e-10,
        }
    }
}

impl BTUTConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.n == 0 {
            return Err("Agent count must be positive".to_string());
        }
        if self.gamma <= 0.0 {
            return Err("Gamma must be positive".to_string());
        }
        if !(0.0..=1.0).contains(&self.tau) {
            return Err("Tau must be in [0, 1]".to_string());
        }
        if self.m == 0 {
            return Err("Minimum degree must be positive".to_string());
        }
        Ok(())
    }

    /// High cooperation preset
    pub fn high_cooperation() -> Self {
        Self {
            gamma: 2.0,
            c_a_sh: 0.2,
            c_b_sh: 0.05,
            ..Default::default()
        }
    }

    /// Low cooperation preset
    pub fn low_cooperation() -> Self {
        Self {
            gamma: 1.1,
            c_a_sh: 0.6,
            c_b_sh: 0.3,
            ..Default::default()
        }
    }

    /// Hub-dominated network preset
    pub fn hub_dominated() -> Self {
        Self {
            tau: 0.8,
            m: 5,
            ..Default::default()
        }
    }

    /// Large-scale preset (1M agents)
    pub fn large_scale() -> Self {
        Self {
            n: 1_000_000,
            iterations: 30,
            ..Default::default()
        }
    }
}

/// Compute expected utilities for both strategies
/// Returns (U_A, U_B) where U_A is utility for strategy A (cooperate)
pub fn compute_expected_utilities(
    p: f64,  // Fraction of cooperators
    config: &BTUTConfig,
) -> (f64, f64) {
    // Base payoffs (log-scaled for numerical stability)
    let u_a_pd = (2.0_f64).ln();
    let u_b_pd = (1.2_f64).ln();
    let u_a_hd = (2.2_f64).ln();
    let u_b_hd = (1.5_f64).ln();
    let u_a_sh = (2.5_f64).ln();
    let u_b_sh = (1.2_f64).ln();

    // Prisoner's Dilemma expected utilities
    let eu_pd_a = 0.5 * (u_a_pd + (p * u_a_pd + (1.0 - p) * u_b_pd))
        - config.c_a_pd * u_a_pd;
    let eu_pd_b = 0.5 * (u_b_pd + (p * u_a_pd + (1.0 - p) * u_b_pd))
        - config.c_b_pd * u_b_pd;

    // Hawk-Dove expected utilities
    let eu_hd_a = 0.5 * (u_a_hd + p * u_a_hd + (1.0 - p) * u_b_hd)
        * (p * config.alpha + (1.0 - p))
        - 0.22 * u_a_hd;
    let eu_hd_b = 0.5 * (u_b_hd + p * u_a_hd + (1.0 - p) * u_b_hd)
        * (p + (1.0 - p) * 1.08)
        - 0.12 * u_b_hd;

    // Stag Hunt expected utilities
    let eu_sh_a = 0.5 * (u_a_sh + p * u_a_sh + (1.0 - p) * u_b_sh)
        * (p * config.gamma + (1.0 - p) * 0.03)
        - config.c_a_sh * u_a_sh;
    let eu_sh_b = 0.5 * (u_b_sh + p * u_a_sh + (1.0 - p) * u_b_sh)
        * (p * 0.03 + (1.0 - p))
        - config.c_b_sh * u_b_sh;

    // Mixed game utility (equal weighting)
    let mix = 1.0 / 3.0;
    let u_a_total = mix * (eu_pd_a + eu_hd_a + eu_sh_a);
    let u_b_total = mix * (eu_pd_b + eu_hd_b + eu_sh_b);

    (u_a_total, u_b_total)
}

/// Kernel function for hub influence
/// K(k) = (k / k_max)^τ
pub fn kernel_weight(degree: f64, max_degree: f64, tau: f64) -> f64 {
    if tau == 0.0 || max_degree == 0.0 {
        1.0
    } else {
        (degree / max_degree).powf(tau)
    }
}

/// Mean-field update: compute new fraction of cooperators
/// Uses momentum-based averaging for stability
pub fn mean_field_update(
    p_current: f64,
    agents_data: &[(f64, f64)],  // (degree, weight) pairs
    config: &BTUTConfig,
    momentum: f64,
) -> f64 {
    let (u_a, u_b) = compute_expected_utilities(p_current, config);

    // Count weighted votes for strategy A
    let mut weighted_a = 0.0;
    let mut total_weight = 0.0;

    for &(degree, weight) in agents_data {
        let agent_u_a = u_a * weight * degree;
        let agent_u_b = u_b * weight * degree;

        if agent_u_a > agent_u_b {
            weighted_a += weight;
        }
        total_weight += weight;
    }

    let p_new = if total_weight > 0.0 {
        weighted_a / total_weight
    } else {
        p_current
    };

    // Momentum averaging for stability
    momentum * p_current + (1.0 - momentum) * p_new
}

/// Check convergence based on variance of recent history
pub fn check_convergence(history: &[f64], threshold: f64, window: usize) -> bool {
    if history.len() < window {
        return false;
    }

    let recent = &history[history.len() - window..];
    let mean = recent.iter().sum::<f64>() / (window as f64);
    let variance = recent.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (window as f64);

    variance < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = BTUTConfig::default();
        assert!(config.validate().is_ok());

        let bad_config = BTUTConfig {
            n: 0,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_expected_utilities() {
        let config = BTUTConfig::default();
        let (u_a, u_b) = compute_expected_utilities(0.5, &config);

        // Utilities should be finite and non-NaN
        assert!(u_a.is_finite());
        assert!(u_b.is_finite());
    }

    #[test]
    fn test_kernel_weight() {
        let w = kernel_weight(10.0, 100.0, 0.5);
        assert!(w > 0.0 && w <= 1.0);

        let w_no_tau = kernel_weight(10.0, 100.0, 0.0);
        assert_eq!(w_no_tau, 1.0);
    }

    #[test]
    fn test_convergence_detection() {
        let stable = vec![0.99, 0.991, 0.989, 0.99, 0.991];
        assert!(check_convergence(&stable, 1e-4, 5));

        let unstable = vec![0.5, 0.7, 0.3, 0.9, 0.2];
        assert!(!check_convergence(&unstable, 1e-4, 5));
    }
}
