// Agent structures and network topology for BTUT

use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand::rngs::SmallRng;

/// Individual agent in the multi-agent system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: usize,
    pub degree: f64,         // Sampled from power-law distribution
    pub weight: f64,         // Kernel weight: (k/k_max)^τ
    pub strategy: Strategy,  // Current strategy (A=cooperate, B=defect)
    pub utility: f64,        // Current utility
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Strategy {
    A, // Cooperate
    B, // Defect
}

impl Strategy {
    pub fn to_u8(self) -> u8 {
        match self {
            Strategy::A => 1,
            Strategy::B => 0,
        }
    }

    pub fn from_u8(val: u8) -> Self {
        if val == 1 {
            Strategy::A
        } else {
            Strategy::B
        }
    }
}

impl Agent {
    pub fn new(id: usize, degree: f64, weight: f64, strategy: Strategy) -> Self {
        Self {
            id,
            degree,
            weight,
            strategy,
            utility: 0.0,
        }
    }
}

/// Network topology generator
pub struct NetworkTopology {
    rng: SmallRng,
}

impl NetworkTopology {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Sample degrees from Barabási-Albert power-law distribution
    /// P(k) ~ k^(-3) for large k
    /// Uses inverse transform sampling: k = m / sqrt(1 - U)
    pub fn sample_barabasi_albert_degrees(&mut self, n: usize, m: usize) -> Vec<f64> {
        let m_f64 = m as f64;
        let mut degrees = Vec::with_capacity(n);

        for _ in 0..n {
            let u: f64 = self.rng.gen();
            // Inverse transform: k = m / sqrt(1 - U)
            let k = (m_f64 / (1.0 - u).sqrt()).floor().max(m_f64);
            degrees.push(k);
        }

        degrees
    }

    /// Sample degrees with custom power-law exponent
    /// P(k) ~ k^(-γ)
    pub fn sample_power_law_degrees(
        &mut self,
        n: usize,
        m: usize,
        gamma: f64,
    ) -> Vec<f64> {
        let m_f64 = m as f64;
        let mut degrees = Vec::with_capacity(n);

        for _ in 0..n {
            let u: f64 = self.rng.gen();
            // Generalized inverse transform
            let k = m_f64 * (1.0 - u).powf(-1.0 / (gamma - 1.0));
            degrees.push(k.max(m_f64));
        }

        degrees
    }

    /// Create initial agent population with random strategies
    pub fn create_agents(
        &mut self,
        n: usize,
        m: usize,
        tau: f64,
    ) -> Vec<Agent> {
        let degrees = self.sample_barabasi_albert_degrees(n, m);
        let max_degree = degrees.iter().cloned().fold(0.0f64, f64::max);

        let mut agents = Vec::with_capacity(n);

        for (id, &degree) in degrees.iter().enumerate() {
            let weight = if tau > 0.0 && max_degree > 0.0 {
                (degree / max_degree).powf(tau)
            } else {
                1.0
            };

            let strategy = if self.rng.gen::<f64>() > 0.5 {
                Strategy::A
            } else {
                Strategy::B
            };

            agents.push(Agent::new(id, degree, weight, strategy));
        }

        agents
    }

    /// Create agents with heterogeneous initial strategies
    /// initial_p: fraction starting with strategy A
    pub fn create_agents_with_initial_fraction(
        &mut self,
        n: usize,
        m: usize,
        tau: f64,
        initial_p: f64,
    ) -> Vec<Agent> {
        let degrees = self.sample_barabasi_albert_degrees(n, m);
        let max_degree = degrees.iter().cloned().fold(0.0f64, f64::max);

        let mut agents = Vec::with_capacity(n);

        for (id, &degree) in degrees.iter().enumerate() {
            let weight = if tau > 0.0 && max_degree > 0.0 {
                (degree / max_degree).powf(tau)
            } else {
                1.0
            };

            let strategy = if self.rng.gen::<f64>() < initial_p {
                Strategy::A
            } else {
                Strategy::B
            };

            agents.push(Agent::new(id, degree, weight, strategy));
        }

        agents
    }
}

/// Agent statistics and metrics
pub struct AgentStats {
    pub total: usize,
    pub strategy_a_count: usize,
    pub strategy_b_count: usize,
    pub fraction_a: f64,
    pub avg_degree: f64,
    pub max_degree: f64,
    pub avg_utility: f64,
}

impl AgentStats {
    pub fn compute(agents: &[Agent]) -> Self {
        let total = agents.len();
        let strategy_a_count = agents.iter().filter(|a| a.strategy == Strategy::A).count();
        let strategy_b_count = total - strategy_a_count;
        let fraction_a = (strategy_a_count as f64) / (total as f64);

        let avg_degree = agents.iter().map(|a| a.degree).sum::<f64>() / (total as f64);
        let max_degree = agents.iter().map(|a| a.degree).fold(0.0f64, f64::max);
        let avg_utility = agents.iter().map(|a| a.utility).sum::<f64>() / (total as f64);

        Self {
            total,
            strategy_a_count,
            strategy_b_count,
            fraction_a,
            avg_degree,
            max_degree,
            avg_utility,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_topology_creation() {
        let mut topology = NetworkTopology::new(42);
        let agents = topology.create_agents(1000, 3, 0.5);

        assert_eq!(agents.len(), 1000);
        assert!(agents.iter().all(|a| a.degree >= 3.0));
    }

    #[test]
    fn test_degree_distribution() {
        let mut topology = NetworkTopology::new(42);
        let degrees = topology.sample_barabasi_albert_degrees(10000, 3);

        // Check minimum degree
        assert!(degrees.iter().all(|&k| k >= 3.0));

        // Check power-law tail (rough check)
        let high_degree = degrees.iter().filter(|&&k| k > 100.0).count();
        assert!(high_degree > 0 && high_degree < 1000); // Should have some hubs but not too many
    }

    #[test]
    fn test_agent_stats() {
        let agents = vec![
            Agent::new(0, 10.0, 0.5, Strategy::A),
            Agent::new(1, 20.0, 0.7, Strategy::A),
            Agent::new(2, 15.0, 0.6, Strategy::B),
        ];

        let stats = AgentStats::compute(&agents);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.strategy_a_count, 2);
        assert!((stats.fraction_a - 2.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_initial_fraction() {
        let mut topology = NetworkTopology::new(42);
        let agents = topology.create_agents_with_initial_fraction(1000, 3, 0.5, 0.8);

        let stats = AgentStats::compute(&agents);
        // Should be approximately 80% strategy A
        assert!((stats.fraction_a - 0.8).abs() < 0.1);
    }
}
