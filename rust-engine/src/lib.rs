use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand::rngs::SmallRng;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Agent {
    pub id: usize,
    pub degree: f64,
    pub weight: f64,
    pub strategy: u8, // 0 = B, 1 = A
    pub utility: f64,
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationConfig {
    pub n: usize,
    pub gamma: f64,
    pub tau: f64,
    pub c_a_sh: f64,
    pub c_b_sh: f64,
    pub c_a_pd: f64,
    pub c_b_pd: f64,
    pub alpha: f64,
    pub iterations: usize,
    pub m: usize,
    pub seed: u64,
}

#[wasm_bindgen]
impl SimulationConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        n: usize,
        gamma: f64,
        tau: f64,
        c_a_sh: f64,
        c_b_sh: f64,
        c_a_pd: f64,
        c_b_pd: f64,
        alpha: f64,
        iterations: usize,
        m: usize,
        seed: u64,
    ) -> SimulationConfig {
        SimulationConfig {
            n,
            gamma,
            tau,
            c_a_sh,
            c_b_sh,
            c_a_pd,
            c_b_pd,
            alpha,
            iterations,
            m,
            seed,
        }
    }

    #[wasm_bindgen]
    pub fn quick_preset() -> SimulationConfig {
        SimulationConfig {
            n: 10_000,
            gamma: 1.45,
            tau: 0.30,
            c_a_sh: 0.40,
            c_b_sh: 0.10,
            c_a_pd: 0.20,
            c_b_pd: 0.08,
            alpha: 0.60,
            iterations: 20,
            m: 3,
            seed: 42,
        }
    }

    #[wasm_bindgen]
    pub fn standard_preset() -> SimulationConfig {
        SimulationConfig {
            n: 100_000,
            gamma: 1.45,
            tau: 0.30,
            c_a_sh: 0.40,
            c_b_sh: 0.10,
            c_a_pd: 0.20,
            c_b_pd: 0.08,
            alpha: 0.60,
            iterations: 25,
            m: 3,
            seed: 42,
        }
    }

    #[wasm_bindgen]
    pub fn massive_preset() -> SimulationConfig {
        SimulationConfig {
            n: 1_000_000,
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
        }
    }
}

#[wasm_bindgen]
pub struct BTUTSimulator {
    config: SimulationConfig,
    agents: Vec<Agent>,
    history: Vec<f64>,
    p: f64,
    rng: SmallRng,
}

#[wasm_bindgen]
impl BTUTSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(config: SimulationConfig) -> BTUTSimulator {
        let mut sim = BTUTSimulator {
            config: config.clone(),
            agents: Vec::with_capacity(config.n),
            history: Vec::new(),
            p: 0.5,
            rng: SmallRng::seed_from_u64(config.seed),
        };
        sim.initialize_agents();
        console_log!("BTUT Simulator initialized with {} agents", config.n);
        sim
    }

    fn initialize_agents(&mut self) {
        let n = self.config.n;
        let m = self.config.m as f64;
        let tau = self.config.tau;

        // Sample degrees from Barabási-Albert distribution
        let mut degrees = Vec::with_capacity(n);
        let mut max_degree = 0.0f64;

        for _ in 0..n {
            let u: f64 = self.rng.gen();
            let k = (m / (1.0 - u).sqrt()).floor().max(m);
            degrees.push(k);
            max_degree = max_degree.max(k);
        }

        // Create agents with kernel weights
        for id in 0..n {
            let degree = degrees[id];
            let weight = if tau > 0.0 {
                (degree / max_degree).powf(tau)
            } else {
                1.0
            };

            let strategy = if self.rng.gen::<f64>() > 0.5 { 1 } else { 0 };

            self.agents.push(Agent {
                id,
                degree,
                weight,
                strategy,
                utility: 0.0,
            });
        }
    }

    fn compute_utilities(&mut self) -> (f64, f64) {
        let p = self.p;
        let gamma = self.config.gamma;
        let c_a_sh = self.config.c_a_sh;
        let c_b_sh = self.config.c_b_sh;
        let c_a_pd = self.config.c_a_pd;
        let c_b_pd = self.config.c_b_pd;
        let alpha = self.config.alpha;

        // Base payoffs (log-scaled)
        let u_a_pd = 2.0f64.ln();
        let u_b_pd = 1.2f64.ln();
        let u_a_hd = 2.2f64.ln();
        let u_b_hd = 1.5f64.ln();
        let u_a_sh = 2.5f64.ln();
        let u_b_sh = 1.2f64.ln();

        // Expected utilities for each game type
        let eu_pd_a = 0.5 * (u_a_pd + (p * u_a_pd + (1.0 - p) * u_b_pd)) - c_a_pd * u_a_pd;
        let eu_pd_b = 0.5 * (u_b_pd + (p * u_a_pd + (1.0 - p) * u_b_pd)) - c_b_pd * u_b_pd;

        let eu_hd_a = 0.5 * (u_a_hd + p * u_a_hd + (1.0 - p) * u_b_hd) 
            * (p * alpha + (1.0 - p)) - 0.22 * u_a_hd;
        let eu_hd_b = 0.5 * (u_b_hd + p * u_a_hd + (1.0 - p) * u_b_hd) 
            * (p + (1.0 - p) * 1.08) - 0.12 * u_b_hd;

        let eu_sh_a = 0.5 * (u_a_sh + p * u_a_sh + (1.0 - p) * u_b_sh) 
            * (p * gamma + (1.0 - p) * 0.03) - c_a_sh * u_a_sh;
        let eu_sh_b = 0.5 * (u_b_sh + p * u_a_sh + (1.0 - p) * u_b_sh) 
            * (p * 0.03 + (1.0 - p)) - c_b_sh * u_b_sh;

        // Mixed game utility
        let mix = 1.0 / 3.0;

        for agent in &mut self.agents {
            let u_a = agent.weight * agent.degree * mix * (eu_pd_a + eu_hd_a + eu_sh_a);
            let u_b = agent.weight * agent.degree * mix * (eu_pd_b + eu_hd_b + eu_sh_b);
            agent.utility = if agent.strategy == 1 { u_a } else { u_b };
        }

        (eu_pd_a + eu_hd_a + eu_sh_a, eu_pd_b + eu_hd_b + eu_sh_b)
    }

    fn update_strategies(&mut self) {
        let (u_a_total, u_b_total) = self.compute_utilities();

        let mut count_a = 0usize;
        for agent in &mut self.agents {
            let choose_a = u_a_total * agent.weight * agent.degree 
                > u_b_total * agent.weight * agent.degree;
            
            if choose_a {
                agent.strategy = 1;
                count_a += 1;
            } else {
                agent.strategy = 0;
            }
        }

        let p_new = count_a as f64 / self.agents.len() as f64;

        // Momentum update
        self.p = 0.5 * self.p + 0.5 * p_new;
        self.history.push(self.p);
    }

    #[wasm_bindgen]
    pub fn step(&mut self) -> JsValue {
        self.update_strategies();

        let state = SimulationState {
            iteration: self.history.len(),
            fraction_a: self.p,
            convergence_history: self.history.clone(),
            is_complete: self.history.len() >= self.config.iterations,
            agent_count: self.agents.len(),
            strategy_a_count: self.agents.iter().filter(|a| a.strategy == 1).count(),
        };

        serde_wasm_bindgen::to_value(&state).unwrap()
    }

    #[wasm_bindgen]
    pub fn run_complete(&mut self) -> JsValue {
        for _ in 0..self.config.iterations {
            self.update_strategies();
        }

        let state = SimulationState {
            iteration: self.history.len(),
            fraction_a: self.p,
            convergence_history: self.history.clone(),
            is_complete: true,
            agent_count: self.agents.len(),
            strategy_a_count: self.agents.iter().filter(|a| a.strategy == 1).count(),
        };

        console_log!("Simulation complete: p* = {:.6}", self.p);
        serde_wasm_bindgen::to_value(&state).unwrap()
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.history.clear();
        self.p = 0.5;
        self.rng = SmallRng::seed_from_u64(self.config.seed);
        self.agents.clear();
        self.initialize_agents();
    }

    #[wasm_bindgen]
    pub fn get_state(&self) -> JsValue {
        let state = SimulationState {
            iteration: self.history.len(),
            fraction_a: self.p,
            convergence_history: self.history.clone(),
            is_complete: self.history.len() >= self.config.iterations,
            agent_count: self.agents.len(),
            strategy_a_count: self.agents.iter().filter(|a| a.strategy == 1).count(),
        };

        serde_wasm_bindgen::to_value(&state).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_agents_sample(&self, max_count: usize) -> JsValue {
        let step = (self.agents.len() / max_count.min(self.agents.len())).max(1);
        let sample: Vec<&Agent> = self.agents.iter().step_by(step).collect();
        serde_wasm_bindgen::to_value(&sample).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_convergence_history(&self) -> Vec<f64> {
        self.history.clone()
    }

    #[wasm_bindgen]
    pub fn get_fraction_a(&self) -> f64 {
        self.p
    }

    #[wasm_bindgen]
    pub fn get_iteration(&self) -> usize {
        self.history.len()
    }
}

#[derive(Serialize, Deserialize)]
struct SimulationState {
    iteration: usize,
    fraction_a: f64,
    convergence_history: Vec<f64>,
    is_complete: bool,
    agent_count: usize,
    strategy_a_count: usize,
}

// Performance benchmark function
#[wasm_bindgen]
pub fn benchmark_scaling(max_n: usize, step: usize) -> JsValue {
    let mut results = Vec::new();
    
    let mut n = step;
    while n <= max_n {
        let config = SimulationConfig {
            n,
            gamma: 1.45,
            tau: 0.30,
            c_a_sh: 0.40,
            c_b_sh: 0.10,
            c_a_pd: 0.20,
            c_b_pd: 0.08,
            alpha: 0.60,
            iterations: 10,
            m: 3,
            seed: 42,
        };

        let start = js_sys::Date::now();
        let mut sim = BTUTSimulator::new(config);
        sim.run_complete();
        let elapsed = js_sys::Date::now() - start;

        results.push(BenchmarkResult {
            n,
            time_ms: elapsed,
            throughput: (n * 10) as f64 / elapsed,
        });

        console_log!("Benchmark N={}: {:.2}ms", n, elapsed);
        n += step;
    }

    serde_wasm_bindgen::to_value(&results).unwrap()
}

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    n: usize,
    time_ms: f64,
    throughput: f64,
}
