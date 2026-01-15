/**
 * TypeScript implementation of BTUT simulation engine
 * Pure TypeScript fallback for environments where WASM is not available
 */

export type Strategy = 'A' | 'B';

export interface SimulationConfig {
  N: number;
  gamma: number;
  tau: number;
  cA_SH: number;
  cB_SH: number;
  cA_PD: number;
  cB_PD: number;
  alpha: number;
  iterations: number;
  m: number;
  seed: number;
  topology?: 'barabasi_albert' | 'regular' | 'random';
  convergence_threshold?: number;
  convergence_window?: number;
}

export interface Agent {
  id: number;
  strategy: Strategy;
  degree: number;
  weight: number;
  utility: number;
  neighbors?: number[];
}

export interface SimulationState {
  iteration: number;
  fractionA: number;
  convergenceHistory: number[];
  agents: Agent[];
  isComplete: boolean;
  runtime?: number;
  agentCount: number;
  strategyACount: number;
}

export class BTUTSimulator {
  private config: SimulationConfig;
  private state: SimulationState;
  private rng: () => number;

  constructor(config: SimulationConfig) {
    this.config = {
      topology: 'barabasi_albert',
      convergence_threshold: 1e-6,
      convergence_window: 100,
      ...config,
    };
    this.rng = Math.random;
    this.state = this.initializeState();
  }

  private initializeState(): SimulationState {
    const { N, m, seed, tau } = this.config;
    const agents: Agent[] = [];

    // Use seed for reproducible random number generation
    if (seed !== undefined) {
      // Simple seeded RNG
      let s = seed;
      this.rng = () => {
        s = (s * 9301 + 49297) % 233280;
        return s / 233280;
      };
    }

    // Sample degrees from Barabási-Albert power-law distribution
    const degrees = this.sampleDegrees(N, m);
    const maxDegree = Math.max(...degrees);

    // Initialize agents with p0 = 0.5
    for (let i = 0; i < N; i++) {
      const degree = degrees[i];
      const weight = Math.pow(degree / maxDegree, tau);

      agents.push({
        id: i,
        strategy: this.rng() < 0.5 ? 'A' : 'B',
        degree,
        weight,
        utility: 0,
      });
    }

    // Compute initial cooperation fraction
    const fractionA = agents.filter(a => a.strategy === 'A').length / N;

    return {
      iteration: 0,
      fractionA,
      convergenceHistory: [fractionA],
      agents,
      isComplete: false,
      agentCount: N,
      strategyACount: agents.filter(a => a.strategy === 'A').length,
    };
  }

  private sampleDegrees(n: number, m: number): number[] {
    const degrees: number[] = [];
    for (let i = 0; i < n; i++) {
      const u = this.rng();
      const k = Math.max(m, Math.floor(m / Math.sqrt(1 - u)));
      degrees.push(k);
    }
    return degrees;
  }

  private kernelWeight(k: number, tau: number): number {
    return Math.pow(k, -tau);
  }

  private computeExpectedUtilities(p: number): [number, number] {
    const { gamma, tau, cA_SH, cB_SH, cA_PD, cB_PD, alpha } = this.config;

    // Base payoffs (log-scaled for stability)
    const u_a_pd = Math.log(2);
    const u_b_pd = Math.log(1.2);
    const u_a_hd = Math.log(2.2);
    const u_b_hd = Math.log(1.5);
    const u_a_sh = Math.log(2.5);
    const u_b_sh = Math.log(1.2);

    // Expected utilities for each game type
    const eu_pd_a = 0.5 * (u_a_pd + (p * u_a_pd + (1 - p) * u_b_pd)) - cA_PD * u_a_pd;
    const eu_pd_b = 0.5 * (u_b_pd + (p * u_a_pd + (1 - p) * u_b_pd)) - cB_PD * u_b_pd;

    const eu_hd_a = 0.5 * (u_a_hd + p * u_a_hd + (1 - p) * u_b_hd)
        * (p * alpha + (1 - p)) - 0.22 * u_a_hd;
    const eu_hd_b = 0.5 * (u_b_hd + p * u_a_hd + (1 - p) * u_b_hd)
        * (p + (1 - p) * 1.08) - 0.12 * u_b_hd;

    const eu_sh_a = 0.5 * (u_a_sh + p * u_a_sh + (1 - p) * u_b_sh)
        * (p * gamma + (1 - p) * 0.03) - cA_SH * u_a_sh;
    const eu_sh_b = 0.5 * (u_b_sh + p * u_a_sh + (1 - p) * u_b_sh)
        * (p * 0.03 + (1 - p)) - cB_SH * u_b_sh;

    // Mixed game utility
    const mix = 1 / 3;
    const u_a_total = mix * (eu_pd_a + eu_hd_a + eu_sh_a);
    const u_b_total = mix * (eu_pd_b + eu_hd_b + eu_sh_b);

    return [u_a_total, u_b_total];
  }

  private updateStrategies(fractionA: number): void {
    const { gamma } = this.config;
    const { agents } = this.state;

    // Compute global utilities
    const [u_a, u_b] = this.computeExpectedUtilities(fractionA);

    // O(N) mean-field update
    for (const agent of agents) {
      const weight = agent.weight;

      // Compute switching probability based on weighted utilities
      const current_u = agent.strategy === 'A' ? u_a : u_b;
      const other_u = agent.strategy === 'A' ? u_b : u_a;

      const delta_u = other_u - current_u;
      const prob_switch = 1 / (1 + Math.exp(-gamma * weight * delta_u));

      if (this.rng() < prob_switch) {
        agent.strategy = agent.strategy === 'A' ? 'B' : 'A';
      }

      agent.utility = current_u;
    }
  }

  private checkConvergence(): boolean {
    const { convergence_window, convergence_threshold } = this.config;
    const { convergenceHistory } = this.state;

    if (convergenceHistory.length < (convergence_window || 100)) {
      return false;
    }

    const recent = convergenceHistory.slice(-(convergence_window || 100));
    const mean_p = recent.reduce((sum, val) => sum + val, 0) / recent.length;
    const variance = recent.reduce((sum, val) => sum + Math.pow(val - mean_p, 2), 0) / recent.length;

    return variance < (convergence_threshold || 1e-6);
  }

  public step(): SimulationState {
    if (this.state.isComplete || this.state.iteration >= this.config.iterations) {
      this.state.isComplete = true;
      return this.state;
    }

    // Update strategies
    this.updateStrategies(this.state.fractionA);

    // Compute new cooperation fraction
    const fractionA = this.state.agents.filter(a => a.strategy === 'A').length / this.config.N;
    const strategyACount = this.state.agents.filter(a => a.strategy === 'A').length;

    // Check convergence
    const converged = this.checkConvergence();

    // Update state
    this.state.iteration++;
    this.state.fractionA = fractionA;
    this.state.strategyACount = strategyACount;
    this.state.convergenceHistory.push(fractionA);
    this.state.isComplete = converged || this.state.iteration >= this.config.iterations;

    return this.state;
  }

  public run(): SimulationState[] {
    const states: SimulationState[] = [{ ...this.state }];

    while (!this.state.isComplete && this.state.iteration < this.config.iterations) {
      this.step();
      states.push({ ...this.state });
    }

    return states;
  }

  public getState(): SimulationState {
    return this.state;
  }

  public reset(): void {
    this.state = this.initializeState();
  }
}

// Preset configurations
export const PRESETS = {
  quick: {
    N: 1000,
    gamma: 1.2,
    tau: 0.30,
    cA_SH: 0.40,
    cB_SH: 0.10,
    cA_PD: 0.20,
    cB_PD: 0.08,
    alpha: 0.60,
    iterations: 100,
    m: 3,
    seed: 42,
  },
  default: {
    N: 1000,
    gamma: 1.2,
    tau: 0.30,
    cA_SH: 0.40,
    cB_SH: 0.10,
    cA_PD: 0.20,
    cB_PD: 0.08,
    alpha: 0.60,
    iterations: 1000,
    m: 3,
    seed: 42,
  },
  small: {
    N: 100,
    gamma: 1.2,
    tau: 0.30,
    cA_SH: 0.40,
    cB_SH: 0.10,
    cA_PD: 0.20,
    cB_PD: 0.08,
    alpha: 0.60,
    iterations: 500,
    m: 3,
    seed: 42,
  },
  large: {
    N: 10000,
    gamma: 1.2,
    tau: 0.30,
    cA_SH: 0.40,
    cB_SH: 0.10,
    cA_PD: 0.20,
    cB_PD: 0.08,
    alpha: 0.60,
    iterations: 2000,
    m: 3,
    seed: 42,
  },
  highCooperation: {
    N: 1000,
    gamma: 2.0,
    tau: 0.20,
    cA_SH: 0.30,
    cB_SH: 0.15,
    cA_PD: 0.15,
    cB_PD: 0.05,
    alpha: 0.80,
    iterations: 1000,
    m: 5,
    seed: 42,
  },
  lowCooperation: {
    N: 1000,
    gamma: 0.8,
    tau: 0.40,
    cA_SH: 0.50,
    cB_SH: 0.05,
    cA_PD: 0.30,
    cB_PD: 0.10,
    alpha: 0.40,
    iterations: 1000,
    m: 2,
    seed: 42,
  },
} as const;
