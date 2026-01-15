/**
 * TypeScript interfaces for BTUT validation data
 * Data sources: integrations/sumo/*.json
 */

// Baseline strategy comparison
export interface BaselineStrategy {
  name: string
  description: string
  cooperation_rate: number
  avg_speed: number
  avg_waiting_time: number
  throughput: number
  stability: number
}

export interface BaselineResults {
  [key: string]: BaselineStrategy
}

// Stress test results
export interface StressTestResult {
  scenario: string
  tau: number
  gamma: number
  peak_vehicles: number
  final_cooperation: number
  avg_speed: number
  min_speed: number
  avg_waiting_time: number
  max_waiting_time: number
  throughput: number
  gridlock_occurred: boolean
  recovery_time: number | null
}

export interface StressTestData {
  results: {
    [key: string]: StressTestResult
  }
  report: string
}

// Large scale results
export interface ScaleTestPoint {
  num_agents: number
  tau: number
  gamma: number
  cooperation: number
  convergence_iterations: number
  hub_cooperation: number
  periphery_cooperation: number
  degree_variance: number
  effective_speed: number
  effective_wait: number
}

export interface LargeScaleResults {
  results: {
    [key: string]: ScaleTestPoint[]
  }
  analysis: {
    scale_effects: Array<{
      agents: number
      speed_gain: number
      wait_reduction: number
    }>
    conclusion: string
  }
}

// Validation results with improvements
export interface ValidationSummaryPoint {
  tau: number
  gamma: number
  cooperation: number
  avg_speed: number
  avg_waiting: number
  throughput: number
}

export interface ValidationImprovement {
  'speed_improvement_%': number
  'wait_reduction_%': number
}

export interface ValidationResults {
  timestamp: string
  analysis: {
    summary: {
      [key: string]: ValidationSummaryPoint
    }
    improvements: {
      [key: string]: ValidationImprovement
    }
  }
}

// Calibration data
export interface CriticalPointResult {
  gamma_critical: number
  gamma_lower: number
  gamma_upper: number
  transition_sharpness: number
  recommended_gamma: number
  recommended_margin: number
  search_iterations: number
  confidence: number
  domain: string
  config_used: {
    N: number
    tau: number
    tolerance: number
    cA_SH: number
    cB_SH: number
    cA_PD: number
    cB_PD: number
    alpha: number
  }
  timestamp: string
}

export interface CalibrationData {
  [domain: string]: CriticalPointResult
}

// Drone swarm results
export interface SwarmTestResult {
  n_drones: number
  gamma: number
  tau: number
  iterations: number
  final_cooperation: number
  final_formation_error: number
  total_collisions: number
  avg_speed: number
  final_energy: number
  convergence_iteration: number
  drones_per_second: number
}

export interface SwarmResults {
  [key: string]: SwarmTestResult
}
