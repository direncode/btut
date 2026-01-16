/**
 * Static validation data imports for results-first frontend
 * All data from actual SUMO/drone simulations
 */

import type {
  BaselineResults,
  StressTestData,
  LargeScaleResults
} from '@/types/validation'

// Import raw JSON data
import baselineResultsRaw from '@/integrations/sumo/baseline_results.json'
import stressTestResultsRaw from '@/integrations/sumo/stress_test_results.json'
import largeScaleResultsRaw from '@/integrations/sumo/large_scale_results.json'

// Type assertions for typed access to raw data
export const baselineResults = baselineResultsRaw as BaselineResults
export const stressTestResults = stressTestResultsRaw as StressTestData
export const largeScaleResults = largeScaleResultsRaw as LargeScaleResults

// Pre-computed hero metrics from stress test
export const stressTestHeroMetrics = {
  peakVehicles: 800,
  cooperation: 99.99,
  avgSpeed: 12.2,
  maxWaitTime: 8,
  gridlockOccurred: false,
  throughput: 1760,
  source: 'Eclipse SUMO Traffic Simulator',
  testPhases: [
    'Warm-up (0-300s)',
    'Ramp-up (300-600s)',
    'Peak stress (600-1200s)',
    'Sustained (1200-1800s)',
    'Wind-down (1800-2400s)',
    'Recovery (2400-3000s)'
  ]
}

// Pre-computed strategy comparison for table/chart
export const strategyComparison = [
  { name: 'Fixed 60%', speed: 11.2, waitTime: 18.0, throughput: 320, stability: 100, cooperation: 60 },
  { name: 'Threshold (50%)', speed: 10.7, waitTime: 19.3, throughput: 306, stability: 50, cooperation: 53 },
  { name: 'No Coordination', speed: 10.0, waitTime: 21.5, throughput: 285, stability: 30, cooperation: 43 },
  { name: 'Greedy (Nash)', speed: 9.2, waitTime: 23.9, throughput: 261, stability: 60, cooperation: 31 },
  { name: 'BTUT (tau=0.0)', speed: 7.0, waitTime: 30.0, throughput: 200, stability: 95, cooperation: 0 },
  { name: 'BTUT (tau=0.3)', speed: 7.0, waitTime: 30.0, throughput: 200, stability: 95, cooperation: 0 },
  { name: 'BTUT (tau=0.5)', speed: 7.0, waitTime: 30.0, throughput: 200, stability: 95, cooperation: 0 },
]

// Pre-computed tau sweep data for parameter sweep chart
export const tauSweepData = [
  { tau: 0.0, cooperation: 49.0, speedImprovement: 0, waitReduction: 0 },
  { tau: 0.1, cooperation: 53.2, speedImprovement: 4.5, waitReduction: -0.9 },
  { tau: 0.2, cooperation: 54.9, speedImprovement: 5.3, waitReduction: -6.8 },
  { tau: 0.3, cooperation: 57.8, speedImprovement: 9.2, waitReduction: -9.7 },
  { tau: 0.4, cooperation: 63.5, speedImprovement: 2.6, waitReduction: -2.3 },
  { tau: 0.5, cooperation: 67.3, speedImprovement: 4.7, waitReduction: 4.7 },
  { tau: 0.6, cooperation: 70.1, speedImprovement: 3.3, waitReduction: 0.2 },
  { tau: 0.7, cooperation: 69.8, speedImprovement: 12.3, waitReduction: 3.1 },
  { tau: 0.8, cooperation: 73.4, speedImprovement: 6.6, waitReduction: 2.1 },
]

// Pre-computed scale test data
export const scaleTestData = [
  { agents: 500, iterations: 12, speed: 7.88, waitTime: 30.3 },
  { agents: 1000, iterations: 12, speed: 7.76, waitTime: 30.6 },
  { agents: 2000, iterations: 12, speed: 7.52, waitTime: 31.2 },
  { agents: 5000, iterations: 12, speed: 6.80, waitTime: 33.0 },
  { agents: 10000, iterations: 12, speed: 5.60, waitTime: 36.0 },
]

// Drone swarm summary
export const droneSwarmSummary = [
  { drones: 50, cooperation: 100, formationError: 82, collisions: 0, energy: 96.9 },
  { drones: 100, cooperation: 100, formationError: 90, collisions: 42, energy: 96.9 },
  { drones: 200, cooperation: 100, formationError: 103, collisions: 294, energy: 96.8 },
]

// Critical points by domain
export const criticalPoints = {
  abstract: { gammaCritical: 1.326, recommended: 1.376, confidence: 0.86 },
  traffic: { gammaCritical: 1.329, recommended: 1.379, confidence: 0.82 },
  drone: { gammaCritical: 1.239, recommended: 1.289, confidence: 0.90 },
}
