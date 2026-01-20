// Simulation data types and mock data generator

export interface MarketMetrics {
  timestep: number
  midPrice: number
  spread: number
  spreadBps: number
  bidVolume: number
  askVolume: number
  imbalance: number
  volatility: number
  cooperation: number
  regeneration: number
  drag: number
  isCrash: boolean
}

export interface AgentStats {
  type: string
  count: number
  cooperation: number
  color: string
}

export interface CrashEvent {
  startTimestep: number
  endTimestep: number
  maxDrawdown: number
  peakSpread: number
  recovered: boolean
}

export interface SimulationConfig {
  numAgents: number
  numTimesteps: number
  kernelDecay: number
  momentum: number
  temperature: number
  cooperationBias: number
}

export interface SimulationState {
  isRunning: boolean
  currentTimestep: number
  progress: number
  metrics: MarketMetrics[]
  crashes: CrashEvent[]
}

// Generate realistic-looking simulation data
export function generateSimulationData(
  numTimesteps: number,
  config: SimulationConfig
): MarketMetrics[] {
  const data: MarketMetrics[] = []
  let price = 100
  let cooperation = 0.5 + config.cooperationBias
  let volatility = 0.02

  // Simulate a crash around timestep 200-250
  const crashStart = Math.floor(numTimesteps * 0.4)
  const crashEnd = crashStart + 50

  for (let t = 0; t < numTimesteps; t++) {
    const inCrash = t >= crashStart && t <= crashEnd

    // Volatility regime
    if (inCrash) {
      volatility = Math.min(0.08, volatility + 0.002)
      cooperation = Math.max(0.15, cooperation - 0.015)
    } else if (t > crashEnd) {
      volatility = Math.max(0.015, volatility * 0.98)
      cooperation = Math.min(0.85, cooperation + 0.008)
    } else {
      volatility = 0.02 + Math.random() * 0.01
      cooperation = Math.min(0.8, cooperation + (Math.random() - 0.48) * 0.02)
    }

    // Price movement
    const returns = (Math.random() - 0.5) * volatility * (inCrash ? 3 : 1)
    price = Math.max(80, Math.min(120, price * (1 + returns)))

    // Spread widens during crash
    const baseSpread = 0.02
    const spreadMultiplier = inCrash ? 3 + Math.random() * 2 : 1 + Math.random() * 0.5
    const spread = baseSpread * spreadMultiplier

    // Volume imbalance
    const imbalance = (Math.random() - 0.5) * (inCrash ? 0.6 : 0.3)

    // Regeneration and drag scores
    const regeneration = cooperation * 0.4 + (1 - volatility * 10) * 0.3 + (1 - spread * 10) * 0.3
    const drag = (1 - cooperation) * 0.4 + volatility * 10 * 0.3 + spread * 10 * 0.3

    data.push({
      timestep: t,
      midPrice: price,
      spread: spread,
      spreadBps: spread / price * 10000,
      bidVolume: 5000 + Math.random() * 3000 * (inCrash ? 0.3 : 1),
      askVolume: 5000 + Math.random() * 3000 * (inCrash ? 0.3 : 1),
      imbalance,
      volatility,
      cooperation,
      regeneration: Math.max(0, Math.min(1, regeneration)),
      drag: Math.max(0, Math.min(1, drag)),
      isCrash: inCrash,
    })
  }

  return data
}

export function generateAgentStats(numAgents: number): AgentStats[] {
  return [
    { type: 'Retail', count: Math.floor(numAgents * 0.7), cooperation: 0.52, color: '#0071e3' },
    { type: 'HFT', count: Math.floor(numAgents * 0.05), cooperation: 0.61, color: '#af52de' },
    { type: 'Institution', count: Math.floor(numAgents * 0.1), cooperation: 0.58, color: '#ff9500' },
    { type: 'Market Maker', count: Math.floor(numAgents * 0.15), cooperation: 0.73, color: '#34c759' },
  ]
}

export const defaultConfig: SimulationConfig = {
  numAgents: 100000,
  numTimesteps: 500,
  kernelDecay: 0.1,
  momentum: 0.7,
  temperature: 1.0,
  cooperationBias: 0.0,
}
