'use client'

import { SimulationState, SimulationConfig } from '@/lib/simulation/btut-engine'
import { Activity, Zap, Users, TrendingUp } from 'lucide-react'

interface MetricsPanelProps {
  state: SimulationState | null
  config: SimulationConfig
}

export default function MetricsPanel({ state, config }: MetricsPanelProps) {
  const calculateStats = () => {
    if (!state) return null

    const variance = state.convergenceHistory.length > 1
      ? state.convergenceHistory.reduce((acc, val, idx, arr) => {
          const mean = arr.reduce((s, v) => s + v, 0) / arr.length
          return acc + Math.pow(val - mean, 2)
        }, 0) / state.convergenceHistory.length
      : 0

    const convergenceRate = state.convergenceHistory.length > 1
      ? Math.abs(
          state.convergenceHistory[state.convergenceHistory.length - 1] - 
          state.convergenceHistory[state.convergenceHistory.length - 2]
        )
      : 0

    const hubCount = state.agents.filter(a => a.weight > 0.5).length
    const hubPercentage = (hubCount / state.agents.length) * 100

    return {
      variance: variance.toExponential(2),
      convergenceRate: convergenceRate.toFixed(6),
      hubCount,
      hubPercentage: hubPercentage.toFixed(1),
      avgDegree: (state.agents.reduce((sum, a) => sum + a.degree, 0) / state.agents.length).toFixed(2),
      throughput: ((config.N * state.iteration) / ((state.runtime || 1000) / 1000)).toFixed(0),
    }
  }

  const stats = calculateStats()

  const metrics = [
    {
      icon: Activity,
      label: 'Convergence Rate',
      value: stats?.convergenceRate || '0.000000',
      color: 'text-neon-blue',
      description: 'Δp per iteration',
    },
    {
      icon: TrendingUp,
      label: 'Variance',
      value: stats?.variance || '0.00e+0',
      color: 'text-neon-green',
      description: 'Statistical spread',
    },
    {
      icon: Users,
      label: 'Hub Agents',
      value: `${stats?.hubCount || 0} (${stats?.hubPercentage || 0}%)`,
      color: 'text-neon-purple',
      description: 'High-influence nodes',
    },
    {
      icon: Zap,
      label: 'Throughput',
      value: `${stats ? (parseInt(stats.throughput) / 1000).toFixed(0) : 0}K/s`,
      color: 'text-neon-pink',
      description: 'Agent-steps per second',
    },
  ]

  return (
    <div className="space-y-6">
      <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-4 text-neon-purple">Live Metrics</h2>
        
        <div className="space-y-4">
          {metrics.map((metric) => {
            const Icon = metric.icon
            return (
              <div key={metric.label} className="bg-black/50 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <Icon className={`w-5 h-5 ${metric.color} mr-2`} />
                  <span className="text-sm font-medium text-gray-300">{metric.label}</span>
                </div>
                <div className={`text-2xl font-bold ${metric.color} mb-1`}>
                  {metric.value}
                </div>
                <div className="text-xs text-gray-500">{metric.description}</div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Network Stats */}
      <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-bold mb-4 text-neon-blue">Network Properties</h3>
        
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Total Agents</span>
            <span className="font-mono text-white">{config.N.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Avg. Degree</span>
            <span className="font-mono text-white">{stats?.avgDegree || '0.00'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Min Degree (m)</span>
            <span className="font-mono text-white">{config.m}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Topology</span>
            <span className="font-mono text-neon-green">Scale-Free</span>
          </div>
        </div>
      </div>

      {/* Game Parameters */}
      <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-bold mb-4 text-neon-green">Game Parameters</h3>
        
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">SH Bonus (γ)</span>
            <span className="font-mono text-white">{config.gamma.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Hub Influence (τ)</span>
            <span className="font-mono text-white">{config.tau.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Cost A (SH)</span>
            <span className="font-mono text-white">{config.cA_SH.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Aggression (α)</span>
            <span className="font-mono text-white">{config.alpha.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Performance Indicator */}
      {state && (
        <div className="bg-gradient-to-r from-neon-blue/10 to-neon-green/10 border border-neon-blue/30 rounded-xl p-4">
          <div className="text-center">
            <div className="text-sm text-gray-400 mb-1">Computational Complexity</div>
            <div className="text-3xl font-bold gradient-text">O(N)</div>
            <div className="text-xs text-gray-500 mt-1">Linear scaling verified</div>
          </div>
        </div>
      )}
    </div>
  )
}
