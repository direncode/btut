'use client'

import { useState, useCallback } from 'react'
import Link from 'next/link'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { BTUTSimulator, SimulationState, SimulationConfig, PRESETS } from '@/lib/simulation/btut-engine'

const defaultConfig: SimulationConfig = {
  N: 1000,
  gamma: 1.5,
  tau: 0.5,
  cA_SH: 0.40,
  cB_SH: 0.10,
  cA_PD: 0.20,
  cB_PD: 0.08,
  alpha: 0.6,
  iterations: 25,
  m: 3,
  seed: 42
}

export function EmbeddedSimulator() {
  const [config, setConfig] = useState<SimulationConfig>(defaultConfig)
  const [state, setState] = useState<SimulationState | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [executionTime, setExecutionTime] = useState<number | null>(null)

  const runSimulation = useCallback(() => {
    setIsRunning(true)
    setState(null)
    setExecutionTime(null)

    // Run in next tick to allow UI update
    setTimeout(() => {
      const startTime = performance.now()
      const simulator = new BTUTSimulator(config)
      const states = simulator.run()
      const endTime = performance.now()

      // Get the final state
      const finalState = states[states.length - 1]
      setState(finalState)
      setExecutionTime(endTime - startTime)
      setIsRunning(false)
    }, 50)
  }, [config])

  const chartData = state?.convergenceHistory.map((value, index) => ({
    iteration: index + 1,
    fractionA: value,
  })) || []

  const finalCooperation = state?.convergenceHistory[state.convergenceHistory.length - 1] ?? 0

  return (
    <section className="py-16 bg-gray-950/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="gradient-text">Try It Yourself</span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Run the actual BTUT simulation in your browser.
            Adjust parameters and watch cooperation emerge.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Parameters</h3>

            {/* N Agents */}
            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-1">
                Agents (N): <span className="text-neon-blue">{config.N.toLocaleString()}</span>
              </label>
              <input
                type="range"
                min={100}
                max={10000}
                step={100}
                value={config.N}
                onChange={(e) => setConfig({ ...config, N: parseInt(e.target.value) })}
                className="w-full accent-neon-blue"
              />
            </div>

            {/* Gamma */}
            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-1">
                Gamma (cooperation bonus): <span className="text-neon-green">{config.gamma.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min={1.0}
                max={2.0}
                step={0.05}
                value={config.gamma}
                onChange={(e) => setConfig({ ...config, gamma: parseFloat(e.target.value) })}
                className="w-full accent-neon-green"
              />
            </div>

            {/* Tau */}
            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-1">
                Tau (hub influence): <span className="text-neon-purple">{config.tau.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={config.tau}
                onChange={(e) => setConfig({ ...config, tau: parseFloat(e.target.value) })}
                className="w-full accent-purple-500"
              />
            </div>

            {/* Iterations */}
            <div className="mb-6">
              <label className="block text-sm text-gray-400 mb-1">
                Iterations: <span className="text-gray-200">{config.iterations}</span>
              </label>
              <input
                type="range"
                min={10}
                max={50}
                step={5}
                value={config.iterations}
                onChange={(e) => setConfig({ ...config, iterations: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Presets */}
            <div className="mb-6">
              <label className="block text-sm text-gray-400 mb-2">Quick Presets</label>
              <div className="flex flex-wrap gap-2">
                {['quick', 'standard', 'high_cooperation'].map((preset) => (
                  <button
                    key={preset}
                    onClick={() => setConfig({ ...defaultConfig, ...PRESETS[preset as keyof typeof PRESETS] })}
                    className="px-3 py-1 text-xs rounded bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-neon-blue/50 transition-colors"
                  >
                    {preset.replace('_', ' ')}
                  </button>
                ))}
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={runSimulation}
              disabled={isRunning}
              className={`
                w-full py-3 rounded-lg font-semibold transition-all
                ${isRunning
                  ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                  : 'bg-neon-green/20 border border-neon-green/50 text-neon-green hover:bg-neon-green/30'
                }
              `}
            >
              {isRunning ? 'Running...' : 'Run Simulation'}
            </button>

            {/* Execution Time */}
            {executionTime !== null && (
              <div className="mt-3 text-center text-sm text-gray-400">
                Completed in <span className="text-neon-blue font-mono">{executionTime.toFixed(1)}ms</span>
              </div>
            )}
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2 bg-gray-900/50 border border-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-200">Convergence Dynamics</h3>
              {state && (
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-gray-400">
                    Final: <span className={`font-bold ${finalCooperation > 0.9 ? 'text-neon-green' : 'text-yellow-400'}`}>
                      {(finalCooperation * 100).toFixed(1)}%
                    </span>
                  </span>
                </div>
              )}
            </div>

            {chartData.length > 0 ? (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <defs>
                      <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#3b82f6" />
                        <stop offset="100%" stopColor="#00ff88" />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey="iteration"
                      stroke="#9ca3af"
                      label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      stroke="#9ca3af"
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Cooperation']}
                    />
                    <ReferenceLine y={1.0} stroke="#00ff88" strokeDasharray="5 5" />
                    <ReferenceLine y={0.5} stroke="#eab308" strokeDasharray="3 3" />
                    <Line
                      type="monotone"
                      dataKey="fractionA"
                      stroke="url(#lineGradient)"
                      strokeWidth={3}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-80 flex items-center justify-center border border-dashed border-gray-700 rounded-lg">
                <div className="text-center text-gray-500">
                  <div className="text-4xl mb-2">&#x25B6;</div>
                  <p>Click "Run Simulation" to see results</p>
                </div>
              </div>
            )}

            {/* Result Summary */}
            {state && (
              <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <div className="text-2xl font-bold text-neon-blue">{config.N.toLocaleString()}</div>
                  <div className="text-xs text-gray-400">Agents</div>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <div className={`text-2xl font-bold ${finalCooperation > 0.9 ? 'text-neon-green' : 'text-yellow-400'}`}>
                    {(finalCooperation * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-400">Cooperation</div>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <div className="text-2xl font-bold text-gray-200">{chartData.length}</div>
                  <div className="text-xs text-gray-400">Iterations</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Link to Full Simulator */}
        <div className="mt-8 text-center">
          <Link
            href="/simulator"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-gray-800 border border-gray-700 hover:border-neon-blue/50 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            Open Full Simulator with Network Visualization
          </Link>
        </div>
      </div>
    </section>
  )
}
