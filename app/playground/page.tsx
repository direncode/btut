'use client'

import { useState } from 'react'
import { BTUTSimulator, PRESETS } from '@/lib/simulation/btut-engine'
import type { SimulationConfig } from '@/lib/simulation/btut-engine'
import { Play, Download, Share2, Code, Sparkles } from 'lucide-react'

export default function PlaygroundPage() {
  const [config, setConfig] = useState<SimulationConfig>(PRESETS.quick)
  const [results, setResults] = useState<any>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [code, setCode] = useState('')

  const runSimulation = () => {
    setIsRunning(true)
    
    const sim = new BTUTSimulator(config)
    const states = sim.run()
    const finalState = states[states.length - 1]
    
    setResults({
      finalFractionA: finalState.fractionA,
      convergenceHistory: finalState.convergenceHistory,
      iterations: states.length,
      agentCount: config.N,
    })
    
    setIsRunning(false)
  }

  const generateCode = () => {
    const codeTemplate = `import { BTUTSimulator } from '@/lib/simulation/btut-engine'

// Configure your simulation
const config = {
  N: ${config.N},
  gamma: ${config.gamma},
  tau: ${config.tau},
  cA_SH: ${config.cA_SH},
  cB_SH: ${config.cB_SH},
  cA_PD: ${config.cA_PD},
  cB_PD: ${config.cB_PD},
  alpha: ${config.alpha},
  iterations: ${config.iterations},
  m: ${config.m},
  seed: ${config.seed || 42},
}

// Run simulation
const simulator = new BTUTSimulator(config)
const results = simulator.run()

// Get final state
const finalState = results[results.length - 1]
console.log('Final cooperation:', finalState.fractionA)
console.log('Convergence history:', finalState.convergenceHistory)
`
    setCode(codeTemplate)
  }

  const exportResults = () => {
    if (!results) return
    
    const data = {
      config,
      results,
      timestamp: new Date().toISOString(),
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `btut-playground-${Date.now()}.json`
    a.click()
  }

  return (
    <div className="min-h-screen bg-black pt-20">
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-3 mb-2">
            <Sparkles className="w-8 h-8 text-neon-purple" />
            <h1 className="text-4xl font-bold gradient-text">Playground</h1>
          </div>
          <p className="text-gray-400">Experiment with BTUT configurations and see instant results</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Configuration */}
          <div className="space-y-6">
            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <h2 className="text-2xl font-bold mb-6 text-neon-blue">Quick Experiments</h2>
              
              <div className="space-y-4">
                <button
                  onClick={() => setConfig({...PRESETS.quick, gamma: 1.8})}
                  className="w-full text-left px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition"
                >
                  <div className="font-bold text-neon-green">High Cooperation</div>
                  <div className="text-sm text-gray-400">γ = 1.8 (strong incentive)</div>
                </button>

                <button
                  onClick={() => setConfig({...PRESETS.quick, gamma: 1.2})}
                  className="w-full text-left px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition"
                >
                  <div className="font-bold text-neon-blue">Moderate Cooperation</div>
                  <div className="text-sm text-gray-400">γ = 1.2 (weak incentive)</div>
                </button>

                <button
                  onClick={() => setConfig({...PRESETS.quick, tau: 0.6})}
                  className="w-full text-left px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition"
                >
                  <div className="font-bold text-neon-purple">Hub Dominated</div>
                  <div className="text-sm text-gray-400">τ = 0.6 (strong hub influence)</div>
                </button>

                <button
                  onClick={() => setConfig({...PRESETS.quick, N: 50000, iterations: 30})}
                  className="w-full text-left px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition"
                >
                  <div className="font-bold text-neon-pink">Large Scale</div>
                  <div className="text-sm text-gray-400">50K agents, 30 iterations</div>
                </button>
              </div>
            </div>

            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <h2 className="text-2xl font-bold mb-6 text-neon-green">Custom Configuration</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Agents (N)</label>
                  <input
                    type="number"
                    value={config.N}
                    onChange={(e) => setConfig({...config, N: parseInt(e.target.value)})}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Cooperation Bonus (γ)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={config.gamma}
                    onChange={(e) => setConfig({...config, gamma: parseFloat(e.target.value)})}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Hub Influence (τ)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={config.tau}
                    onChange={(e) => setConfig({...config, tau: parseFloat(e.target.value)})}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Iterations</label>
                  <input
                    type="number"
                    value={config.iterations}
                    onChange={(e) => setConfig({...config, iterations: parseInt(e.target.value)})}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
                  />
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={runSimulation}
                disabled={isRunning}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition disabled:opacity-50"
              >
                <Play className="w-5 h-5" />
                {isRunning ? 'Running...' : 'Run Experiment'}
              </button>

              <button
                onClick={generateCode}
                className="px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white font-semibold rounded-lg transition flex items-center gap-2"
              >
                <Code className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {results ? (
              <>
                <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold text-neon-purple">Results</h2>
                    <button
                      onClick={exportResults}
                      className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition flex items-center gap-2 text-sm"
                    >
                      <Download className="w-4 h-4" />
                      Export
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="bg-black/50 rounded-lg p-4">
                      <div className="text-3xl font-bold text-neon-green mb-1">
                        {results.finalFractionA.toFixed(4)}
                      </div>
                      <div className="text-sm text-gray-400">Final Cooperation</div>
                    </div>

                    <div className="bg-black/50 rounded-lg p-4">
                      <div className="text-3xl font-bold text-neon-blue mb-1">
                        {results.iterations}
                      </div>
                      <div className="text-sm text-gray-400">Iterations</div>
                    </div>

                    <div className="bg-black/50 rounded-lg p-4">
                      <div className="text-3xl font-bold text-neon-purple mb-1">
                        {results.agentCount.toLocaleString()}
                      </div>
                      <div className="text-sm text-gray-400">Agents</div>
                    </div>

                    <div className="bg-black/50 rounded-lg p-4">
                      <div className="text-3xl font-bold text-neon-pink mb-1">
                        {results.finalFractionA > 0.99 ? 'YES' : 'NO'}
                      </div>
                      <div className="text-sm text-gray-400">Converged</div>
                    </div>
                  </div>

                  <div className="bg-black/50 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-2">Convergence Path</div>
                    <div className="flex items-end gap-1 h-20">
                      {results.convergenceHistory.map((val: number, i: number) => (
                        <div
                          key={i}
                          className="flex-1 bg-gradient-to-t from-neon-blue to-neon-green rounded-t"
                          style={{ height: `${val * 100}%` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>

                <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
                  <h3 className="text-xl font-bold mb-4 text-neon-blue">Interpretation</h3>
                  <div className="space-y-3 text-sm">
                    {results.finalFractionA > 0.99 ? (
                      <p className="text-neon-green">
                        ✅ <strong>Full Cooperation Achieved</strong> - The network converged to universal strategy A. 
                        Hubs successfully coordinated the entire population.
                      </p>
                    ) : results.finalFractionA > 0.8 ? (
                      <p className="text-neon-blue">
                        📊 <strong>High Cooperation</strong> - Most agents adopted strategy A, but some defectors remain. 
                        Consider increasing γ or τ.
                      </p>
                    ) : (
                      <p className="text-neon-purple">
                        ⚠️ <strong>Mixed Equilibrium</strong> - The system reached a balance between strategies. 
                        Cooperation incentives may be too weak.
                      </p>
                    )}
                    
                    <p className="text-gray-400">
                      With γ={config.gamma}, the cooperation bonus is {config.gamma > 1.5 ? 'strong' : config.gamma > 1.3 ? 'moderate' : 'weak'}.
                      Hub influence (τ={config.tau}) is {config.tau > 0.4 ? 'high' : config.tau > 0.2 ? 'moderate' : 'low'}.
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6 h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="text-6xl mb-4">🧪</div>
                  <p className="text-gray-400">Configure parameters and run an experiment</p>
                  <p className="text-sm text-gray-500 mt-2">Results will appear here</p>
                </div>
              </div>
            )}

            {code && (
              <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-neon-green">Generated Code</h3>
                  <button
                    onClick={() => navigator.clipboard.writeText(code)}
                    className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-sm"
                  >
                    Copy
                  </button>
                </div>
                <pre className="bg-black/50 rounded-lg p-4 overflow-x-auto text-xs">
                  <code className="text-neon-green">{code}</code>
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
