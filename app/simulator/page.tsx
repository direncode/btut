'use client'

import { useState, useEffect } from 'react'
import SimulationControls from '@/components/simulator/SimulationControls'
import VisualizationPanel from '@/components/simulator/VisualizationPanel'
import MetricsPanel from '@/components/simulator/MetricsPanel'
import NetworkView from '@/components/simulator/NetworkView'
import { BTUTUnifiedSimulator, PRESETS, initWasm, isWasmAvailable } from '@/lib/simulation'
import type { SimulationState, SimulationConfig } from '@/lib/simulation/btut-engine'
import { Play, Pause, RotateCcw, Download, Zap, Code } from 'lucide-react'

export default function SimulatorPage() {
  const [config, setConfig] = useState<SimulationConfig>(PRESETS.quick)
  const [simulator, setSimulator] = useState<BTUTUnifiedSimulator | null>(null)
  const [state, setState] = useState<SimulationState | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [engineType, setEngineType] = useState<'wasm' | 'typescript' | 'loading'>('loading')

  useEffect(() => {
    const init = async () => {
      const success = await initWasm()
      setEngineType(success ? 'wasm' : 'typescript')
    }
    init()
  }, [])

  const initializeSimulator = () => {
    const sim = new BTUTUnifiedSimulator(config)
    setSimulator(sim)
    setState(sim.getState())
    setEngineType(sim.getEngineType())
  }

  const runSimulation = async () => {
    if (!simulator) {
      initializeSimulator()
      return
    }

    setIsRunning(true)
    setIsPaused(false)

    const runStep = () => {
      if (!simulator) return

      const newState = simulator.step()
      setState(newState)

      if (!newState.isComplete && !isPaused) {
        requestAnimationFrame(runStep)
      } else {
        setIsRunning(false)
      }
    }

    requestAnimationFrame(runStep)
  }

  const pauseSimulation = () => {
    setIsPaused(true)
    setIsRunning(false)
  }

  const resetSimulation = () => {
    if (simulator) {
      simulator.reset()
      setState(simulator.getState())
    }
    setIsRunning(false)
    setIsPaused(false)
  }

  const exportData = () => {
    if (!state) return
    
    const data = {
      config,
      finalState: state,
      convergenceHistory: state.convergenceHistory,
      engineType,
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `btut-simulation-${Date.now()}.json`
    a.click()
  }

  return (
    <div className="min-h-screen bg-black pt-20">
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold gradient-text mb-2">BTUT Simulator</h1>
              <p className="text-gray-400">Real-time multi-agent coordination engine</p>
            </div>
            
            <div className="flex items-center gap-2">
              {engineType === 'loading' ? (
                <div className="px-4 py-2 bg-gray-800 rounded-lg text-gray-400 text-sm">
                  Loading...
                </div>
              ) : engineType === 'wasm' ? (
                <div className="px-4 py-2 bg-gradient-to-r from-neon-blue/20 to-neon-green/20 border border-neon-blue/50 rounded-lg flex items-center gap-2">
                  <Zap className="w-4 h-4 text-neon-blue animate-pulse" />
                  <span className="text-sm font-bold text-neon-blue">Rust/WASM Accelerated</span>
                </div>
              ) : (
                <div className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg flex items-center gap-2">
                  <Code className="w-4 h-4 text-gray-400" />
                  <span className="text-sm font-mono text-gray-400">TypeScript Engine</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <h2 className="text-xl font-bold mb-4 text-neon-blue">Simulation Controls</h2>
              
              <SimulationControls 
                config={config}
                onConfigChange={setConfig}
                disabled={isRunning}
              />

              <div className="space-y-3 mt-6">
                <button
                  onClick={isRunning ? pauseSimulation : runSimulation}
                  disabled={!simulator && !config}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isRunning ? (
                    <>
                      <Pause className="w-5 h-5" />
                      Pause
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      {state?.iteration > 0 ? 'Resume' : 'Run'}
                    </>
                  )}
                </button>

                <button
                  onClick={resetSimulation}
                  disabled={!simulator}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gray-800 hover:bg-gray-700 text-white font-semibold rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <RotateCcw className="w-5 h-5" />
                  Reset
                </button>

                <button
                  onClick={exportData}
                  disabled={!state}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gray-800 hover:bg-gray-700 text-white font-semibold rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Download className="w-5 h-5" />
                  Export Data
                </button>
              </div>
            </div>

            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-bold mb-4 text-neon-green">Quick Presets</h3>
              <div className="space-y-2">
                {Object.entries(PRESETS).map(([name, preset]) => (
                  <button
                    key={name}
                    onClick={() => {
                      setConfig(preset)
                      resetSimulation()
                    }}
                    disabled={isRunning}
                    className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {name.charAt(0).toUpperCase() + name.slice(1)} ({preset.N.toLocaleString()} agents)
                  </button>
                ))}
              </div>
            </div>

            {engineType === 'wasm' && (
              <div className="bg-gradient-to-r from-neon-blue/10 to-neon-green/10 border border-neon-blue/30 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-2">⚡ Performance Mode</p>
                <p className="text-sm text-gray-300">
                  Rust/WASM acceleration active. 10-100x faster than TypeScript.
                </p>
              </div>
            )}
          </div>

          <div className="lg:col-span-2 space-y-6">
            <VisualizationPanel state={state} config={config} />
            <NetworkView state={state} />
          </div>

          <div className="lg:col-span-1">
            <MetricsPanel state={state} config={config} />
          </div>
        </div>
      </div>
    </div>
  )
}
