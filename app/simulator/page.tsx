'use client'

import { useState, useEffect } from 'react'
import SimulationControls from '@/components/simulator/SimulationControls'
import VisualizationPanel from '@/components/simulator/VisualizationPanel'
import MetricsPanel from '@/components/simulator/MetricsPanel'
import NetworkView from '@/components/simulator/NetworkView'
import { BTUTUnifiedSimulator, PRESETS, initWasm, isWasmAvailable } from '@/lib/simulation'
import type { SimulationState, SimulationConfig } from '@/lib/simulation/btut-engine'
import { Play, Pause, RotateCcw, Download, Zap, Code, Wifi, Radio, Car, Cloud } from 'lucide-react'

export default function SimulatorPage() {
  const [config, setConfig] = useState<SimulationConfig>(PRESETS.quick)
  const [simulator, setSimulator] = useState<BTUTUnifiedSimulator | null>(null)
  const [state, setState] = useState<SimulationState | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [engineType, setEngineType] = useState<'wasm' | 'typescript' | 'loading'>('loading')
  const [connectionMode, setConnectionMode] = useState<'standalone' | 'ros' | 'sumo' | 'api'>('standalone')
  const [showConnectionConfig, setShowConnectionConfig] = useState(false)

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
        {/* Connection Mode Selector */}
        <div className="mb-6 card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white">Connection Mode</h3>
            <button
              onClick={() => setShowConnectionConfig(!showConnectionConfig)}
              className="text-sm text-gray-400 hover:text-neon-blue transition"
            >
              {showConnectionConfig ? 'Hide' : 'Configure'}
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <button
              onClick={() => setConnectionMode('standalone')}
              className={`p-4 rounded-lg border-2 transition ${
                connectionMode === 'standalone'
                  ? 'border-neon-blue bg-neon-blue/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Wifi className={`w-6 h-6 mx-auto mb-2 ${connectionMode === 'standalone' ? 'text-neon-blue' : 'text-gray-400'}`} />
              <div className={`text-sm font-semibold ${connectionMode === 'standalone' ? 'text-neon-blue' : 'text-gray-400'}`}>
                Standalone
              </div>
              <div className="text-xs text-gray-500 mt-1">Browser only</div>
            </button>

            <button
              onClick={() => setConnectionMode('ros')}
              className={`p-4 rounded-lg border-2 transition ${
                connectionMode === 'ros'
                  ? 'border-neon-green bg-neon-green/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Radio className={`w-6 h-6 mx-auto mb-2 ${connectionMode === 'ros' ? 'text-neon-green' : 'text-gray-400'}`} />
              <div className={`text-sm font-semibold ${connectionMode === 'ros' ? 'text-neon-green' : 'text-gray-400'}`}>
                ROS
              </div>
              <div className="text-xs text-gray-500 mt-1">Connect robots</div>
            </button>

            <button
              onClick={() => setConnectionMode('sumo')}
              className={`p-4 rounded-lg border-2 transition ${
                connectionMode === 'sumo'
                  ? 'border-neon-purple bg-neon-purple/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Car className={`w-6 h-6 mx-auto mb-2 ${connectionMode === 'sumo' ? 'text-neon-purple' : 'text-gray-400'}`} />
              <div className={`text-sm font-semibold ${connectionMode === 'sumo' ? 'text-neon-purple' : 'text-gray-400'}`}>
                SUMO
              </div>
              <div className="text-xs text-gray-500 mt-1">Traffic sim</div>
            </button>

            <button
              onClick={() => setConnectionMode('api')}
              className={`p-4 rounded-lg border-2 transition ${
                connectionMode === 'api'
                  ? 'border-neon-pink bg-neon-pink/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Cloud className={`w-6 h-6 mx-auto mb-2 ${connectionMode === 'api' ? 'text-neon-pink' : 'text-gray-400'}`} />
              <div className={`text-sm font-semibold ${connectionMode === 'api' ? 'text-neon-pink' : 'text-gray-400'}`}>
                API
              </div>
              <div className="text-xs text-gray-500 mt-1">Backend server</div>
            </button>
          </div>

          {/* Connection Configuration */}
          {showConnectionConfig && connectionMode !== 'standalone' && (
            <div className="mt-4 p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
              {connectionMode === 'ros' && (
                <div>
                  <div className="text-sm text-gray-400 mb-3">ROS Configuration</div>
                  <input
                    type="text"
                    placeholder="ws://localhost:9090"
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white text-sm mb-2"
                  />
                  <div className="text-xs text-gray-500">
                    Make sure rosbridge_server is running.{' '}
                    <a href="/robotics" className="text-neon-green hover:underline">
                      Go to Robotics Dashboard →
                    </a>
                  </div>
                </div>
              )}

              {connectionMode === 'sumo' && (
                <div>
                  <div className="text-sm text-gray-400 mb-3">SUMO Configuration</div>
                  <input
                    type="text"
                    placeholder="localhost:8813"
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white text-sm mb-2"
                  />
                  <div className="text-xs text-gray-500">
                    Make sure SUMO is running with TraCI.{' '}
                    <a href="/traffic" className="text-neon-purple hover:underline">
                      Go to Traffic Dashboard →
                    </a>
                  </div>
                </div>
              )}

              {connectionMode === 'api' && (
                <div>
                  <div className="text-sm text-gray-400 mb-3">API Configuration</div>
                  <input
                    type="text"
                    placeholder="https://btut-api.fly.dev"
                    defaultValue="https://btut-api.fly.dev"
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white text-sm mb-2"
                  />
                  <div className="text-xs text-gray-500">
                    Using production API server on Fly.io
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

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
                      {(state?.iteration ?? 0) > 0 ? 'Resume' : 'Run'}
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
