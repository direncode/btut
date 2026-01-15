'use client'

import { useState } from 'react'
import { Play, Square, Upload, Download, AlertCircle } from 'lucide-react'
import { useSUMOConnection, useSimulationControl, useVehicleData, useTrafficMetrics, useBTUTCoordination, useComparisonMode } from '@/lib/sumo'
import SUMOConnection from '@/components/traffic/SUMOConnection'
import TrafficMap from '@/components/traffic/TrafficMap'
import TrafficMetrics from '@/components/traffic/TrafficMetrics'
import ComparisonView from '@/components/traffic/ComparisonView'
import ReportGenerator from '@/components/traffic/ReportGenerator'

export default function TrafficPage() {
  const [networkFile, setNetworkFile] = useState('')
  const [comparisonMode, setComparisonMode] = useState<'baseline' | 'btut' | 'both'>('baseline')

  // SUMO Connection
  const { connected, connecting, error, client, connect, disconnect } = useSUMOConnection({
    host: 'localhost',
    port: 8813,
    autoConnect: false
  })

  // Simulation Control
  const { isRunning, isPaused, currentTime, start, stop, pause, resume, step } = useSimulationControl(client)

  // Vehicle Data
  const { vehicles, vehicleCount } = useVehicleData(client)

  // Traffic Metrics
  const { metrics, refreshMetrics } = useTrafficMetrics(client)

  // BTUT Coordination
  const { enabled: btutEnabled, gamma, tau, setGamma, setTau, applyCoordination } = useBTUTCoordination(client)

  // Comparison Mode
  const {
    baselineMetrics,
    btutMetrics,
    mode,
    setMode,
    runBaseline,
    runWithBTUT,
    getImprovement
  } = useComparisonMode(client)

  const handleConnect = (host: string, port: number) => {
    connect()
  }

  const handleDisconnect = () => {
    disconnect()
  }

  const handleLoadNetwork = () => {
    if (!networkFile) {
      alert('Please enter a network file path')
      return
    }
    // This would load the SUMO network file
    console.log('Loading network:', networkFile)
  }

  const handleStartSimulation = async () => {
    if (!connected) {
      alert('Please connect to SUMO first')
      return
    }

    try {
      await start(networkFile || 'default.sumocfg')
    } catch (err) {
      console.error('Failed to start simulation:', err)
    }
  }

  const handleStopSimulation = async () => {
    try {
      await stop()
    } catch (err) {
      console.error('Failed to stop simulation:', err)
    }
  }

  const handleApplyBTUT = async () => {
    if (!connected) {
      alert('Please connect to SUMO first')
      return
    }

    try {
      await applyCoordination(gamma, tau)
      alert(`BTUT coordination applied with γ=${gamma}, τ=${tau}`)
    } catch (err) {
      console.error('Failed to apply BTUT:', err)
    }
  }

  const handleRunComparison = async () => {
    if (!connected) {
      alert('Please connect to SUMO and load a network first')
      return
    }

    try {
      // Run baseline
      await runBaseline(networkFile || 'default.sumocfg')

      // Run with BTUT
      await runWithBTUT(networkFile || 'default.sumocfg', gamma, tau)

      alert('Comparison complete! Check the results below.')
    } catch (err) {
      console.error('Comparison failed:', err)
    }
  }

  return (
    <div className="min-h-screen bg-black pt-20">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold gradient-text mb-2">Traffic Control Center</h1>
              <p className="text-gray-400">SUMO traffic simulation with BTUT coordination</p>
            </div>

            {/* Status */}
            <div className="flex items-center gap-4">
              {connected && (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-neon-green rounded-full animate-pulse"></div>
                  <span className="text-neon-green font-semibold">{vehicleCount} vehicles</span>
                </div>
              )}
              {btutEnabled && (
                <div className="px-3 py-1 bg-neon-blue/10 border border-neon-blue/30 rounded-lg">
                  <span className="text-xs text-neon-blue font-semibold">BTUT Active</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Connection Warning */}
        {!connected && !connecting && (
          <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-yellow-500 mb-1">Not Connected to SUMO</div>
              <div className="text-sm text-gray-400">
                Start SUMO with TraCI enabled to begin traffic simulation:
                <code className="block mt-2 px-3 py-2 bg-black/50 rounded text-xs">
                  sumo -c network.sumocfg --remote-port 8813
                </code>
              </div>
            </div>
          </div>
        )}

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Column - Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Connection */}
            <SUMOConnection
              connected={connected}
              connecting={connecting}
              error={error}
              host="localhost"
              port={8813}
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
              vehicleCount={vehicleCount}
              currentTime={currentTime}
            />

            {/* Network File */}
            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-bold text-white mb-4">Network File</h3>
              <div className="space-y-3">
                <input
                  type="text"
                  value={networkFile}
                  onChange={(e) => setNetworkFile(e.target.value)}
                  placeholder="network.sumocfg"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                />
                <button
                  onClick={handleLoadNetwork}
                  disabled={!connected}
                  className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold rounded-lg transition flex items-center justify-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  Load Network
                </button>
              </div>
            </div>

            {/* Traffic Metrics */}
            {metrics && (
              <TrafficMetrics
                averageSpeed={metrics.averageSpeed}
                totalVehicles={metrics.totalVehicles}
                waitingVehicles={metrics.totalWaitingTime / (metrics.totalVehicles || 1)}
                meanWaitingTime={metrics.totalWaitingTime / (metrics.totalVehicles || 1)}
                throughput={metrics.throughput}
                totalFuel={metrics.fuelConsumption}
                totalCO2={metrics.co2Emissions}
                congestionLevel={0}
              />
            )}

            {/* Report Generator */}
            <ReportGenerator
              baselineMetrics={baselineMetrics}
              btutMetrics={btutMetrics}
            />
          </div>

          {/* Center Column - Visualization */}
          <div className="lg:col-span-2 space-y-6">
            {/* Traffic Map */}
            <div>
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold text-white">Live Traffic Map</h2>
                  <p className="text-sm text-gray-400">Real-time vehicle positions and strategies</p>
                </div>
                {isRunning && (
                  <div className="px-3 py-1 bg-neon-green/10 border border-neon-green/30 rounded-lg">
                    <span className="text-sm text-neon-green font-semibold">● Running</span>
                  </div>
                )}
              </div>

              <TrafficMap
                vehicles={vehicles.map(v => ({
                  id: v.id,
                  position: v.position,
                  speed: v.speed,
                  angle: v.angle,
                  roadId: v.roadId,
                  laneId: v.laneId,
                  strategy: v.strategy
                }))}
                showHeatmap={true}
              />
            </div>

            {/* Control Panel */}
            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-bold text-white mb-4">Simulation Control</h3>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <button
                  onClick={isRunning ? handleStopSimulation : handleStartSimulation}
                  disabled={!connected}
                  className="px-4 py-3 bg-neon-green hover:bg-neon-green/80 disabled:bg-gray-700 disabled:text-gray-500 text-black font-bold rounded-lg transition flex items-center justify-center gap-2"
                >
                  {isRunning ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isRunning ? 'Stop' : 'Start'}
                </button>

                <button
                  onClick={isPaused ? resume : pause}
                  disabled={!connected || !isRunning}
                  className="px-4 py-3 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold rounded-lg transition"
                >
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
              </div>

              {/* BTUT Parameters */}
              <div className="pt-4 border-t border-gray-800 space-y-3">
                <h4 className="text-sm font-semibold text-gray-300">BTUT Parameters</h4>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm text-gray-400">Gamma (γ)</label>
                    <span className="text-sm font-mono text-white">{gamma.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="2.0"
                    step="0.1"
                    value={gamma}
                    onChange={(e) => setGamma(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-neon-green"
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm text-gray-400">Tau (τ)</label>
                    <span className="text-sm font-mono text-white">{tau.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.6"
                    step="0.05"
                    value={tau}
                    onChange={(e) => setTau(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-neon-blue"
                  />
                </div>

                <button
                  onClick={handleApplyBTUT}
                  disabled={!connected}
                  className="w-full px-4 py-2 bg-neon-blue hover:bg-neon-blue/80 disabled:bg-gray-700 disabled:text-gray-500 text-black font-semibold rounded-lg transition"
                >
                  Apply BTUT Coordination
                </button>
              </div>

              {/* Comparison Mode */}
              <div className="pt-4 border-t border-gray-800">
                <button
                  onClick={handleRunComparison}
                  disabled={!connected}
                  className="w-full px-4 py-2 bg-neon-purple hover:bg-neon-purple/80 disabled:bg-gray-700 disabled:text-gray-500 text-black font-semibold rounded-lg transition"
                >
                  Run Baseline vs BTUT Comparison
                </button>
              </div>
            </div>
          </div>

          {/* Right Column - Comparison */}
          <div className="lg:col-span-1">
            <ComparisonView
              baselineMetrics={baselineMetrics}
              btutMetrics={btutMetrics}
            />
          </div>
        </div>

        {/* Documentation */}
        <div className="mt-8 p-6 bg-gray-900/50 border border-gray-800 rounded-xl">
          <h3 className="text-lg font-bold text-white mb-2">Need Help?</h3>
          <p className="text-sm text-gray-400 mb-4">
            Check the SUMO integration guide for network setup, TraCI configuration, and troubleshooting.
          </p>
          <a
            href="/docs/integrations/sumo"
            className="inline-flex items-center gap-2 px-4 py-2 bg-neon-green/10 border border-neon-green/50 text-neon-green rounded-lg hover:bg-neon-green/20 transition text-sm font-semibold"
          >
            View SUMO Documentation →
          </a>
        </div>
      </div>
    </div>
  )
}
