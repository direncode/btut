'use client'

import { useState, useEffect } from 'react'
import { Download, Play, Square, RotateCcw, AlertCircle } from 'lucide-react'
import { useROSConnection, useAgentStates, useCoordinationResults, useParameterUpdate, useROSSystemInfo } from '@/lib/ros'
import ROSConnectionStatus from '@/components/robotics/ROSConnectionStatus'
import RobotMap from '@/components/robotics/RobotMap'
import SwarmMetrics from '@/components/robotics/SwarmMetrics'
import ROSTerminal from '@/components/robotics/ROSTerminal'
import ParameterPanel from '@/components/robotics/ParameterPanel'

export default function RoboticsPage() {
  const [namespace, setNamespace] = useState('/btut')
  const [logs, setLogs] = useState<any[]>([])
  const [isRunning, setIsRunning] = useState(false)

  // ROS Connection
  const { connected, connecting, error, ros, connect, disconnect } = useROSConnection({
    url: 'ws://localhost:9090',
    autoConnect: false
  })

  // ROS System Info
  const { nodes, topics, services, refresh: refreshSystemInfo } = useROSSystemInfo(ros)

  // Agent States
  const { agents, lastUpdate } = useAgentStates(ros, namespace)

  // Coordination Results
  const { result } = useCoordinationResults(ros, namespace)

  // Parameter Updates
  const { updateParameters } = useParameterUpdate(ros, namespace)

  // Refresh system info when connected
  useEffect(() => {
    if (connected) {
      refreshSystemInfo()
      addLog('INFO', 'system', 'Connected to ROS master')
    }
  }, [connected, refreshSystemInfo])

  // Log agent updates
  useEffect(() => {
    if (agents.length > 0) {
      addLog('DEBUG', 'btut_node', `Received ${agents.length} agent states`)
    }
  }, [lastUpdate])

  // Log coordination results
  useEffect(() => {
    if (result) {
      addLog('INFO', 'coordinator', `Coordination complete: ${(result.finalFractionA * 100).toFixed(1)}% cooperation`)
    }
  }, [result])

  const addLog = (level: string, node: string, message: string) => {
    const newLog = {
      timestamp: new Date().toLocaleTimeString(),
      level: level as any,
      node,
      message
    }
    setLogs(prev => [...prev, newLog].slice(-100)) // Keep last 100 logs
  }

  const handleConnect = (url: string) => {
    connect({ url })
  }

  const handleDisconnect = () => {
    disconnect()
    addLog('INFO', 'system', 'Disconnected from ROS master')
  }

  const handleApplyParameters = (params: any) => {
    if (!connected) {
      addLog('WARN', 'system', 'Cannot update parameters: not connected to ROS')
      return
    }

    updateParameters(params)
    addLog('INFO', 'btut_node', `Parameters updated: γ=${params.gamma}, τ=${params.tau}`)
  }

  const handleExportROSBag = () => {
    addLog('INFO', 'system', 'ROS bag export started')
    // This would trigger ROS bag recording
    alert('ROS bag export feature requires rosbag record integration')
  }

  const calculateCooperationPercentage = (): number => {
    if (agents.length === 0) return 0
    const cooperating = agents.filter(a => a.strategy === 'A').length
    return (cooperating / agents.length) * 100
  }

  const calculateAverageUtility = (): number => {
    if (agents.length === 0) return 0
    const sum = agents.reduce((acc, a) => acc + a.utility, 0)
    return sum / agents.length
  }

  return (
    <div className="min-h-screen bg-black pt-20">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold gradient-text mb-2">Robotics Dashboard</h1>
              <p className="text-gray-400">Live ROS swarm coordination and monitoring</p>
            </div>

            {/* Status Indicator */}
            <div className="flex items-center gap-4">
              {connected && (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-neon-green rounded-full animate-pulse"></div>
                  <span className="text-neon-green font-semibold">{agents.length} robots online</span>
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
              <div className="font-semibold text-yellow-500 mb-1">Not Connected to ROS</div>
              <div className="text-sm text-gray-400">
                Connect to your ROS master to start monitoring robots. Make sure rosbridge_server is running:
                <code className="block mt-2 px-3 py-2 bg-black/50 rounded text-xs">
                  roslaunch rosbridge_server rosbridge_websocket.launch
                </code>
              </div>
            </div>
          </div>
        )}

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Column - Controls & Connection */}
          <div className="lg:col-span-1 space-y-6">
            {/* Connection Status */}
            <ROSConnectionStatus
              connected={connected}
              connecting={connecting}
              error={error}
              url="ws://localhost:9090"
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
              nodes={nodes}
              topics={topics}
            />

            {/* Swarm Metrics */}
            <SwarmMetrics
              totalRobots={agents.length}
              cooperationPercentage={calculateCooperationPercentage()}
              averageUtility={calculateAverageUtility()}
              convergenceProgress={result?.totalIterations || 0}
              collisionsAvoided={0}
            />

            {/* Parameters */}
            <ParameterPanel
              onApply={handleApplyParameters}
              savedConfigs={[
                {
                  name: 'High Cooperation',
                  params: {
                    gamma: 1.8,
                    tau: 0.4,
                    cA_SH: 0.8,
                    cB_SH: 0.6,
                    cA_PD: 0.5,
                    cB_PD: 0.3,
                    alpha: 0.1,
                    m: 10
                  }
                },
                {
                  name: 'Fast Convergence',
                  params: {
                    gamma: 1.6,
                    tau: 0.5,
                    cA_SH: 0.8,
                    cB_SH: 0.6,
                    cA_PD: 0.5,
                    cB_PD: 0.3,
                    alpha: 0.2,
                    m: 15
                  }
                }
              ]}
            />
          </div>

          {/* Center Column - Map */}
          <div className="lg:col-span-2 space-y-6">
            {/* Robot Map */}
            <div>
              <div className="mb-4">
                <h2 className="text-xl font-bold text-white mb-2">Live Robot Map</h2>
                <p className="text-sm text-gray-400">
                  Real-time positions and strategies. Click robots for details.
                </p>
              </div>

              <RobotMap
                robots={agents.map(a => ({
                  id: a.id,
                  position: a.position,
                  strategy: a.strategy,
                  utility: a.utility
                }))}
                showIds={agents.length <= 50}
                onRobotClick={(robot) => {
                  addLog('INFO', 'user', `Selected robot: ${robot.id} (strategy: ${robot.strategy})`)
                }}
              />
            </div>

            {/* Control Buttons */}
            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-white">Coordination Control</h3>
                <div className="flex items-center gap-2">
                  {isRunning ? (
                    <div className="px-3 py-1 bg-neon-green/10 border border-neon-green/30 rounded-lg">
                      <span className="text-xs text-neon-green font-semibold">● Running</span>
                    </div>
                  ) : (
                    <div className="px-3 py-1 bg-gray-800 border border-gray-700 rounded-lg">
                      <span className="text-xs text-gray-400">Stopped</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => {
                    setIsRunning(!isRunning)
                    addLog('INFO', 'coordinator', isRunning ? 'Coordination stopped' : 'Coordination started')
                  }}
                  disabled={!connected}
                  className="px-4 py-3 bg-neon-blue hover:bg-neon-blue/80 disabled:bg-gray-700 disabled:text-gray-500 text-black font-bold rounded-lg transition flex items-center justify-center gap-2"
                >
                  {isRunning ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isRunning ? 'Stop' : 'Start'}
                </button>

                <button
                  onClick={() => {
                    addLog('INFO', 'coordinator', 'Coordination reset')
                  }}
                  disabled={!connected}
                  className="px-4 py-3 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold rounded-lg transition flex items-center justify-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </button>

                <button
                  onClick={handleExportROSBag}
                  disabled={!connected}
                  className="col-span-2 px-4 py-3 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold rounded-lg transition flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Export ROS Bag
                </button>
              </div>
            </div>
          </div>

          {/* Right Column - Terminal */}
          <div className="lg:col-span-1">
            <ROSTerminal logs={logs} autoScroll={true} />
          </div>
        </div>

        {/* Documentation Link */}
        <div className="mt-8 p-6 bg-gray-900/50 border border-gray-800 rounded-xl">
          <h3 className="text-lg font-bold text-white mb-2">Need Help?</h3>
          <p className="text-sm text-gray-400 mb-4">
            Check the complete ROS integration guide for setup instructions, message definitions, and troubleshooting.
          </p>
          <a
            href="/docs/integrations/ros"
            className="inline-flex items-center gap-2 px-4 py-2 bg-neon-blue/10 border border-neon-blue/50 text-neon-blue rounded-lg hover:bg-neon-blue/20 transition text-sm font-semibold"
          >
            View ROS Documentation →
          </a>
        </div>
      </div>
    </div>
  )
}
