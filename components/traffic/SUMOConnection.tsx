'use client'

import { useState } from 'react'
import { Car, WifiOff, RefreshCw, Settings, AlertCircle } from 'lucide-react'

interface SUMOConnectionProps {
  connected: boolean
  connecting: boolean
  error: string | null
  host: string
  port: number
  onConnect: (host: string, port: number) => void
  onDisconnect: () => void
  vehicleCount?: number
  currentTime?: number
}

export default function SUMOConnection({
  connected,
  connecting,
  error,
  host,
  port,
  onConnect,
  onDisconnect,
  vehicleCount = 0,
  currentTime = 0
}: SUMOConnectionProps) {
  const [sumoHost, setSumoHost] = useState(host)
  const [sumoPort, setSumoPort] = useState(port.toString())
  const [showSettings, setShowSettings] = useState(false)

  const handleConnect = () => {
    onConnect(sumoHost, parseInt(sumoPort))
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          {connected ? (
            <Car className="w-5 h-5 text-neon-green" />
          ) : (
            <WifiOff className="w-5 h-5 text-gray-500" />
          )}
          SUMO Connection
        </h3>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 hover:bg-gray-800 rounded-lg transition"
        >
          <Settings className="w-4 h-4 text-gray-400" />
        </button>
      </div>

      {/* Connection Status */}
      <div className="mb-4">
        {connected && (
          <div className="flex items-center gap-2 px-3 py-2 bg-neon-green/10 border border-neon-green/30 rounded-lg">
            <div className="w-2 h-2 bg-neon-green rounded-full animate-pulse"></div>
            <span className="text-sm text-neon-green font-semibold">Connected to SUMO</span>
          </div>
        )}
        {connecting && (
          <div className="flex items-center gap-2 px-3 py-2 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />
            <span className="text-sm text-blue-500 font-semibold">Connecting...</span>
          </div>
        )}
        {!connected && !connecting && (
          <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg">
            <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
            <span className="text-sm text-gray-400">Disconnected</span>
          </div>
        )}
        {error && (
          <div className="mt-2 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
            <span className="text-sm text-red-400">{error}</span>
          </div>
        )}
      </div>

      {/* Connection Settings */}
      {(showSettings || !connected) && (
        <div className="space-y-3 mb-4">
          <div>
            <label className="text-sm text-gray-400 mb-1 block">SUMO Host</label>
            <input
              type="text"
              value={sumoHost}
              onChange={(e) => setSumoHost(e.target.value)}
              placeholder="localhost"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:border-neon-green focus:outline-none"
            />
          </div>
          <div>
            <label className="text-sm text-gray-400 mb-1 block">TraCI Port</label>
            <input
              type="number"
              value={sumoPort}
              onChange={(e) => setSumoPort(e.target.value)}
              placeholder="8813"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:border-neon-green focus:outline-none"
            />
          </div>
        </div>
      )}

      {/* Connection Actions */}
      <div className="flex gap-2">
        {!connected ? (
          <button
            onClick={handleConnect}
            disabled={connecting}
            className="flex-1 px-4 py-2 bg-neon-green hover:bg-neon-green/80 disabled:bg-gray-700 disabled:text-gray-500 text-black font-semibold rounded-lg transition"
          >
            {connecting ? 'Connecting...' : 'Connect'}
          </button>
        ) : (
          <button
            onClick={onDisconnect}
            className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-lg transition"
          >
            Disconnect
          </button>
        )}
      </div>

      {/* Simulation Info */}
      {connected && (
        <div className="mt-4 pt-4 border-t border-gray-800 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Active Vehicles</span>
            <span className="text-white font-semibold">{vehicleCount}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Simulation Time</span>
            <span className="text-white font-semibold">{currentTime.toFixed(1)}s</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Server</span>
            <span className="text-xs text-gray-500 font-mono">{sumoHost}:{sumoPort}</span>
          </div>
        </div>
      )}

      {/* Setup Instructions */}
      {!connected && !connecting && (
        <div className="mt-4 pt-4 border-t border-gray-800">
          <div className="text-xs text-gray-400 mb-2">Quick Start:</div>
          <code className="block px-2 py-2 bg-black/50 rounded text-xs text-gray-500 font-mono">
            sumo -c network.sumocfg --remote-port 8813
          </code>
        </div>
      )}
    </div>
  )
}
