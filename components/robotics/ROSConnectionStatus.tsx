'use client'

import { useState } from 'react'
import { Wifi, WifiOff, RefreshCw, Settings } from 'lucide-react'

interface ROSConnectionStatusProps {
  connected: boolean
  connecting: boolean
  error: string | null
  url: string
  onConnect: (url: string) => void
  onDisconnect: () => void
  nodes?: string[]
  topics?: string[]
}

export default function ROSConnectionStatus({
  connected,
  connecting,
  error,
  url,
  onConnect,
  onDisconnect,
  nodes = [],
  topics = []
}: ROSConnectionStatusProps) {
  const [rosUrl, setRosUrl] = useState(url)
  const [showSettings, setShowSettings] = useState(false)

  const handleConnect = () => {
    onConnect(rosUrl)
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          {connected ? (
            <Wifi className="w-5 h-5 text-neon-green" />
          ) : (
            <WifiOff className="w-5 h-5 text-gray-500" />
          )}
          ROS Connection
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
            <span className="text-sm text-neon-green font-semibold">Connected to roscore</span>
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
          <div className="mt-2 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg">
            <span className="text-sm text-red-400">{error}</span>
          </div>
        )}
      </div>

      {/* Connection Settings */}
      {(showSettings || !connected) && (
        <div className="space-y-3 mb-4">
          <div>
            <label className="text-sm text-gray-400 mb-1 block">ROS Master URI</label>
            <input
              type="text"
              value={rosUrl}
              onChange={(e) => setRosUrl(e.target.value)}
              placeholder="ws://localhost:9090"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:border-neon-blue focus:outline-none"
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
            className="flex-1 px-4 py-2 bg-neon-blue hover:bg-neon-blue/80 disabled:bg-gray-700 disabled:text-gray-500 text-black font-semibold rounded-lg transition"
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

      {/* System Info */}
      {connected && (
        <div className="mt-4 pt-4 border-t border-gray-800 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Active Nodes</span>
            <span className="text-white font-semibold">{nodes.length}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Active Topics</span>
            <span className="text-white font-semibold">{topics.length}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">URI</span>
            <span className="text-xs text-gray-500 font-mono">{rosUrl}</span>
          </div>
        </div>
      )}

      {/* Node List (collapsed) */}
      {connected && nodes.length > 0 && showSettings && (
        <div className="mt-4 pt-4 border-t border-gray-800">
          <div className="text-xs text-gray-400 mb-2 font-semibold">ROS Nodes:</div>
          <div className="max-h-32 overflow-y-auto space-y-1">
            {nodes.slice(0, 10).map((node, i) => (
              <div key={i} className="text-xs text-gray-500 font-mono px-2 py-1 bg-gray-800/50 rounded">
                {node}
              </div>
            ))}
            {nodes.length > 10 && (
              <div className="text-xs text-gray-600 px-2 py-1">
                ... and {nodes.length - 10} more
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
