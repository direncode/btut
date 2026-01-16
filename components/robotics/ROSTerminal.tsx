'use client'

import { useState, useRef, useEffect } from 'react'
import { Terminal, Search, Download, Trash2, Filter } from 'lucide-react'

interface LogEntry {
  timestamp: string
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG'
  node: string
  message: string
}

interface ROSTerminalProps {
  logs?: LogEntry[]
  autoScroll?: boolean
}

export default function ROSTerminal({ logs = [], autoScroll = true }: ROSTerminalProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [levelFilter, setLevelFilter] = useState<string>('ALL')
  const [nodeFilter, setNodeFilter] = useState<string>('ALL')
  const terminalRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const filteredLogs = logs.filter(log => {
    const matchesSearch = searchTerm === '' ||
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.node.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesLevel = levelFilter === 'ALL' || log.level === levelFilter
    const matchesNode = nodeFilter === 'ALL' || log.node === nodeFilter

    return matchesSearch && matchesLevel && matchesNode
  })

  const uniqueNodes = Array.from(new Set(logs.map(log => log.node)))

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return 'text-red-400'
      case 'WARN': return 'text-yellow-400'
      case 'INFO': return 'text-blue-400'
      case 'DEBUG': return 'text-gray-500'
      default: return 'text-gray-400'
    }
  }

  const handleExportLogs = () => {
    const logText = filteredLogs
      .map(log => `[${log.timestamp}] [${log.level}] [${log.node}] ${log.message}`)
      .join('\n')

    const blob = new Blob([logText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ros-logs-${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleClearLogs = () => {
    // This would typically call a prop function to clear logs
    console.log('Clear logs')
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gray-900/80 border-b border-gray-800 p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-neon-green" />
            <h3 className="text-lg font-bold text-white">ROS Terminal</h3>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleExportLogs}
              className="p-2 hover:bg-gray-800 rounded-lg transition"
              title="Export logs"
            >
              <Download className="w-4 h-4 text-gray-400" />
            </button>
            <button
              onClick={handleClearLogs}
              className="p-2 hover:bg-gray-800 rounded-lg transition"
              title="Clear logs"
            >
              <Trash2 className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search logs..."
              className="w-full pl-10 pr-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:border-neon-blue focus:outline-none"
            />
          </div>

          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:border-neon-blue focus:outline-none"
          >
            <option value="ALL">All Levels</option>
            <option value="ERROR">ERROR</option>
            <option value="WARN">WARN</option>
            <option value="INFO">INFO</option>
            <option value="DEBUG">DEBUG</option>
          </select>

          <select
            value={nodeFilter}
            onChange={(e) => setNodeFilter(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:border-neon-blue focus:outline-none"
          >
            <option value="ALL">All Nodes</option>
            {uniqueNodes.map(node => (
              <option key={node} value={node}>{node}</option>
            ))}
          </select>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
          <span>{filteredLogs.length} of {logs.length} logs</span>
          <span className="text-red-400">{logs.filter(l => l.level === 'ERROR').length} errors</span>
          <span className="text-yellow-400">{logs.filter(l => l.level === 'WARN').length} warnings</span>
        </div>
      </div>

      {/* Terminal Output */}
      <div
        ref={terminalRef}
        className="h-96 overflow-y-auto font-mono text-xs p-4 bg-black/50"
      >
        {filteredLogs.length === 0 && (
          <div className="text-gray-600 text-center py-8">
            {logs.length === 0 ? 'No logs yet. Waiting for ROS messages...' : 'No logs match your filters.'}
          </div>
        )}

        {filteredLogs.map((log, i) => (
          <div key={i} className="mb-1 hover:bg-gray-900/50 px-2 py-1 rounded">
            <span className="text-gray-600">[{log.timestamp}]</span>
            <span className={`ml-2 font-semibold ${getLevelColor(log.level)}`}>[{log.level}]</span>
            <span className="ml-2 text-gray-500">[{log.node}]</span>
            <span className="ml-2 text-gray-300">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
