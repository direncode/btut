'use client'

import { SimulationState } from '@/lib/simulation/btut-engine'
import { useEffect, useRef } from 'react'

interface NetworkViewProps {
  state: SimulationState | null
}

export default function NetworkView({ state }: NetworkViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current || !state) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = rect.height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const width = rect.width
    const height = rect.height

    // Clear canvas
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, width, height)

    // Sample agents for visualization (max 2000 for performance)
    const maxNodes = 2000
    const step = Math.ceil(state.agents.length / maxNodes)
    const sampledAgents = state.agents.filter((_, i) => i % step === 0)

    // Position nodes in a force-directed layout simulation (simplified)
    const nodes = sampledAgents.map((agent, i) => {
      const angle = (i / sampledAgents.length) * 2 * Math.PI
      const radius = 150 + agent.weight * 100
      return {
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        agent,
      }
    })

    // Draw edges (connections) - sample for performance
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)'
    ctx.lineWidth = 0.5
    const maxEdges = 1000
    for (let i = 0; i < Math.min(nodes.length, maxEdges); i += 3) {
      const node1 = nodes[i]
      const node2 = nodes[(i + Math.floor(Math.random() * 10)) % nodes.length]
      
      ctx.beginPath()
      ctx.moveTo(node1.x, node1.y)
      ctx.lineTo(node2.x, node2.y)
      ctx.stroke()
    }

    // Draw nodes
    nodes.forEach(({ x, y, agent }) => {
      const radius = 2 + agent.weight * 6
      const color = agent.strategy === 'A' ? '#00ff88' : '#ff0080'
      const alpha = 0.6 + agent.weight * 0.4

      // Glow effect for hubs
      if (agent.weight > 0.5) {
        ctx.shadowBlur = 15
        ctx.shadowColor = color
      } else {
        ctx.shadowBlur = 0
      }

      // Draw node
      ctx.fillStyle = color
      ctx.globalAlpha = alpha
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, 2 * Math.PI)
      ctx.fill()

      ctx.globalAlpha = 1
      ctx.shadowBlur = 0
    })

    // Draw legend
    const legendX = 20
    const legendY = height - 80

    ctx.fillStyle = 'rgba(17, 24, 39, 0.8)'
    ctx.fillRect(legendX - 10, legendY - 10, 200, 70)

    ctx.font = '12px Inter'
    ctx.fillStyle = '#00ff88'
    ctx.beginPath()
    ctx.arc(legendX + 8, legendY + 8, 5, 0, 2 * Math.PI)
    ctx.fill()
    ctx.fillStyle = '#9ca3af'
    ctx.fillText('Strategy A (Cooperation)', legendX + 20, legendY + 12)

    ctx.fillStyle = '#ff0080'
    ctx.beginPath()
    ctx.arc(legendX + 8, legendY + 32, 5, 0, 2 * Math.PI)
    ctx.fill()
    ctx.fillStyle = '#9ca3af'
    ctx.fillText('Strategy B (Defection)', legendX + 20, legendY + 36)

    ctx.fillStyle = '#6b7280'
    ctx.font = '10px Inter'
    ctx.fillText(`Showing ${sampledAgents.length}/${state.agents.length} agents`, legendX, legendY + 56)

  }, [state])

  return (
    <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
      <h2 className="text-xl font-bold mb-4 text-neon-blue">Network Topology</h2>
      
      {state ? (
        <div className="relative">
          <canvas
            ref={canvasRef}
            className="w-full h-[400px] rounded-lg bg-black"
          />
          <div className="absolute top-4 right-4 bg-black/80 backdrop-blur px-3 py-2 rounded-lg text-xs">
            <div className="text-neon-green font-mono">
              A: {((state.agents.filter(a => a.strategy === 'A').length / state.agents.length) * 100).toFixed(1)}%
            </div>
            <div className="text-neon-pink font-mono">
              B: {((state.agents.filter(a => a.strategy === 'B').length / state.agents.length) * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      ) : (
        <div className="h-[400px] flex items-center justify-center border-2 border-dashed border-gray-700 rounded-lg">
          <div className="text-center">
            <div className="text-6xl mb-4">🌐</div>
            <p className="text-gray-400">Network visualization will appear here</p>
            <p className="text-sm text-gray-500 mt-2">Green = Strategy A | Pink = Strategy B</p>
          </div>
        </div>
      )}
    </div>
  )
}
