'use client'

import { useRef, useEffect, useState } from 'react'
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'

interface Robot {
  id: string
  position: { x: number; y: number; z: number }
  strategy: 'A' | 'B'
  utility: number
}

interface RobotMapProps {
  robots: Robot[]
  bounds?: { xMin: number; xMax: number; yMin: number; yMax: number }
  showTrajectories?: boolean
  showIds?: boolean
  onRobotClick?: (robot: Robot) => void
}

export default function RobotMap({
  robots,
  bounds = { xMin: -10, xMax: 10, yMin: -10, yMax: 10 },
  showTrajectories = false,
  showIds = false,
  onRobotClick
}: RobotMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid
    drawGrid(ctx, canvas.width, canvas.height, bounds, zoom, pan)

    // Draw robots
    robots.forEach(robot => {
      drawRobot(ctx, robot, canvas.width, canvas.height, bounds, zoom, pan, showIds)
    })

  }, [robots, bounds, zoom, pan, showIds])

  const drawGrid = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    bounds: any,
    zoom: number,
    pan: { x: number; y: number }
  ) => {
    const gridSize = 1
    const { xMin, xMax, yMin, yMax } = bounds

    ctx.strokeStyle = '#1a1a1a'
    ctx.lineWidth = 1

    // Vertical lines
    for (let x = xMin; x <= xMax; x += gridSize) {
      const screenX = worldToScreen(x, 0, width, height, bounds, zoom, pan).x
      ctx.beginPath()
      ctx.moveTo(screenX, 0)
      ctx.lineTo(screenX, height)
      ctx.stroke()
    }

    // Horizontal lines
    for (let y = yMin; y <= yMax; y += gridSize) {
      const screenY = worldToScreen(0, y, width, height, bounds, zoom, pan).y
      ctx.beginPath()
      ctx.moveTo(0, screenY)
      ctx.lineTo(width, screenY)
      ctx.stroke()
    }

    // Draw axes
    ctx.strokeStyle = '#333333'
    ctx.lineWidth = 2

    // X-axis
    const xAxis = worldToScreen(0, 0, width, height, bounds, zoom, pan).y
    ctx.beginPath()
    ctx.moveTo(0, xAxis)
    ctx.lineTo(width, xAxis)
    ctx.stroke()

    // Y-axis
    const yAxis = worldToScreen(0, 0, width, height, bounds, zoom, pan).x
    ctx.beginPath()
    ctx.moveTo(yAxis, 0)
    ctx.lineTo(yAxis, height)
    ctx.stroke()
  }

  const drawRobot = (
    ctx: CanvasRenderingContext2D,
    robot: Robot,
    width: number,
    height: number,
    bounds: any,
    zoom: number,
    pan: { x: number; y: number },
    showId: boolean
  ) => {
    const screen = worldToScreen(robot.position.x, robot.position.y, width, height, bounds, zoom, pan)

    // Robot color based on strategy
    const color = robot.strategy === 'A' ? '#00ff88' : '#ff0055'
    const radius = 6 * zoom

    // Draw robot circle
    ctx.fillStyle = color
    ctx.strokeStyle = color
    ctx.lineWidth = 2

    ctx.beginPath()
    ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2)
    ctx.fill()

    // Draw outer glow
    ctx.strokeStyle = color + '40'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.arc(screen.x, screen.y, radius + 3, 0, Math.PI * 2)
    ctx.stroke()

    // Draw ID if enabled
    if (showId) {
      ctx.fillStyle = '#ffffff'
      ctx.font = '10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(robot.id.substring(0, 4), screen.x, screen.y - radius - 5)
    }

    // Draw utility indicator (small bar)
    const utilityBarWidth = 20
    const utilityBarHeight = 3
    const utilityFill = robot.utility * utilityBarWidth

    ctx.fillStyle = '#333333'
    ctx.fillRect(
      screen.x - utilityBarWidth / 2,
      screen.y + radius + 5,
      utilityBarWidth,
      utilityBarHeight
    )

    ctx.fillStyle = color
    ctx.fillRect(
      screen.x - utilityBarWidth / 2,
      screen.y + radius + 5,
      utilityFill,
      utilityBarHeight
    )
  }

  const worldToScreen = (
    worldX: number,
    worldY: number,
    width: number,
    height: number,
    bounds: any,
    zoom: number,
    pan: { x: number; y: number }
  ) => {
    const { xMin, xMax, yMin, yMax } = bounds
    const worldWidth = xMax - xMin
    const worldHeight = yMax - yMin

    const screenX = ((worldX - xMin) / worldWidth) * width * zoom + pan.x
    const screenY = height - ((worldY - yMin) / worldHeight) * height * zoom + pan.y

    return { x: screenX, y: screenY }
  }

  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.2, 5))
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.2, 0.5))
  const handleResetView = () => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onRobotClick) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top

    // Find clicked robot
    const clickedRobot = robots.find(robot => {
      const screen = worldToScreen(
        robot.position.x,
        robot.position.y,
        canvas.width,
        canvas.height,
        bounds,
        zoom,
        pan
      )
      const distance = Math.sqrt(
        Math.pow(screen.x - clickX, 2) + Math.pow(screen.y - clickY, 2)
      )
      return distance < 10
    })

    if (clickedRobot) {
      onRobotClick(clickedRobot)
    }
  }

  return (
    <div className="relative bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl overflow-hidden">
      {/* Map Canvas */}
      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="w-full h-full cursor-crosshair"
        onClick={handleCanvasClick}
      />

      {/* Controls Overlay */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button
          onClick={handleZoomIn}
          className="p-2 bg-gray-900/80 hover:bg-gray-800 border border-gray-700 rounded-lg transition"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4 text-gray-400" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 bg-gray-900/80 hover:bg-gray-800 border border-gray-700 rounded-lg transition"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4 text-gray-400" />
        </button>
        <button
          onClick={handleResetView}
          className="p-2 bg-gray-900/80 hover:bg-gray-800 border border-gray-700 rounded-lg transition"
          title="Reset View"
        >
          <Maximize2 className="w-4 h-4 text-gray-400" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-900/80 backdrop-blur border border-gray-700 rounded-lg p-3">
        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-neon-green"></div>
            <span className="text-gray-300">Strategy A (Cooperate)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-neon-pink"></div>
            <span className="text-gray-300">Strategy B (Defect)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
            <span className="text-gray-300">Transitioning</span>
          </div>
        </div>
      </div>

      {/* Robot Count */}
      <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur border border-gray-700 rounded-lg px-3 py-2">
        <span className="text-sm text-gray-300">
          <span className="font-bold text-white">{robots.length}</span> robots
        </span>
      </div>
    </div>
  )
}

