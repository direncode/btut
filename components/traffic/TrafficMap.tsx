'use client'

import { useRef, useEffect, useState } from 'react'
import { ZoomIn, ZoomOut, Maximize2, Eye, EyeOff } from 'lucide-react'

interface Vehicle {
  id: string
  position: { x: number; y: number }
  speed: number
  angle: number
  roadId: string
  laneId: string
  strategy: 'A' | 'B' | null
}

interface TrafficMapProps {
  vehicles: Vehicle[]
  networkBounds?: { xMin: number; xMax: number; yMin: number; yMax: number }
  showHeatmap?: boolean
  onVehicleClick?: (vehicle: Vehicle) => void
}

export default function TrafficMap({
  vehicles,
  networkBounds = { xMin: -100, xMax: 100, yMin: -100, yMax: 100 },
  showHeatmap = false,
  onVehicleClick
}: TrafficMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [showStrategies, setShowStrategies] = useState(true)
  const [showSpeeds, setShowSpeeds] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw road network (simplified - would use actual network data)
    drawRoadNetwork(ctx, canvas.width, canvas.height, networkBounds, zoom, pan)

    // Draw heatmap if enabled
    if (showHeatmap) {
      drawCongestionHeatmap(ctx, vehicles, canvas.width, canvas.height, networkBounds, zoom, pan)
    }

    // Draw vehicles
    vehicles.forEach(vehicle => {
      drawVehicle(ctx, vehicle, canvas.width, canvas.height, networkBounds, zoom, pan, showStrategies, showSpeeds)
    })

  }, [vehicles, networkBounds, zoom, pan, showHeatmap, showStrategies, showSpeeds])

  const drawRoadNetwork = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    bounds: any,
    zoom: number,
    pan: { x: number; y: number }
  ) => {
    // Draw simplified grid-based road network
    const roadSpacing = 20
    const { xMin, xMax, yMin, yMax } = bounds

    ctx.strokeStyle = '#222222'
    ctx.lineWidth = 4

    // Horizontal roads
    for (let y = yMin; y <= yMax; y += roadSpacing) {
      const start = worldToScreen(xMin, y, width, height, bounds, zoom, pan)
      const end = worldToScreen(xMax, y, width, height, bounds, zoom, pan)

      ctx.beginPath()
      ctx.moveTo(start.x, start.y)
      ctx.lineTo(end.x, end.y)
      ctx.stroke()
    }

    // Vertical roads
    for (let x = xMin; x <= xMax; x += roadSpacing) {
      const start = worldToScreen(x, yMin, width, height, bounds, zoom, pan)
      const end = worldToScreen(x, yMax, width, height, bounds, zoom, pan)

      ctx.beginPath()
      ctx.moveTo(start.x, start.y)
      ctx.lineTo(end.x, end.y)
      ctx.stroke()
    }

    // Draw lane markings
    ctx.strokeStyle = '#333333'
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])

    // Add center lines
    for (let y = yMin; y <= yMax; y += roadSpacing) {
      const start = worldToScreen(xMin, y, width, height, bounds, zoom, pan)
      const end = worldToScreen(xMax, y, width, height, bounds, zoom, pan)

      ctx.beginPath()
      ctx.moveTo(start.x, start.y)
      ctx.lineTo(end.x, end.y)
      ctx.stroke()
    }

    ctx.setLineDash([])
  }

  const drawCongestionHeatmap = (
    ctx: CanvasRenderingContext2D,
    vehicles: Vehicle[],
    width: number,
    height: number,
    bounds: any,
    zoom: number,
    pan: { x: number; y: number }
  ) => {
    // Create heatmap based on vehicle density
    const gridSize = 10
    const heatGrid: number[][] = []

    // Initialize grid
    for (let i = 0; i < Math.ceil(width / gridSize); i++) {
      heatGrid[i] = []
      for (let j = 0; j < Math.ceil(height / gridSize); j++) {
        heatGrid[i][j] = 0
      }
    }

    // Count vehicles in each grid cell
    vehicles.forEach(vehicle => {
      const screen = worldToScreen(vehicle.position.x, vehicle.position.y, width, height, bounds, zoom, pan)
      const gridX = Math.floor(screen.x / gridSize)
      const gridY = Math.floor(screen.y / gridSize)

      if (gridX >= 0 && gridX < heatGrid.length && gridY >= 0 && gridY < heatGrid[0].length) {
        heatGrid[gridX][gridY]++
      }
    })

    // Draw heatmap
    const maxDensity = Math.max(...heatGrid.flat())

    for (let i = 0; i < heatGrid.length; i++) {
      for (let j = 0; j < heatGrid[i].length; j++) {
        if (heatGrid[i][j] > 0) {
          const intensity = heatGrid[i][j] / maxDensity
          const alpha = intensity * 0.6

          // Red for high density, yellow for medium, green for low
          if (intensity > 0.7) {
            ctx.fillStyle = `rgba(255, 0, 0, ${alpha})`
          } else if (intensity > 0.4) {
            ctx.fillStyle = `rgba(255, 255, 0, ${alpha})`
          } else {
            ctx.fillStyle = `rgba(0, 255, 0, ${alpha})`
          }

          ctx.fillRect(i * gridSize, j * gridSize, gridSize, gridSize)
        }
      }
    }
  }

  const drawVehicle = (
    ctx: CanvasRenderingContext2D,
    vehicle: Vehicle,
    width: number,
    height: number,
    bounds: any,
    zoom: number,
    pan: { x: number; y: number },
    showStrategy: boolean,
    showSpeed: boolean
  ) => {
    const screen = worldToScreen(vehicle.position.x, vehicle.position.y, width, height, bounds, zoom, pan)

    // Vehicle color based on strategy or speed
    let color = '#888888'
    if (showStrategy && vehicle.strategy) {
      color = vehicle.strategy === 'A' ? '#00ff88' : '#ff0055'
    } else if (showSpeed) {
      // Speed-based coloring (red = slow, green = fast)
      const speedRatio = Math.min(vehicle.speed / 30, 1)
      const r = Math.floor(255 * (1 - speedRatio))
      const g = Math.floor(255 * speedRatio)
      color = `rgb(${r}, ${g}, 0)`
    }

    const length = 8 * zoom
    const width_v = 4 * zoom

    // Save context
    ctx.save()

    // Translate and rotate
    ctx.translate(screen.x, screen.y)
    ctx.rotate((vehicle.angle * Math.PI) / 180)

    // Draw vehicle rectangle
    ctx.fillStyle = color
    ctx.fillRect(-length / 2, -width_v / 2, length, width_v)

    // Draw vehicle outline
    ctx.strokeStyle = color
    ctx.lineWidth = 1
    ctx.strokeRect(-length / 2, -width_v / 2, length, width_v)

    // Draw direction indicator (front of vehicle)
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(length / 3, -width_v / 4, length / 6, width_v / 2)

    // Restore context
    ctx.restore()

    // Draw speed label if enabled
    if (showSpeed) {
      ctx.fillStyle = '#ffffff'
      ctx.font = '9px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`${vehicle.speed.toFixed(0)}`, screen.x, screen.y - 8)
    }
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
    if (!onVehicleClick) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top

    // Find clicked vehicle
    const clickedVehicle = vehicles.find(vehicle => {
      const screen = worldToScreen(
        vehicle.position.x,
        vehicle.position.y,
        canvas.width,
        canvas.height,
        networkBounds,
        zoom,
        pan
      )
      const distance = Math.sqrt(
        Math.pow(screen.x - clickX, 2) + Math.pow(screen.y - clickY, 2)
      )
      return distance < 12
    })

    if (clickedVehicle) {
      onVehicleClick(clickedVehicle)
    }
  }

  return (
    <div className="relative bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl overflow-hidden">
      {/* Map Canvas */}
      <canvas
        ref={canvasRef}
        width={1000}
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

        <div className="h-px bg-gray-700 my-1"></div>

        <button
          onClick={() => setShowStrategies(!showStrategies)}
          className={`p-2 bg-gray-900/80 hover:bg-gray-800 border rounded-lg transition ${
            showStrategies ? 'border-neon-green' : 'border-gray-700'
          }`}
          title="Toggle Strategy Colors"
        >
          {showStrategies ? <Eye className="w-4 h-4 text-neon-green" /> : <EyeOff className="w-4 h-4 text-gray-400" />}
        </button>
        <button
          onClick={() => setShowSpeeds(!showSpeeds)}
          className={`p-2 bg-gray-900/80 hover:bg-gray-800 border rounded-lg transition ${
            showSpeeds ? 'border-neon-blue' : 'border-gray-700'
          }`}
          title="Toggle Speed Display"
        >
          <span className="text-xs font-mono text-gray-400">km/h</span>
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-900/80 backdrop-blur border border-gray-700 rounded-lg p-3">
        <div className="space-y-2 text-xs">
          {showStrategies && (
            <>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-neon-green"></div>
                <span className="text-gray-300">Strategy A (Cooperate)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-neon-pink"></div>
                <span className="text-gray-300">Strategy B (Defect)</span>
              </div>
            </>
          )}
          {showSpeeds && (
            <div className="flex items-center gap-2">
              <div className="flex gap-1">
                <div className="w-2 h-3 bg-red-500"></div>
                <div className="w-2 h-3 bg-yellow-500"></div>
                <div className="w-2 h-3 bg-green-500"></div>
              </div>
              <span className="text-gray-300">Speed (slow → fast)</span>
            </div>
          )}
        </div>
      </div>

      {/* Vehicle Count */}
      <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur border border-gray-700 rounded-lg px-3 py-2">
        <span className="text-sm text-gray-300">
          <span className="font-bold text-white">{vehicles.length}</span> vehicles
        </span>
      </div>
    </div>
  )
}
