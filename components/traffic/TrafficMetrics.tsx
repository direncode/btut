'use client'

import { TrendingUp, TrendingDown, Gauge, Users, Clock, Zap, Droplet } from 'lucide-react'

interface TrafficMetricsProps {
  averageSpeed: number
  totalVehicles: number
  waitingVehicles: number
  meanWaitingTime: number
  throughput: number
  totalFuel?: number
  totalCO2?: number
  congestionLevel?: number
}

export default function TrafficMetrics({
  averageSpeed,
  totalVehicles,
  waitingVehicles,
  meanWaitingTime,
  throughput,
  totalFuel = 0,
  totalCO2 = 0,
  congestionLevel = 0
}: TrafficMetricsProps) {
  const movingVehicles = totalVehicles - waitingVehicles
  const waitingPercentage = totalVehicles > 0 ? (waitingVehicles / totalVehicles) * 100 : 0

  const getCongestionColor = (level: number) => {
    if (level > 0.7) return 'text-red-400'
    if (level > 0.4) return 'text-yellow-400'
    return 'text-neon-green'
  }

  const getCongestionLabel = (level: number) => {
    if (level > 0.7) return 'High'
    if (level > 0.4) return 'Medium'
    return 'Low'
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <Gauge className="w-5 h-5 text-neon-blue" />
        Traffic Metrics
      </h3>

      <div className="space-y-4">
        {/* Average Speed */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Average Speed</span>
            </div>
            <span className="text-2xl font-bold text-neon-blue">{averageSpeed.toFixed(1)}</span>
          </div>
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>km/h</span>
            <span className="text-neon-blue">↑ {((averageSpeed / 50) * 100).toFixed(0)}% of limit</span>
          </div>
        </div>

        {/* Total Vehicles */}
        <div className="pt-4 border-t border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Total Vehicles</span>
            </div>
            <span className="text-2xl font-bold text-white">{totalVehicles}</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-neon-green">{movingVehicles} moving</span>
            <span className="text-yellow-500">{waitingVehicles} waiting</span>
          </div>
        </div>

        {/* Waiting Time */}
        <div className="pt-4 border-t border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Mean Waiting Time</span>
            </div>
            <span className="text-2xl font-bold text-yellow-500">{meanWaitingTime.toFixed(1)}s</span>
          </div>
          <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-yellow-500 transition-all duration-500"
              style={{ width: `${Math.min((waitingPercentage / 50) * 100, 100)}%` }}
            />
          </div>
        </div>

        {/* Throughput */}
        <div className="pt-4 border-t border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Throughput</span>
            </div>
            <span className="text-2xl font-bold text-neon-purple">{throughput}</span>
          </div>
          <div className="text-xs text-gray-500">vehicles/hour</div>
        </div>

        {/* Congestion Level */}
        {congestionLevel > 0 && (
          <div className="pt-4 border-t border-gray-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Congestion</span>
              <span className={`text-xl font-bold ${getCongestionColor(congestionLevel)}`}>
                {getCongestionLabel(congestionLevel)}
              </span>
            </div>
            <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  congestionLevel > 0.7 ? 'bg-red-500' : congestionLevel > 0.4 ? 'bg-yellow-500' : 'bg-neon-green'
                }`}
                style={{ width: `${congestionLevel * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Environmental Metrics */}
        {(totalFuel > 0 || totalCO2 > 0) && (
          <div className="pt-4 border-t border-gray-800">
            <div className="text-xs text-gray-400 mb-3">Environmental Impact</div>
            <div className="grid grid-cols-2 gap-3">
              {totalFuel > 0 && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Droplet className="w-3 h-3 text-blue-400" />
                    <span className="text-xs text-gray-400">Fuel Used</span>
                  </div>
                  <div className="text-lg font-bold text-blue-400">{totalFuel.toFixed(1)}L</div>
                </div>
              )}
              {totalCO2 > 0 && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingDown className="w-3 h-3 text-gray-400" />
                    <span className="text-xs text-gray-400">CO₂</span>
                  </div>
                  <div className="text-lg font-bold text-gray-400">{totalCO2.toFixed(1)}kg</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Queue Statistics */}
        <div className="pt-4 border-t border-gray-800">
          <div className="text-xs text-gray-400 mb-3">Queue Statistics</div>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">Waiting Vehicles</span>
              <span className="text-yellow-500 font-semibold">{waitingVehicles}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">Waiting %</span>
              <span className="text-yellow-500 font-semibold">{waitingPercentage.toFixed(1)}%</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">Avg Queue Time</span>
              <span className="text-yellow-500 font-semibold">{meanWaitingTime.toFixed(1)}s</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
