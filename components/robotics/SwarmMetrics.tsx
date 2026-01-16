'use client'

import { TrendingUp, Users, Target, Zap, Activity } from 'lucide-react'

interface SwarmMetricsProps {
  totalRobots: number
  cooperationPercentage: number
  averageUtility: number
  convergenceProgress: number
  collisionsAvoided?: number
}

export default function SwarmMetrics({
  totalRobots,
  cooperationPercentage,
  averageUtility,
  convergenceProgress,
  collisionsAvoided = 0
}: SwarmMetricsProps) {
  const cooperatingRobots = Math.round((cooperationPercentage / 100) * totalRobots)
  const defectingRobots = totalRobots - cooperatingRobots

  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-neon-blue" />
        Swarm Metrics
      </h3>

      <div className="space-y-4">
        {/* Total Robots */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Total Robots</span>
            </div>
            <span className="text-2xl font-bold text-white">{totalRobots}</span>
          </div>
        </div>

        {/* Cooperation Percentage */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Cooperation</span>
            <span className="text-xl font-bold text-neon-green">{cooperationPercentage.toFixed(1)}%</span>
          </div>
          <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-neon-green to-neon-blue transition-all duration-500"
              style={{ width: `${cooperationPercentage}%` }}
            />
          </div>
          <div className="flex items-center justify-between mt-1 text-xs text-gray-500">
            <span>{cooperatingRobots} cooperating</span>
            <span>{defectingRobots} defecting</span>
          </div>
        </div>

        {/* Average Utility */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Average Utility</span>
            </div>
            <span className="text-xl font-bold text-neon-purple">{averageUtility.toFixed(3)}</span>
          </div>
          <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-neon-purple transition-all duration-500"
              style={{ width: `${averageUtility * 100}%` }}
            />
          </div>
        </div>

        {/* Convergence Progress */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Convergence</span>
            </div>
            <span className="text-xl font-bold text-neon-blue">{convergenceProgress}/20</span>
          </div>
          <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-neon-blue transition-all duration-500"
              style={{ width: `${(convergenceProgress / 20) * 100}%` }}
            />
          </div>
          {convergenceProgress >= 20 && (
            <div className="mt-2 px-3 py-1 bg-neon-green/10 border border-neon-green/30 rounded-lg">
              <span className="text-xs text-neon-green font-semibold">✓ Converged</span>
            </div>
          )}
        </div>

        {/* Collisions Avoided */}
        {collisionsAvoided > 0 && (
          <div className="pt-4 border-t border-gray-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-500" />
                <span className="text-sm text-gray-400">Collisions Avoided</span>
              </div>
              <span className="text-xl font-bold text-yellow-500">{collisionsAvoided}</span>
            </div>
          </div>
        )}

        {/* Strategy Distribution */}
        <div className="pt-4 border-t border-gray-800">
          <div className="text-xs text-gray-400 mb-3">Strategy Distribution</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-neon-green/10 border border-neon-green/30 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Strategy A</div>
              <div className="text-lg font-bold text-neon-green">{cooperatingRobots}</div>
              <div className="text-xs text-gray-500">{cooperationPercentage.toFixed(1)}%</div>
            </div>
            <div className="bg-neon-pink/10 border border-neon-pink/30 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Strategy B</div>
              <div className="text-lg font-bold text-neon-pink">{defectingRobots}</div>
              <div className="text-xs text-gray-500">{(100 - cooperationPercentage).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
