'use client'

import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface ComparisonMetrics {
  averageSpeed: number
  throughput: number
  waitingTime: number
  fuelConsumption: number
  co2Emissions: number
}

interface ComparisonViewProps {
  baselineMetrics: ComparisonMetrics | null
  btutMetrics: ComparisonMetrics | null
}

export default function ComparisonView({ baselineMetrics, btutMetrics }: ComparisonViewProps) {
  if (!baselineMetrics || !btutMetrics) {
    return (
      <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8">
        <div className="text-center text-gray-500">
          <p className="mb-2">No comparison data available</p>
          <p className="text-sm">Run both baseline and BTUT simulations to see the comparison</p>
        </div>
      </div>
    )
  }

  const calculateImprovement = (baseline: number, btut: number, higherIsBetter: boolean = true) => {
    const diff = btut - baseline
    const percentage = (diff / baseline) * 100

    if (higherIsBetter) {
      return { value: percentage, improved: percentage > 0 }
    } else {
      return { value: -percentage, improved: percentage < 0 }
    }
  }

  const speedImprovement = calculateImprovement(baselineMetrics.averageSpeed, btutMetrics.averageSpeed, true)
  const throughputImprovement = calculateImprovement(baselineMetrics.throughput, btutMetrics.throughput, true)
  const waitingImprovement = calculateImprovement(baselineMetrics.waitingTime, btutMetrics.waitingTime, false)
  const fuelImprovement = calculateImprovement(baselineMetrics.fuelConsumption, btutMetrics.fuelConsumption, false)
  const co2Improvement = calculateImprovement(baselineMetrics.co2Emissions, btutMetrics.co2Emissions, false)

  const ImprovementIndicator = ({ improvement }: { improvement: { value: number; improved: boolean } }) => {
    if (Math.abs(improvement.value) < 1) {
      return (
        <div className="flex items-center gap-1 text-gray-500">
          <Minus className="w-4 h-4" />
          <span className="text-sm font-semibold">~0%</span>
        </div>
      )
    }

    return (
      <div className={`flex items-center gap-1 ${improvement.improved ? 'text-neon-green' : 'text-red-400'}`}>
        {improvement.improved ? (
          <TrendingUp className="w-4 h-4" />
        ) : (
          <TrendingDown className="w-4 h-4" />
        )}
        <span className="text-sm font-semibold">
          {improvement.improved ? '+' : ''}{improvement.value.toFixed(1)}%
        </span>
      </div>
    )
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gray-900/80 border-b border-gray-800 p-6">
        <h3 className="text-2xl font-bold text-white mb-2">Performance Comparison</h3>
        <p className="text-sm text-gray-400">Baseline traffic vs BTUT-coordinated traffic</p>
      </div>

      {/* Comparison Grid */}
      <div className="p-6">
        <div className="space-y-6">
          {/* Average Speed */}
          <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-white">Average Speed</h4>
              <ImprovementIndicator improvement={speedImprovement} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-500 mb-1">Baseline</div>
                <div className="text-2xl font-bold text-gray-400">{baselineMetrics.averageSpeed.toFixed(1)}</div>
                <div className="text-xs text-gray-600">km/h</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">With BTUT</div>
                <div className="text-2xl font-bold text-neon-blue">{btutMetrics.averageSpeed.toFixed(1)}</div>
                <div className="text-xs text-gray-600">km/h</div>
              </div>
            </div>
          </div>

          {/* Throughput */}
          <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-white">Throughput</h4>
              <ImprovementIndicator improvement={throughputImprovement} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-500 mb-1">Baseline</div>
                <div className="text-2xl font-bold text-gray-400">{baselineMetrics.throughput}</div>
                <div className="text-xs text-gray-600">vehicles/hour</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">With BTUT</div>
                <div className="text-2xl font-bold text-neon-purple">{btutMetrics.throughput}</div>
                <div className="text-xs text-gray-600">vehicles/hour</div>
              </div>
            </div>
          </div>

          {/* Waiting Time */}
          <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-white">Mean Waiting Time</h4>
              <ImprovementIndicator improvement={waitingImprovement} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-500 mb-1">Baseline</div>
                <div className="text-2xl font-bold text-gray-400">{baselineMetrics.waitingTime.toFixed(1)}</div>
                <div className="text-xs text-gray-600">seconds</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">With BTUT</div>
                <div className="text-2xl font-bold text-yellow-500">{btutMetrics.waitingTime.toFixed(1)}</div>
                <div className="text-xs text-gray-600">seconds</div>
              </div>
            </div>
          </div>

          {/* Environmental Impact */}
          <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-5">
            <h4 className="text-lg font-semibold text-white mb-4">Environmental Impact</h4>

            {/* Fuel */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Fuel Consumption</span>
                <ImprovementIndicator improvement={fuelImprovement} />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-lg font-bold text-gray-400">{baselineMetrics.fuelConsumption.toFixed(2)}L</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-blue-400">{btutMetrics.fuelConsumption.toFixed(2)}L</div>
                </div>
              </div>
            </div>

            {/* CO2 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">CO₂ Emissions</span>
                <ImprovementIndicator improvement={co2Improvement} />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-lg font-bold text-gray-400">{baselineMetrics.co2Emissions.toFixed(2)}kg</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-gray-400">{btutMetrics.co2Emissions.toFixed(2)}kg</div>
                </div>
              </div>
            </div>
          </div>

          {/* Overall Summary */}
          <div className="bg-gradient-to-r from-neon-blue/10 to-neon-green/10 border border-neon-blue/30 rounded-xl p-5">
            <h4 className="text-lg font-semibold text-white mb-3">Overall Impact</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Speed Increase</span>
                <span className={speedImprovement.improved ? 'text-neon-green font-semibold' : 'text-red-400'}>
                  {speedImprovement.improved ? '+' : ''}{speedImprovement.value.toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Queue Reduction</span>
                <span className={waitingImprovement.improved ? 'text-neon-green font-semibold' : 'text-red-400'}>
                  {waitingImprovement.improved ? '-' : '+'}{Math.abs(waitingImprovement.value).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Throughput Increase</span>
                <span className={throughputImprovement.improved ? 'text-neon-green font-semibold' : 'text-red-400'}>
                  {throughputImprovement.improved ? '+' : ''}{throughputImprovement.value.toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Fuel Savings</span>
                <span className={fuelImprovement.improved ? 'text-neon-green font-semibold' : 'text-red-400'}>
                  {fuelImprovement.improved ? '-' : '+'}{Math.abs(fuelImprovement.value).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
