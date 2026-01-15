'use client'

import { useState } from 'react'
import { Plus, Trash2, Play } from 'lucide-react'

interface ParameterRange {
  param: string
  min: number
  max: number
  step: number
  values: number[]
}

interface ExperimentConfigProps {
  onRun?: (config: ExperimentConfiguration) => void
}

interface ExperimentConfiguration {
  parameters: ParameterRange[]
  seeds: number[]
  totalExperiments: number
  estimatedTime: number
}

const AVAILABLE_PARAMS = [
  { key: 'gamma', label: 'Gamma (γ)', min: 1.0, max: 2.0, step: 0.1, default: 1.5 },
  { key: 'tau', label: 'Tau (τ)', min: 0.1, max: 0.6, step: 0.05, default: 0.3 },
  { key: 'N', label: 'Agents (N)', min: 1000, max: 100000, step: 1000, default: 10000 },
  { key: 'alpha', label: 'Alpha (α)', min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
  { key: 'm', label: 'Hub Neighbors (m)', min: 3, max: 20, step: 1, default: 10 }
]

export default function ExperimentConfig({ onRun }: ExperimentConfigProps) {
  const [parameters, setParameters] = useState<ParameterRange[]>([])
  const [seedStart, setSeedStart] = useState(1)
  const [seedEnd, setSeedEnd] = useState(5)

  const addParameter = (param: typeof AVAILABLE_PARAMS[0]) => {
    const values = []
    for (let v = param.min; v <= param.max; v += param.step) {
      values.push(Number(v.toFixed(3)))
    }

    setParameters(prev => [...prev, {
      param: param.key,
      min: param.min,
      max: param.max,
      step: param.step,
      values
    }])
  }

  const removeParameter = (index: number) => {
    setParameters(prev => prev.filter((_, i) => i !== index))
  }

  const updateParameterRange = (index: number, field: 'min' | 'max' | 'step', value: number) => {
    setParameters(prev => {
      const updated = [...prev]
      updated[index][field] = value

      // Recalculate values
      const values = []
      const { min, max, step } = updated[index]
      for (let v = min; v <= max; v += step) {
        values.push(Number(v.toFixed(3)))
      }
      updated[index].values = values

      return updated
    })
  }

  const calculateTotalExperiments = () => {
    const paramCombos = parameters.reduce((acc, p) => acc * p.values.length, 1)
    const seeds = seedEnd - seedStart + 1
    return paramCombos * seeds
  }

  const estimateTime = () => {
    const total = calculateTotalExperiments()
    // Assume ~1 second per experiment
    return total
  }

  const handleRun = () => {
    const seeds = []
    for (let s = seedStart; s <= seedEnd; s++) {
      seeds.push(s)
    }

    const config: ExperimentConfiguration = {
      parameters,
      seeds,
      totalExperiments: calculateTotalExperiments(),
      estimatedTime: estimateTime()
    }

    if (onRun) {
      onRun(config)
    }
  }

  const totalExperiments = calculateTotalExperiments()
  const estimatedTimeStr = estimateTime() > 60
    ? `${Math.ceil(estimateTime() / 60)} minutes`
    : `${estimateTime()} seconds`

  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
      <h3 className="text-lg font-bold text-white mb-4">Parameter Sweep Configuration</h3>

      {/* Add Parameter */}
      <div className="mb-6">
        <label className="text-sm text-gray-400 mb-2 block">Add Parameter</label>
        <div className="grid grid-cols-2 gap-2">
          {AVAILABLE_PARAMS.map(param => (
            <button
              key={param.key}
              onClick={() => addParameter(param)}
              disabled={parameters.some(p => p.param === param.key)}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 disabled:bg-gray-900 disabled:text-gray-600 text-white rounded-lg text-sm transition flex items-center justify-center gap-2"
            >
              <Plus className="w-3 h-3" />
              {param.label}
            </button>
          ))}
        </div>
      </div>

      {/* Parameter Ranges */}
      {parameters.length > 0 && (
        <div className="mb-6 space-y-4">
          {parameters.map((param, index) => {
            const paramDef = AVAILABLE_PARAMS.find(p => p.key === param.param)!
            return (
              <div key={index} className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-semibold text-white">{paramDef.label}</span>
                  <button
                    onClick={() => removeParameter(index)}
                    className="p-1 hover:bg-red-500/20 rounded transition"
                  >
                    <Trash2 className="w-4 h-4 text-red-400" />
                  </button>
                </div>

                <div className="grid grid-cols-3 gap-3 mb-2">
                  <div>
                    <label className="text-xs text-gray-500 block mb-1">Min</label>
                    <input
                      type="number"
                      value={param.min}
                      onChange={(e) => updateParameterRange(index, 'min', parseFloat(e.target.value))}
                      step={param.step}
                      className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-white text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 block mb-1">Max</label>
                    <input
                      type="number"
                      value={param.max}
                      onChange={(e) => updateParameterRange(index, 'max', parseFloat(e.target.value))}
                      step={param.step}
                      className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-white text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 block mb-1">Step</label>
                    <input
                      type="number"
                      value={param.step}
                      onChange={(e) => updateParameterRange(index, 'step', parseFloat(e.target.value))}
                      step={0.01}
                      className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-white text-sm"
                    />
                  </div>
                </div>

                <div className="text-xs text-gray-500">
                  {param.values.length} values: [{param.values.slice(0, 3).join(', ')}
                  {param.values.length > 3 && `, ... ${param.values.length - 3} more`}]
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Seeds */}
      <div className="mb-6">
        <label className="text-sm text-gray-400 mb-2 block">Random Seeds</label>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <input
              type="number"
              value={seedStart}
              onChange={(e) => setSeedStart(parseInt(e.target.value))}
              placeholder="Start"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
            />
          </div>
          <div>
            <input
              type="number"
              value={seedEnd}
              onChange={(e) => setSeedEnd(parseInt(e.target.value))}
              placeholder="End"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
            />
          </div>
        </div>
        <div className="text-xs text-gray-500 mt-1">
          {seedEnd - seedStart + 1} seeds ({seedStart} to {seedEnd})
        </div>
      </div>

      {/* Summary */}
      <div className="mb-6 p-4 bg-gradient-to-r from-neon-purple/10 to-neon-pink/10 border border-neon-purple/30 rounded-lg">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-gray-400 mb-1">Total Experiments</div>
            <div className="text-2xl font-bold text-neon-purple">{totalExperiments}</div>
          </div>
          <div>
            <div className="text-gray-400 mb-1">Estimated Time</div>
            <div className="text-2xl font-bold text-neon-pink">{estimatedTimeStr}</div>
          </div>
        </div>
      </div>

      {/* Run Button */}
      <button
        onClick={handleRun}
        disabled={parameters.length === 0}
        className="w-full px-6 py-3 bg-gradient-to-r from-neon-purple to-neon-pink hover:shadow-lg hover:shadow-neon-purple/50 disabled:from-gray-700 disabled:to-gray-700 disabled:text-gray-500 text-black font-bold rounded-lg transition flex items-center justify-center gap-2"
      >
        <Play className="w-5 h-5" />
        Run {totalExperiments} Experiments
      </button>
    </div>
  )
}
