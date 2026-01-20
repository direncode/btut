'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, RotateCcw, Settings2, ChevronDown } from 'lucide-react'
import { SimulationConfig, defaultConfig } from '@/lib/simulation-data'
import { cn, formatCompactNumber } from '@/lib/utils'

interface SimulationControlsProps {
  config: SimulationConfig
  onConfigChange: (config: SimulationConfig) => void
  onRun: () => void
  onReset: () => void
  isRunning: boolean
  progress: number
}

export function SimulationControls({
  config,
  onConfigChange,
  onRun,
  onReset,
  isRunning,
  progress,
}: SimulationControlsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const presets = [
    { name: 'Quick Test', agents: 10000, timesteps: 200 },
    { name: 'Standard', agents: 100000, timesteps: 500 },
    { name: 'Large Scale', agents: 1000000, timesteps: 1000 },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="chart-container"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-apple-gray-900">
            Simulation Controls
          </h3>
          <p className="text-sm text-apple-gray-500">
            Configure and run the market simulation
          </p>
        </div>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm text-apple-gray-500 hover:text-apple-gray-700 transition-colors"
        >
          <Settings2 className="w-4 h-4" />
          Advanced
          <ChevronDown className={cn(
            'w-4 h-4 transition-transform',
            showAdvanced && 'rotate-180'
          )} />
        </button>
      </div>

      {/* Presets */}
      <div className="flex gap-2 mb-6">
        {presets.map((preset) => (
          <button
            key={preset.name}
            onClick={() => onConfigChange({
              ...config,
              numAgents: preset.agents,
              numTimesteps: preset.timesteps,
            })}
            className={cn(
              'px-4 py-2 rounded-full text-sm font-medium transition-all',
              config.numAgents === preset.agents && config.numTimesteps === preset.timesteps
                ? 'bg-apple-blue text-white shadow-glow'
                : 'bg-apple-gray-100 text-apple-gray-700 hover:bg-apple-gray-200'
            )}
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Basic controls */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-apple-gray-700 mb-2">
            Number of Agents
          </label>
          <input
            type="range"
            min={1000}
            max={1000000}
            step={1000}
            value={config.numAgents}
            onChange={(e) => onConfigChange({
              ...config,
              numAgents: parseInt(e.target.value),
            })}
            className="w-full accent-apple-blue"
          />
          <div className="flex justify-between mt-1">
            <span className="text-xs text-apple-gray-500">1K</span>
            <span className="text-sm font-semibold text-apple-blue tabular-nums">
              {formatCompactNumber(config.numAgents)}
            </span>
            <span className="text-xs text-apple-gray-500">1M</span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-apple-gray-700 mb-2">
            Timesteps
          </label>
          <input
            type="range"
            min={100}
            max={2000}
            step={100}
            value={config.numTimesteps}
            onChange={(e) => onConfigChange({
              ...config,
              numTimesteps: parseInt(e.target.value),
            })}
            className="w-full accent-apple-blue"
          />
          <div className="flex justify-between mt-1">
            <span className="text-xs text-apple-gray-500">100</span>
            <span className="text-sm font-semibold text-apple-blue tabular-nums">
              {config.numTimesteps}
            </span>
            <span className="text-xs text-apple-gray-500">2000</span>
          </div>
        </div>
      </div>

      {/* Advanced options */}
      {showAdvanced && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mb-6 p-4 bg-apple-gray-50 rounded-apple"
        >
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-apple-gray-600 mb-1">
                Kernel Decay (λ)
              </label>
              <input
                type="number"
                step={0.01}
                min={0.01}
                max={1}
                value={config.kernelDecay}
                onChange={(e) => onConfigChange({
                  ...config,
                  kernelDecay: parseFloat(e.target.value),
                })}
                className="input-field text-sm py-2"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-apple-gray-600 mb-1">
                Momentum (α)
              </label>
              <input
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={config.momentum}
                onChange={(e) => onConfigChange({
                  ...config,
                  momentum: parseFloat(e.target.value),
                })}
                className="input-field text-sm py-2"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-apple-gray-600 mb-1">
                Temperature
              </label>
              <input
                type="number"
                step={0.1}
                min={0.1}
                max={2}
                value={config.temperature}
                onChange={(e) => onConfigChange({
                  ...config,
                  temperature: parseFloat(e.target.value),
                })}
                className="input-field text-sm py-2"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-apple-gray-600 mb-1">
                Cooperation Bias
              </label>
              <input
                type="number"
                step={0.05}
                min={-0.5}
                max={0.5}
                value={config.cooperationBias}
                onChange={(e) => onConfigChange({
                  ...config,
                  cooperationBias: parseFloat(e.target.value),
                })}
                className="input-field text-sm py-2"
              />
            </div>
          </div>
        </motion.div>
      )}

      {/* Progress bar */}
      {isRunning && (
        <div className="mb-6">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-apple-gray-600">Simulating...</span>
            <span className="text-sm font-medium text-apple-blue tabular-nums">
              {Math.round(progress)}%
            </span>
          </div>
          <div className="h-2 bg-apple-gray-100 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              className="h-full bg-gradient-to-r from-apple-blue to-apple-purple rounded-full"
            />
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          onClick={onRun}
          disabled={isRunning}
          className={cn(
            'flex-1 btn-primary',
            isRunning && 'opacity-50 cursor-not-allowed'
          )}
        >
          <Play className="w-4 h-4 mr-2" />
          {isRunning ? 'Running...' : 'Run Simulation'}
        </button>
        <button
          onClick={onReset}
          disabled={isRunning}
          className="btn-secondary"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  )
}
