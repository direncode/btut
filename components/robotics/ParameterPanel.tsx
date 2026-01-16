'use client'

import { useState } from 'react'
import { Settings, Save, RotateCcw, Send } from 'lucide-react'

interface ParameterPanelProps {
  initialParams?: SimulationParams
  onApply?: (params: SimulationParams) => void
  onSave?: (name: string, params: SimulationParams) => void
  savedConfigs?: SavedConfig[]
}

interface SimulationParams {
  gamma: number
  tau: number
  cA_SH: number
  cB_SH: number
  cA_PD: number
  cB_PD: number
  alpha: number
  m: number
}

interface SavedConfig {
  name: string
  params: SimulationParams
}

const DEFAULT_PARAMS: SimulationParams = {
  gamma: 1.5,
  tau: 0.3,
  cA_SH: 0.8,
  cB_SH: 0.6,
  cA_PD: 0.5,
  cB_PD: 0.3,
  alpha: 0.1,
  m: 10
}

export default function ParameterPanel({
  initialParams = DEFAULT_PARAMS,
  onApply,
  onSave,
  savedConfigs = []
}: ParameterPanelProps) {
  const [params, setParams] = useState<SimulationParams>(initialParams)
  const [configName, setConfigName] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)

  const handleParamChange = (key: keyof SimulationParams, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }

  const handleApply = () => {
    if (onApply) {
      onApply(params)
    }
  }

  const handleReset = () => {
    setParams(DEFAULT_PARAMS)
  }

  const handleSave = () => {
    if (onSave && configName.trim()) {
      onSave(configName, params)
      setConfigName('')
      setShowSaveDialog(false)
    }
  }

  const loadConfig = (config: SavedConfig) => {
    setParams(config.params)
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Settings className="w-5 h-5 text-neon-purple" />
          Parameters
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowSaveDialog(!showSaveDialog)}
            className="p-2 hover:bg-gray-800 rounded-lg transition"
            title="Save configuration"
          >
            <Save className="w-4 h-4 text-gray-400" />
          </button>
          <button
            onClick={handleReset}
            className="p-2 hover:bg-gray-800 rounded-lg transition"
            title="Reset to defaults"
          >
            <RotateCcw className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Save Dialog */}
      {showSaveDialog && (
        <div className="mb-4 p-3 bg-gray-800 border border-gray-700 rounded-lg">
          <input
            type="text"
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            placeholder="Configuration name..."
            className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white text-sm mb-2"
          />
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={!configName.trim()}
              className="flex-1 px-3 py-2 bg-neon-purple hover:bg-neon-purple/80 disabled:bg-gray-700 disabled:text-gray-500 text-black font-semibold rounded-lg text-sm transition"
            >
              Save
            </button>
            <button
              onClick={() => setShowSaveDialog(false)}
              className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Parameters */}
      <div className="space-y-4 mb-6">
        {/* Gamma */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm text-gray-400">Cooperation Bonus (γ)</label>
            <span className="text-sm font-mono text-white">{params.gamma.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="1.0"
            max="2.0"
            step="0.1"
            value={params.gamma}
            onChange={(e) => handleParamChange('gamma', parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-neon-green"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>1.0</span>
            <span>2.0</span>
          </div>
        </div>

        {/* Tau */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm text-gray-400">Hub Influence (τ)</label>
            <span className="text-sm font-mono text-white">{params.tau.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.6"
            step="0.05"
            value={params.tau}
            onChange={(e) => handleParamChange('tau', parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-neon-blue"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>0.1</span>
            <span>0.6</span>
          </div>
        </div>

        {/* Alpha */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm text-gray-400">Learning Rate (α)</label>
            <span className="text-sm font-mono text-white">{params.alpha.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={params.alpha}
            onChange={(e) => handleParamChange('alpha', parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-neon-purple"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>0.01</span>
            <span>0.5</span>
          </div>
        </div>

        {/* m */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm text-gray-400">Hub Neighbors (m)</label>
            <span className="text-sm font-mono text-white">{params.m}</span>
          </div>
          <input
            type="range"
            min="3"
            max="20"
            step="1"
            value={params.m}
            onChange={(e) => handleParamChange('m', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-neon-pink"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>3</span>
            <span>20</span>
          </div>
        </div>

        {/* Advanced Parameters (Collapsible) */}
        <details className="mt-4">
          <summary className="text-sm text-gray-400 cursor-pointer hover:text-white transition">
            Advanced Parameters
          </summary>
          <div className="space-y-4 mt-4 pl-2">
            {/* cA_SH */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-gray-500">Cost A (Same Hub)</label>
                <span className="text-xs font-mono text-white">{params.cA_SH.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={params.cA_SH}
                onChange={(e) => handleParamChange('cA_SH', parseFloat(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-gray-500"
              />
            </div>

            {/* cB_SH */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-gray-500">Cost B (Same Hub)</label>
                <span className="text-xs font-mono text-white">{params.cB_SH.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={params.cB_SH}
                onChange={(e) => handleParamChange('cB_SH', parseFloat(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-gray-500"
              />
            </div>

            {/* cA_PD */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-gray-500">Cost A (Peer Discovery)</label>
                <span className="text-xs font-mono text-white">{params.cA_PD.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={params.cA_PD}
                onChange={(e) => handleParamChange('cA_PD', parseFloat(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-gray-500"
              />
            </div>

            {/* cB_PD */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-gray-500">Cost B (Peer Discovery)</label>
                <span className="text-xs font-mono text-white">{params.cB_PD.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={params.cB_PD}
                onChange={(e) => handleParamChange('cB_PD', parseFloat(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-gray-500"
              />
            </div>
          </div>
        </details>
      </div>

      {/* Apply Button */}
      <button
        onClick={handleApply}
        className="w-full px-4 py-3 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition flex items-center justify-center gap-2"
      >
        <Send className="w-4 h-4" />
        Apply to ROS
      </button>

      {/* Saved Configurations */}
      {savedConfigs.length > 0 && (
        <div className="mt-6 pt-6 border-t border-gray-800">
          <div className="text-sm text-gray-400 mb-3">Saved Configurations</div>
          <div className="space-y-2">
            {savedConfigs.map((config, i) => (
              <button
                key={i}
                onClick={() => loadConfig(config)}
                className="w-full px-3 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg text-sm text-left transition"
              >
                {config.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
