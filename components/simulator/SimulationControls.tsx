'use client'

import { SimulationConfig } from '@/lib/simulation/btut-engine'

interface SimulationControlsProps {
  config: SimulationConfig
  onConfigChange: (config: SimulationConfig) => void
  disabled?: boolean
}

export default function SimulationControls({ config, onConfigChange, disabled }: SimulationControlsProps) {
  const updateConfig = (key: keyof SimulationConfig, value: number) => {
    onConfigChange({ ...config, [key]: value })
  }

  const parameters = [
    { key: 'N' as const, label: 'Agents (N)', min: 1000, max: 1000000, step: 1000 },
    { key: 'gamma' as const, label: 'SH Bonus (γ)', min: 1.0, max: 2.0, step: 0.05 },
    { key: 'tau' as const, label: 'Hub Influence (τ)', min: 0.0, max: 1.0, step: 0.05 },
    { key: 'cA_SH' as const, label: 'Cost A (SH)', min: 0.0, max: 1.0, step: 0.05 },
    { key: 'alpha' as const, label: 'Aggression (α)', min: 0.0, max: 1.0, step: 0.05 },
    { key: 'iterations' as const, label: 'Iterations', min: 5, max: 50, step: 1 },
  ]

  return (
    <div className="space-y-4">
      {parameters.map(({ key, label, min, max, step }) => (
        <div key={key}>
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium text-gray-300">{label}</label>
            <span className="text-sm font-mono text-neon-blue">
              {key === 'N' ? config[key].toLocaleString() : config[key].toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={config[key]}
            onChange={(e) => updateConfig(key, parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-neon-blue disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{min}</span>
            <span>{max}</span>
          </div>
        </div>
      ))}
    </div>
  )
}
