'use client'

import { SimulationState, SimulationConfig } from '@/lib/simulation/btut-engine'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

interface VisualizationPanelProps {
  state: SimulationState | null
  config: SimulationConfig
}

export default function VisualizationPanel({ state, config }: VisualizationPanelProps) {
  const chartData = state?.convergenceHistory.map((value, index) => ({
    iteration: index + 1,
    fractionA: value,
  })) || []

  return (
    <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
      <h2 className="text-xl font-bold mb-4 text-neon-green">Convergence Dynamics</h2>
      
      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis 
              dataKey="iteration" 
              stroke="#6b7280"
              label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#6b7280' }}
            />
            <YAxis 
              domain={[0, 1]}
              stroke="#6b7280"
              label={{ value: 'Fraction A', angle: -90, position: 'insideLeft', fill: '#6b7280' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#111827', 
                border: '1px solid #374151',
                borderRadius: '8px'
              }}
              labelStyle={{ color: '#9ca3af' }}
            />
            <ReferenceLine y={1.0} stroke="#00ff88" strokeDasharray="3 3" label="Full Cooperation" />
            <Line 
              type="monotone" 
              dataKey="fractionA" 
              stroke="url(#colorGradient)" 
              strokeWidth={3}
              dot={false}
              animationDuration={300}
            />
            <defs>
              <linearGradient id="colorGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#00f0ff" />
                <stop offset="100%" stopColor="#00ff88" />
              </linearGradient>
            </defs>
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="h-[400px] flex items-center justify-center border-2 border-dashed border-gray-700 rounded-lg">
          <div className="text-center">
            <div className="text-6xl mb-4">📊</div>
            <p className="text-gray-400">Configure parameters and run simulation</p>
            <p className="text-sm text-gray-500 mt-2">Convergence data will appear here</p>
          </div>
        </div>
      )}

      {/* Phase Indicator */}
      {state && state.fractionA > 0 && (
        <div className="mt-4 grid grid-cols-3 gap-4">
          <div className="bg-black/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-neon-blue">{state.fractionA.toFixed(4)}</div>
            <div className="text-xs text-gray-400 mt-1">Current p*</div>
          </div>
          <div className="bg-black/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-neon-green">
              {state.iteration}/{config.iterations}
            </div>
            <div className="text-xs text-gray-400 mt-1">Progress</div>
          </div>
          <div className="bg-black/50 rounded-lg p-4 text-center">
            <div className={`text-2xl font-bold ${
              state.fractionA > 0.99 ? 'text-neon-green' : 
              state.fractionA > 0.8 ? 'text-neon-blue' : 
              'text-neon-purple'
            }`}>
              {state.fractionA > 0.99 ? 'LOCKED' : 
               state.fractionA > 0.8 ? 'CONVERGING' : 
               'EVOLVING'}
            </div>
            <div className="text-xs text-gray-400 mt-1">Phase</div>
          </div>
        </div>
      )}
    </div>
  )
}
