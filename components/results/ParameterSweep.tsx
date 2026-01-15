'use client'

import Image from 'next/image'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { tauSweepData } from '@/lib/data/validation-data'

export function ParameterSweep() {
  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="gradient-text">How Tau Affects Cooperation</span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Parameter sweep from tau=0.0 to tau=0.8.
            Higher tau gives network hubs more influence.
          </p>
        </div>

        {/* Key Finding */}
        <div className="bg-gradient-to-r from-neon-blue/10 to-neon-green/10 border border-neon-blue/30 rounded-lg p-6 mb-12">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h3 className="text-xl font-semibold text-white mb-2">Key Finding</h3>
              <p className="text-gray-300">
                Cooperation increases from <span className="text-neon-blue font-bold">49%</span> at tau=0.0
                to <span className="text-neon-green font-bold">73%</span> at tau=0.8
              </p>
            </div>
            <div className="flex gap-8">
              <div className="text-center">
                <div className="text-3xl font-bold text-neon-blue">+12.3%</div>
                <div className="text-sm text-gray-400">Best Speed Gain</div>
                <div className="text-xs text-gray-500">at tau=0.7</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-neon-green">+4.7%</div>
                <div className="text-sm text-gray-400">Best Wait Reduction</div>
                <div className="text-xs text-gray-500">at tau=0.5</div>
              </div>
            </div>
          </div>
        </div>

        {/* Interactive Chart */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Cooperation vs Tau */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Cooperation Rate vs Tau</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tauSweepData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="tau" stroke="#9ca3af" label={{ value: 'tau', position: 'bottom', fill: '#9ca3af' }} />
                  <YAxis stroke="#9ca3af" domain={[40, 80]} label={{ value: 'Cooperation %', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#f3f4f6' }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, 'Cooperation']}
                  />
                  <Line
                    type="monotone"
                    dataKey="cooperation"
                    stroke="#00ff88"
                    strokeWidth={2}
                    dot={{ fill: '#00ff88', strokeWidth: 2 }}
                    activeDot={{ r: 6, fill: '#00ff88' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Speed Improvement vs Tau */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Speed Improvement vs Tau</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tauSweepData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="tau" stroke="#9ca3af" label={{ value: 'tau', position: 'bottom', fill: '#9ca3af' }} />
                  <YAxis stroke="#9ca3af" label={{ value: 'Improvement %', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#f3f4f6' }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, 'Improvement']}
                  />
                  <Line
                    type="monotone"
                    dataKey="speedImprovement"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ fill: '#3b82f6', strokeWidth: 2 }}
                    activeDot={{ r: 6, fill: '#3b82f6' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Full Tau Sweep Chart Image */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">
            Complete Parameter Sweep Visualization
          </h3>
          <div className="relative aspect-[3/1] w-full">
            <Image
              src="/validation/tau_sweep.png"
              alt="Tau parameter sweep showing cooperation, speed, and congestion across tau values 0.0 to 0.8"
              fill
              className="object-contain rounded"
            />
          </div>
          <div className="mt-4 text-sm text-gray-500">
            Three panels showing: Phase transition in cooperation (left), Traffic flow efficiency (center), Congestion reduction (right)
          </div>
        </div>

        {/* Data Table */}
        <div className="mt-8 bg-gray-900/50 border border-gray-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-800">
              <thead className="bg-gray-900/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Tau</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Cooperation</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Speed Gain</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Wait Reduction</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {tauSweepData.map((row) => (
                  <tr key={row.tau} className="hover:bg-gray-800/50">
                    <td className="px-4 py-3 text-gray-200 font-mono">{row.tau.toFixed(1)}</td>
                    <td className="px-4 py-3 text-neon-green">{row.cooperation.toFixed(1)}%</td>
                    <td className="px-4 py-3">
                      <span className={row.speedImprovement > 0 ? 'text-neon-blue' : 'text-gray-400'}>
                        {row.speedImprovement > 0 ? '+' : ''}{row.speedImprovement.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={row.waitReduction > 0 ? 'text-neon-green' : 'text-red-400'}>
                        {row.waitReduction > 0 ? '+' : ''}{row.waitReduction.toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-3 bg-gray-900/30 border-t border-gray-800 text-xs text-gray-500">
            Data source: integrations/sumo/validation_results/validation_results_20260115_154049.json
          </div>
        </div>
      </div>
    </section>
  )
}
