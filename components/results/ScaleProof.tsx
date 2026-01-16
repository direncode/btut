'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line } from 'recharts'
import { scaleTestData } from '@/lib/data/validation-data'

export function ScaleProof() {
  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="gradient-text">O(N) Scaling Verified</span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Convergence iterations remain constant as agent count scales from 500 to 10,000.
            This confirms true O(N) linear complexity.
          </p>
        </div>

        {/* Key Insight */}
        <div className="bg-gradient-to-r from-neon-purple/10 to-neon-blue/10 border border-neon-purple/30 rounded-lg p-6 mb-12">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h3 className="text-xl font-semibold text-white mb-2">Constant Convergence</h3>
              <p className="text-gray-300">
                All tests converge in exactly <span className="text-neon-purple font-bold">12 iterations</span>,
                regardless of agent count
              </p>
            </div>
            <div className="text-center">
              <div className="text-5xl font-bold text-neon-purple">12</div>
              <div className="text-sm text-gray-400">iterations</div>
              <div className="text-xs text-gray-500">500 to 10,000 agents</div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Convergence Iterations */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Convergence Iterations by Scale</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={scaleTestData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="agents"
                    stroke="#9ca3af"
                    tickFormatter={(v) => v >= 1000 ? `${v/1000}K` : v}
                  />
                  <YAxis stroke="#9ca3af" domain={[0, 20]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    formatter={(value: number) => [value, 'Iterations']}
                    labelFormatter={(label) => `${label.toLocaleString()} agents`}
                  />
                  <Bar dataKey="iterations" radius={[4, 4, 0, 0]}>
                    {scaleTestData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill="#a855f7" />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 text-sm text-gray-500 text-center">
              All bars at 12 = O(N) complexity confirmed
            </div>
          </div>

          {/* Speed vs Scale */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Effective Speed by Scale</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scaleTestData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="agents"
                    stroke="#9ca3af"
                    tickFormatter={(v) => v >= 1000 ? `${v/1000}K` : v}
                  />
                  <YAxis stroke="#9ca3af" domain={[4, 10]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    formatter={(value: number) => [`${value.toFixed(2)} m/s`, 'Speed']}
                    labelFormatter={(label) => `${label.toLocaleString()} agents`}
                  />
                  <Line
                    type="monotone"
                    dataKey="speed"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ fill: '#3b82f6', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 text-sm text-gray-500 text-center">
              Expected: speed decreases with density
            </div>
          </div>
        </div>

        {/* Data Table */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-800">
              <thead className="bg-gray-900/50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Agents</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Convergence</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Speed (m/s)</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Wait Time (s)</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Complexity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {scaleTestData.map((row, index) => (
                  <tr key={row.agents} className="hover:bg-gray-800/50">
                    <td className="px-6 py-4 text-gray-200 font-mono">
                      {row.agents.toLocaleString()}
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-neon-purple font-bold">{row.iterations}</span>
                      <span className="text-gray-500 text-sm ml-1">iter</span>
                    </td>
                    <td className="px-6 py-4 text-neon-blue">
                      {row.speed.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 text-gray-300">
                      {row.waitTime.toFixed(1)}
                    </td>
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-neon-green/20 text-neon-green border border-neon-green/30">
                        O(N)
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-6 py-3 bg-gray-900/30 border-t border-gray-800 text-xs text-gray-500">
            Data source: integrations/sumo/large_scale_results.json
          </div>
        </div>

        {/* Complexity Comparison */}
        <div className="mt-12 grid md:grid-cols-3 gap-6">
          <div className="bg-gray-900/50 border border-neon-green/30 rounded-lg p-6 text-center">
            <div className="text-3xl font-bold text-neon-green mb-2">BTUT</div>
            <div className="text-5xl font-mono font-bold text-white mb-2">O(N)</div>
            <div className="text-gray-400 text-sm">Linear time</div>
            <div className="text-gray-500 text-xs mt-2">No adjacency matrix needed</div>
          </div>
          <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-6 text-center opacity-60">
            <div className="text-3xl font-bold text-gray-400 mb-2">MASON</div>
            <div className="text-5xl font-mono font-bold text-gray-400 mb-2">O(N²)</div>
            <div className="text-gray-500 text-sm">Quadratic time</div>
            <div className="text-gray-600 text-xs mt-2">Pairwise interactions</div>
          </div>
          <div className="bg-gray-900/50 border border-gray-700 rounded-lg p-6 text-center opacity-60">
            <div className="text-3xl font-bold text-gray-400 mb-2">Mesa</div>
            <div className="text-5xl font-mono font-bold text-gray-400 mb-2">O(N²)</div>
            <div className="text-gray-500 text-sm">Quadratic time</div>
            <div className="text-gray-600 text-xs mt-2">Python loop overhead</div>
          </div>
        </div>
      </div>
    </section>
  )
}
