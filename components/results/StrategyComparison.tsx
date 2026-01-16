'use client'

import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { strategyComparison } from '@/lib/data/validation-data'

type SortKey = 'name' | 'speed' | 'waitTime' | 'throughput' | 'stability'

export function StrategyComparison() {
  const [sortKey, setSortKey] = useState<SortKey>('speed')
  const [sortAsc, setSortAsc] = useState(false)

  const sortedData = [...strategyComparison].sort((a, b) => {
    const aVal = a[sortKey]
    const bVal = b[sortKey]
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
    }
    return sortAsc ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number)
  })

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc)
    } else {
      setSortKey(key)
      setSortAsc(false)
    }
  }

  const SortHeader = ({ label, sortKeyName }: { label: string; sortKeyName: SortKey }) => (
    <th
      className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors"
      onClick={() => handleSort(sortKeyName)}
    >
      <div className="flex items-center gap-1">
        {label}
        {sortKey === sortKeyName && (
          <span className="text-neon-blue">{sortAsc ? '↑' : '↓'}</span>
        )}
      </div>
    </th>
  )

  // Colors for chart bars
  const getBarColor = (name: string) => {
    if (name.includes('BTUT')) return '#00ff88'
    if (name === 'Fixed 60%') return '#3b82f6'
    return '#6b7280'
  }

  return (
    <section className="py-16 bg-gray-950/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="gradient-text">7 Coordination Strategies Compared</span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Side-by-side comparison from baseline_results.json.
            Click column headers to sort.
          </p>
        </div>

        {/* Charts */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Speed Chart */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Average Speed (m/s)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={strategyComparison} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" stroke="#9ca3af" domain={[0, 12]} />
                  <YAxis dataKey="name" type="category" stroke="#9ca3af" width={100} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#f3f4f6' }}
                  />
                  <Bar dataKey="speed" radius={[0, 4, 4, 0]}>
                    {strategyComparison.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getBarColor(entry.name)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Wait Time Chart */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-200">Wait Time (seconds)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={strategyComparison} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" stroke="#9ca3af" domain={[0, 35]} />
                  <YAxis dataKey="name" type="category" stroke="#9ca3af" width={100} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                    labelStyle={{ color: '#f3f4f6' }}
                  />
                  <Bar dataKey="waitTime" radius={[0, 4, 4, 0]}>
                    {strategyComparison.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.waitTime < 20 ? '#00ff88' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Data Table */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-800">
              <thead className="bg-gray-900/50">
                <tr>
                  <SortHeader label="Strategy" sortKeyName="name" />
                  <SortHeader label="Speed (m/s)" sortKeyName="speed" />
                  <SortHeader label="Wait (s)" sortKeyName="waitTime" />
                  <SortHeader label="Throughput" sortKeyName="throughput" />
                  <SortHeader label="Stability" sortKeyName="stability" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {sortedData.map((strategy, index) => (
                  <tr
                    key={strategy.name}
                    className={`${strategy.name.includes('BTUT') ? 'bg-neon-green/5' : ''} hover:bg-gray-800/50 transition-colors`}
                  >
                    <td className="px-4 py-4 whitespace-nowrap">
                      <span className={strategy.name.includes('BTUT') ? 'text-neon-green font-medium' : 'text-gray-200'}>
                        {strategy.name}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-gray-300">
                      {strategy.speed.toFixed(1)}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-gray-300">
                      {strategy.waitTime.toFixed(1)}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-gray-300">
                      {strategy.throughput}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-neon-blue h-2 rounded-full"
                            style={{ width: `${strategy.stability}%` }}
                          />
                        </div>
                        <span className="text-gray-400 text-sm">{strategy.stability}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-3 bg-gray-900/30 border-t border-gray-800 text-xs text-gray-500">
            Data source: integrations/sumo/baseline_results.json
          </div>
        </div>
      </div>
    </section>
  )
}
