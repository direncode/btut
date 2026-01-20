'use client'

import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { MarketMetrics } from '@/lib/simulation-data'
import { formatNumber } from '@/lib/utils'

interface CooperationChartProps {
  data: MarketMetrics[]
  height?: number
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass rounded-apple p-4 shadow-apple-lg border border-apple-gray-200/50">
        <p className="text-sm font-medium text-apple-gray-500 mb-2">
          Timestep {label}
        </p>
        <div className="space-y-2">
          {payload.map((entry: any) => (
            <div key={entry.name} className="flex items-center justify-between gap-6">
              <div className="flex items-center gap-2">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-sm text-apple-gray-600">{entry.name}</span>
              </div>
              <span className="text-sm font-semibold tabular-nums">
                {formatNumber(entry.value * 100, 1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }
  return null
}

export function CooperationChart({ data, height = 280 }: CooperationChartProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.3 }}
      className="chart-container"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-apple-gray-900">
            Market Dynamics
          </h3>
          <p className="text-sm text-apple-gray-500">
            Cooperation, regeneration, and drag over time
          </p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#e8e8ed"
            vertical={false}
          />

          <XAxis
            dataKey="timestep"
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#86868b', fontSize: 12 }}
            tickFormatter={(value) => value % 100 === 0 ? value : ''}
          />

          <YAxis
            domain={[0, 1]}
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#86868b', fontSize: 12 }}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            width={45}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            verticalAlign="top"
            align="right"
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ paddingBottom: 20 }}
            formatter={(value) => (
              <span className="text-xs text-apple-gray-600">{value}</span>
            )}
          />

          <Line
            type="monotone"
            dataKey="cooperation"
            name="Cooperation"
            stroke="#0071e3"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#0071e3', stroke: '#fff', strokeWidth: 2 }}
          />

          <Line
            type="monotone"
            dataKey="regeneration"
            name="Regeneration"
            stroke="#34c759"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#34c759', stroke: '#fff', strokeWidth: 2 }}
          />

          <Line
            type="monotone"
            dataKey="drag"
            name="Drag"
            stroke="#ff3b30"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#ff3b30', stroke: '#fff', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  )
}
