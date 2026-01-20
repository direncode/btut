'use client'

import { motion } from 'framer-motion'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { MarketMetrics } from '@/lib/simulation-data'
import { formatNumber } from '@/lib/utils'

interface PriceChartProps {
  data: MarketMetrics[]
  height?: number
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as MarketMetrics
    return (
      <div className="glass rounded-apple p-4 shadow-apple-lg border border-apple-gray-200/50">
        <p className="text-sm font-medium text-apple-gray-500 mb-2">
          Timestep {label}
        </p>
        <div className="space-y-1">
          <div className="flex items-center justify-between gap-6">
            <span className="text-sm text-apple-gray-600">Price</span>
            <span className="text-sm font-semibold tabular-nums">
              ${formatNumber(data.midPrice, 2)}
            </span>
          </div>
          <div className="flex items-center justify-between gap-6">
            <span className="text-sm text-apple-gray-600">Spread</span>
            <span className="text-sm font-semibold tabular-nums">
              {formatNumber(data.spreadBps, 1)} bps
            </span>
          </div>
          <div className="flex items-center justify-between gap-6">
            <span className="text-sm text-apple-gray-600">Volatility</span>
            <span className="text-sm font-semibold tabular-nums">
              {formatNumber(data.volatility * 100, 2)}%
            </span>
          </div>
          {data.isCrash && (
            <div className="mt-2 px-2 py-1 bg-apple-red/10 rounded text-apple-red text-xs font-medium">
              Flash Crash Active
            </div>
          )}
        </div>
      </div>
    )
  }
  return null
}

export function PriceChart({ data, height = 350 }: PriceChartProps) {
  const minPrice = Math.min(...data.map(d => d.midPrice)) * 0.99
  const maxPrice = Math.max(...data.map(d => d.midPrice)) * 1.01
  const crashStart = data.findIndex(d => d.isCrash)
  const crashEnd = data.findLastIndex(d => d.isCrash)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.2 }}
      className="chart-container"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-apple-gray-900">
            Price History
          </h3>
          <p className="text-sm text-apple-gray-500">
            Mid-price over simulation timesteps
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-apple-blue" />
            <span className="text-xs text-apple-gray-500">Price</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-apple-red/30" />
            <span className="text-xs text-apple-gray-500">Crash Zone</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0071e3" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#0071e3" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="crashGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ff3b30" stopOpacity={0.1} />
              <stop offset="100%" stopColor="#ff3b30" stopOpacity={0.05} />
            </linearGradient>
          </defs>

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
            domain={[minPrice, maxPrice]}
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#86868b', fontSize: 12 }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
            width={50}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Crash zone highlight */}
          {crashStart >= 0 && (
            <ReferenceLine
              x={crashStart}
              stroke="#ff3b30"
              strokeDasharray="5 5"
              strokeOpacity={0.5}
            />
          )}
          {crashEnd >= 0 && (
            <ReferenceLine
              x={crashEnd}
              stroke="#34c759"
              strokeDasharray="5 5"
              strokeOpacity={0.5}
            />
          )}

          <Area
            type="monotone"
            dataKey="midPrice"
            stroke="#0071e3"
            strokeWidth={2}
            fill="url(#priceGradient)"
            dot={false}
            activeDot={{
              r: 6,
              fill: '#0071e3',
              stroke: '#fff',
              strokeWidth: 2,
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </motion.div>
  )
}
