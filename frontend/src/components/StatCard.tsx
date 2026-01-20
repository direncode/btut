'use client'

import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { LucideIcon } from 'lucide-react'

interface StatCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: LucideIcon
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
  color?: 'blue' | 'green' | 'red' | 'orange' | 'purple'
  delay?: number
}

const colorMap = {
  blue: {
    bg: 'bg-apple-blue/10',
    text: 'text-apple-blue',
    glow: 'shadow-glow',
  },
  green: {
    bg: 'bg-apple-green/10',
    text: 'text-apple-green',
    glow: 'shadow-glow-green',
  },
  red: {
    bg: 'bg-apple-red/10',
    text: 'text-apple-red',
    glow: 'shadow-glow-red',
  },
  orange: {
    bg: 'bg-apple-orange/10',
    text: 'text-apple-orange',
    glow: '',
  },
  purple: {
    bg: 'bg-apple-purple/10',
    text: 'text-apple-purple',
    glow: '',
  },
}

export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendValue,
  color = 'blue',
  delay = 0,
}: StatCardProps) {
  const colors = colorMap[color]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="stat-card group hover:shadow-apple-lg transition-all duration-300"
    >
      <div className="flex items-start justify-between mb-4">
        <div className={cn(
          'w-12 h-12 rounded-apple flex items-center justify-center transition-all duration-300',
          colors.bg,
          'group-hover:scale-110'
        )}>
          <Icon className={cn('w-6 h-6', colors.text)} />
        </div>
        {trend && trendValue && (
          <div className={cn(
            'flex items-center gap-1 text-sm font-medium',
            trend === 'up' && 'text-apple-green',
            trend === 'down' && 'text-apple-red',
            trend === 'neutral' && 'text-apple-gray-500'
          )}>
            {trend === 'up' && '↑'}
            {trend === 'down' && '↓'}
            {trendValue}
          </div>
        )}
      </div>

      <div className="space-y-1">
        <p className="text-sm text-apple-gray-500 font-medium">{title}</p>
        <p className={cn(
          'text-3xl font-semibold tracking-tight tabular-nums',
          colors.text
        )}>
          {value}
        </p>
        {subtitle && (
          <p className="text-xs text-apple-gray-400">{subtitle}</p>
        )}
      </div>

      {/* Decorative gradient */}
      <div className={cn(
        'absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl opacity-0 group-hover:opacity-20 transition-opacity duration-500',
        color === 'blue' && 'bg-apple-blue',
        color === 'green' && 'bg-apple-green',
        color === 'red' && 'bg-apple-red',
        color === 'orange' && 'bg-apple-orange',
        color === 'purple' && 'bg-apple-purple'
      )} />
    </motion.div>
  )
}
