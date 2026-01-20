'use client'

import { motion } from 'framer-motion'
import { AgentStats } from '@/lib/simulation-data'
import { formatCompactNumber, formatNumber } from '@/lib/utils'
import { Users, Zap, Building2, TrendingUp } from 'lucide-react'

interface AgentDistributionProps {
  agents: AgentStats[]
}

const iconMap = {
  Retail: Users,
  HFT: Zap,
  Institution: Building2,
  'Market Maker': TrendingUp,
}

export function AgentDistribution({ agents }: AgentDistributionProps) {
  const totalAgents = agents.reduce((sum, a) => sum + a.count, 0)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="chart-container"
    >
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-apple-gray-900">
          Agent Distribution
        </h3>
        <p className="text-sm text-apple-gray-500">
          {formatCompactNumber(totalAgents)} total agents by type
        </p>
      </div>

      <div className="space-y-4">
        {agents.map((agent, index) => {
          const Icon = iconMap[agent.type as keyof typeof iconMap] || Users
          const percentage = (agent.count / totalAgents) * 100

          return (
            <motion.div
              key={agent.type}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.5 + index * 0.1 }}
              className="group"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-apple flex items-center justify-center transition-transform group-hover:scale-110"
                    style={{ backgroundColor: `${agent.color}15` }}
                  >
                    <Icon className="w-5 h-5" style={{ color: agent.color }} />
                  </div>
                  <div>
                    <p className="font-medium text-apple-gray-900">{agent.type}</p>
                    <p className="text-xs text-apple-gray-500">
                      {formatCompactNumber(agent.count)} agents
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-apple-gray-900 tabular-nums">
                    {formatNumber(percentage, 0)}%
                  </p>
                  <p className="text-xs text-apple-gray-500">
                    Coop: {formatNumber(agent.cooperation * 100, 0)}%
                  </p>
                </div>
              </div>

              {/* Progress bar */}
              <div className="h-2 bg-apple-gray-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 0.8, delay: 0.6 + index * 0.1, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: agent.color }}
                />
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Summary */}
      <div className="mt-6 pt-6 border-t border-apple-gray-200">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-apple-gray-500 mb-1">Avg Cooperation</p>
            <p className="text-xl font-semibold text-apple-blue tabular-nums">
              {formatNumber(
                agents.reduce((sum, a) => sum + a.cooperation * a.count, 0) / totalAgents * 100,
                1
              )}%
            </p>
          </div>
          <div>
            <p className="text-xs text-apple-gray-500 mb-1">Liquidity Providers</p>
            <p className="text-xl font-semibold text-apple-green tabular-nums">
              {formatNumber(
                (agents.find(a => a.type === 'Market Maker')?.count || 0) / totalAgents * 100,
                1
              )}%
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
