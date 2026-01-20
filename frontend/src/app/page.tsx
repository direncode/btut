'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Users,
  AlertTriangle,
  Sparkles,
  BarChart3,
} from 'lucide-react'

import { Header } from '@/components/Header'
import { Hero } from '@/components/Hero'
import { StatCard } from '@/components/StatCard'
import { PriceChart } from '@/components/PriceChart'
import { CooperationChart } from '@/components/CooperationChart'
import { AgentDistribution } from '@/components/AgentDistribution'
import { SimulationControls } from '@/components/SimulationControls'

import {
  SimulationConfig,
  MarketMetrics,
  AgentStats,
  defaultConfig,
  generateSimulationData,
  generateAgentStats,
} from '@/lib/simulation-data'
import { formatNumber, formatCompactNumber } from '@/lib/utils'

export default function Home() {
  const [config, setConfig] = useState<SimulationConfig>(defaultConfig)
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [metrics, setMetrics] = useState<MarketMetrics[]>([])
  const [agents, setAgents] = useState<AgentStats[]>([])

  // Initialize with data
  useEffect(() => {
    const initialData = generateSimulationData(config.numTimesteps, config)
    const initialAgents = generateAgentStats(config.numAgents)
    setMetrics(initialData)
    setAgents(initialAgents)
  }, [])

  const runSimulation = useCallback(() => {
    setIsRunning(true)
    setProgress(0)

    // Simulate progress
    const duration = 2000 // 2 seconds
    const interval = 50
    const steps = duration / interval
    let currentStep = 0

    const progressInterval = setInterval(() => {
      currentStep++
      setProgress((currentStep / steps) * 100)

      if (currentStep >= steps) {
        clearInterval(progressInterval)
        // Generate new data
        const newData = generateSimulationData(config.numTimesteps, config)
        const newAgents = generateAgentStats(config.numAgents)
        setMetrics(newData)
        setAgents(newAgents)
        setIsRunning(false)
      }
    }, interval)
  }, [config])

  const resetSimulation = useCallback(() => {
    setConfig(defaultConfig)
    setProgress(0)
    const newData = generateSimulationData(defaultConfig.numTimesteps, defaultConfig)
    const newAgents = generateAgentStats(defaultConfig.numAgents)
    setMetrics(newData)
    setAgents(newAgents)
  }, [])

  // Calculate summary statistics
  const latestMetrics = metrics[metrics.length - 1]
  const crashCount = metrics.filter(m => m.isCrash).length
  const avgCooperation = metrics.reduce((sum, m) => sum + m.cooperation, 0) / metrics.length
  const maxVolatility = Math.max(...metrics.map(m => m.volatility))
  const minPrice = Math.min(...metrics.map(m => m.midPrice))
  const maxPrice = Math.max(...metrics.map(m => m.midPrice))
  const maxDrawdown = ((maxPrice - minPrice) / maxPrice) * 100

  return (
    <main className="min-h-screen">
      <Header />
      <Hero />

      {/* Dashboard Section */}
      <section id="simulator" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Section header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h2 className="text-headline text-apple-gray-900 mb-4">
              Real-Time Market Dynamics
            </h2>
            <p className="text-body-lg text-apple-gray-500 max-w-2xl mx-auto">
              Watch agent strategies evolve through BTUT coordination.
              Observe emergent market behaviors including flash crashes and recovery patterns.
            </p>
          </motion.div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard
              title="Total Agents"
              value={formatCompactNumber(config.numAgents)}
              subtitle="Heterogeneous traders"
              icon={Users}
              color="blue"
              delay={0}
            />
            <StatCard
              title="Avg Cooperation"
              value={`${formatNumber(avgCooperation * 100, 1)}%`}
              subtitle={avgCooperation > 0.5 ? 'Market stable' : 'Stress detected'}
              icon={avgCooperation > 0.5 ? TrendingUp : TrendingDown}
              trend={avgCooperation > 0.5 ? 'up' : 'down'}
              trendValue={avgCooperation > 0.5 ? '+2.3%' : '-3.1%'}
              color={avgCooperation > 0.5 ? 'green' : 'red'}
              delay={0.1}
            />
            <StatCard
              title="Crash Events"
              value={crashCount > 0 ? '1' : '0'}
              subtitle={crashCount > 0 ? 'Recovery in progress' : 'No crashes detected'}
              icon={AlertTriangle}
              color={crashCount > 0 ? 'red' : 'green'}
              delay={0.2}
            />
            <StatCard
              title="Regeneration"
              value={formatNumber(latestMetrics?.regeneration * 100 || 0, 0)}
              subtitle="Market health score"
              icon={Sparkles}
              color="purple"
              delay={0.3}
            />
          </div>

          {/* Main Dashboard Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left column - Charts */}
            <div className="lg:col-span-2 space-y-6">
              <PriceChart data={metrics} />
              <CooperationChart data={metrics} />
            </div>

            {/* Right column - Controls & Stats */}
            <div className="space-y-6">
              <SimulationControls
                config={config}
                onConfigChange={setConfig}
                onRun={runSimulation}
                onReset={resetSimulation}
                isRunning={isRunning}
                progress={progress}
              />
              <AgentDistribution agents={agents} />
            </div>
          </div>

          {/* Summary Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-8 p-6 rounded-apple-lg bg-gradient-to-r from-apple-gray-900 to-apple-gray-800 text-white"
          >
            <div className="flex items-center gap-3 mb-6">
              <BarChart3 className="w-6 h-6 text-apple-blue-light" />
              <h3 className="text-lg font-semibold">Simulation Summary</h3>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
              <div>
                <p className="text-sm text-apple-gray-400 mb-1">Timesteps</p>
                <p className="text-2xl font-semibold tabular-nums">{metrics.length}</p>
              </div>
              <div>
                <p className="text-sm text-apple-gray-400 mb-1">Price Range</p>
                <p className="text-2xl font-semibold tabular-nums">
                  ${formatNumber(minPrice, 0)} - ${formatNumber(maxPrice, 0)}
                </p>
              </div>
              <div>
                <p className="text-sm text-apple-gray-400 mb-1">Max Drawdown</p>
                <p className="text-2xl font-semibold tabular-nums text-apple-red-light">
                  {formatNumber(maxDrawdown, 1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-apple-gray-400 mb-1">Max Volatility</p>
                <p className="text-2xl font-semibold tabular-nums">
                  {formatNumber(maxVolatility * 100, 2)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-apple-gray-400 mb-1">BTUT Convergence</p>
                <p className="text-2xl font-semibold tabular-nums text-apple-green-light">
                  25 iter
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6 bg-apple-gray-50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-headline text-apple-gray-900 mb-4">
              BTUT Technology
            </h2>
            <p className="text-body-lg text-apple-gray-500 max-w-2xl mx-auto">
              Bivariate Trajectory-Undercurrent Theory enables unprecedented scale
              through kernel-weighted mean-field dynamics.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: 'O(N) Complexity',
                description: 'No explicit graph storage. Kernel-weighted interactions enable linear scaling to millions of agents.',
                icon: '⚡',
              },
              {
                title: 'Fast Convergence',
                description: 'Strategy equilibrium reached in 20-30 iterations regardless of agent count. Mathematical guarantees.',
                icon: '🎯',
              },
              {
                title: 'Emergent Behavior',
                description: 'Flash crashes, liquidity spirals, and market regeneration emerge naturally from agent interactions.',
                icon: '🌊',
              },
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="card card-hover p-8"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-title text-apple-gray-900 mb-2">{feature.title}</h3>
                <p className="text-body text-apple-gray-500">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-apple-gray-200">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-sm text-apple-gray-500">
            Built with BTUT (Bivariate Trajectory-Undercurrent Theory)
          </p>
          <p className="text-xs text-apple-gray-400 mt-2">
            Planetary-scale multi-agent financial simulation
          </p>
        </div>
      </footer>
    </main>
  )
}
