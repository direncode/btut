'use client'

import Image from 'next/image'
import Link from 'next/link'
import { stressTestHeroMetrics } from '@/lib/data/validation-data'

interface MetricCardProps {
  value: string | number
  label: string
  highlight?: boolean
}

function MetricCard({ value, label, highlight = false }: MetricCardProps) {
  return (
    <div className={`
      p-6 rounded-lg border backdrop-blur-sm
      ${highlight
        ? 'bg-neon-green/10 border-neon-green/30'
        : 'bg-gray-900/50 border-gray-800'
      }
    `}>
      <div className={`
        text-4xl md:text-5xl font-bold mb-2
        ${highlight ? 'text-neon-green' : 'gradient-text'}
      `}>
        {value}
      </div>
      <div className="text-gray-400 text-sm uppercase tracking-wider">
        {label}
      </div>
    </div>
  )
}

export function StressTestHero() {
  const { peakVehicles, cooperation, avgSpeed, gridlockOccurred, throughput, source } = stressTestHeroMetrics

  return (
    <section className="py-16 md:py-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gray-900/50 border border-gray-800 mb-6">
            <span className="w-2 h-2 bg-neon-green rounded-full animate-pulse" />
            <span className="text-sm text-gray-400">Validated with {source}</span>
          </div>

          <h1 className="text-4xl md:text-6xl font-bold mb-4">
            <span className="gradient-text">{peakVehicles} Vehicles.</span>
            {' '}
            <span className="text-neon-green">Zero Gridlock.</span>
          </h1>

          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            BTUT coordination tested under peak traffic stress.
            Every metric from real Eclipse SUMO simulations.
          </p>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          <MetricCard
            value={peakVehicles}
            label="Peak Vehicles"
          />
          <MetricCard
            value={`${cooperation}%`}
            label="Cooperation Rate"
            highlight
          />
          <MetricCard
            value={`${avgSpeed} m/s`}
            label="Average Speed"
          />
          <MetricCard
            value={gridlockOccurred ? 'YES' : '0'}
            label="Gridlocks"
            highlight={!gridlockOccurred}
          />
        </div>

        {/* Convergence Chart */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6 mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">
            Convergence Dynamics Under Stress
          </h3>
          <div className="relative aspect-[2/1] w-full">
            <Image
              src="/validation/convergence_dynamics.png"
              alt="BTUT convergence dynamics showing cooperation, speed, congestion, and density over time"
              fill
              className="object-contain rounded"
              priority
            />
          </div>
          <div className="mt-4 flex flex-wrap gap-4 text-sm text-gray-500">
            <span>Test duration: 3000 seconds</span>
            <span>Peak at 600-1200s</span>
            <span>Throughput: {throughput} veh/hr</span>
          </div>
        </div>

        {/* Data Source */}
        <div className="flex flex-wrap justify-center gap-4">
          <Link
            href="/integrations/sumo/stress_test_results.json"
            target="_blank"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-900/50 border border-gray-800 hover:border-neon-blue/50 transition-colors text-sm"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            View Raw JSON
          </Link>
          <Link
            href="/simulator"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-neon-blue/20 border border-neon-blue/50 hover:bg-neon-blue/30 transition-colors text-sm"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Run Simulator
          </Link>
        </div>
      </div>
    </section>
  )
}
