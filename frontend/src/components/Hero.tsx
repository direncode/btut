'use client'

import { motion } from 'framer-motion'
import { ArrowDown, Sparkles, Cpu, Network } from 'lucide-react'

export function Hero() {
  return (
    <section className="relative overflow-hidden py-20 lg:py-32">
      {/* Background gradient orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-1/4 w-96 h-96 bg-apple-blue/20 rounded-full blur-3xl animate-float" />
        <div className="absolute top-1/3 -right-1/4 w-96 h-96 bg-apple-purple/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '-2s' }} />
        <div className="absolute bottom-1/4 left-1/3 w-64 h-64 bg-apple-green/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '-4s' }} />
      </div>

      <div className="relative max-w-7xl mx-auto px-6">
        <div className="text-center max-w-4xl mx-auto">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-apple-blue/10 text-apple-blue text-sm font-medium mb-8"
          >
            <Sparkles className="w-4 h-4" />
            Powered by BTUT — O(N) Multi-Agent Coordination
          </motion.div>

          {/* Main headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-display-lg md:text-display-xl text-apple-gray-900 mb-6"
          >
            Market Simulation.{' '}
            <span className="gradient-text">Reimagined.</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-body-lg text-apple-gray-500 mb-12 max-w-2xl mx-auto text-balance"
          >
            Experience planetary-scale financial market simulation with up to 1 million
            heterogeneous agents. Witness emergent flash crashes, liquidity spirals,
            and market regeneration in real-time.
          </motion.p>

          {/* Feature highlights */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-wrap justify-center gap-8 mb-12"
          >
            {[
              { icon: Cpu, label: '1M+ Agents', sublabel: 'Single machine' },
              { icon: Network, label: '20-30 Iterations', sublabel: 'Fast convergence' },
              { icon: Sparkles, label: 'Emergent Behavior', sublabel: 'Flash crashes & regeneration' },
            ].map((feature, index) => (
              <div key={feature.label} className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-apple bg-apple-gray-100 flex items-center justify-center">
                  <feature.icon className="w-6 h-6 text-apple-gray-600" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-apple-gray-900">{feature.label}</p>
                  <p className="text-sm text-apple-gray-500">{feature.sublabel}</p>
                </div>
              </div>
            ))}
          </motion.div>

          {/* CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="flex flex-col items-center gap-4"
          >
            <a
              href="#simulator"
              className="btn-primary text-lg px-8 py-4"
            >
              Launch Simulator
            </a>
            <a
              href="#simulator"
              className="flex items-center gap-2 text-apple-gray-500 hover:text-apple-gray-700 transition-colors animate-bounce"
            >
              <ArrowDown className="w-4 h-4" />
              <span className="text-sm">Scroll to explore</span>
            </a>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
