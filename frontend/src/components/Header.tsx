'use client'

import { motion } from 'framer-motion'
import { Activity, Github, Zap } from 'lucide-react'

export function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="sticky top-0 z-50 glass border-b border-apple-gray-200/50"
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-apple bg-gradient-to-br from-apple-blue to-apple-purple flex items-center justify-center shadow-glow">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-apple-gray-900">
                BTUT Market Simulator
              </h1>
              <p className="text-xs text-apple-gray-500">
                Planetary-Scale Multi-Agent Simulation
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-apple-green/10 text-apple-green text-sm font-medium">
              <Zap className="w-4 h-4" />
              <span>O(N) Complexity</span>
            </div>

            <a
              href="https://github.com/btut/btut_market_simulator"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-full hover:bg-apple-gray-100 transition-colors"
            >
              <Github className="w-5 h-5 text-apple-gray-600" />
            </a>
          </div>
        </div>
      </div>
    </motion.header>
  )
}
