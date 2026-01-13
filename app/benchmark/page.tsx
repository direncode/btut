'use client'

import { useState } from 'react'
import { BTUTSimulator as TypeScriptSimulator } from '@/lib/simulation/btut-engine'
import { BTUTWasmSimulator, isWasmAvailable } from '@/lib/simulation/btut-wasm-wrapper'
import { Zap, Code, Trophy, TrendingUp } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

interface BenchmarkResult {
  n: number
  typescriptTime: number
  wasmTime: number
  speedup: number
}

export default function BenchmarkPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<BenchmarkResult[]>([])
  const [progress, setProgress] = useState(0)

  const runBenchmark = async () => {
    if (!isWasmAvailable()) {
      alert('WASM not available. Please build the Rust module first.')
      return
    }

    setIsRunning(true)
    setResults([])
    
    const sizes = [1000, 5000, 10000, 25000, 50000, 100000]
    const newResults: BenchmarkResult[] = []

    for (let i = 0; i < sizes.length; i++) {
      const n = sizes[i]
      setProgress(((i + 1) / sizes.length) * 100)

      const config = {
        N: n,
        gamma: 1.45,
        tau: 0.30,
        cA_SH: 0.40,
        cB_SH: 0.10,
        cA_PD: 0.20,
        cB_PD: 0.08,
        alpha: 0.60,
        iterations: 10,
        m: 3,
        seed: 42,
      }

      // Benchmark TypeScript
      const tsStart = performance.now()
      const tsSim = new TypeScriptSimulator(config)
      for (let j = 0; j < 10; j++) {
        tsSim.step()
      }
      const tsTime = performance.now() - tsStart

      // Benchmark WASM
      const wasmStart = performance.now()
      const wasmSim = new BTUTWasmSimulator(config)
      wasmSim.runComplete()
      const wasmTime = performance.now() - wasmStart

      const speedup = tsTime / wasmTime

      newResults.push({
        n,
        typescriptTime: tsTime,
        wasmTime,
        speedup,
      })

      setResults([...newResults])
    }

    setIsRunning(false)
    setProgress(100)
  }

  const avgSpeedup = results.length > 0
    ? results.reduce((sum, r) => sum + r.speedup, 0) / results.length
    : 0

  return (
    <div className="min-h-screen bg-black pt-20">
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <h1 className="text-4xl font-bold gradient-text mb-2">Performance Benchmark</h1>
          <p className="text-gray-400">Rust/WASM vs TypeScript engine comparison</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
            <div className="flex items-center mb-4">
              <Zap className="w-8 h-8 text-neon-blue mr-3" />
              <h3 className="text-xl font-bold">Rust/WASM</h3>
            </div>
            <div className="text-3xl font-bold text-neon-blue mb-2">
              {results.length > 0 ? `${results[results.length - 1].wasmTime.toFixed(0)}ms` : '—'}
            </div>
            <div className="text-sm text-gray-400">Last benchmark (100K agents)</div>
          </div>

          <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
            <div className="flex items-center mb-4">
              <Code className="w-8 h-8 text-neon-purple mr-3" />
              <h3 className="text-xl font-bold">TypeScript</h3>
            </div>
            <div className="text-3xl font-bold text-neon-purple mb-2">
              {results.length > 0 ? `${results[results.length - 1].typescriptTime.toFixed(0)}ms` : '—'}
            </div>
            <div className="text-sm text-gray-400">Last benchmark (100K agents)</div>
          </div>

          <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
            <div className="flex items-center mb-4">
              <Trophy className="w-8 h-8 text-neon-green mr-3" />
              <h3 className="text-xl font-bold">Speedup</h3>
            </div>
            <div className="text-3xl font-bold text-neon-green mb-2">
              {avgSpeedup > 0 ? `${avgSpeedup.toFixed(1)}x` : '—'}
            </div>
            <div className="text-sm text-gray-400">Average across all sizes</div>
          </div>
        </div>

        {/* Run Button */}
        <div className="mb-8 text-center">
          <button
            onClick={runBenchmark}
            disabled={isRunning}
            className="px-8 py-4 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? `Running... ${progress.toFixed(0)}%` : 'Run Benchmark'}
          </button>
          {!isWasmAvailable() && (
            <p className="text-red-400 text-sm mt-4">
              ⚠️ WASM not available. Run <code className="bg-gray-800 px-2 py-1 rounded">npm run build:wasm</code> first.
            </p>
          )}
        </div>

        {/* Chart */}
        {results.length > 0 && (
          <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6 mb-8">
            <h2 className="text-2xl font-bold mb-6">Runtime Comparison</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={results}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis 
                  dataKey="n" 
                  stroke="#6b7280"
                  label={{ value: 'Number of Agents', position: 'insideBottom', offset: -5, fill: '#6b7280' }}
                />
                <YAxis 
                  stroke="#6b7280"
                  label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft', fill: '#6b7280' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#111827', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="typescriptTime" 
                  stroke="#b000ff" 
                  name="TypeScript"
                  strokeWidth={2}
                  dot={{ fill: '#b000ff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="wasmTime" 
                  stroke="#00f0ff" 
                  name="Rust/WASM"
                  strokeWidth={2}
                  dot={{ fill: '#00f0ff' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Results Table */}
        {results.length > 0 && (
          <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-6">Detailed Results</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-3 px-4 text-gray-400">Agents</th>
                    <th className="text-right py-3 px-4 text-gray-400">TypeScript</th>
                    <th className="text-right py-3 px-4 text-gray-400">Rust/WASM</th>
                    <th className="text-right py-3 px-4 text-gray-400">Speedup</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result) => (
                    <tr key={result.n} className="border-b border-gray-800">
                      <td className="py-3 px-4 font-mono">{result.n.toLocaleString()}</td>
                      <td className="text-right py-3 px-4 font-mono text-neon-purple">
                        {result.typescriptTime.toFixed(2)}ms
                      </td>
                      <td className="text-right py-3 px-4 font-mono text-neon-blue">
                        {result.wasmTime.toFixed(2)}ms
                      </td>
                      <td className="text-right py-3 px-4 font-mono text-neon-green font-bold">
                        {result.speedup.toFixed(2)}x
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
