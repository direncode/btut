'use client'

import { useState } from 'react'
import Link from 'next/link'

const pythonCode = `from btut import Simulator, Presets

# Quick test with 10K agents
sim = Simulator(Presets.STANDARD)
results = sim.run()

print(f"Cooperation: {results.cooperation_rate:.1%}")
print(f"Converged in {results.convergence_time} iterations")`

const apiCode = `curl -X POST https://api.btut.dev/simulate \\
  -H "Content-Type: application/json" \\
  -d '{
    "N": 10000,
    "gamma": 1.5,
    "tau": 0.3,
    "iterations": 25
  }'`

const typescriptCode = `import { BTUTSimulator } from '@btut/engine'

const sim = new BTUTSimulator({
  N: 10000,
  gamma: 1.5,
  tau: 0.3,
  iterations: 25
})

const result = sim.runComplete()
console.log(\`Cooperation: \${result.fractionA * 100}%\`)`

type CodeTab = 'python' | 'api' | 'typescript'

export function GetStarted() {
  const [activeTab, setActiveTab] = useState<CodeTab>('python')
  const [copied, setCopied] = useState(false)

  const codeMap: Record<CodeTab, string> = {
    python: pythonCode,
    api: apiCode,
    typescript: typescriptCode,
  }

  const copyToClipboard = () => {
    navigator.clipboard.writeText(codeMap[activeTab])
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <section className="py-16 bg-gray-950/50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="gradient-text">Get Started</span>
          </h2>
          <p className="text-gray-400">
            Three ways to use BTUT: Python SDK, REST API, or TypeScript.
          </p>
        </div>

        {/* Install Command */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 mb-8">
          <div className="flex items-center justify-between">
            <code className="text-neon-green font-mono text-lg">
              pip install btut
            </code>
            <button
              onClick={() => {
                navigator.clipboard.writeText('pip install btut')
                setCopied(true)
                setTimeout(() => setCopied(false), 2000)
              }}
              className="px-3 py-1 text-sm bg-gray-800 hover:bg-gray-700 rounded transition-colors"
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>

        {/* Code Tabs */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg overflow-hidden">
          {/* Tab Headers */}
          <div className="flex border-b border-gray-800">
            {(['python', 'api', 'typescript'] as CodeTab[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`
                  px-6 py-3 text-sm font-medium transition-colors
                  ${activeTab === tab
                    ? 'bg-gray-800 text-white border-b-2 border-neon-blue'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                  }
                `}
              >
                {tab === 'python' && 'Python'}
                {tab === 'api' && 'REST API'}
                {tab === 'typescript' && 'TypeScript'}
              </button>
            ))}
          </div>

          {/* Code Block */}
          <div className="relative">
            <pre className="p-6 overflow-x-auto text-sm">
              <code className="text-gray-300 font-mono whitespace-pre">
                {codeMap[activeTab]}
              </code>
            </pre>
            <button
              onClick={copyToClipboard}
              className="absolute top-4 right-4 px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded transition-colors"
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>

        {/* Links */}
        <div className="mt-12 grid md:grid-cols-4 gap-4">
          <Link
            href="https://github.com/direncode/btut"
            target="_blank"
            className="flex items-center justify-center gap-2 px-4 py-3 bg-gray-900/50 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            GitHub
          </Link>
          <Link
            href="/docs"
            className="flex items-center justify-center gap-2 px-4 py-3 bg-gray-900/50 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            Documentation
          </Link>
          <Link
            href="/simulator"
            className="flex items-center justify-center gap-2 px-4 py-3 bg-gray-900/50 border border-gray-800 rounded-lg hover:border-gray-600 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Simulator
          </Link>
          <Link
            href="/traffic"
            className="flex items-center justify-center gap-2 px-4 py-3 bg-neon-blue/20 border border-neon-blue/50 rounded-lg hover:bg-neon-blue/30 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
            SUMO Traffic
          </Link>
        </div>

        {/* Version Info */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <span>BTUT v1.0.0</span>
          <span className="mx-2">|</span>
          <span>MIT License</span>
          <span className="mx-2">|</span>
          <Link href="https://github.com/direncode/btut/releases" className="hover:text-gray-400">
            Release Notes
          </Link>
        </div>
      </div>
    </section>
  )
}
