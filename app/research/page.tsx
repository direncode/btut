'use client'

import { useState } from 'react'
import { Beaker, Play, FileText, TrendingUp } from 'lucide-react'
import ExperimentConfig from '@/components/research/ExperimentConfig'

export default function ResearchPage() {
  const [experimentResults, setExperimentResults] = useState<any[]>([])

  const handleRunExperiment = (config: any) => {
    console.log('Running experiments with config:', config)
    // This would trigger the actual experiment runner
    alert(`Starting ${config.totalExperiments} experiments. This will take approximately ${Math.ceil(config.estimatedTime / 60)} minutes.`)
  }

  return (
    <div className="min-h-screen bg-black pt-20">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold gradient-text mb-2">Research Workbench</h1>
              <p className="text-gray-400">Parameter sweeps, experiments, and analysis tools</p>
            </div>

            <div className="flex items-center gap-2">
              <Beaker className="w-8 h-8 text-neon-purple" />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Experiment Configuration */}
          <div className="lg:col-span-1">
            <ExperimentConfig onRun={handleRunExperiment} />
          </div>

          {/* Center Column - Results Viewer */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-neon-green" />
                  Results Viewer
                </h3>
              </div>

              {experimentResults.length === 0 ? (
                <div className="text-center py-12">
                  <Play className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                  <p className="text-gray-500 text-sm">No experiments run yet</p>
                  <p className="text-gray-600 text-xs mt-1">Configure and run experiments to see results</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {experimentResults.map((result, idx) => (
                    <div key={idx} className="p-3 bg-gray-800/50 border border-gray-700 rounded-lg">
                      <div className="text-sm text-white font-semibold mb-1">Experiment {idx + 1}</div>
                      <div className="text-xs text-gray-400">
                        Convergence: {result.convergence}% | Time: {result.time}s
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Analysis Tools */}
            <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-bold text-white mb-4">Analysis Tools</h3>
              <div className="space-y-2">
                <button className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition text-sm">
                  Statistical Summary
                </button>
                <button className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition text-sm">
                  Convergence Analysis
                </button>
                <button className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition text-sm">
                  Parameter Sensitivity
                </button>
                <button className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition text-sm">
                  Compare Configurations
                </button>
              </div>
            </div>
          </div>

          {/* Right Column - Report Generator */}
          <div className="lg:col-span-1">
            <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <FileText className="w-5 h-5 text-neon-pink" />
                  Publication Tools
                </h3>
              </div>

              {/* Figure Generator */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Figure Generator</h4>
                <div className="space-y-2">
                  <button className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition text-sm">
                    Convergence Plots
                  </button>
                  <button className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition text-sm">
                    Parameter Heatmaps
                  </button>
                  <button className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition text-sm">
                    Phase Diagrams
                  </button>
                </div>
              </div>

              {/* Export Options */}
              <div className="mb-6 p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
                <div className="text-xs text-gray-400 mb-2">Export Formats:</div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="text-xs text-gray-500 text-center py-2 bg-gray-900 rounded">PDF</div>
                  <div className="text-xs text-gray-500 text-center py-2 bg-gray-900 rounded">LaTeX</div>
                  <div className="text-xs text-gray-500 text-center py-2 bg-gray-900 rounded">CSV</div>
                  <div className="text-xs text-gray-500 text-center py-2 bg-gray-900 rounded">JSON</div>
                </div>
              </div>

              {/* Citation Helper */}
              <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Citation Helper</h4>
                <div className="p-3 bg-gray-800/50 border border-gray-700 rounded-lg">
                  <div className="text-xs text-gray-400 mb-2">BibTeX Entry:</div>
                  <code className="text-xs text-gray-500 font-mono block">
                    @article&#123;btut2024,<br />
                    &nbsp;&nbsp;title=&#123;BTUT Platform&#125;,<br />
                    &nbsp;&nbsp;author=&#123;...&#125;,<br />
                    &nbsp;&nbsp;year=&#123;2024&#125;<br />
                    &#125;
                  </code>
                </div>
              </div>

              {/* Generate Report Button */}
              <button className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-neon-purple to-neon-pink hover:shadow-lg hover:shadow-neon-purple/50 text-black font-bold rounded-lg transition">
                Generate Research Report
              </button>
            </div>
          </div>
        </div>

        {/* Documentation Section */}
        <div className="mt-8 p-6 bg-gray-900/50 border border-gray-800 rounded-xl">
          <h3 className="text-lg font-bold text-white mb-2">Research Resources</h3>
          <p className="text-sm text-gray-400 mb-4">
            The research workbench provides tools for systematic parameter exploration, statistical analysis, and publication-ready figure generation.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
              <h4 className="font-semibold text-white mb-2 text-sm">Parameter Sweeps</h4>
              <p className="text-xs text-gray-400">
                Automatically run experiments across parameter ranges to identify optimal configurations
              </p>
            </div>
            <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
              <h4 className="font-semibold text-white mb-2 text-sm">Statistical Analysis</h4>
              <p className="text-xs text-gray-400">
                Built-in tools for convergence analysis, sensitivity testing, and significance testing
              </p>
            </div>
            <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
              <h4 className="font-semibold text-white mb-2 text-sm">Publication Tools</h4>
              <p className="text-xs text-gray-400">
                Export publication-ready figures in multiple formats with automatic citation generation
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
