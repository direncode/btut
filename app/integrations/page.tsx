'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Radio, Cpu, Cloud, FlaskConical, ExternalLink, Download, Play, Settings, Terminal, Activity } from 'lucide-react'

export default function IntegrationsPage() {
  const [activeTab, setActiveTab] = useState<'ros' | 'sumo' | 'cloud' | 'research'>('ros')

  return (
    <div className="min-h-screen bg-black pt-20">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold gradient-text mb-2">Real-World Integrations</h1>
              <p className="text-gray-400">Connect BTUT to robotics, traffic systems, cloud platforms, and research pipelines</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Integration Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {/* ROS Integration Card */}
          <Link href="/robotics">
            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8 hover:border-neon-blue transition-all cursor-pointer group">
              <div className="flex items-start justify-between mb-4">
                <div className="p-3 bg-neon-blue/10 rounded-lg">
                  <Radio className="w-8 h-8 text-neon-blue" />
                </div>
                <ExternalLink className="w-5 h-5 text-gray-600 group-hover:text-neon-blue transition" />
              </div>

              <h2 className="text-2xl font-bold text-white mb-2">ROS Integration</h2>
              <p className="text-gray-400 mb-4">
                Connect to Robot Operating System for swarm coordination, multi-robot planning, and distributed control
              </p>

              <div className="space-y-2 mb-6">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-blue rounded-full"></div>
                  <span>Real-time robot coordination</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-blue rounded-full"></div>
                  <span>Strategy assignment via topics/services</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-blue rounded-full"></div>
                  <span>Live visualization and monitoring</span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">ROS Noetic</span>
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">rosbridge_suite</span>
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">WebSocket</span>
              </div>

              <div className="mt-6 pt-4 border-t border-gray-800">
                <button className="w-full px-4 py-2 bg-neon-blue/10 hover:bg-neon-blue/20 border border-neon-blue/50 text-neon-blue rounded-lg transition font-semibold">
                  Open Robotics Dashboard →
                </button>
              </div>
            </div>
          </Link>

          {/* SUMO Integration Card */}
          <Link href="/traffic">
            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8 hover:border-neon-green transition-all cursor-pointer group">
              <div className="flex items-start justify-between mb-4">
                <div className="p-3 bg-neon-green/10 rounded-lg">
                  <Cpu className="w-8 h-8 text-neon-green" />
                </div>
                <ExternalLink className="w-5 h-5 text-gray-600 group-hover:text-neon-green transition" />
              </div>

              <h2 className="text-2xl font-bold text-white mb-2">SUMO Traffic Integration</h2>
              <p className="text-gray-400 mb-4">
                Optimize traffic flow with BTUT-coordinated vehicles. Reduce congestion, improve throughput, save fuel
              </p>

              <div className="space-y-2 mb-6">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-green rounded-full"></div>
                  <span>Before/after traffic comparison</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-green rounded-full"></div>
                  <span>Real-time congestion heatmaps</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-green rounded-full"></div>
                  <span>Automated performance reports</span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">SUMO 1.12+</span>
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">TraCI</span>
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">Python</span>
              </div>

              <div className="mt-6 pt-4 border-t border-gray-800">
                <button className="w-full px-4 py-2 bg-neon-green/10 hover:bg-neon-green/20 border border-neon-green/50 text-neon-green rounded-lg transition font-semibold">
                  Open Traffic Dashboard →
                </button>
              </div>
            </div>
          </Link>

          {/* Cloud Deployment Card */}
          <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8 hover:border-neon-purple transition-all group">
            <div className="flex items-start justify-between mb-4">
              <div className="p-3 bg-neon-purple/10 rounded-lg">
                <Cloud className="w-8 h-8 text-neon-purple" />
              </div>
              <Settings className="w-5 h-5 text-gray-600 group-hover:text-neon-purple transition" />
            </div>

            <h2 className="text-2xl font-bold text-white mb-2">Cloud Deployment</h2>
            <p className="text-gray-400 mb-4">
              Deploy BTUT as serverless functions on AWS Lambda, Google Cloud Functions, or Azure Functions
            </p>

            <div className="space-y-2 mb-6">
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-neon-purple rounded-full"></div>
                <span>One-click Lambda deployment</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-neon-purple rounded-full"></div>
                <span>Auto-scaling up to 10M+ agents</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-neon-purple rounded-full"></div>
                <span>Real-time cost tracking</span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">AWS Lambda</span>
              <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">$0.20 / 100K</span>
            </div>

            <div className="mt-6 pt-4 border-t border-gray-800">
              <button
                className="w-full px-4 py-2 bg-neon-purple/10 hover:bg-neon-purple/20 border border-neon-purple/50 text-neon-purple rounded-lg transition font-semibold"
                onClick={() => setActiveTab('cloud')}
              >
                Configure Cloud Deployment
              </button>
            </div>
          </div>

          {/* Research Pipeline Card */}
          <Link href="/research">
            <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8 hover:border-neon-pink transition-all cursor-pointer group">
              <div className="flex items-start justify-between mb-4">
                <div className="p-3 bg-neon-pink/10 rounded-lg">
                  <FlaskConical className="w-8 h-8 text-neon-pink" />
                </div>
                <ExternalLink className="w-5 h-5 text-gray-600 group-hover:text-neon-pink transition" />
              </div>

              <h2 className="text-2xl font-bold text-white mb-2">Research Workbench</h2>
              <p className="text-gray-400 mb-4">
                Run parameter sweeps, generate publication figures, export to LaTeX, connect to Jupyter notebooks
              </p>

              <div className="space-y-2 mb-6">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-pink rounded-full"></div>
                  <span>Automated parameter sweeps</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-pink rounded-full"></div>
                  <span>Publication-ready figures</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-neon-pink rounded-full"></div>
                  <span>BibTeX citation generator</span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">Python SDK</span>
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">Jupyter</span>
                <span className="px-3 py-1 bg-gray-800 rounded text-xs text-gray-400">LaTeX</span>
              </div>

              <div className="mt-6 pt-4 border-t border-gray-800">
                <button className="w-full px-4 py-2 bg-neon-pink/10 hover:bg-neon-pink/20 border border-neon-pink/50 text-neon-pink rounded-lg transition font-semibold">
                  Open Research Workbench →
                </button>
              </div>
            </div>
          </Link>
        </div>

        {/* Quick Start Guide */}
        <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 text-white">Quick Start Guide</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-neon-blue mb-4 flex items-center gap-2">
                <Radio className="w-5 h-5" />
                ROS Setup
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-neon-blue/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-neon-blue">1</span>
                  </div>
                  <div>
                    <p className="text-gray-300">Install rosbridge_suite:</p>
                    <code className="block mt-1 px-2 py-1 bg-black/50 rounded text-xs text-gray-400">
                      sudo apt install ros-noetic-rosbridge-suite
                    </code>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-neon-blue/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-neon-blue">2</span>
                  </div>
                  <div>
                    <p className="text-gray-300">Launch rosbridge server:</p>
                    <code className="block mt-1 px-2 py-1 bg-black/50 rounded text-xs text-gray-400">
                      roslaunch rosbridge_server rosbridge_websocket.launch
                    </code>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-neon-blue/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-neon-blue">3</span>
                  </div>
                  <div>
                    <p className="text-gray-300">Connect from BTUT dashboard</p>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-neon-green mb-4 flex items-center gap-2">
                <Cpu className="w-5 h-5" />
                SUMO Setup
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-neon-green/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-neon-green">1</span>
                  </div>
                  <div>
                    <p className="text-gray-300">Install SUMO:</p>
                    <code className="block mt-1 px-2 py-1 bg-black/50 rounded text-xs text-gray-400">
                      sudo apt install sumo sumo-tools
                    </code>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-neon-green/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-neon-green">2</span>
                  </div>
                  <div>
                    <p className="text-gray-300">Install Python integration:</p>
                    <code className="block mt-1 px-2 py-1 bg-black/50 rounded text-xs text-gray-400">
                      pip install traci
                    </code>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-neon-green/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-neon-green">3</span>
                  </div>
                  <div>
                    <p className="text-gray-300">Load your network file in dashboard</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-800">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-white mb-1">Download ROS Package</h3>
                <p className="text-sm text-gray-400">Complete ROS package with messages, services, and nodes</p>
              </div>
              <a
                href="/integrations/ros/btut_ros_package.zip"
                className="px-6 py-3 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition flex items-center gap-2"
              >
                <Download className="w-5 h-5" />
                Download Package
              </a>
            </div>
          </div>
        </div>

        {/* Documentation Links */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <a href="/docs/integrations/ros" className="p-6 bg-gray-900/50 border border-gray-800 rounded-xl hover:border-neon-blue transition">
            <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
              <Terminal className="w-4 h-4 text-neon-blue" />
              ROS Documentation
            </h3>
            <p className="text-sm text-gray-400">Complete guide for ROS integration, messages, and services</p>
          </a>

          <a href="/docs/integrations/sumo" className="p-6 bg-gray-900/50 border border-gray-800 rounded-xl hover:border-neon-green transition">
            <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
              <Activity className="w-4 h-4 text-neon-green" />
              SUMO Documentation
            </h3>
            <p className="text-sm text-gray-400">Traffic simulation setup, network files, and TraCI protocol</p>
          </a>

          <a href="/docs/integrations/cloud" className="p-6 bg-gray-900/50 border border-gray-800 rounded-xl hover:border-neon-purple transition">
            <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
              <Cloud className="w-4 h-4 text-neon-purple" />
              Cloud Deployment Guide
            </h3>
            <p className="text-sm text-gray-400">Deploy to AWS Lambda, GCP, or Azure with serverless framework</p>
          </a>
        </div>
      </div>
    </div>
  )
}
