import Link from 'next/link'
import { Play } from 'lucide-react'

export default function LiveDemo() {
  return (
    <section className="py-24 bg-gray-900">
      <div className="max-w-6xl mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-5xl font-bold mb-4 gradient-text">See It In Action</h2>
          <p className="text-xl text-gray-400">Watch real-time convergence with live parameter tuning</p>
        </div>
        
        <div className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8 relative overflow-hidden">
          {/* Preview Video/Image Placeholder */}
          <div className="aspect-video bg-gradient-to-br from-gray-900 to-black rounded-lg flex items-center justify-center relative">
            <div className="absolute inset-0 grid-background opacity-30"></div>
            <div className="relative text-center z-10">
              <div className="text-8xl mb-6 animate-pulse">⚡</div>
              <h3 className="text-3xl font-bold mb-4">Live Simulation Engine</h3>
              <p className="text-gray-400 mb-8 max-w-md mx-auto">
                Interactive controls • Real-time graphs • Network visualization
              </p>
              <Link 
                href="/simulator"
                className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition transform hover:scale-105"
              >
                <Play className="w-5 h-5 mr-2" />
                Launch Full Simulator
              </Link>
            </div>
          </div>
          
          {/* Stats Strip */}
          <div className="grid grid-cols-3 gap-4 mt-8">
            <div className="bg-black/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-neon-blue">500K</div>
              <div className="text-xs text-gray-400 mt-1">Agents in real-time</div>
            </div>
            <div className="bg-black/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-neon-green">20ms</div>
              <div className="text-xs text-gray-400 mt-1">Per iteration</div>
            </div>
            <div className="bg-black/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-neon-purple">~0</div>
              <div className="text-xs text-gray-400 mt-1">Variance</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
