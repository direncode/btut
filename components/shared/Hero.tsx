import Link from 'next/link'
import { ArrowRight, Play, Github } from 'lucide-react'

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden grid-background pt-20">
      {/* Gradient Overlays */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-black/50 to-black"></div>
      
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute w-96 h-96 bg-neon-blue/10 rounded-full blur-3xl -top-48 -left-48 animate-float"></div>
        <div className="absolute w-96 h-96 bg-neon-green/10 rounded-full blur-3xl -bottom-48 -right-48 animate-float" style={{ animationDelay: '2s' }}></div>
        <div className="absolute w-96 h-96 bg-neon-purple/10 rounded-full blur-3xl top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 animate-float" style={{ animationDelay: '4s' }}></div>
      </div>
      
      <div className="relative max-w-6xl mx-auto px-6 text-center z-10">
        {/* Badge */}
        <div className="mb-8 inline-block animate-pulse">
          <span className="px-4 py-2 rounded-full bg-neon-blue/10 border border-neon-blue/30 text-neon-blue text-sm font-semibold">
            🚀 Addressing DARPA Mathematical Challenge 13
          </span>
        </div>
        
        {/* Main Headline */}
        <h1 className="text-6xl md:text-8xl font-black mb-6 leading-tight">
          <span className="gradient-text glow-text">Multi-Agent Simulation</span>
          <br />
          <span className="gradient-text glow-text">At Massive Scale</span>
        </h1>
        
        {/* Subheadline */}
        <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto">
          Build, simulate, and deploy <strong className="text-neon-green">coordination systems</strong> with 
          <strong className="text-neon-blue"> O(N) complexity</strong>. 
          No PDEs. Real-time convergence. <strong className="text-neon-purple">1M+ agents</strong>.
        </p>
        
        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
          <Link 
            href="/simulator"
            className="group px-8 py-4 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition transform hover:scale-105 flex items-center justify-center"
          >
            <Play className="w-5 h-5 mr-2 group-hover:animate-pulse" />
            Launch Simulator
            <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition" />
          </Link>
          <a 
            href="https://github.com"
            className="px-8 py-4 bg-transparent border-2 border-neon-blue text-neon-blue font-bold rounded-lg hover:bg-neon-blue/10 transition flex items-center justify-center"
          >
            <Github className="w-5 h-5 mr-2" />
            View on GitHub
          </a>
        </div>
        
        {/* Key Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
          <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6 card-glow">
            <div className="text-4xl font-bold text-neon-blue mb-2">O(N)</div>
            <div className="text-sm text-gray-400">Linear Complexity</div>
          </div>
          <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6 card-glow" style={{ animationDelay: '0.1s' }}>
            <div className="text-4xl font-bold text-neon-green mb-2">1.000</div>
            <div className="text-sm text-gray-400">Equilibrium Lock</div>
          </div>
          <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6 card-glow" style={{ animationDelay: '0.2s' }}>
            <div className="text-4xl font-bold text-neon-purple mb-2">&lt;1s</div>
            <div className="text-sm text-gray-400">Convergence Time</div>
          </div>
          <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-6 card-glow" style={{ animationDelay: '0.3s' }}>
            <div className="text-4xl font-bold text-neon-pink mb-2">1M+</div>
            <div className="text-sm text-gray-400">Agent Capacity</div>
          </div>
        </div>
      </div>
      
      {/* Scroll Indicator */}
      <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 animate-bounce">
        <svg className="w-8 h-8 text-neon-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
        </svg>
      </div>
    </section>
  )
}
