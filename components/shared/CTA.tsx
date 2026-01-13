import Link from 'next/link'
import { ArrowRight, Sparkles } from 'lucide-react'

export default function CTA() {
  return (
    <section className="py-24 bg-black relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 grid-background opacity-20"></div>
      <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-black"></div>
      
      <div className="relative max-w-4xl mx-auto px-6 text-center">
        <div className="mb-8">
          <Sparkles className="w-16 h-16 text-neon-blue mx-auto animate-pulse" />
        </div>
        
        <h2 className="text-5xl md:text-6xl font-black mb-6 leading-tight">
          <span className="gradient-text">Ready to Scale?</span>
        </h2>
        
        <p className="text-xl text-gray-300 mb-12 max-w-2xl mx-auto">
          Join researchers and engineers building the next generation of 
          coordination systems with BTUT.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link 
            href="/simulator"
            className="group px-8 py-4 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition transform hover:scale-105 flex items-center justify-center"
          >
            Start Building Now
            <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition" />
          </Link>
          <Link 
            href="/documentation"
            className="px-8 py-4 bg-transparent border-2 border-gray-700 text-white font-bold rounded-lg hover:border-neon-blue hover:bg-neon-blue/10 transition flex items-center justify-center"
          >
            Read Documentation
          </Link>
        </div>
      </div>
    </section>
  )
}
