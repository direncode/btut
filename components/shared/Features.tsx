import { Zap, Network, BarChart3, Code2, Shield, Rocket } from 'lucide-react'

export default function Features() {
  const features = [
    {
      icon: Zap,
      title: 'O(N) Complexity',
      description: 'Linear scaling proven empirically. Process millions of agents without exponential blowup.',
      color: 'text-neon-blue',
    },
    {
      icon: Network,
      title: 'Scale-Free Networks',
      description: 'Built-in support for Barabási-Albert topologies with hub-driven convergence dynamics.',
      color: 'text-neon-green',
    },
    {
      icon: BarChart3,
      title: 'Real-Time Analytics',
      description: 'Live convergence tracking, variance analysis, and phase transition detection.',
      color: 'text-neon-purple',
    },
    {
      icon: Code2,
      title: 'React SDK',
      description: 'Drop-in components for building custom multi-agent simulations in your app.',
      color: 'text-neon-pink',
    },
    {
      icon: Shield,
      title: 'Proven Theory',
      description: 'Kernel-weighted mean-field dynamics with mathematical convergence guarantees.',
      color: 'text-neon-yellow',
    },
    {
      icon: Rocket,
      title: 'Deploy Anywhere',
      description: 'Run in browser, Node.js, or cloud. Export simulations as JSON or Python.',
      color: 'text-neon-blue',
    },
  ]

  return (
    <section className="py-24 bg-gradient-to-b from-black to-gray-900">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold mb-4 gradient-text">Why BTUT?</h2>
          <p className="text-xl text-gray-400">The only simulation engine built for real-world scale</p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature) => {
            const Icon = feature.icon
            return (
              <div key={feature.title} className="card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8 hover:border-gray-700 transition">
                <div className={`w-12 h-12 ${feature.color.replace('text', 'bg')}/20 rounded-lg flex items-center justify-center mb-4`}>
                  <Icon className={`w-6 h-6 ${feature.color}`} />
                </div>
                <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
