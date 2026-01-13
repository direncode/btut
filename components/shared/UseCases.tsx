import { Car, Plane, Building2, Bot, Dna, TrendingUp } from 'lucide-react'

export default function UseCases() {
  const useCases = [
    {
      icon: Car,
      title: 'Autonomous Fleet Coordination',
      description: 'Optimize routing, traffic flow, and collision avoidance for thousands of autonomous vehicles in real-time.',
      stats: '10K+ vehicles',
    },
    {
      icon: Plane,
      title: 'Drone Swarm Control',
      description: 'Coordinate massive drone formations for search & rescue, surveillance, or delivery operations.',
      stats: '5K+ drones',
    },
    {
      icon: Building2,
      title: 'Smart City Optimization',
      description: 'Model and optimize resource allocation, energy distribution, and infrastructure usage across urban networks.',
      stats: '1M+ agents',
    },
    {
      icon: Bot,
      title: 'Multi-Agent AI Training',
      description: 'Train cooperative AI agents at scale. Perfect for reinforcement learning and emergent behavior research.',
      stats: 'GPU accelerated',
    },
    {
      icon: Dna,
      title: 'Biological Systems',
      description: 'Simulate cellular interactions, immune responses, and population dynamics with high fidelity.',
      stats: '100K+ cells',
    },
    {
      icon: TrendingUp,
      title: 'Market Simulation',
      description: 'Model trading behavior, market dynamics, and economic coordination in complex financial systems.',
      stats: 'Real-time',
    },
  ]

  return (
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold mb-4 gradient-text">Built For Real Problems</h2>
          <p className="text-xl text-gray-400">From autonomous systems to AI research</p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {useCases.map((useCase) => {
            const Icon = useCase.icon
            return (
              <div key={useCase.title} className="group card-glow bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8 hover:border-neon-blue/50 transition">
                <div className="w-14 h-14 bg-neon-blue/10 rounded-lg flex items-center justify-center mb-4 group-hover:bg-neon-blue/20 transition">
                  <Icon className="w-7 h-7 text-neon-blue" />
                </div>
                <div className="mb-2 text-xs font-mono text-neon-green">{useCase.stats}</div>
                <h3 className="text-xl font-bold mb-3">{useCase.title}</h3>
                <p className="text-gray-400 text-sm">{useCase.description}</p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
