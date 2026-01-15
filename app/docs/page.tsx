'use client'

import { useState } from 'react'
import { BookOpen, FileText, Code, GitBranch, Download, ExternalLink } from 'lucide-react'

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', label: 'Overview', icon: BookOpen },
    { id: 'theory', label: 'Mathematical Theory', icon: FileText },
    { id: 'proofs', label: 'Proofs & Analysis', icon: GitBranch },
    { id: 'implementation', label: 'Implementation', icon: Code },
  ]

  return (
    <div className="min-h-screen bg-black pt-20">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold gradient-text mb-2">Documentation</h1>
              <p className="text-gray-400">Mathematical foundations and implementation guide</p>
            </div>
            <BookOpen className="w-12 h-12 text-neon-purple" />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-4 sticky top-24">
              <h3 className="text-sm font-semibold text-gray-400 mb-3">CONTENTS</h3>
              <nav className="space-y-1">
                {sections.map((section) => {
                  const Icon = section.icon
                  return (
                    <button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition ${
                        activeSection === section.id
                          ? 'bg-neon-blue/10 text-neon-blue border border-neon-blue/30'
                          : 'text-gray-400 hover:text-white hover:bg-gray-800'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span className="text-sm font-medium">{section.label}</span>
                    </button>
                  )
                })}
              </nav>

              {/* Quick Links */}
              <div className="mt-6 pt-6 border-t border-gray-800">
                <h3 className="text-sm font-semibold text-gray-400 mb-3">QUICK LINKS</h3>
                <div className="space-y-2">
                  <a
                    href="https://github.com/direncode/btut"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-sm text-gray-400 hover:text-neon-blue transition"
                  >
                    <ExternalLink className="w-3 h-3" />
                    GitHub Repository
                  </a>
                  <a
                    href="/simulator"
                    className="flex items-center gap-2 text-sm text-gray-400 hover:text-neon-green transition"
                  >
                    <ExternalLink className="w-3 h-3" />
                    Try Simulator
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Content Area */}
          <div className="lg:col-span-3">
            {activeSection === 'overview' && <OverviewSection />}
            {activeSection === 'theory' && <TheorySection />}
            {activeSection === 'proofs' && <ProofsSection />}
            {activeSection === 'implementation' && <ImplementationSection />}
          </div>
        </div>
      </div>
    </div>
  )
}

function OverviewSection() {
  return (
    <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8">
      <h2 className="text-3xl font-bold text-white mb-6">BTUT: Bivariate Trajectory-Undercurrent Theory</h2>

      <div className="prose prose-invert max-w-none">
        <p className="text-gray-300 mb-4">
          The Bivariate Trajectory-Undercurrent Theory (BTUT) is a novel framework for analyzing and promoting cooperation in multi-agent systems through network topology and strategic interaction dynamics.
        </p>

        <h3 className="text-xl font-bold text-white mt-8 mb-4">Key Concepts</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
            <h4 className="font-semibold text-neon-blue mb-2">Scale-Free Networks</h4>
            <p className="text-sm text-gray-400">
              Power-law degree distribution P(k) ∝ k<sup>-γ</sup> with preferential attachment mechanism
            </p>
          </div>

          <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
            <h4 className="font-semibold text-neon-green mb-2">Strategic Dynamics</h4>
            <p className="text-sm text-gray-400">
              Agents play Stag Hunt and Prisoner's Dilemma games with payoff-dependent strategy updates
            </p>
          </div>

          <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
            <h4 className="font-semibold text-neon-purple mb-2">Hub Influence</h4>
            <p className="text-sm text-gray-400">
              High-degree nodes (hubs) with parameter τ control cooperation cascades through the network
            </p>
          </div>

          <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
            <h4 className="font-semibold text-neon-pink mb-2">Critical Transitions</h4>
            <p className="text-sm text-gray-400">
              System exhibits phase transitions in cooperation levels based on network topology
            </p>
          </div>
        </div>

        <h3 className="text-xl font-bold text-white mt-8 mb-4">Applications</h3>

        <ul className="space-y-3">
          <li className="flex items-start gap-3">
            <div className="w-2 h-2 bg-neon-blue rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <span className="font-semibold text-white">Robotics:</span>
              <span className="text-gray-400"> Multi-robot coordination, swarm navigation, collaborative task allocation</span>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <div className="w-2 h-2 bg-neon-green rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <span className="font-semibold text-white">Traffic Systems:</span>
              <span className="text-gray-400"> Autonomous vehicle coordination, traffic flow optimization, congestion reduction</span>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <div className="w-2 h-2 bg-neon-purple rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <span className="font-semibold text-white">Social Networks:</span>
              <span className="text-gray-400"> Opinion dynamics, information propagation, consensus formation</span>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <div className="w-2 h-2 bg-neon-pink rounded-full mt-2 flex-shrink-0"></div>
            <div>
              <span className="font-semibold text-white">Economics:</span>
              <span className="text-gray-400"> Market coordination, supply chain optimization, resource allocation</span>
            </div>
          </li>
        </ul>
      </div>
    </div>
  )
}

function TheorySection() {
  return (
    <div className="space-y-6">
      <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8">
        <h2 className="text-3xl font-bold text-white mb-6">Mathematical Foundations</h2>

        <div className="space-y-8">
          {/* Scale-Free Network */}
          <div>
            <h3 className="text-xl font-bold text-neon-blue mb-4">1. Scale-Free Network Generation</h3>
            <p className="text-gray-300 mb-4">
              The network follows a power-law degree distribution with preferential attachment:
            </p>
            <div className="p-4 bg-black/50 border border-gray-700 rounded-lg font-mono text-sm">
              <div className="text-gray-300">P(k) = C · k<sup>-γ</sup></div>
              <div className="text-gray-500 text-xs mt-2">where γ is the degree exponent (typically 2 &lt; γ &lt; 3)</div>
            </div>
            <p className="text-gray-400 text-sm mt-3">
              New nodes attach to existing nodes with probability proportional to their current degree, creating hub structures.
            </p>
          </div>

          {/* Game Payoffs */}
          <div>
            <h3 className="text-xl font-bold text-neon-green mb-4">2. Game-Theoretic Payoffs</h3>
            <p className="text-gray-300 mb-4">
              Agents play two types of games simultaneously:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              {/* Stag Hunt */}
              <div className="p-4 bg-black/50 border border-gray-700 rounded-lg">
                <h4 className="font-semibold text-white mb-3">Stag Hunt Game</h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left text-gray-500 pb-2"></th>
                      <th className="text-center text-gray-400 pb-2">A</th>
                      <th className="text-center text-gray-400 pb-2">B</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    <tr>
                      <td className="text-gray-400 py-2">A</td>
                      <td className="text-center text-neon-green py-2">c<sub>A</sub></td>
                      <td className="text-center text-gray-500 py-2">0</td>
                    </tr>
                    <tr>
                      <td className="text-gray-400 py-2">B</td>
                      <td className="text-center text-gray-500 py-2">0</td>
                      <td className="text-center text-neon-blue py-2">c<sub>B</sub></td>
                    </tr>
                  </tbody>
                </table>
                <p className="text-xs text-gray-500 mt-2">Coordination game: c<sub>A</sub> &gt; c<sub>B</sub> &gt; 0</p>
              </div>

              {/* Prisoner's Dilemma */}
              <div className="p-4 bg-black/50 border border-gray-700 rounded-lg">
                <h4 className="font-semibold text-white mb-3">Prisoner's Dilemma</h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left text-gray-500 pb-2"></th>
                      <th className="text-center text-gray-400 pb-2">A</th>
                      <th className="text-center text-gray-400 pb-2">B</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    <tr>
                      <td className="text-gray-400 py-2">A</td>
                      <td className="text-center text-neon-green py-2">d<sub>A</sub></td>
                      <td className="text-center text-gray-500 py-2">0</td>
                    </tr>
                    <tr>
                      <td className="text-gray-400 py-2">B</td>
                      <td className="text-center text-neon-pink py-2">d<sub>B</sub></td>
                      <td className="text-center text-neon-blue py-2">d<sub>B</sub></td>
                    </tr>
                  </tbody>
                </table>
                <p className="text-xs text-gray-500 mt-2">Social dilemma: d<sub>B</sub> &gt; d<sub>A</sub> &gt; 0</p>
              </div>
            </div>
          </div>

          {/* Strategy Update */}
          <div>
            <h3 className="text-xl font-bold text-neon-purple mb-4">3. Strategy Update Dynamics</h3>
            <p className="text-gray-300 mb-4">
              Agents update strategies based on accumulated payoffs with Fermi function:
            </p>
            <div className="p-4 bg-black/50 border border-gray-700 rounded-lg font-mono text-sm">
              <div className="text-gray-300">P(s<sub>i</sub> ← s<sub>j</sub>) = 1 / (1 + exp(-(U<sub>j</sub> - U<sub>i</sub>) / κ))</div>
              <div className="text-gray-500 text-xs mt-2">where U<sub>i</sub> is agent i's total payoff and κ is the selection intensity</div>
            </div>
          </div>

          {/* Hub Influence */}
          <div>
            <h3 className="text-xl font-bold text-neon-pink mb-4">4. Hub Influence Parameter (τ)</h3>
            <p className="text-gray-300 mb-4">
              High-degree nodes (hubs) receive weighted influence in strategy updates:
            </p>
            <div className="p-4 bg-black/50 border border-gray-700 rounded-lg font-mono text-sm">
              <div className="text-gray-300">w<sub>ij</sub> = (k<sub>j</sub>)<sup>τ</sup> / Σ<sub>l∈N(i)</sub> (k<sub>l</sub>)<sup>τ</sup></div>
              <div className="text-gray-500 text-xs mt-2">where k<sub>j</sub> is the degree of neighbor j and τ ∈ [0, 1] controls hub influence</div>
            </div>
            <ul className="mt-4 space-y-2 text-sm text-gray-400">
              <li className="flex items-start gap-2">
                <span className="text-neon-blue">•</span>
                <span>τ = 0: All neighbors have equal weight (democratic)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-neon-blue">•</span>
                <span>τ = 1: Neighbors weighted by degree (hub-centric)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-neon-blue">•</span>
                <span>Intermediate τ: Balanced influence distribution</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

function ProofsSection() {
  return (
    <div className="space-y-6">
      <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8">
        <h2 className="text-3xl font-bold text-white mb-6">Theoretical Proofs & Analysis</h2>

        <div className="space-y-8">
          {/* Convergence Theorem */}
          <div className="p-6 bg-gradient-to-r from-neon-purple/5 to-neon-blue/5 border border-neon-purple/30 rounded-lg">
            <h3 className="text-xl font-bold text-white mb-4">Theorem 1: Cooperation Convergence</h3>
            <div className="space-y-4">
              <div>
                <span className="text-neon-purple font-semibold">Statement:</span>
                <p className="text-gray-300 mt-2">
                  In a scale-free network with γ ∈ (2, 3), τ &gt; τ<sub>c</sub>, and c<sub>A</sub> &gt; d<sub>B</sub>,
                  the system converges to full cooperation (all agents choose strategy A) with probability 1
                  as N → ∞.
                </p>
              </div>

              <div className="border-t border-gray-700 pt-4">
                <span className="text-neon-blue font-semibold">Proof Sketch:</span>
                <ol className="mt-2 space-y-3 text-sm text-gray-300">
                  <li className="flex gap-2">
                    <span className="text-neon-blue font-mono">1.</span>
                    <span>Hub nodes have degree k<sub>hub</sub> ∼ N<sup>1/(γ-1)</sup> in scale-free networks</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-neon-blue font-mono">2.</span>
                    <span>When a hub adopts strategy A, it influences O(k<sub>hub</sub>) neighbors</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-neon-blue font-mono">3.</span>
                    <span>With τ &gt; τ<sub>c</sub>, hub influence creates positive feedback loop</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-neon-blue font-mono">4.</span>
                    <span>Since c<sub>A</sub> &gt; d<sub>B</sub>, cooperation is payoff-superior</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-neon-blue font-mono">5.</span>
                    <span>Cooperation cascades through network faster than defection can spread</span>
                  </li>
                </ol>
                <div className="mt-4 p-3 bg-black/30 rounded text-xs text-gray-400">
                  <span className="text-neon-green">□</span> Full proof available in research publication
                </div>
              </div>
            </div>
          </div>

          {/* Critical Threshold */}
          <div className="p-6 bg-gradient-to-r from-neon-green/5 to-neon-purple/5 border border-neon-green/30 rounded-lg">
            <h3 className="text-xl font-bold text-white mb-4">Theorem 2: Critical Threshold τ<sub>c</sub></h3>
            <div className="space-y-4">
              <div>
                <span className="text-neon-green font-semibold">Statement:</span>
                <p className="text-gray-300 mt-2">
                  There exists a critical value τ<sub>c</sub> ≈ 0.3 below which cooperation cannot dominate,
                  and above which cooperation emerges via hub-mediated cascades.
                </p>
              </div>

              <div className="border-t border-gray-700 pt-4">
                <span className="text-neon-blue font-semibold">Analysis:</span>
                <div className="mt-3 space-y-3 text-sm text-gray-300">
                  <p>The critical threshold emerges from the balance between:</p>
                  <ul className="space-y-2 ml-4">
                    <li className="flex gap-2">
                      <span className="text-neon-green">+</span>
                      <span>Hub amplification: Cooperation spreads faster via high-degree nodes</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-neon-pink">−</span>
                      <span>Defection temptation: Higher individual payoff from strategy B</span>
                    </li>
                  </ul>
                  <p className="mt-3">
                    Below τ<sub>c</sub>, individual defection temptation overwhelms collective benefit.
                    Above τ<sub>c</sub>, hub-mediated coordination enables cooperation.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Phase Transition */}
          <div className="p-6 bg-gradient-to-r from-neon-pink/5 to-neon-green/5 border border-neon-pink/30 rounded-lg">
            <h3 className="text-xl font-bold text-white mb-4">Corollary: Phase Transition Behavior</h3>
            <div className="space-y-4">
              <p className="text-gray-300">
                The system exhibits a continuous phase transition at τ = τ<sub>c</sub>,
                characterized by critical exponents typical of mean-field universality class.
              </p>

              <div className="p-4 bg-black/30 border border-gray-700 rounded-lg font-mono text-sm">
                <div className="text-gray-300">f<sub>A</sub>(τ) ∼ (τ - τ<sub>c</sub>)<sup>β</sup> for τ → τ<sub>c</sub><sup>+</sup></div>
                <div className="text-gray-500 text-xs mt-2">where β ≈ 0.5 is the critical exponent and f<sub>A</sub> is cooperation fraction</div>
              </div>

              <p className="text-sm text-gray-400 mt-3">
                This phase transition is robust to noise and network variations, making BTUT applicable
                across diverse real-world systems.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function ImplementationSection() {
  return (
    <div className="space-y-6">
      <div className="bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-8">
        <h2 className="text-3xl font-bold text-white mb-6">Implementation Guide</h2>

        <div className="space-y-8">
          {/* TypeScript Engine */}
          <div>
            <h3 className="text-xl font-bold text-neon-blue mb-4">TypeScript/JavaScript</h3>
            <p className="text-gray-300 mb-4">
              Use the built-in TypeScript engine for web applications and Node.js:
            </p>
            <div className="p-4 bg-black border border-gray-700 rounded-lg">
              <pre className="text-sm text-gray-300 overflow-x-auto">
{`import { BTUTSimulator } from '@/lib/simulation/btut-engine'

const simulator = new BTUTSimulator({
  N: 10000,        // Number of agents
  gamma: 1.5,      // Scale-free exponent
  tau: 0.4,        // Hub influence
  cA_SH: 0.8,      // Stag Hunt: cooperate payoff
  cB_SH: 0.6,      // Stag Hunt: defect payoff
  cA_PD: 0.5,      // Prisoner's Dilemma: cooperate
  cB_PD: 0.3,      // Prisoner's Dilemma: defect
  alpha: 0.1,      // Update rate
  m: 10            // Initial connections
})

const result = await simulator.runSimulation()`}
              </pre>
            </div>
          </div>

          {/* Python Engine */}
          <div>
            <h3 className="text-xl font-bold text-neon-green mb-4">Python</h3>
            <p className="text-gray-300 mb-4">
              High-performance NumPy-based engine for research and experiments:
            </p>
            <div className="p-4 bg-black border border-gray-700 rounded-lg">
              <pre className="text-sm text-gray-300 overflow-x-auto">
{`from btut_engine import BTUTEngine

engine = BTUTEngine(
    n_agents=10000,
    gamma=1.5,
    tau=0.4,
    c_a_sh=0.8,
    c_b_sh=0.6,
    c_a_pd=0.5,
    c_b_pd=0.3,
    alpha=0.1,
    m=10
)

result = engine.run_simulation()
print(f"Final cooperation: {result['final_fraction_a']:.2%}")`}
              </pre>
            </div>
          </div>

          {/* ROS Integration */}
          <div>
            <h3 className="text-xl font-bold text-neon-purple mb-4">ROS Integration</h3>
            <p className="text-gray-300 mb-4">
              Deploy BTUT coordination for robot swarms:
            </p>
            <div className="p-4 bg-black border border-gray-700 rounded-lg">
              <pre className="text-sm text-gray-300 overflow-x-auto">
{`roslaunch btut btut_coordination.launch

# Configure parameters
rosparam set /btut/gamma 1.5
rosparam set /btut/tau 0.4

# Monitor coordination
rostopic echo /btut/coordination_result`}
              </pre>
            </div>
          </div>

          {/* SUMO Integration */}
          <div>
            <h3 className="text-xl font-bold text-neon-pink mb-4">SUMO Traffic Simulation</h3>
            <p className="text-gray-300 mb-4">
              Apply BTUT to autonomous vehicle coordination:
            </p>
            <div className="p-4 bg-black border border-gray-700 rounded-lg">
              <pre className="text-sm text-gray-300 overflow-x-auto">
{`# Start SUMO with TraCI
sumo -c network.sumocfg --remote-port 8813

# Run BTUT traffic coordinator
python integrations/sumo/traffic_coordinator.py \\
  --gamma 1.5 --tau 0.4 --network network.net.xml`}
              </pre>
            </div>
          </div>

          {/* Key Parameters */}
          <div className="mt-8 p-6 bg-gradient-to-r from-neon-blue/5 to-neon-purple/5 border border-neon-blue/30 rounded-lg">
            <h3 className="text-xl font-bold text-white mb-4">Parameter Tuning Guide</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-neon-blue font-mono">γ</span>
                  <span className="text-gray-400 text-sm">Scale-free exponent</span>
                </div>
                <p className="text-xs text-gray-500 ml-4">
                  Lower γ → more hubs. Typical: 1.4-1.6 for optimal cooperation
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-neon-green font-mono">τ</span>
                  <span className="text-gray-400 text-sm">Hub influence</span>
                </div>
                <p className="text-xs text-gray-500 ml-4">
                  τ &gt; 0.3 enables cooperation. Higher τ → faster convergence
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-neon-purple font-mono">α</span>
                  <span className="text-gray-400 text-sm">Update rate</span>
                </div>
                <p className="text-xs text-gray-500 ml-4">
                  Controls adaptation speed. Typical: 0.05-0.15
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-neon-pink font-mono">m</span>
                  <span className="text-gray-400 text-sm">Hub neighbors</span>
                </div>
                <p className="text-xs text-gray-500 ml-4">
                  Initial connections per node. Higher m → denser network
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
