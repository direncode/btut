import Link from 'next/link'
import { Github, Twitter, Linkedin, Mail } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="bg-black border-t border-gray-800 py-12">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div>
            <h4 className="font-bold text-lg mb-4 gradient-text">BTUT Platform</h4>
            <p className="text-sm text-gray-400 mb-4">
              Next-generation multi-agent simulation engine for scalable coordination problems.
            </p>
            <div className="flex space-x-4">
              <a href="https://github.com" className="text-gray-400 hover:text-neon-blue transition">
                <Github className="w-5 h-5" />
              </a>
              <a href="https://twitter.com" className="text-gray-400 hover:text-neon-blue transition">
                <Twitter className="w-5 h-5" />
              </a>
              <a href="https://linkedin.com" className="text-gray-400 hover:text-neon-blue transition">
                <Linkedin className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Product */}
          <div>
            <h4 className="font-bold mb-4">Product</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><Link href="/simulator" className="hover:text-neon-blue transition">Simulator</Link></li>
              <li><Link href="/playground" className="hover:text-neon-blue transition">Playground</Link></li>
              <li><Link href="/documentation" className="hover:text-neon-blue transition">Documentation</Link></li>
              <li><Link href="/pricing" className="hover:text-neon-blue transition">Pricing</Link></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-bold mb-4">Resources</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><a href="#" className="hover:text-neon-green transition">API Reference</a></li>
              <li><a href="#" className="hover:text-neon-green transition">GitHub</a></li>
              <li><a href="#" className="hover:text-neon-green transition">Research Paper</a></li>
              <li><a href="#" className="hover:text-neon-green transition">Zenodo Archive</a></li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="font-bold mb-4">Company</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><a href="#" className="hover:text-neon-purple transition">About</a></li>
              <li><a href="#" className="hover:text-neon-purple transition">Blog</a></li>
              <li><a href="mailto:contact@btut.ai" className="hover:text-neon-purple transition">Contact</a></li>
              <li><a href="#" className="hover:text-neon-purple transition">Careers</a></li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-800 pt-8 flex flex-col md:flex-row justify-between items-center">
          <div className="text-sm text-gray-400 mb-4 md:mb-0">
            © 2025 BTUT Platform. Apache License 2.0.
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-400">Powered by</span>
            <a 
              href="https://x.ai" 
              target="_blank"
              className="flex items-center space-x-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition"
            >
              <span className="text-sm font-bold gradient-text">Grok-4</span>
            </a>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 text-center text-xs text-gray-500">
          <p>
            <strong>Disclaimer:</strong> BTUT is research software under active development. 
            Results shown are computational demonstrations. Formal peer review pending.
          </p>
        </div>
      </div>
    </footer>
  )
}
