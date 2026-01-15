'use client'

import Link from 'next/link'
import { useState, useRef, useEffect } from 'react'
import { Menu, X, Zap, ChevronDown } from 'lucide-react'
import ThemeToggle from '@/app/components/ThemeToggle'

export default function Navigation() {
  const [isOpen, setIsOpen] = useState(false)
  const [integrationsOpen, setIntegrationsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIntegrationsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const navItems = [
    { label: 'Home', href: '/' },
    { label: 'Simulator', href: '/simulator' },
    { label: 'Playground', href: '/playground' },
    { label: 'Benchmark', href: '/benchmark' },
  ]

  const integrationItems = [
    { label: 'Integration Hub', href: '/integrations' },
    { label: 'Robotics (ROS)', href: '/robotics' },
    { label: 'Traffic (SUMO)', href: '/traffic' },
    { label: 'Research Workbench', href: '/research' },
  ]

  return (
    <nav className="fixed top-0 w-full bg-black/80 backdrop-blur-lg border-b border-gray-800 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2 group">
            <Zap className="w-8 h-8 text-neon-blue group-hover:text-neon-green transition" />
            <span className="text-2xl font-bold gradient-text">BTUT Platform</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-gray-300 hover:text-neon-blue transition font-medium"
              >
                {item.label}
              </Link>
            ))}

            {/* Integrations Dropdown */}
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setIntegrationsOpen(!integrationsOpen)}
                className="flex items-center gap-1 text-gray-300 hover:text-neon-blue transition font-medium"
              >
                Integrations
                <ChevronDown className={`w-4 h-4 transition-transform ${integrationsOpen ? 'rotate-180' : ''}`} />
              </button>

              {integrationsOpen && (
                <div className="absolute top-full mt-2 right-0 w-56 bg-gray-900 border border-gray-800 rounded-lg shadow-xl overflow-hidden">
                  {integrationItems.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setIntegrationsOpen(false)}
                      className="block px-4 py-3 text-gray-300 hover:bg-gray-800 hover:text-neon-blue transition"
                    >
                      {item.label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* CTA Button + Theme Toggle */}
          <div className="hidden md:flex items-center space-x-4">
            <ThemeToggle />
            <Link
              href="/simulator"
              className="px-6 py-2 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-blue/50 transition"
            >
              Launch Simulator
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-white"
          >
            {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden mt-4 pb-4 space-y-4">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsOpen(false)}
                className="block text-gray-300 hover:text-neon-blue transition"
              >
                {item.label}
              </Link>
            ))}

            {/* Mobile Integrations */}
            <div className="border-t border-gray-800 pt-4">
              <div className="text-xs text-gray-500 mb-2 px-2">INTEGRATIONS</div>
              {integrationItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setIsOpen(false)}
                  className="block text-gray-300 hover:text-neon-blue transition py-2 px-2"
                >
                  {item.label}
                </Link>
              ))}
            </div>

            <div className="flex items-center justify-between pt-4">
              <ThemeToggle />
              <Link
                href="/simulator"
                onClick={() => setIsOpen(false)}
                className="px-6 py-2 bg-gradient-to-r from-neon-blue to-neon-green text-black font-bold rounded-lg text-center"
              >
                Launch Simulator
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}
