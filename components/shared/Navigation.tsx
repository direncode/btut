'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Menu, X, Zap } from 'lucide-react'
import ThemeToggle from '@/app/components/ThemeToggle'

export default function Navigation() {
  const [isOpen, setIsOpen] = useState(false)

  const navItems = [
    { label: 'Home', href: '/' },
    { label: 'Simulator', href: '/simulator' },
    { label: 'Playground', href: '/playground' },
    { label: 'Benchmark', href: '/benchmark' },
    { label: 'Documentation', href: '/documentation' },
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
            <div className="flex items-center justify-between">
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
