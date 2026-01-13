import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navigation from '@/components/shared/Navigation'
import Footer from '@/components/shared/Footer'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'BTUT Platform - Multi-Agent Simulation Engine',
  description: 'Revolutionary PDE-free framework for scalable multi-agent differential games. Build, simulate, and deploy coordination systems at massive scale.',
  keywords: ['BTUT', 'game theory', 'multi-agent', 'simulation', 'AI', 'coordination', 'DARPA'],
  authors: [{ name: 'BTUT Team' }],
  openGraph: {
    title: 'BTUT Platform - Multi-Agent Simulation Engine',
    description: 'O(N) complexity. 1M+ agents. Real-time coordination.',
    type: 'website',
    url: 'https://btut.ai',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'BTUT Platform',
    description: 'Scalable multi-agent simulation engine',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth dark">
      <body className={inter.className}>
        <Navigation />
        <main className="min-h-screen">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}
