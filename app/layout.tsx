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
  manifest: '/manifest.json',
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
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#3b82f6" />
        <link rel="apple-touch-icon" href="/icon-192.png" />
      </head>
      <body className={inter.className}>
        <Navigation />
        <main className="min-h-screen">
          {children}
        </main>
        <Footer />
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', () => {
                  navigator.serviceWorker.register('/sw.js').then(
                    (registration) => {
                      console.log('Service Worker registered:', registration.scope);
                    },
                    (error) => {
                      console.error('Service Worker registration failed:', error);
                    }
                  );
                });
              }
            `,
          }}
        />
      </body>
    </html>
  )
}
