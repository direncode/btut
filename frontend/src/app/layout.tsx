import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'BTUT Market Simulator',
  description: 'Planetary-scale multi-agent financial market simulation using Bivariate Trajectory-Undercurrent Theory',
  keywords: ['market simulation', 'BTUT', 'multi-agent', 'finance', 'trading'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen gradient-bg">
        {children}
      </body>
    </html>
  )
}
