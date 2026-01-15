/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  experimental: {
    optimizePackageImports: ['lucide-react', 'recharts'],
  },
  // Disable source maps in production to reduce build time
  productionBrowserSourceMaps: false,
}

module.exports = nextConfig
