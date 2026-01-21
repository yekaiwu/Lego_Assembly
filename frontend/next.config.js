/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/api/image/**',
      },
      {
        protocol: 'https',
        hostname: '*.up.railway.app',
        pathname: '/api/image/**',
      },
    ],
  },
}

module.exports = nextConfig




