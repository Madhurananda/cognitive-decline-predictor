/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    unoptimized: true,
  },
  allowedDevOrigins: ['143.167.102.50', 'localhost', '127.0.0.1'],
};

module.exports = nextConfig;
