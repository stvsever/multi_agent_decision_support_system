import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// The built client is served by the FastAPI process from `dist/`, so asset
// paths stay relative to the app root. In dev, `/api` proxies to that process.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.COMPASS_API_URL ?? 'http://127.0.0.1:5005',
        changeOrigin: true,
        // Server-sent events must not be buffered by the proxy.
        configure: (proxy) => {
          proxy.on('proxyRes', (res) => {
            if (res.headers['content-type']?.includes('text/event-stream')) {
              res.headers['cache-control'] = 'no-cache'
            }
          })
        },
      },
    },
  },
  build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 1400 },
})
