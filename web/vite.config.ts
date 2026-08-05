import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Local dev only. The `/api` proxy forwards browser requests to the Bento
// service so the frontend never hard-codes a backend origin.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        // Strip /api — Bento mounts routes at the root, not under /api.
        '/api': {
          target: env.VITE_BENTO_URL || 'http://localhost:3000',
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
