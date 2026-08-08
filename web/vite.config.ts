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
      // Fixed local-dev ports: strictPort fails fast instead of silently
      // moving, so an occupied port is reported clearly. 127.0.0.1 keeps the
      // dashboard on the IPv4 loopback, matching the Bento bind host.
      host: '127.0.0.1',
      port: 5173,
      strictPort: true,
      proxy: {
        // Strip /api — Bento mounts routes at the root, not under /api.
        // 127.0.0.1 matches the Bento bind host in scripts/dev.sh.
        '/api': {
          target: env.VITE_BENTO_URL || 'http://127.0.0.1:3000',
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
